"""LLM judge arms through any OpenAI-compatible server (vLLM, llama.cpp server, SGLang, Ollama).

Three output formats share the same query, candidate fields, semantic labels and backbone (plan §6):

  rich          JSON with decision, confidence, evidence and rubric fields (public product/citation rubric)
  minimal_json  {"decision": -1 | j}  (control that removes the auxiliary-supervision confound)
  symbols       one constrained symbol token from {A..., NONE} (guided choice when the server supports it)

Failures (timeouts, unparsable output, out-of-range index) are recorded as FAIL with a reason; they are
never converted to -1. Requested and resolved model versions, token usage and latency are recorded.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..data.serialization import NONE_SYMBOL, SYMBOLS, serialize_record
from ..hashing import sha256_text
from ..schema import Decision, DecisionInstance, DecisionKind, NONE_INDEX


@dataclass
class JudgeConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "google/gemma-3-12b-it"
    mode: str = "rich"  # rich | minimal_json | symbols
    prompt_file: str = "configs/mrd/prompts/judge_rich_v1.txt"
    serialization: str = "rich_v1"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 120.0
    max_retries: int = 2
    concurrency: int = 8
    guided: bool = True  # constrain the output (choice / JSON schema) when the server supports it
    # "structured_outputs" (vLLM >= 0.10: extra_body.structured_outputs.{choice,json}) or "guided" (older guided_choice /
    # guided_json, removed from current vLLM, where it is silently ignored)
    structured_api: str = "structured_outputs"
    extra_body: dict[str, Any] = field(default_factory=dict)
    seed: int | None = 0


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def render(template: str, inst: DecisionInstance, policy: str, serialization: str, field_order: Sequence[str] | None = None, compact_fields: Sequence[str] | None = None) -> str:
    q = serialize_record(inst.query_fields, field_order, serialization, compact_fields)
    lines = [f"[{j}] (symbol {SYMBOLS[j]}) {serialize_record(c.fields, field_order, serialization, compact_fields)}" for j, c in enumerate(inst.candidates)]
    return template.format(policy=policy, query=q, candidates="\n".join(lines), k=inst.k, k_minus_1=max(inst.k - 1, 0), none_symbol=NONE_SYMBOL)


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_output(text: str, mode: str, k: int) -> tuple[DecisionKind, int, dict[str, Any], str | None]:
    """Return (kind, index, raw_fields, failure_reason)."""
    t = text.strip()
    if mode == "symbols":
        sym = t.split()[0].strip(".,:;[]()") if t else ""
        if sym.upper() == NONE_SYMBOL:
            return DecisionKind.NONE, NONE_INDEX, {"symbol": sym}, None
        if sym in SYMBOLS[:k]:
            return DecisionKind.SELECT, SYMBOLS.index(sym), {"symbol": sym}, None
        return DecisionKind.FAIL, NONE_INDEX, {"text": t}, "unparsable_symbol"
    m = _JSON_RE.search(t)
    if not m:
        return DecisionKind.FAIL, NONE_INDEX, {"text": t}, "no_json"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return DecisionKind.FAIL, NONE_INDEX, {"text": t}, "bad_json"
    if mode == "pairwise_batch":
        # batched pairwise questions: one boolean per candidate; listwise gain vs request batching screen (plan §6)
        flags = obj.get("matches")
        if not isinstance(flags, list) or len(flags) != k or not all(isinstance(f, bool) for f in flags):
            return DecisionKind.FAIL, NONE_INDEX, obj, "bad_matches_vector"
        hits = [j for j, f in enumerate(flags) if f]
        obj = {**obj, "n_positive": len(hits)}
        if not hits:
            return DecisionKind.NONE, NONE_INDEX, obj, None
        return DecisionKind.SELECT, hits[0], obj, None  # lowest index; multiple positives recorded in n_positive
    dec = obj.get("decision", obj.get("match_index", obj.get("index")))
    if dec is None or isinstance(dec, bool):
        return DecisionKind.FAIL, NONE_INDEX, obj, "missing_decision"
    if isinstance(dec, str):
        s = dec.strip()
        if s.upper() in (NONE_SYMBOL, "NULL", "-1"):
            return DecisionKind.NONE, NONE_INDEX, obj, None
        if s in SYMBOLS[:k]:
            return DecisionKind.SELECT, SYMBOLS.index(s), obj, None
        if re.fullmatch(r"-?\d+", s):
            dec = int(s)
        else:
            return DecisionKind.FAIL, NONE_INDEX, obj, "unparsable_decision"
    if isinstance(dec, float):
        if not dec.is_integer():
            return DecisionKind.FAIL, NONE_INDEX, obj, "non_integer_decision"
        dec = int(dec)
    if dec == NONE_INDEX:
        return DecisionKind.NONE, NONE_INDEX, obj, None
    if 0 <= dec < k:
        return DecisionKind.SELECT, int(dec), obj, None
    return DecisionKind.FAIL, NONE_INDEX, obj, "index_out_of_range"


class LLMJudge:
    def __init__(self, cfg: JudgeConfig, policy_text: str, field_order: Sequence[str] | None = None, compact_fields: Sequence[str] | None = None):
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_s, max_retries=0)
        self.template = load_prompt(cfg.prompt_file)
        self.prompt_hash = sha256_text(self.template)
        self.policy = policy_text
        self.field_order = field_order
        self.compact_fields = compact_fields
        self.resolved_model: str | None = None

    def _extra_body(self, inst: DecisionInstance) -> dict[str, Any]:
        eb = {k: v for k, v in self.cfg.extra_body.items() if v is not None}  # null in a config drops an inherited field
        if not self.cfg.guided:
            return eb
        spec: dict[str, Any] = {}
        if self.cfg.mode == "symbols":
            spec = {"choice": SYMBOLS[: inst.k] + [NONE_SYMBOL]}
        elif self.cfg.mode == "minimal_json":
            spec = {"json": {"type": "object", "properties": {"decision": {"type": "integer", "minimum": -1, "maximum": max(inst.k - 1, -1)}}, "required": ["decision"], "additionalProperties": False}}
        elif self.cfg.mode == "pairwise_batch":
            spec = {"json": {"type": "object", "properties": {"matches": {"type": "array", "items": {"type": "boolean"}, "minItems": inst.k, "maxItems": inst.k}}, "required": ["matches"], "additionalProperties": False}}
        if spec and self.cfg.structured_api == "structured_outputs":
            eb["structured_outputs"] = spec
        elif spec:
            eb.update({f"guided_{k}": v for k, v in spec.items()})
        return eb

    def judge_one(self, inst: DecisionInstance) -> Decision:
        prompt = render(self.template, inst, self.policy, self.cfg.serialization, self.field_order, self.compact_fields)
        t0 = time.perf_counter()
        last_err = None
        attempts = 0
        for attempt in range(self.cfg.max_retries + 1):
            attempts += 1
            try:
                kwargs: dict[str, Any] = dict(model=self.cfg.model, messages=[{"role": "user", "content": prompt}], temperature=self.cfg.temperature, max_tokens=8 if self.cfg.mode == "symbols" else self.cfg.max_tokens)
                if self.cfg.seed is not None:
                    kwargs["seed"] = self.cfg.seed
                eb = self._extra_body(inst)
                if eb:
                    kwargs["extra_body"] = eb
                resp = self.client.chat.completions.create(**kwargs)
                self.resolved_model = getattr(resp, "model", None) or self.resolved_model
                text = resp.choices[0].message.content or ""
                kind, idx, raw, reason = parse_output(text, self.cfg.mode, inst.k)
                usage = getattr(resp, "usage", None)
                return Decision(
                    inst.instance_id,
                    kind,
                    idx,
                    inst.index_to_row_id(idx) if kind == DecisionKind.SELECT else None,
                    probs=None,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None),
                    failure_reason=reason,
                    raw={"text": text[:2000], "parsed": raw, "attempts": attempts, "requested_model": self.cfg.model, "resolved_model": self.resolved_model, "prompt_hash": self.prompt_hash, "mode": self.cfg.mode},
                )
            except Exception as e:  # network / timeout / server error
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(min(2**attempt, 8))
        return Decision(inst.instance_id, DecisionKind.FAIL, NONE_INDEX, None, latency_ms=(time.perf_counter() - t0) * 1000, failure_reason=f"request_failed: {last_err}", raw={"attempts": attempts, "requested_model": self.cfg.model, "mode": self.cfg.mode})

    def judge_many(self, insts: Sequence[DecisionInstance], on_result: Callable[[Decision], None] | None = None) -> list[Decision]:
        out: list[Decision] = []
        with ThreadPoolExecutor(self.cfg.concurrency) as ex:
            futs = {ex.submit(self.judge_one, i): i for i in insts}
            for f in as_completed(futs):
                d = f.result()
                out.append(d)
                if on_result:
                    on_result(d)
        return out

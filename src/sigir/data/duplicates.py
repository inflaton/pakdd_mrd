"""Probable unlinked duplicates (review note R1, Nov 5, 2026).

The DBLP-Scholar "perfect mapping" misses Scholar records that the frozen identity policy calls matches: citation
entries with shortened, misspelt or run-together titles, abbreviated author lists, or no venue/year ("Hilbert r-tree
fractals" for "Hilbert R-tree: An Improved R-tree using Fractals"; "Hector Garcia-Molina Speaks Out Regarding
Startups, ..." for "Hector Garcia-Molina Speaks Out"). A pool that keeps such a record is not a certified no-match
pool. This module flags them with a deterministic, model-free rule written from the policy text and frozen before
any calibration outcome:

  near-equal titles (token Jaccard >= 0.8): corroborated by a shared author surname, an identical specific title
      (>= 4 content tokens), or the same year with >= 3 shared content tokens
  truncated titles (one title's tokens all inside the other's, >= 3 tokens, sharing a word outside the 211 most
      common DBLP-Scholar title words and outside generic editorial words, years not disagreeing): corroborated by
      a shared author surname, or by a missing author list together with the same year and a compatible venue
  never when a field distinguishes the publications (policy: preprint, conference and journal versions differ):
      years at least 2 apart, different venue kinds (conference / journal / report; a report or draft record against
      an unknown venue also counts), or a "Response to /
      Comments on / Review of ..." title on one side only

Checked by hand on the Nov 5 inspection cases (tests/test_duplicates_r1.py); the remaining errors are a
label-noise limitation reported with the counts.

Uses: `remove_all_acceptable` removes flagged records together with the acceptable ones (catalog absence of the
publication), and natural certified no-match pools containing a flagged record are withdrawn from N-FMR
denominators (label screen written by 03_constructed_cohort.py). Product datasets have no screen: exact-title
duplicates are not policy matches there (variants share titles).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

COMMON = frozenset({'access', 'acm', 'active', 'adaptive', 'agent', 'aggregate', 'aggregation', 'algorithm',
    'analysi', 'application', 'approach', 'approximate', 'architecture', 'as', 'association', 'automatic', 'base',
    'based', 'building', 'business', 'cache', 'caching', 'case', 'challenge', 'client', 'clustering', 'commerce',
    'complex', 'computation', 'computer', 'computing', 'concurrency', 'conference', 'constraint', 'content',
    'continuou', 'control', 'cost', 'cube', 'data', 'database', 'dbm', 'decision', 'deductive', 'design',
    'detection', 'development', 'digital', 'dimensional', 'discovery', 'disk', 'distributed', 'document', 'driven',
    'dynamic', 'editor', 'effective', 'efficient', 'electronic', 'engine', 'environment', 'estimation',
    'evaluation', 'execution', 'experience', 'expression', 'extended', 'fast', 'file', 'framework', 'generation',
    'graph', 'heterogeneou', 'high', 'histogram', 'how', 'image', 'implementation', 'improving', 'incremental',
    'index', 'indexing', 'information', 'integrating', 'integration', 'interactive', 'interface', 'international',
    'internet', 'introduction', 'issue', 'join', 'knowledge', 'language', 'large', 'learning', 'level',
    'maintenance', 'management', 'managing', 'matching', 'materialized', 'memory', 'method', 'microsoft', 'mining',
    'mobile', 'model', 'modeling', 'multi', 'multidimensional', 'multimedia', 'multiple', 'network', 'new',
    'object', 'olap', 'online', 'open', 'optimization', 'optimizing', 'oriented', 'over', 'overview', 'parallel',
    'path', 'pattern', 'peer', 'performance', 'problem', 'processing', 'programming', 'project', 'protocol',
    'quality', 'query', 'querying', 'quot', 'real', 'recovery', 'relation', 'relational', 'report', 'research',
    'resource', 'retrieval', 'rule', 'sampling', 'scalable', 'scale', 'scheduling', 'schema', 'scheme', 'search',
    'selection', 'selectivity', 'self', 'semantic', 'semistructured', 'sensor', 'server', 'service', 'sery', 'set',
    'sigmod', 'similarity', 'site', 'software', 'source', 'space', 'spatial', 'specification', 'sql', 'storage',
    'strategy', 'stream', 'structure', 'structured', 'study', 'support', 'supporting', 'survey', 'system', 'table',
    'technique', 'technology', 'temporal', 'text', 'through', 'time', 'tool', 'transaction', 'tree', 'tutorial',
    'type', 'update', 'user', 'version', 'very', 'video', 'view', 'visual', 'warehouse', 'warehousing', 'web',
    'what', 'wide', 'workflow', 'workshop', 'world', 'xml'})
# editorial / front-matter words: never evidence that two titles name the same publication
GENERIC = frozenset({"guest", "editor", "editorial", "introduction", "preface", "foreword", "note", "column", "review", "book", "keynote", "address", "message", "issue", "special", "panel", "announcement", "obituary", "letter", "chair", "welcome", "remark", "invited", "talk", "abstract", "session", "director", "information", "reminiscence"})
STOP = {"a", "an", "the", "of", "on", "in", "for", "and", "to", "with", "using", "by", "from", "at", "via", "its", "is", "are", "towards", "toward", "over"}
NEGATING = ("response to", "comments on", "comment on", "reply to", "a reply", "erratum", "corrigendum", "review of", "a review of")
PLACEHOLDER = {"n/a", "na", "anthology", "acms", "et", "al", "others", "unknown"}


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", str(s or "")).lower()).strip()


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def title_tokens(title: Any) -> set[str]:
    return {_stem(w) for w in _norm(title).split() if len(w) > 1 and w not in STOP}


def surnames(authors: Any) -> set[str]:
    out = set()
    for name in re.split(r",|;| and ", unicodedata.normalize("NFKC", str(authors or ""))):
        toks = [t for t in re.findall(r"[a-z]+", name.lower()) if len(t) > 1]
        if toks and toks[-1] not in PLACEHOLDER:
            out.add(toks[-1])
    return out


def year(v: Any) -> int | None:
    m = re.search(r"(19|20)\d\d", str(v or ""))
    return int(m.group(0)) if m else None


def venue_kind(v: Any) -> str | None:
    s = _norm(v)
    if not s or s == "n a":
        return None
    if re.search(r"\breport\b|\btechnical\b|\btech rep|\bthesis\b|\btr\b|\bpreprint\b|\bdraft\b|\bw3c\b|\biso\b|\bstandard\b|\bworking\b", s):
        return "report"
    if re.search(r"\bjournal\b|\bj\b|\btransactions\b|\btrans\b|\btods\b|\btkde\b", s):
        return "journal"
    if re.search(r"\bconf|\bproc|\bsymposium\b|\bworkshop\b|\bicde\b|\bedbt\b|\bvldb\b|\bsigmod\b", s):
        return "conference"
    return None


VENUE_GENERIC = {"proc", "proceeding", "proceedings", "of", "the", "conference", "conf", "journal", "international", "acm", "ieee", "on", "j", "in", "and", "int", "intl", "na", "n", "a"}


def venue_tokens(v: Any) -> set[str]:
    return {w for w in _norm(v).split() if w not in VENUE_GENERIC and not w.isdigit()}


def dblp_probable_duplicate(q: dict[str, Any], c: dict[str, Any]) -> bool:
    tq, tc = title_tokens(q.get("title")), title_tokens(c.get("title"))
    if not tq or not tc:
        return False
    nq, nc = _norm(q.get("title")), _norm(c.get("title"))
    if nc.startswith(NEGATING) != nq.startswith(NEGATING):
        return False
    yq, yc = year(q.get("year")), year(c.get("year"))
    if yq and yc and abs(yq - yc) >= 2:
        return False
    kq, kc = venue_kind(q.get("venue")), venue_kind(c.get("venue"))
    if (kq and kc and kq != kc) or ("report" in (kq, kc) and kq != kc):
        return False  # a report / draft record distinguishes itself from the published version
    inter = tq & tc
    shared_author = bool(surnames(q.get("authors")) & surnames(c.get("authors")))
    same_year = bool(yq and yc and yq == yc)
    if len(inter) / len(tq | tc) >= 0.8:
        return shared_author or (tq == tc and len(tq) >= 4) or (same_year and len(inter) >= 3)
    truncated = (inter == tc and len(tc) >= 3) or (inter == tq and len(tq) >= 3)
    if not truncated or not (inter - COMMON - GENERIC) or (yq and yc and yq != yc):
        return False  # truncation needs a distinctive shared word and must not disagree on the year
    if shared_author:
        return True
    no_authors = not surnames(q.get("authors")) or not surnames(c.get("authors"))
    vq, vc = venue_tokens(q.get("venue")), venue_tokens(c.get("venue"))
    return no_authors and same_year and (not vq or not vc or bool(vq & vc))


DUPLICATE_SCREENS: dict[str, Callable[[dict[str, Any], dict[str, Any]], bool]] = {"dblp_scholar": dblp_probable_duplicate}


def probable_duplicates(dataset: str, query_fields: dict[str, Any], candidates) -> list[str]:
    """Row ids of candidates flagged as probable unlinked duplicates of the query (empty for unscreened datasets)."""
    screen = DUPLICATE_SCREENS.get(dataset)
    return [c.row_id for c in candidates if screen and screen(query_fields, c.fields)] if screen else []

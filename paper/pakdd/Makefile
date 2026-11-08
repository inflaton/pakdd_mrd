PYTHON ?= python3
RUNS_DIR ?= ../../runs

.PHONY: numbers check
numbers:
	$(PYTHON) generate_results.py --runs-dir "$(RUNS_DIR)" --output-dir generated

check:
	@set -eu; \
	  output=$$(mktemp -d); \
	  trap 'rm -rf "$$output"' EXIT; \
	  $(PYTHON) generate_results.py --runs-dir "$(RUNS_DIR)" --output-dir "$$output"; \
	  diff -r generated "$$output"

# List available recipes when you type `make`
.PHONY: default test test-full lint format check smoke

# pytest-xdist worker count. The two tiers are CPU-bound on the researcher's laptop.
# Set to 1 to run serially, e.g. when debugging or reading a traceback:
#   make test PYTEST_WORKERS=1
PYTEST_WORKERS ?= 3

# Extra pytest arguments, e.g. `make test PYTEST_ARGS="-k boozer"`.
PYTEST_ARGS ?=

default:
	@grep -E '^[A-Za-z][A-Za-z0-9_-]*:' $(MAKEFILE_LIST) | cut -d: -f1

# PR-CI subset: fast tests only. Budget: under 2 min, no single test over ~20 s
# (DESIGN.md section 22.5). The durations report is the early warning; read it.
test:
	python -m pytest -n $(PYTEST_WORKERS) -m "not slow" --durations=15 $(PYTEST_ARGS)

# Complete suite, including slow tests. Budget: under 5 min, no single slow test
# over ~90 s (DESIGN.md section 22.5). This is the gate before a PR goes ready.
test-full:
	python -m pytest -n $(PYTEST_WORKERS) --durations=25 $(PYTEST_ARGS)

# Formatting check. DESIGN.md section 2 requires new code to be black-formatted.
lint:
	black --check --diff alpha_analysis test

# Apply the formatting rather than just reporting it.
format:
	black alpha_analysis test

# The gate. This is what "done" means for a milestone PR.
check: lint test-full

# Clean-environment install and import check. Catches a module that only imports
# because it was found in the source tree, and optional deps leaking into the base
# package (DESIGN.md section 19.2).
smoke:
	rm -rf /tmp/alpha-analysis-smoke
	python -m venv /tmp/alpha-analysis-smoke
	/tmp/alpha-analysis-smoke/bin/python -m pip install -q .
	/tmp/alpha-analysis-smoke/bin/python -c "import alpha_analysis; print(alpha_analysis.__file__)"

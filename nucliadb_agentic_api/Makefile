include ../hyperforge.mk

.PHONY: test-cov
test-cov:
	uv run $(PYTEST) $(pytest_cov_report_flags) --cov=nucliadb_agentic_api --cov-config=../../.coveragerc tests/
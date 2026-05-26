include ../hyperforge.mk

.PHONY: test-cov
test-cov:
	uv run $(PYTEST) $(pytest_cov_report_flags) --cov=hyperforge --cov-config=../../.coveragerc tests/
# Common Makefile variables and patterns. Include this inside other makefiles with:
#
# include path/to/hyperforge.mk
#
# Set REPO_ROOT before including if the default (../..) doesn't match your depth.
# REPO_ROOT is used to locate mypy.ini.

REPO_ROOT ?= ../..

pytest_flags := -s -rfE -v --tb=native 
pytest_extra_flags :=
pytest_record_flags := --record-mode=rewrite
pytest_play_record_flags := --record-mode=none
pytest_cov_report_flags := --cov-report xml --cov-report term-missing:skip-covered

PYTEST := pytest $(pytest_flags) $(pytest_extra_flags)


.PHONY: format
format fmt:
	uv run ruff check --fix .
	uv run ruff format .


.PHONY: lint
lint:
	uv run ruff check . && \
	uv run ruff format --check . && \
	uv run ty check src && \
	uv run mypy --config-file=$(REPO_ROOT)/mypy.ini src

.PHONY: test
test:
	uv run $(PYTEST) $(pytest_play_record_flags) tests/ $(ARGS)

record:
	uv run $(PYTEST) $(pytest_record_flags) tests/ $(ARGS)
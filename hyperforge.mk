# Common Makefile variables and patterns. Include this inside other makefiles with:
#
# include path/to/hyperforge.mk
#

pytest_flags := -s -rfE -v --tb=native 
pytest_extra_flags :=
pytest_record_flags := --record-mode=rewrite
pytest_play_record_flags := --record-mode=none
pytest_cov_report_flags := --cov-report xml --cov-report term-missing:skip-covered

PYTEST := pytest $(pytest_flags) $(pytest_extra_flags)


.PHONY: format
format:
	uv run ruff check --fix --config=../../ruff.toml .
	uv run ruff format --config=../../ruff.toml .


.PHONY: lint
lint:
	uv run ruff check --config=../../ruff.toml .
	uv run ruff format --check --config=../../ruff.toml .
	uv run ty check .
	uv run mypy --config-file=../../mypy.ini src

.PHONY: test
test:
	uv run $(PYTEST) $(pytest_play_record_flags) tests/ $(ARGS)

record:
	uv run $(PYTEST) $(pytest_record_flags) tests/ $(ARGS)
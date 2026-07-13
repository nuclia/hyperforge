COMPONENTS := \
	hyperforge \
	agents/conditional \
	agents/external \
	agents/generate \
	agents/google \
	agents/historical \
	agents/http \
	agents/mcp \
	agents/nucliadb \
	agents/passthrough \
	agents/perplexity \
	agents/perplexity_search \
	agents/related \
	agents/remi \
	agents/rephrase \
	agents/restart \
	agents/restricted \
	agents/smart \
	agents/static \
	agents/static_string \
	agents/summarize

install:
	uv sync

install-test:
	uv sync --group dev

fmt:
	@for dir in $(COMPONENTS); do $(MAKE) -C $$dir format || exit 1; done

extract-openai:
	uv run arag-extract-openapi  $(DOCS_FILE) $(API_VERSION) $(HASH)

lint:
	@for dir in $(COMPONENTS); do $(MAKE) -C $$dir lint || exit 1; done

start_local_db:
	brew services start postgresql

stop_local_db:
	brew services stop postgresql

create_db:
	POSTGRESQL_DSN=postgresql:///postgres alembic upgrade head

reset_db:
	psql -d postgres -c "DELETE FROM alembic_version;" || true
	POSTGRESQL_DSN=postgresql:///postgres alembic stamp head

reset_db_hard:
	psql -d postgres -c "DROP TABLE IF EXISTS alembic_version CASCADE;" || true
	psql -d postgres -c "DROP TABLE IF EXISTS download_requests CASCADE;" || true
	POSTGRESQL_DSN=postgresql:///postgres alembic upgrade head

check_db_version:
	psql -d postgres -c "SELECT * FROM alembic_version;" || echo "No alembic_version table found"

alembic_history:
	POSTGRESQL_DSN=postgresql:///postgres alembic history

generate_alembic_version:
	POSTGRESQL_DSN=postgresql:///postgres alembic revision --autogenerate

dockers:
	docker build -t arag . -f HYPERFORGE.Dockerfile

all-tests:
	@for dir in $(COMPONENTS); do $(MAKE) -C $$dir test || exit 1; done

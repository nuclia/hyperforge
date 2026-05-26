install:
	uv sync

install-test:
	uv sync --group dev

fmt:
	uv run ruff format hyperforge agents nucliadb_agentic_api 
	uv run ruff check hyperforge agents nucliadb_agentic_api --select I --fix 

extract-openai:
	uv run arag-extract-openapi  $(DOCS_FILE) $(API_VERSION) $(HASH)

lint:
	uv run ruff check hyperforge agents nucliadb_agentic_api 
	uv run ruff format --check hyperforge agents nucliadb_agentic_api 
	uv run mypy hyperforge agents nucliadb_agentic_api

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


build-ui:
	cd frontend && npm ci && npm run build

dev-ui:
	cd frontend && npm install && npm run dev

dockers:
	docker build -t arag . -f HYPERFORGE.Dockerfile

FROM python:3.12 AS build

RUN pip install uv &&  apt update -y && apt install -y npm && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install dependencies
RUN uv venv /app
ENV VIRTUAL_ENV=/app
ENV CHROME_DOCKER=true
COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml
COPY agents/hyperforge/pyproject.toml /app/agents/hyperforge/pyproject.toml
COPY libraries /app/libraries
RUN --mount=type=secret,id=PYTHON_KEY,env=PYTHON_KEY UV_INDEX_NUCLIA_USERNAME=oauth2accesstoken UV_INDEX_NUCLIA_PASSWORD=$(echo $PYTHON_KEY | sed -E 's/.*:(.*)@.*/\1/') uv sync --active --frozen --only-group forge --no-install-workspace --directory /app
# RUN --mount=type=secret,id=UV_INDEX_NUCLIA_PASSWORD,env=UV_INDEX_NUCLIA_PASSWORD UV_INDEX_NUCLIA_USERNAME=oauth2accesstoken uv sync --active --frozen --only-group rao --no-install-workspace --directory /app/forge


# Copy and install project
COPY libraries /app/libraries
COPY . /app/.
RUN uv sync --active --frozen --only-group forge --directory /app --compile-bytecode

#
# Only copy the virtual env to the final image.
#
FROM python:3.12
COPY --from=build /app /app
ENV PATH=/app/bin:$PATH

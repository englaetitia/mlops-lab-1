# syntax=docker/dockerfile:1

###############################################################################
# Stage 1 — builder
#
# Resolves and installs dependencies into /app/.venv. Everything expensive
# (uv, compilers, wheel downloads, the uv cache) lives here and is thrown away
# when the stage ends; only the finished virtualenv is carried forward.
###############################################################################
FROM python:3.12-slim AS builder

# uv is copied in from its official image rather than pip-installed, so no
# extra Python packages end up in this stage.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# ONLY the dependency manifests are copied here — not the source.
# This layer's cache key is the content of these two files, so editing
# serve.py does not invalidate it and the dependency install is reused.
COPY pyproject.toml uv.lock ./

# --frozen            : fail rather than silently re-resolving; the lock file is
#                       the contract, exactly as in Lab 1's Q1.
# --no-dev            : skip dev-only dependency groups.
# --no-install-project: install the dependencies but not the project itself,
#                       so this layer never depends on src/.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

###############################################################################
# Stage 2 — runtime
#
# A fresh slim image with no uv, no build cache and no compilers: just Python,
# the finished virtualenv and the application code.
###############################################################################
FROM python:3.12-slim AS runtime

# Running as root inside a container is a needless risk; the service only ever
# reads its own code and writes to a temp dir.
RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
# Source last: the most frequently changed input invalidates the fewest layers.
COPY --chown=app:app src/ ./src/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_URI=models:/food11@champion \
    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

USER app
EXPOSE 8000

# No model weights are baked in. The image knows only *which* model to ask for;
# the bytes are fetched from the tracking server at startup, so promoting a new
# version to @champion changes what this same image serves.
ENTRYPOINT ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
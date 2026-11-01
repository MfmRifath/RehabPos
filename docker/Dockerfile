# CPU-only image. Pose models are not included: mount data/ (see docker-compose.yml).
FROM node:24-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.12.23
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY configs ./configs
# The lock resolves torch from the CPU index on Linux (ADR 0002).
RUN uv sync --frozen --no-dev
COPY --from=frontend /build/frontend/dist ./frontend/dist
RUN useradd --create-home --uid 1000 rehabpose \
    && mkdir -p /app/data /app/reports \
    && chown -R rehabpose:rehabpose /app/data /app/reports
ENV PATH="/app/.venv/bin:$PATH" \
    REHABPOSE_HOME=/app \
    REHABPOSE_HOST=0.0.0.0 \
    MPLCONFIGDIR=/tmp/matplotlib
USER rehabpose
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health', timeout=8)"]
CMD ["rehabpose", "serve"]

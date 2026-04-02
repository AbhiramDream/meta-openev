# ──────────────────────────────────────────────────────────────────────────────
# Supply Chain OpenEnv — Root Dockerfile
# Targets port 7860 (required by Hugging Face Spaces)
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL description="OpenEnv Supply Chain RL Environment"
LABEL version="1.0.0"

# ── Standard runtime vars ─────────────────────────────────────────────────────
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Mandatory inference vars (override at runtime via -e / HF Secrets) ────────
# API endpoint for the LLM
ENV API_BASE_URL="https://api.openai.com/v1"
# Model identifier
ENV MODEL_NAME="gpt-4o-mini"
# Hugging Face / API authentication token
ENV HF_TOKEN=""

# ── Optional inference vars ───────────────────────────────────────────────────
# Which tasks to run (comma-separated)
ENV TASKS="easy,medium,hard"
# Reproducibility seed
ENV SEED="42"
# OpenEnv server URL (default = this container itself)
ENV ENV_BASE_URL="http://localhost:7860"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Port ─────────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Liveness probe ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Start the OpenEnv server ──────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]

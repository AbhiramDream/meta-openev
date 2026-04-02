# ──────────────────────────────────────────────
# Supply Chain OpenEnv — Docker Image
# ──────────────────────────────────────────────
#
# Build:
#   docker build -t supply-chain-openenv .
#
# Run locally:
#   docker run -p 7860:7860 supply-chain-openenv
#
# For Hugging Face Spaces, this Dockerfile should be placed
# at the root of the Space repository (or set as the build
# context in the Space settings).
# ──────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="supply-chain-openenv"
LABEL description="OpenEnv-compatible Supply Chain RL environment"
LABEL version="1.0.0"

# HF Spaces expects the app to run on port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────
# Copy only requirements first to exploit Docker layer caching
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────
COPY .. .

# ── Expose port ───────────────────────────────
EXPOSE 7860

# ── Health-check ──────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Entrypoint ────────────────────────────────
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]

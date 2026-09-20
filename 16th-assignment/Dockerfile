# One image, two entrypoints (API and UI) -- compose picks the command.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --create-home --uid 10001 appuser && mkdir -p /srv && chown appuser /srv
USER appuser
WORKDIR /srv

# Bake the MiniLM embedding model into the image (under the runtime user's cache)
# so a cold start does not download 80 MB before serving its first request.
RUN python -c "\
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; \
DefaultEmbeddingFunction()(['warm the onnx embedding model'])"

COPY --chown=appuser app ./app
COPY --chown=appuser data ./data
COPY --chown=appuser scripts ./scripts
COPY --chown=appuser models ./models

# Build the vector index at image build time -- the container starts ready.
RUN python -c "from app import rag; print(rag.ingest())"

EXPOSE 8080 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8080/health',timeout=4).status_code==200 else 1)"

# Single async worker: Chroma's persistent client owns a SQLite file, so scale
# out with replicas + a shared Chroma server rather than more workers per box.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

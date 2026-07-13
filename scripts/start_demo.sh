#!/usr/bin/env bash
# Single-process hosted demo: API (localhost:8000) + Streamlit console on $PORT.
# Zero external dependencies — in-memory vector store + demo embeddings by default.
# Set PINECONE_API_KEY (and VECTOR_BACKEND=pinecone) for a persistent gallery.
set -euo pipefail

export VECTOR_BACKEND="${VECTOR_BACKEND:-memory}"
export EMBEDDING_MODE="${EMBEDDING_MODE:-demo}"
export API_KEY="${API_KEY:-dev-local-key-change-me}"
export API_URL="http://127.0.0.1:8000"
# pyarrow's bundled mimalloc allocator segfaults on some hosts when Streamlit
# serialises a dataframe off the main thread; use Arrow's system allocator.
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"
PORT="${PORT:-8501}"

uvicorn app.main:app --host 127.0.0.1 --port 8000 &

for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  sleep 1
done

# Reverse-proxy-friendly flags so the console works behind a hosted load balancer
# (Render / Railway / Fly): disable CORS + XSRF origin checks, force the light theme.
exec streamlit run ui/app.py \
  --server.port "$PORT" --server.address 0.0.0.0 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false \
  --browser.gatherUsageStats false --theme.base light

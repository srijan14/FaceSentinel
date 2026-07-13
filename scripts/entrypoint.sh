#!/usr/bin/env bash
# Container entrypoint: provision models on first boot, then run the given command.
set -e

if [ "${AUTO_DOWNLOAD_MODELS:-true}" = "true" ]; then
  if ! ls models/*.onnx >/dev/null 2>&1; then
    echo "[entrypoint] No ONNX models found in models/ — downloading (buffalo_l)..."
    python scripts/download_models.py || {
      echo "[entrypoint] Model download failed. Provide SCRFD + ArcFace .onnx in models/ and retry." >&2
    }
  fi
fi

exec "$@"

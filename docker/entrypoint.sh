#!/usr/bin/env bash
set -e
echo "CUDA available? ->" $(python - <<'PY'
import torch; print(torch.cuda.is_available())
PY
)
exec "$@"

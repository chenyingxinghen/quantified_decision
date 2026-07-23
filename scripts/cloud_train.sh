#!/usr/bin/env bash
# 云端一键训练 + 检验（SUFE Gemini AI Platform, Linux 容器）
# 用法: bash scripts/cloud_train.sh [--build-from-raw] [--stocks 6000]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python3}"

echo ">>> 云端一键训练启动 ($(date))"
"$PY" scripts/cloud_train.py "$@"
echo ">>> 完成 ($(date))"

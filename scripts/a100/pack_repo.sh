#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

archive="${1:-$(dirname "$REPO_ROOT")/CoLT_a100_$(date +%Y%m%d_%H%M%S).tar.gz}"
COPYFILE_DISABLE=1 tar \
  --exclude='./.DS_Store' \
  --exclude='*/.DS_Store' \
  --exclude='./._*' \
  --exclude='*/._*' \
  --exclude='./__MACOSX' \
  --exclude='*/__MACOSX' \
  --exclude='./__pycache__' \
  --exclude='*/__pycache__' \
  -C "$(dirname "$REPO_ROOT")" \
  -czf "$archive" \
  "$(basename "$REPO_ROOT")"

echo "Created: $archive"
du -h "$archive"

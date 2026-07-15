#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/eval_common.sh"

bash "$REPO_ROOT/scripts/a100/09_download_eval_data.sh" MMStar

python - "$EVAL_DATA_ROOT/MMStar.tsv" "$EVAL_DATA_ROOT/COLT_SMOKE_MMSTAR.tsv" <<'PY'
import sys
import pandas as pd

source, target = sys.argv[1:]
data = pd.read_csv(source, sep="\t", nrows=8)
assert len(data) == 8
data.to_csv(target, sep="\t", index=False)
print(f"Created smoke subset: {target} ({len(data)} records)")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
smoke_run_id="$(date +%Y%m%d_%H%M%S)"
export MMEVAL_ROOT="$EVAL_OUTPUT_ROOT/smoke/$smoke_run_id"
smoke_log="$EVAL_LOG_ROOT/colt_eval_smoke_${smoke_run_id}.log"
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE

cd "$VLMEVAL_ROOT"
python run.py \
  --data COLT_SMOKE_MMSTAR \
  --model "$EVAL_MODEL_NAME" \
  --work-dir "$MMEVAL_ROOT" \
  --mode all \
  --judge exact_matching \
  --verbose \
  2>&1 | tee -a "$smoke_log"

python - "$MMEVAL_ROOT" "$EVAL_MODEL_NAME" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
model = sys.argv[2]
dataset = "COLT_SMOKE_MMSTAR"
base = f"{model}_{dataset}"

predictions = [
    path
    for path in root.rglob("*")
    if path.is_file()
    and not path.is_symlink()
    and path.stem == base
    and path.suffix.lower() in {".xlsx", ".csv"}
]
if len(predictions) != 1:
    raise RuntimeError(f"Expected one smoke prediction file, found: {predictions}")

prediction_file = predictions[0]
if prediction_file.suffix.lower() == ".xlsx":
    data = pd.read_excel(prediction_file)
else:
    data = pd.read_csv(prediction_file)

if len(data) != 8 or "prediction" not in data:
    raise RuntimeError(
        f"Invalid smoke predictions: rows={len(data)}, columns={list(data.columns)}"
    )
responses = data["prediction"].fillna("").astype(str)
if (responses.str.strip() == "").any() or responses.str.contains("Failed to obtain answer", regex=False).any():
    raise RuntimeError("Smoke predictions contain an empty or failed response.")

score_files = [
    path
    for path in root.rglob(f"{base}_acc.csv")
    if path.is_file() and not path.is_symlink()
]
if len(score_files) != 1:
    raise RuntimeError(f"Expected one smoke score file, found: {score_files}")

print(f"Validated predictions: {prediction_file}")
print(f"Validated score: {score_files[0]}")
columns = [column for column in ["index", "prediction", "answer"] if column in data]
print(data[columns].to_string(index=False))
PY

find "$MMEVAL_ROOT" -maxdepth 6 -type f -print | sort
echo "Smoke log: $smoke_log"
echo "CoLT 8-sample smoke evaluation completed."

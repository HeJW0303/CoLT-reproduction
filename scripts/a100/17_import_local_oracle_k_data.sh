#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_workspace_layout
activate_colt_env

: "${ORACLE_K_IMPORT_ARCHIVE:?Set ORACLE_K_IMPORT_ARCHIVE to the uploaded .json.gz file.}"
manifest="${ORACLE_K_IMPORT_MANIFEST:-$ORACLE_K_IMPORT_ARCHIVE.manifest.json}"
if [[ ! -f "$ORACLE_K_IMPORT_ARCHIVE" || ! -f "$manifest" ]]; then
  echo "Missing import archive or manifest." >&2
  exit 1
fi

metadata="$(python - "$ORACLE_K_IMPORT_ARCHIVE" "$manifest" "$DATA_ROOT/colt_sft_image.json" "$DATASET_REVISION" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


archive = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
original = Path(sys.argv[3])
expected_revision = sys.argv[4]
with manifest_path.open(encoding="utf-8") as file:
    manifest = json.load(file)
if manifest.get("schema_version") != 1:
    raise ValueError("Unsupported Oracle-K transfer manifest")
if manifest.get("dataset_revision") != expected_revision:
    raise ValueError("Dataset revision does not match the server A100 profile")
annotation = manifest.get("annotation", {})
if (
    (annotation.get("protocol_version"), annotation.get("user_prompt_version")) not in {(3, 1), (4, 2), (5, 3)}
    or not annotation.get("unit_boundary_pattern")
    or not annotation.get("model")
    or not annotation.get("system_prompt_sha256")
):
    raise ValueError("Manifest lacks supported Oracle-K teacher provenance")
if annotation.get("reasoning_effort") not in {None, "minimal", "low", "medium", "high"}:
    raise ValueError("Manifest contains an invalid teacher reasoning effort")
if archive.name != manifest["archive"]["file_name"]:
    raise ValueError("Archive file name does not match manifest")
if archive.stat().st_size != manifest["archive"]["size"] or sha256_file(archive) != manifest["archive"]["sha256"]:
    raise ValueError("Archive checksum does not match manifest")
if original.stat().st_size != manifest["original"]["size"] or sha256_file(original) != manifest["original"]["sha256"]:
    raise ValueError("Server original JSON does not match the local annotation source")
if original.name != manifest["original"]["file_name"]:
    raise ValueError("Original JSON file name does not match manifest")
output_name = manifest["annotated"]["file_name"]
dataset_name = manifest["dataset_name"]
records = manifest["annotated"]["records"]
original_start = manifest["original_start"]
min_k = manifest["min_k"]
max_k = manifest["max_k"]
if Path(output_name).name != output_name or not output_name.endswith(".json"):
    raise ValueError("Annotated output must be a plain .json file name")
if not isinstance(dataset_name, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_name) is None:
    raise ValueError("Dataset name contains unsupported characters")
if isinstance(records, bool) or not isinstance(records, int) or records < 1:
    raise ValueError("Manifest record count must be a positive integer")
if isinstance(original_start, bool) or not isinstance(original_start, int) or original_start < 0:
    raise ValueError("Manifest original_start must be a non-negative integer")
if any(isinstance(value, bool) or not isinstance(value, int) for value in (min_k, max_k)) or not 1 <= min_k <= max_k:
    raise ValueError("Manifest K range is invalid")
k_distribution = manifest["annotated"].get("k_distribution", {})
if (
    not isinstance(k_distribution, dict)
    or not k_distribution
    or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in k_distribution.values())
):
    raise ValueError("Manifest K distribution is invalid")
if sum(k_distribution.values()) != records:
    raise ValueError("Manifest K distribution does not sum to the record count")
if any(not key.isdigit() or not min_k <= int(key) <= max_k for key in k_distribution):
    raise ValueError("Manifest K distribution contains an out-of-range K")
values = [
    output_name,
    manifest["annotated"]["sha256"],
    str(manifest["annotated"]["size"]),
    dataset_name,
    str(original_start),
    str(min_k),
    str(max_k),
    str(records),
]
print("\t".join(values))
PY
)"
IFS=$'\t' read -r output_name expected_sha expected_size dataset_name original_start min_k max_k expected_records <<< "$metadata"

target="$DATA_ROOT/$output_name"
temporary="$DATA_ROOT/.$output_name.import.tmp"
if [[ -e "$target" && "${ORACLE_K_IMPORT_OVERWRITE:-0}" != "1" ]]; then
  echo "Target already exists: $target" >&2
  echo "Set ORACLE_K_IMPORT_OVERWRITE=1 only after verifying this replacement." >&2
  exit 1
fi
trap 'rm -f "$temporary"' EXIT

python - "$ORACLE_K_IMPORT_ARCHIVE" "$temporary" "$expected_sha" "$expected_size" <<'PY'
import gzip
import hashlib
import sys
from pathlib import Path

archive = Path(sys.argv[1])
target = Path(sys.argv[2])
expected_sha = sys.argv[3]
expected_size = int(sys.argv[4])
digest = hashlib.sha256()
size = 0
with gzip.open(archive, "rb") as source, target.open("wb") as output:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
        output.write(chunk)
if size != expected_size or digest.hexdigest() != expected_sha:
    raise ValueError("Decompressed Oracle-K JSON does not match manifest")
PY

python "$REPO_ROOT/scripts/oracle_k/validate_oracle_blocks.py" \
  --input "$temporary" \
  --original "$DATA_ROOT/colt_sft_image.json" \
  --original-start "$original_start" \
  --expected-records "$expected_records" \
  --min-k "$min_k" \
  --max-k "$max_k"
mv -f "$temporary" "$target"
trap - EXIT

python "$REPO_ROOT/scripts/oracle_k/register_oracle_dataset.py" \
  --dataset-info "$DATA_ROOT/dataset_info.json" \
  --file-name "$output_name" \
  --dataset-name "$dataset_name"

echo "Imported local Oracle-K dataset: $target"
echo "Dataset registration: $dataset_name"

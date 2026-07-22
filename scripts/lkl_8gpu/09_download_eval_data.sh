#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/eval_common.sh"
require_free_gib 10

declare -A URL SIZE MD5

URL[SEEDBench_IMG]='https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench_IMG.tsv'
SIZE[SEEDBench_IMG]=619569378
MD5[SEEDBench_IMG]=68017231464752261a2526d6ca3a10c0

URL[MMBench_DEV_EN]='https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv'
SIZE[MMBench_DEV_EN]=37156625
MD5[MMBench_DEV_EN]=b6caf1133a01c6bb705cf753bb527ed8

URL[ChartQA_TEST]='https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv'
SIZE[ChartQA_TEST]=129773428
MD5[ChartQA_TEST]=c902e0aa9be5582a7aad6dcf52734b42

URL[TextVQA_VAL]='https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv'
SIZE[TextVQA_VAL]=1222773494
MD5[TextVQA_VAL]=b233b31f551bbf4056f2f955da3a92cd

URL[ScienceQA_TEST]='https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_TEST.tsv'
SIZE[ScienceQA_TEST]=51398059
MD5[ScienceQA_TEST]=e42e9e00f9c59a80d8a5db35bc32b71f

URL[MMStar]='https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv'
SIZE[MMStar]=59552082
MD5[MMStar]=e1ecd2140806c1b1bbf54b43372efb9e

URL[AI2D_TEST]='https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv'
SIZE[AI2D_TEST]=167443652
MD5[AI2D_TEST]=0f593e0d1c7df9a3d69bf1f947e71975

URL[MMT-Bench_VAL]='https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_VAL.tsv'
SIZE[MMT-Bench_VAL]=631302456
MD5[MMT-Bench_VAL]=8dd4b730f53dbf9c3aed90ca31c928e0

select_group() {
  case "${1:-phase1}" in
    phase1) printf '%s\n' ChartQA_TEST MMStar MMBench_DEV_EN ;;
    phase2) printf '%s\n' TextVQA_VAL MMT-Bench_VAL ;;
    phase3) printf '%s\n' SEEDBench_IMG ScienceQA_TEST AI2D_TEST ;;
    all) printf '%s\n' SEEDBench_IMG MMBench_DEV_EN ChartQA_TEST TextVQA_VAL ScienceQA_TEST MMStar AI2D_TEST MMT-Bench_VAL ;;
    *) return 1 ;;
  esac
}

if (( $# == 0 )); then
  mapfile -t datasets < <(select_group phase1)
elif (( $# == 1 )) && select_group "$1" >/dev/null 2>&1; then
  mapfile -t datasets < <(select_group "$1")
else
  datasets=("$@")
fi

download_one() {
  local dataset="$1"
  local target="$EVAL_DATA_ROOT/$dataset.tsv"
  local partial="$target.part"

  if [[ -z "${URL[$dataset]:-}" ]]; then
    echo "Unknown evaluation dataset: $dataset" >&2
    exit 1
  fi

  if [[ -f "$target" ]] \
    && [[ "$(stat -c '%s' "$target")" == "${SIZE[$dataset]}" ]] \
    && [[ "$(md5sum "$target" | awk '{print $1}')" == "${MD5[$dataset]}" ]]; then
    echo "Already verified: $dataset"
    return
  fi

  rm -f "$target"
  echo "Downloading $dataset (${SIZE[$dataset]} bytes)"
  if ! curl --fail --location --continue-at - --retry 5 --retry-delay 5 \
      --output "$partial" "${URL[$dataset]}"; then
    echo "TLS verification failed; retrying with -k. MD5 verification remains mandatory." >&2
    curl -k --fail --location --continue-at - --retry 5 --retry-delay 5 \
      --output "$partial" "${URL[$dataset]}"
  fi

  if [[ "$(stat -c '%s' "$partial")" != "${SIZE[$dataset]}" ]]; then
    echo "Size verification failed for $partial" >&2
    rm -f "$partial"
    exit 1
  fi
  if [[ "$(md5sum "$partial" | awk '{print $1}')" != "${MD5[$dataset]}" ]]; then
    echo "MD5 verification failed for $partial" >&2
    rm -f "$partial"
    exit 1
  fi
  mv "$partial" "$target"
  records="$(python - "$target" <<'PY'
import sys
import pandas as pd

print(len(pd.read_csv(sys.argv[1], sep="\t", usecols=["index"])))
PY
)"
  echo "Verified: $target ($records records)"
}

for dataset in "${datasets[@]}"; do
  download_one "$dataset"
done

du -sh "$EVAL_DATA_ROOT"
df -h "$WORKSPACE_ROOT"

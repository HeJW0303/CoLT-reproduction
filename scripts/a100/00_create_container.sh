#!/usr/bin/env bash

set -euo pipefail

HOST_WORKSPACE="${HOST_WORKSPACE:-/workspace2/hejianwei02}"
CONTAINER_NAME="${CONTAINER_NAME:-colt-hjw}"
IMAGE="${COLT_DOCKER_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or is not on PATH." >&2
  exit 1
fi

docker image inspect "$IMAGE" >/dev/null

docker_root="$(docker info --format '{{.DockerRootDir}}')"
echo "Docker root: $docker_root"
df -h "$docker_root"
df -h "$HOST_WORKSPACE"
df -i "$HOST_WORKSPACE"
quota -s 2>/dev/null || true

host_free_kb="$(df -Pk "$HOST_WORKSPACE" | awk 'NR==2 {print $4}')"
minimum_kb=$((350 * 1024 * 1024))
if (( host_free_kb < minimum_kb )); then
  echo "Less than 350 GiB is free under $HOST_WORKSPACE; refusing to create a reproduction workspace." >&2
  exit 1
fi

mkdir -p \
  "$HOST_WORKSPACE"/{.root_home,envs,logs/wandb,outputs,ray_tmp,tmp/python,models,data,downloads} \
  "$HOST_WORKSPACE"/cache/{huggingface/{datasets,transformers,hub,modules},pip,torch,torch_extensions,torchinductor,triton,cuda,wandb,matplotlib,numba,pycache}

touch "$HOST_WORKSPACE/.host_write_test"
rm -f "$HOST_WORKSPACE/.host_write_test"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container $CONTAINER_NAME already exists; refusing to replace it." >&2
  exit 1
fi

docker run -d \
  --gpus all \
  --ipc=host \
  --network=host \
  --tmpfs /tmp:rw,nosuid,nodev,size=16g,mode=1777 \
  --init \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HOME=/workspace/.root_home \
  -e XDG_CACHE_HOME=/workspace/cache \
  -e HF_HOME=/workspace/cache/huggingface \
  -e HF_HUB_CACHE=/workspace/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets \
  -e HF_MODULES_CACHE=/workspace/cache/huggingface/modules \
  -e TORCH_HOME=/workspace/cache/torch \
  -e TORCH_EXTENSIONS_DIR=/workspace/cache/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor \
  -e MPLCONFIGDIR=/workspace/cache/matplotlib \
  -e NUMBA_CACHE_DIR=/workspace/cache/numba \
  -e PYTHONPYCACHEPREFIX=/workspace/cache/pycache \
  -e CUDA_CACHE_PATH=/workspace/cache/cuda \
  -e PIP_CACHE_DIR=/workspace/cache/pip \
  -e WANDB_DIR=/workspace/logs/wandb \
  -e WANDB_CACHE_DIR=/workspace/cache/wandb \
  -e TMPDIR=/workspace/tmp \
  -e TMP=/workspace/tmp \
  -e TEMP=/workspace/tmp \
  -e PYTHON_TEMP_ROOT=/workspace/tmp/python \
  -e RAY_TMPDIR=/workspace/ray_tmp \
  -v "$HOST_WORKSPACE:/workspace" \
  -w /workspace \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  sleep infinity

docker exec "$CONTAINER_NAME" bash -lc \
  'touch /workspace/.container_write_test && rm -f /workspace/.container_write_test && nvidia-smi -L'

echo "Container is ready: docker exec -it $CONTAINER_NAME bash"

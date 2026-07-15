# CoLT 8xA100 复现手册

本配置面向以下固定环境：

- 宿主机唯一可写目录：`/workspace2/hejianwei02`
- 容器挂载：`/workspace2/hejianwei02:/workspace`
- 容器内仓库：`/workspace/CoLT`
- 基础镜像：`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- GPU：8 x A100 80GB

A100 YAML 保持作者代码中的训练设置，包括有效全局 batch size `1 x 8 GPU x 8 累积 = 64`。这是“代码忠实复现”，并非按论文中其他 batch 表述修正。锁定数据集包含 122,179 条样本，一轮约 1,910 个 optimizer step。工程侧只修改本地路径、运行记录和 checkpoint 保留策略：第 1000 step 保存一次，最多保留 1 个可恢复的 ZeRO-3 checkpoint。

## 1. Mac 本地打包

```bash
cd /Users/hejianwei/Documents/project/CoLT
bash scripts/a100/pack_repo.sh
```

脚本会在仓库同级生成 `CoLT_a100_时间戳.tar.gz`。上传到服务器后：

```bash
cd /workspace2/hejianwei02
tar -xzf CoLT_a100_*.tar.gz
cd CoLT
```

打包脚本会禁用并排除 macOS AppleDouble `._*`、`.DS_Store` 和 `__MACOSX` 元数据，避免 Transformers 将二进制元数据误当作 Python 源码扫描。

## 2. 宿主机创建容器

```bash
cd /workspace2/hejianwei02/CoLT
bash scripts/a100/00_create_container.sh
docker exec -it colt-hjw bash
```

不要再添加 `--shm-size`；`--ipc=host` 已使用宿主机的共享内存。容器创建脚本要求 `/workspace2/hejianwei02` 至少剩余 350GiB。

## 3. 容器内逐阶段准备

每一步必须成功退出后才能执行下一步：

```bash
cd /workspace/CoLT
bash scripts/a100/01_setup_env.sh
bash scripts/a100/02_download_assets.sh
bash scripts/a100/03_prepare_data.sh
bash scripts/a100/04_verify_ready.sh
bash scripts/a100/05_nccl_smoke.sh
```

`02_download_assets.sh` 下载约 120GB，并要求下载前至少有 250GiB 空闲。若 Hugging Face 官方站不可达，只在下载阶段设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/a100/02_download_assets.sh
unset HF_ENDPOINT
```

下载脚本锁定 revision，并依据 Hub LFS SHA-256 校验两个模型的全部 safetensors、20 个数据 ZIP 和训练 JSON。数据脚本先检查剩余 ZIP 的 CRC，再串行解压；每个 ZIP 成功后删除。中断后可以重新运行。最后会完整遍历约 285MB 的训练 JSON，确认所有消息结构和图片路径存在。

## 4. 正式训练

默认使用 W&B 离线模式。需要在线 W&B 时先执行：

```bash
source /workspace/envs/colt/bin/activate
wandb login
export WANDB_MODE=online
```

在持久终端中启动：

```bash
cd /workspace/CoLT
tmux new -s colt
bash scripts/a100/06_train.sh
```

用 `Ctrl-b d` 离开 tmux。训练日志位于 `/workspace/logs`，输出位于 `/workspace/outputs/colt_codefaithful`。启动器还会把 A100 YAML、DeepSpeed JSON、Git 状态与 diff、白名单环境变量和 `pip freeze` 保存到 `/workspace/logs/colt_run_*`。

默认拒绝覆盖非空输出目录。确认存在完整的 `checkpoint-*` 后，断点续训使用：

```bash
cd /workspace/CoLT
RESUME=1 bash scripts/a100/06_train.sh
```

因为 YAML 设置了 `overwrite_output_dir: false`，LLaMA-Factory 会自动恢复最后一个完整 checkpoint。

## 5. 固定版本

```text
CoLT:                          331cc78df2d4ab542b9a83822a5a69766e194042
Qwen/Qwen3-VL-8B-Instruct:     0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
Qwen/Qwen3-0.6B:               c1899de289a04d12100db370d81485cdf75e47ca
hulianyuyy/CoLT_Train_Dataset: 7f65a2088bd486b38c24a58c699013d008533388
```

环境锁定为 Python 3.11、PyTorch 2.6.0+cu124、TorchVision 0.21.0+cu124、TorchAudio 2.6.0+cu124、FlashAttention 2.7.4.post1、DeepSpeed 0.16.9、Hugging Face Hub 0.36.2、Qwen VL Utils 0.0.14，以及仓库自带的自定义 Transformers 4.57.0。

DeepSpeed 的 Triton autotune cache 默认放在 `/dev/shm/colt-hjw-triton`，避免 NFS cache 在进程退出时引发长时间等待。该缓存随宿主机重启清空，不影响模型、数据或 checkpoint。

## 6. 存储约束

解压完成后不应保留任何数据 ZIP。数据准备要求至少 150GiB 空闲；正式训练前要求至少 200GiB 空闲，因为单个可恢复 ZeRO-3 checkpoint 预计约 140-170GB，保存过程还需要额外峰值空间。不要把 `save_total_limit` 调高于 1，也不要把 optimizer offload 到该 NFS。

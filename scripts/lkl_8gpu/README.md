# `/data/nvme0/lkl` 双机 8 GPU 部署

这套脚本用于两台公网服务器：8x A100-SXM4-80GB 和 8x A800-SXM4-80GB。二者共享
同一套训练与评测代码，只通过机器 profile 校验 GPU 型号。原有 `scripts/a100` 是另一台
机器上已经验证的 Docker 工作流，不要混用。

当前只支持单机 8 卡训练。两台服务器的 `eth0` 互通不等于 RDMA/NVLink 跨机互通，
因此不要直接把 `NNODES` 改为 2。

## 1. 固定目录

```text
/data/nvme0/lkl/
├── miniconda3/
├── conda/{envs,pkgs}/
├── CoLT-reproduction/
│   ├── .colt_gpu_profile
│   ├── checkpoints/
│   ├── eval/
│   ├── logs/
│   ├── cache/
│   └── tmp/
├── models/
├── datasets/
├── hf-cache/
├── torch-cache/
├── clash
└── config.yaml
```

模型、数据集、Hugging Face 缓存和 PyTorch 缓存体积大且可跨实验复用，因此保留在
`/data/nvme0/lkl` 外层。CoLT 专属的 checkpoint、评测结果、日志、预处理缓存、临时文件和
机器 profile 全部收进仓库目录，并由 `.gitignore` 排除。脚本不会覆盖用户的 `HOME`。

## 2. 克隆与代理

在两台机器上分别执行：

```bash
cd /data/nvme0/lkl
git clone https://github.com/HeJW0303/CoLT-reproduction.git
cd CoLT-reproduction
```

Clash 服务运行后，需要访问 GitHub/Hugging Face 时在当前终端设置：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export ALL_PROXY=socks5h://127.0.0.1:7890
```

代理只影响当前 shell 及其子进程。训练模型和数据下载完成后，可以 `unset` 这些变量。

## 3. 绑定机器 profile

A100 服务器：

```bash
cd /data/nvme0/lkl/CoLT-reproduction
bash scripts/lkl_8gpu/00_verify_host.sh a100
```

A800 服务器：

```bash
cd /data/nvme0/lkl/CoLT-reproduction
bash scripts/lkl_8gpu/00_verify_host.sh a800
```

该命令严格检查 8 张 GPU 的型号，并写入
`/data/nvme0/lkl/CoLT-reproduction/.colt_gpu_profile`。后续脚本自动读取；若选错 profile，会在安装或
训练前退出。

## 4. 首次安装与数据准备

新版 Conda 首次访问 Anaconda 官方源会要求接受服务条款。这是交互式确认，不是创建失败。
可以先查看条款，再由服务器账号本人执行接受命令：

```bash
/data/nvme0/lkl/miniconda3/bin/conda tos view \
  --override-channels --channel https://repo.anaconda.com/pkgs/main
/data/nvme0/lkl/miniconda3/bin/conda tos accept \
  --override-channels --channel https://repo.anaconda.com/pkgs/main
/data/nvme0/lkl/miniconda3/bin/conda tos accept \
  --override-channels --channel https://repo.anaconda.com/pkgs/r
```

也可以在 `conda create` 显示 `[(a)ccept/(r)eject/(v)iew]` 时输入 `a`。接受结果会保存在
当前用户配置中，通常只需操作一次。`--yes` 只确认创建环境，不能代替接受服务条款。

两台机器使用相同命令：

```bash
bash scripts/lkl_8gpu/01_setup_env.sh
bash scripts/lkl_8gpu/02_download_assets.sh
bash scripts/lkl_8gpu/03_prepare_data.sh
bash scripts/lkl_8gpu/04_verify_ready.sh
bash scripts/lkl_8gpu/05_nccl_smoke.sh
```

`01_setup_env.sh` 会创建 Python 3.11 环境
`/data/nvme0/lkl/conda/envs/colt`，安装固定版本的 PyTorch 2.6.0+cu124、FlashAttention、
DeepSpeed、LLaMA-Factory 和仓库内 Transformers。DeepSpeed 以 `DS_BUILD_OPS=0` 安装，
当前 ZeRO-3 配置不需要系统 CUDA Toolkit 或 NVCC。脚本不调用 `apt`，如果预检提示缺少
`git`、`curl`、`unzip` 或 `tmux`，需先让管理员安装系统包。

模型、数据均使用固定 Hugging Face revision 下载并校验。下载中断后可直接重跑步骤 2；
数据解压中断后可直接重跑步骤 3。

## 5. 训练与恢复

建议在 tmux 中运行：

```bash
tmux new-session -A -s colt-train
cd /data/nvme0/lkl/CoLT-reproduction
bash scripts/lkl_8gpu/06_train.sh
```

日志位于 `/data/nvme0/lkl/CoLT-reproduction/logs`，训练输出位于
`/data/nvme0/lkl/CoLT-reproduction/checkpoints/colt_codefaithful`。脚本会记录 Git SHA、工作区 diff、依赖版本、
训练 YAML 和 DeepSpeed 配置。

只有确认是同一次训练的完整 checkpoint 后才使用恢复模式：

```bash
RESUME=1 bash scripts/lkl_8gpu/06_train.sh
```

普通启动遇到非空输出目录会拒绝覆盖；`RESUME=1` 找不到完整 `trainer_state.json` 也会拒绝。

## 6. 更新与评测

以后更新代码：

```bash
cd /data/nvme0/lkl/CoLT-reproduction
git pull --ff-only origin main
```

训练完成后：

```bash
bash scripts/lkl_8gpu/07_verify_final_model.sh
bash scripts/lkl_8gpu/08_setup_eval.sh
bash scripts/lkl_8gpu/10_eval_smoke.sh
```

完整评测、8 卡评测和诊断入口见 `scripts/lkl_8gpu/README_EVAL.md`。

## 7. 常用覆盖项

默认值适合这两台机器。确有需要时可临时覆盖：

```bash
COLT_TRITON_CACHE_DIR=/dev/shm/colt-triton bash scripts/lkl_8gpu/06_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_DEBUG=INFO \
  bash scripts/lkl_8gpu/05_nccl_smoke.sh
```

不要把代理、Hugging Face token 或其他密钥提交到仓库。

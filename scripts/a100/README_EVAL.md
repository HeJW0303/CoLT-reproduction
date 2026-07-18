# CoLT A100 评估路线

本评估配置固定使用训练完成后的推理模型：

```text
/workspace/outputs/colt_codefaithful
```

不要将 `global_step1910` 下的 DeepSpeed 分片直接作为推理模型。`checkpoint-1910`
用于精确续训，输出根目录中的 5 个 safetensors 才是独立推理模型。

## 1. 数据体积

| 数据集 | 原始 TSV | 论文 CoLT 分数 | 主要用途 |
|---|---:|---:|---|
| MMBench_DEV_EN | 35.4 MiB | 84.6 | 小体积 MCQ 链路检查 |
| ScienceQA_TEST | 49.0 MiB | 92.8 | 小体积知识问答检查 |
| MMStar | 56.8 MiB | 68.9 | 论文效率与准确率重点任务 |
| ChartQA_TEST | 123.8 MiB | 74.7 | 论文最大增益，约 +9.6 |
| AI2D_TEST | 159.7 MiB | 85.4 | 图解推理 |
| SEEDBench_IMG | 590.9 MiB | 77.5 | 综合图像理解 |
| MMT-Bench_VAL | 602.1 MiB | 67.4 | 论文主要增益，约 +4.1 |
| TextVQA_VAL | 1166.1 MiB | 81.3 | 论文主要增益，约 +6.1 |

八个 TSV 合计约 2.72 GiB。VLMEvalKit 会把 TSV 中的 base64 图像解码到
`/workspace/eval/LMUData/images`，因此完整评估建议预留 8--10 GiB。

## 2. 评估语义

第一轮采用 code-faithful 语义。官方 CoLT `generate()` 会覆盖调用方参数：

```text
do_sample=True
max_new_tokens=256
temperature=0.6
top_k=20
num_latent=3
```

即使 VLMEvalKit 配置写了 `do_sample=False` 和 `max_new_tokens=8192`，当前官方
实现仍会使用上述参数。评估 adapter 固定随机种子为 1234，但采样结果仍应视为
随机生成结果。完成 code-faithful 评估后，再建立独立的 greedy 对照，不能混在同一
结果目录中。

## 3. 服务器执行顺序

全部命令都在容器内执行：

```bash
cd /workspace/CoLT

bash scripts/a100/07_verify_final_model.sh
bash scripts/a100/08_setup_eval.sh
bash scripts/a100/10_eval_smoke.sh
```

指定其他物理 GPU（例如 GPU 6）时使用：

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/a100/10_eval_smoke.sh
```

每个评估进程只暴露一张物理卡，因此模型内部的 `cuda:0` 指向所选物理卡。CoLT
自定义模型固定使用显式单卡 `device_map`；不要改回 `device_map="auto"`，否则
Accelerate 的 tied-parameter 自动切分会在嵌套 decoder 上失败。

`08_setup_eval.sh` 使用 `eval_constraints.txt` 约束训练环境的核心版本，并以
`--no-deps` 安装 VLMEvalKit 本身。不要直接执行无约束的
`pip install -e Evaluation/VLMEvalKit`，否则 pip 可能升级 NumPy、Transformers、
Hugging Face Hub、Gradio、Pydantic 和 safetensors，破坏已经验证过的训练环境。
无界面 A100 服务器固定使用 `opencv-python-headless==4.11.0.86`，避免普通
`opencv-python` 在导入时依赖宿主机的 `libGL.so.1`。

smoke test 只从 MMStar 中取 8 条数据。必须确认：

- 单卡能够完整加载根目录模型；
- 8 条图片都能解码；
- 没有 CUDA OOM、缺权重、缺 adapter 或输入格式错误；
- 结果目录中生成预测 xlsx 和评分 csv。

脚本每次创建独立的带时间戳结果目录，不复用失败运行。只有预测文件包含完整 8 条非空
回答且评分 CSV 存在时才会打印完成；VLMEvalKit 内部捕获的模型/数据组合异常也会转成
非零退出码。

smoke test 通过后，运行第一阶段：

```bash
tmux new-session -A -s colt-eval
cd /workspace/CoLT
bash scripts/a100/11_eval_phase.sh phase1
```

如果要先用全部 8 张 GPU 对完整 MMStar 做数据并行评估：

```bash
COLT_ALLOW_FULL_ROOT=1 bash scripts/a100/12_eval_mmstar_8gpu.sh
```

该脚本使用 `torchrun --nproc_per_node=8`，每张卡加载一份完整 CoLT 模型，每个 rank
处理约 `1500 / 8` 条样本，最后由 rank 0 合并和评分。它不是张量并行，也不使用
vLLM。可通过 `COLT_EVAL_GPUS=0,1,2,3,4,5,6,7` 指定八张物理卡；脚本会拒绝重复、
非法或显存占用不低于 500 MiB 的卡。

结果目录包含 seed 和评估代码/模型索引指纹，避免未提交代码变化后误用旧预测。脚本不使用
VLMEvalKit 的跨历史 `--reuse`；同一指纹运行若中断，仍会从各 rank 的中间 pickle 继续，
成功合并后这些中间文件会自动删除。

MMStar 验收通过后，可用一个 `torchrun` 进程组连续跑完剩余 7 个论文数据集：

```bash
tmux new-session -A -s colt-eval
cd /workspace/CoLT

COLT_EVAL_GPUS=0,1,2,3,4,5,6,7 \
COLT_ALLOW_FULL_ROOT=1 \
bash scripts/a100/13_eval_remaining_8gpu.sh
```

该脚本只下载并运行以下 7 项，不会重复运行已经完成的 MMStar：

```text
ChartQA_TEST
AI2D_TEST
MMBench_DEV_EN
ScienceQA_TEST
TextVQA_VAL
MMT-Bench_VAL
SEEDBench_IMG
```

8 个 rank 各自只加载一次完整模型，随后对每个数据集共同分片；数据集之间不会重复加载
模型。结果隔离在 `seed + 评估代码 + 模型元数据` 的指纹目录中，因此脚本在该目录内启用
`--reuse` 是安全的：已完成的数据集直接复用，中断数据集从各 rank 的 pickle 继续。重新
上传修改后的评估代码或更换模型会生成新指纹，不会串用历史预测。

A100 评估环境启用原子写入：rank pickle、最终 xlsx/csv 和首次解码的图片都会先写入
同目录临时文件，再原子替换正式文件。因此容器在写文件期间被强制停止时，正式路径不会
留下半个 pickle、xlsx 或图片。

执行顺序有意将 ChartQA 等较小且高价值的任务放在前面，将样本最多的 SEEDBench_IMG
放在最后，以便共享服务器再次发生外部中断时尽可能保留更多完整结果。

脚本退出前会逐项验证 TSV 的结构化记录数、完整且唯一的 `index` 集合、非空回答、失败
回答标记和 `_acc.csv`，并输出与论文分数的百分点差。只有 7 项全部通过才打印完成。
中断后使用完全相同的命令重启即可；不要删除对应指纹目录中的 rank pickle。

需要注意，作者 `generate()` 强制使用随机采样。续跑会保留所有已经完成的样本，但新进程
会重新初始化随机数状态，因此续跑后尚未完成样本的随机序列不保证与一次无中断运行逐
token 相同；这不影响结果完整性，但若要做严格的逐 token 可复现对照，应保证整轮不中断。

## 4. 初始 Qwen3-VL 文本推理基线

论文对初始 `Qwen3-VL-8B-Instruct` 报告了两种设置：

| 设置 | 论文 8 项平均分 | 本地脚本 |
|---|---:|---|
| 直接回答 | 69.5 | 未在本轮运行 |
| 文本推理（Textual CoT） | 75.7 | `14_eval_base_qwen3vl_cot_8gpu.sh` |

本轮默认复现文本推理基线，因为它与 CoLT 使用相同的思考提示，是更公平的初始化模型
对照。该脚本在模型构造前关闭 latent 模式，因此：

- 只加载 `/workspace/models/Qwen3-VL-8B-Instruct`；
- 不创建 `decoder`、`backward_decoder`、`prj` 或 `latent_predictor`；
- `forward()` 与 Transformers v4.57.0 官方实现一致；
- `generate()` 使用 Hugging Face 原生 greedy decoding，`max_new_tokens=8192`；
- 结果与 CoLT 目录完全隔离，并与论文文本推理一行逐项比较。

在 tmux 中运行全部 8 个数据集：

```bash
tmux new-session -A -s qwen-base-eval
cd /workspace/CoLT

COLT_EVAL_GPUS=0,1,2,3,4,5,6,7 \
COLT_ALLOW_FULL_ROOT=1 \
bash scripts/a100/14_eval_base_qwen3vl_cot_8gpu.sh
```

模型和评估数据已经在 CoLT 训练与上一轮评估中下载，脚本只会重新校验，不会重复下载
完整文件。输出位置：

```text
/workspace/eval/results/baseline_qwen3vl_cot/all8/
/workspace/logs/eval/qwen3vl_base_cot_all8_8gpu_*.log
```

如被外部停止，使用同一命令重启即可保留已完成预测并续跑。文本推理基线采用 greedy
decoding，因此相同代码、模型和输入下不会出现 CoLT 随机采样的续跑随机序列变化问题。

## 5. Docker 保护与旧版分阶段评估

如果 Docker overlay 所在的共享宿主盘已满，但 `/workspace` 仍有足够空间，先让管理员
释放 Docker 根目录空间。仅在无法立即释放、并且确认本配置中的缓存、临时文件、数据、
日志和结果路径均已指向 `/workspace` 或 `/dev/shm` 时，才可显式启用受控绕过：

```bash
export COLT_ALLOW_FULL_ROOT=1
bash scripts/a100/10_eval_smoke.sh
```

该开关不会改变任何存储路径，只允许通过根文件系统剩余空间保护检查。脚本会再次校验
关键可写路径；不要直接删除保护逻辑，也不要在共享服务器上执行 `docker system prune`。

Phase 1 使用 GPU 0、1、2 并行运行：

```text
ChartQA_TEST
MMStar
MMBench_DEV_EN
```

原始下载量只有约 216 MiB。如果 ChartQA 和 MMStar 均比论文低超过 5 个百分点，
先停止扩大评估，检查模型加载、回答提取和随机生成语义。

Phase 1 正常后运行主要增益任务：

```bash
bash scripts/a100/11_eval_phase.sh phase2
```

Phase 2 使用 GPU 0、1：

```text
TextVQA_VAL
MMT-Bench_VAL
```

最后补齐其余任务：

```bash
bash scripts/a100/11_eval_phase.sh phase3
```

Phase 3 使用 GPU 0、1、2：

```text
SEEDBench_IMG
ScienceQA_TEST
AI2D_TEST
```

如需指定其他空闲 GPU：

```bash
COLT_EVAL_GPUS=3,4,5 bash scripts/a100/11_eval_phase.sh phase1
```

每个任务使用独立结果目录和日志目录：

```text
/workspace/eval/results/codefaithful/<dataset>
/workspace/logs/eval
```

脚本会错开 60 秒加载模型，避免多个进程同时从 NFS 读取约 20 GiB 权重。可调整：

```bash
COLT_EVAL_STAGGER_SECONDS=120 bash scripts/a100/11_eval_phase.sh phase1
```

## 6. 下载与校验

公开 OpenCompass OSS 站点当前证书已过期。下载脚本先尝试正常 TLS；失败后才使用
`curl -k`，并强制核对仓库内记录的文件字节数和 MD5。因此证书绕过不会绕过数据
完整性校验。

也可以只下载指定阶段：

```bash
bash scripts/a100/09_download_eval_data.sh phase1
bash scripts/a100/09_download_eval_data.sh phase2
bash scripts/a100/09_download_eval_data.sh phase3
```

中断后重新运行会继续 `.part` 文件并在完成后校验。

## 7. Checkpoint 清理

`checkpoint-1000` 已由 Trainer 自动删除。`checkpoint-1910` 是当前唯一包含优化器和
随机状态的完整恢复点。由于输出根目录已经另存了完整推理模型，可在以下条件全部满足
后删除 `checkpoint-1910`：

1. `07_verify_final_model.sh` 通过；
2. `10_eval_smoke.sh` 成功加载根目录模型；
3. 训练日志、运行配置和 git diff 已备份；
4. 不再计划按相同目标函数从 step 1910 继续训练。

删除前再次确认路径：

```bash
du -sh /workspace/outputs/colt_codefaithful/checkpoint-1910
test -f /workspace/outputs/colt_codefaithful/model.safetensors.index.json
```

满足条件后仅删除恢复 checkpoint，绝不能删除输出根目录的 5 个 safetensors：

```bash
rm -rf -- /workspace/outputs/colt_codefaithful/checkpoint-1910
```

## 8. 图像预处理 A/B 诊断

`15_eval_preprocess_ab_8gpu.sh` 仅评估 `AI2D_TEST` 和 `TextVQA_VAL`，依次运行：

```text
legacy14_processor_resize
model_patch_processor_resize
model_patch_no_processor_resize
```

第一项复现原适配器的 `image_patch_size=14`；后两项使用 Qwen3-VL processor
报告的 patch size，并强制校验其为 16。第三项还会向 processor 传入
`images_kwargs={"do_resize": false}`。脚本固定使用 baseline 的 greedy decoding，
不需要、也不会执行逐样本重设随机种子。
每个 profile 使用独立的 work directory，且脚本不启用跨 profile 的预测复用。为避免
陈旧的 rank 临时文件混入新实验，每次启动还会创建带时间戳的新实验根目录；若该次运行的
profile 目录意外非空，脚本会直接拒绝运行。

只测试原始 Qwen3-VL textual-CoT baseline：

```bash
cd /workspace/CoLT
COLT_EVAL_GPUS=0,1,2,3,4,5,6,7 \
COLT_ALLOW_FULL_ROOT=1 \
bash scripts/a100/15_eval_preprocess_ab_8gpu.sh
```

该诊断固定评估 `Qwen3-VL-8B-Instruct-BASE-COT`，不加载 CoLT checkpoint，也不支持用环境变量切换到 CoLT。

所有输出与正式结果隔离：

```text
/workspace/eval/results/diagnostic_preprocess/
/workspace/logs/eval/preprocess_ab_baseline_8gpu_*.log
```

完成三项 baseline 评估后会生成 `preprocess_ab_summary.csv`，包含论文 baseline 分数、
相对论文的 gap、相对旧 `image_patch_size=14` profile 的分差、预测变化数、逐题改善/回退数量和长输出计数。

## 9. Baseline 256-token 生成上限诊断

官方 CoLT 潜在推理路径会在模型内部强制 `max_new_tokens=256`，但原始 Qwen3-VL
baseline 不进入该路径，之前的 baseline 评估实际使用 `8192`。以下诊断保持原始
baseline、greedy decoding 和 `legacy14_processor_resize` 不变，只把生成上限改为 256，
并仅运行 `AI2D_TEST` 与 `TextVQA_VAL`：

```bash
cd /workspace/CoLT
COLT_EVAL_GPUS=0,1,2,3,4,5,6,7 \
COLT_ALLOW_FULL_ROOT=1 \
bash scripts/a100/16_eval_base_max256_8gpu.sh
```

结果和日志分别写入：

```text
/workspace/eval/results/diagnostic_generation/base_max256/
/workspace/logs/eval/qwen3vl_base_legacy14_greedy_max256_8gpu_*.log
```

该脚本使用独立模型名、评估 ID、fingerprint 和结果目录，不会读取或覆盖原有
8192-token baseline 与预处理 A/B 结果。中断后用同一命令可以在相同 fingerprint
目录内继续。

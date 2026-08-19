# CoLT-reproduction 项目操作手册（AGENTS.md）

本文件记录本项目**已验证**的路径规范、技术栈、环境、缓存与操作约束，供 agent 与开发者共用。
协作规则、质量与安全要求见全局 `AGENTS.md`（工作区根目录），本文件不重复全局规则。

## 1. 项目定位与当前状态

在 CoLT（3-step latent reasoning）基础上做 CMPO 阶段：先让 latent 携带视觉语义
（visual grounding SFT），再做 multi-path latent RL。当前（2026-08-16）：

- replay step-grounding SFT 全量完成（checkpoint-500/1000/1500/1986 + 最终根目录）；
- reward 可区分性门禁**判 SFT**（answer reward dead、grounding reward 平坦）；
- 下一步：SFT v2 = 新视觉 CoT 数据（每步 bbox + 文本 CoT + 真实 answer，全监督）
  60-70% + OneThinker replay 30-40%，并开启 `COLT_IMAGE_MASK_PROB` 强制 answer→latent 依赖。

详细进度见 `Markdown/会话记录/20260816_ReplayStepGroundingSFT_RewardGate与TrainingConsistent_实验进度与结果记录.md`。

## 2. 目录与路径规范（已核验）

```text
仓库根        /home/dataset-local/lkl/CoLT-reproduction
训练数据      /home/dataset-local/lkl/datasets/CoLT_Train_Dataset/   （json + dataset_info.json 注册表）
原始数据      /home/dataset-local/lkl/datasets/LVR_Train_Dataset/    （visualcot_98k 等）
模型权重      /home/dataset-local/lkl/models/{Qwen3-VL-8B-Instruct, Qwen3-0.6B, CoLT-8B}
checkpoints   checkpoints/colt_paper_faithful_*
训练日志      logs/colt_paper_faithful_train_*.log
编排/后台日志 logs/background/
评测数据      eval/LMUData/*.tsv；评测结果 eval/results/paper-faithful/...
评测工具      scripts/lkl_8gpu/（唯一入口 colt.sh；commands/ lib/ tools/ experiments/）
训练配置      LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_*.yaml
DeepSpeed     LLaMA-Factory/examples/deepspeed/ds_z3_8gpu.json
缓存根        /home/dataset-local/lkl/cache/（HF / torch / cuda / wandb / colt tokenized）
临时文件      /home/dataset-local/lkl/tmp/
```

绝对/相对约定：`colt.sh train/eval` 的 `--config`、`--output-dir`、`--model-path` **必须是绝对路径**。

## 3. 环境与激活

```bash
source /home/dataset-local/lkl/colt-local.env        # 所有 COLT_* 路径变量
source /opt/conda/etc/profile.d/conda.sh
conda activate colt                                   # /home/dataset-local/lkl/envs/colt，Python 3.11
```

技术栈（已验证）：Python 3.11 · PyTorch · transformers 4.57.0（**vendored editable**：
`transformers-4.57.0/src/transformers`，改模型代码改这里）· flash-attn 2 · DeepSpeed ZeRO-3 ·
LLaMA-Factory（`LLaMA-Factory/src/llamafactory`）· Qwen3-VL-8B-Instruct · qwen-vl-utils · PIL。

硬件：8× NVIDIA A100-SXM4-80GB。训练必须 8 卡；grounding 评测脚本要求**恰好 1 张可见 GPU**
（`CUDA_VISIBLE_DEVICES=0`）。

## 4. 训练

唯一入口：

```bash
bash scripts/lkl_8gpu/colt.sh train paper-faithful \
  --config <abs yaml> --output-dir <abs dir> [--batch-aux]
```

关键 env（沿用 2026-08-16 一轮）：

```text
COLT_VISUAL_GROUNDING=1           开启 grounding 损失
COLT_VISUAL_GROUNDING_WEIGHT=0.2  grounding 权重
COLT_STOCHASTIC_LATENT=0          RL 前关闭
COLT_ANSWER_VISIBILITY=full
COLT_BATCH_AUX_DECODERS=1         decode 合批
COLT_KL_ANCHOR=0                  本轮未启用
COLT_IMAGE_MASK_PROB=0            SFT v2 将开启 0.3~0.5（视觉 CoT 行）
```

数据接入：json 须注册进 `/home/dataset-local/lkl/datasets/CoLT_Train_Dataset/dataset_info.json`
（sharegpt 格式；可选 `step_bboxes`、`visual_only` 列）。混合数据集用
`mix_strategy: interleave_over` + `interleave_probs: "0.70,0.30"`。
tokenized 缓存：`/home/dataset-local/lkl/cache/colt/<name>_tokenized`（LLaMA-Factory 生成）。

训练约束（已踩坑）：

- `train.sh` 会校验 config 的 `output_dir` 与 `--output-dir` 一致；
- `save_total_limit` 必须 ≥ 想保留的 checkpoint 数（当前一轮为 10）；
- 1 epoch ≈ 1986 步（88,833 OneThinker + 30,000 GQA，per-device 1 × 8 × grad-accum 8）；
- 训练输出 dir 非空且未声明 `--resume` 会拒绝启动。

## 5. 评测

```bash
bash scripts/lkl_8gpu/colt.sh eval paper-faithful all8 \
  --model-path <abs ckpt> --gpus 0,1,2,3,4,5,6,7 \
  --generation respect-args --empty-response-policy prevent
```

latent transition：**默认已改为 `training-consistent`**（与训练递推一致；`baseline` 目标强制 official）。
需要与官方 CoLT 旧结果对比时显式传 `--latent-transition official`。两种模式进不同 fingerprint，
禁止混用预测。

中间 checkpoint 评测前置（`verify_model.py` 硬校验，缺了必失败）：

- 写最小 `train_results.json` 标记到 checkpoint 目录（缺失时）；
- `export COLT_EXPECTED_GLOBAL_STEP=<checkpoint 的 trainer_state.json 真实 global_step>`。

grounding / 门禁工具（tools/）：

```text
evaluate_grounding_score.py       轨迹级 own/shuffled/drop（1 GPU，n=100）
evaluate_step_grounding_score.py  step 级 hard-negative + 跨图 shuffle（1 GPU）
gate_reward_discriminability.py   reward 门禁（--shard-id/--n-shards 支持多卡分片）
merge_gate_shards.py              合并门禁分片
audit_visual_cot_bbox_alignment.py bbox↔CoT 对齐审计
prepare_gqa_step_grounding.py     GQA functional program → step_bboxes
build_gqa_case_study_html.py      数据案例可视化 HTML
```

## 6. 缓存与网络（本机实测约束）

```text
HuggingFace       不用代理；unset 代理后 export HF_ENDPOINT=https://hf-mirror.com
海外资源          代理 http://127.0.0.1:7890（mihomo 常驻）
国内资源/pip/conda 不走代理，走国内镜像
HF_HOME           /home/dataset-local/lkl/hf-cache（HF_HUB_OFFLINE=1 常规开启）
wandb             WANDB_MODE=offline，日志在 logs/wandb
```

## 7. 已知坑（实战沉淀，改代码前必读）

1. 模型**第一个参数是 float32 标量 `alpha`**：不要用 `next(self.parameters()).dtype` 当计算 dtype，
   会得到 float32；用 `self.lm_head.weight.dtype`（bf16）。
2. eval 下跑训练式 forward（带 labels）必须包 `torch.autocast("cuda", dtype=torch.bfloat16)`，
   否则 `q_proj` dtype mismatch。
3. 复用 tokenized cache 的 `input_ids` 时，图像必须复刻 LLaMA-Factory `mm_plugin._preprocess_image`
   缩放（max_pixels=802816 / min_pixels=1024），否则 image_pad 数与视觉特征数不匹配。
4. grounding 对比损失的语义依赖 **batch=1**（batch>1 会自动引入跨图负样本，改变被测量指标）。
5. training-consistent 递推 `H + α·prj(H)` 在纯 eval 下会 float32 提升（训练被 autocast 掩盖）；
   `_forward_latent_reasoning` 的 latent 注入点已有 `.to(lm_head.weight.dtype)` 修复。
6. 数据侧：面积 >0.5 的 step bbox 会稀释 grounding 指标；select→select→choose 比较题的
   V3 可能指向非答案对象（30K 中 281 条，~52% 错位）。新数据集接入前务必做对齐审计。
7. 后台长任务用 `setsid nohup bash <script> > logs/background/<name>.log 2>&1 < /dev/null &`；
   `TS=$(date ...)` 与 `&` 同句时变量在后台子 shell 内生效，前台读不到（log 文件仍会正确生成）。
8. 门禁结论：当前 checkpoint answer reward dead（最终 latent 不进 answer CE 图）；
   RL 前必须让 answer 依赖 latent（image-mask 信息瓶颈）并重跑门禁验证。

## 8. 编排脚本（experiments/）

```text
auto_replay_step_grounding_pipeline.sh  训练 → 逐 checkpoint all8+grounding → 旧锚点 all8
auto_run_reward_gate.sh                 8 卡分片门禁 → training-consistent 复测（可 COLT_GATE_SKIP_GATE=1）
auto_eval_after_train.sh                SFT 后 all8 + 干预评测
```

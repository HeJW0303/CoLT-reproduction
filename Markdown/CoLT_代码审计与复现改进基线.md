# CoLT 代码审计、复现路线与改进基线

> 审计日期：2026-07-12  
> 本地仓库：`/Users/hejianwei/Documents/project/CoLT`  
> 官方提交：`331cc78df2d4ab542b9a83822a5a69766e194042`  
> 论文：*CoLT: Teaching Multi-Modal Models to Think with Chain of Latent Thoughts*（arXiv:2606.31986v2）
> 二次复核：2026-07-12，在线 PDF、arXiv LaTeX 源码、GitHub `main` 三方交叉核验

## 0. 这份文档的用途

本文不是论文内容的重复讲解，而是后续复现和算法改进的工程基线。它回答五个问题：

1. 当前官方代码实际如何训练和推理；
2. 论文公式与代码实现是否一致；
3. 当前仓库能否直接复现论文结果；
4. 在投入 8×80GB GPU 前必须通过哪些最小验证；
5. 哪些现象适合作为后续改进方向，哪些只是实现错误，不能包装成新方法。

结论分为三种证据等级：

- **代码确认**：可从当前提交的静态代码直接确定；
- **待实验确认**：静态代码显示高风险，但需运行时梯度、权重或 profiler 证据；
- **外部报告**：来自公开 issue，只能作为检查线索，不能当作已证实结论。

### 在线复核证据

- 用户提供的无版本 arXiv URL 当前解析为 `2606.31986v2`（更新于 2026-07-03）。
- 在线 v2 PDF 与本地论文 PDF 的 SHA-256 均为 `f4dbbdaa512bdd13accbbb8bd4c7604cf65dcf70e4bea68c16407ff55e94e731`。
- arXiv v1 与 v2 的式（3）、（5）、（7）、（9）相同；v2 的主要训练描述变化是删除 v1 声称的 GRPO 第二阶段，改为 SFT-only，这一点与当前代码更匹配。
- GitHub `main` 与本地 HEAD 均为 `331cc78df2d4ab542b9a83822a5a69766e194042`；远端当前没有更新修复。
- 远端 `modeling_qwen3_vl.py`、README 与本地对应文件哈希一致。

## 1. 先给结论

### 1.1 当前仓库不具备“干净 clone 后直接复现”的条件

至少存在三个阻断项：

1. README 的训练脚本名写错。README 使用 `run_colt_sft.sh`，实际文件是 `run_colt.sh`。
2. 当前 checkout 和官方数据发布清单都没有 `dataset_info.json`，配置引用的 `onethinker_sft_image` 无法直接解析；需要自行补充 LLaMA-Factory 数据注册。
3. VLMEvalKit 无条件导入 `qwen3_vl` wrapper，但仓库没有对应文件或包，因此 README 的评测命令无法原样启动。

### 1.2 准确总判断：框架匹配，但关键实现存在实质差异

不能笼统地说“整篇论文与代码完全匹配不上”。以下宏观设计是匹配的：

- v2 采用 SFT-only 训练；
- 主干是 Qwen3-VL-8B-Instruct；
- 外部 decoder 使用 Qwen3-0.6B；
- 默认 latent step 数为 `K=3`；
- 三项辅助损失权重均为 0.2；
- textual CoT 动态切为 3 段；
- 最终目标由答案 loss 与 forward、backward、internal 三项监督组成。

但是，决定梯度方向、token 监督位置和推理行为的若干关键细节与论文不一致。这些不是仅由命名差异造成的：

当前代码至少有以下高优先级差异：

| 项目 | 论文 | 当前代码 | 影响 |
|---|---|---|---|
| latent 递推 | 式（3）描述 hidden state 直接作为下一位置输入 | 训练为 residual+`alpha*prj`，推理为纯 `prj` | 训练/推理互不一致，且都加入论文未说明的变换 |
| 内部监督 | cosine similarity + stop-gradient | MSE + stop-gradient | 优化几何与论文不同 |
| backward stop-gradient | 停止 latent target 梯度 | `detach()` decoder hidden | 梯度方向与公式相反 |
| forward CE | 一次 causal shift | 代码手工 shift 后又调用会 shift 的 `ForCausalLMLoss` | 默认训练路径确定发生双重 shift |
| 生成配置 | 评测配置 `do_sample=False` | 强制 `do_sample=True`、最多 256 token | 结果随机且配置记录失真 |

因此，后续工作不能把“论文算法”和“当前 GitHub 实现”视为同一个无歧义对象。第一阶段必须分别建立：

- **paper-faithful 版本**：严格按论文公式实现；
- **code-faithful 版本**：保持官方提交行为，用于复核公开 checkpoint；
- **fixed-code 版本**：只修明确工程错误，用于公平研究比较。

### 1.3 二次复核后的置信等级

| 判断 | 复核结论 | 置信度/边界 |
|---|---|---|
| internal loss：论文 cosine，代码 MSE | 确定 | arXiv v1/v2 式（9）均为 cosine；代码明确调用 `F.mse_loss` |
| backward stop-gradient 对象相反 | 确定 | 论文 `sg(h_k)`；代码 `cot_hidden.detach()`，梯度所有权相反 |
| forward CE 双重 shift | 默认训练路径下确定 | 除非外部运行时覆写 `_loss_function`；仓库中未发现该覆写 |
| 训练/推理 latent dynamics 不同 | 确定 | 两条代码路径的初始状态和递推公式均不同 |
| `generate()` 覆盖调用参数 | 确定 | 强制 sampling 和 256 token，与 VLMEvalKit 配置冲突 |
| `freeze_vision_tower` 未完整冻结 | 确定的配置实现缺口 | 论文未逐模块声明冻结范围，不应称为公式冲突 |
| batch 右 padding 取错 latent | 条件性确定 | batch>1 且样本长度不同时触发；官方 microbatch=1 会掩盖 |
| `K=3` 实际计算位置可能多 1 | 高风险，待 profiler 锁定 | 静态路径显示最终 latent 还会再作为答案前缀输入；术语计数可能不同 |
| 公开 checkpoint latent 权重异常 | 未确认 | 第三方 issue；作者已回复将稍后检查，尚未给结论 |

### 1.4 现在不应直接开始完整训练

完整训练之前，必须先完成：

1. 公开 checkpoint 权重审计；
2. 单 batch 标签解析可视化；
3. 四项 loss 的梯度归属检查；
4. train/eval latent 递推一致性测试；
5. 8 个 benchmark 的精确数据键、judge 和生成参数冻结；
6. 1、8、32 个样本的端到端 smoke test。

否则即使训练跑完，也无法判断结果差异来自算法、数据、loss 错位、随机采样还是评测协议。

## 2. 仓库结构与真实入口

### 2.1 训练入口

实际调用链是：

```text
LLaMA-Factory/local_scripts/run_colt.sh
  -> llamafactory-cli train
  -> examples/train_full/colt_qwen3_sft.yaml
  -> llamafactory.train.tuner._training_function()
  -> run_sft()
  -> load dataset / load model / build collator
  -> CustomSeq2SeqTrainer.train()
  -> Qwen3VLForConditionalGeneration.forward()
```

关键文件：

- [`run_colt.sh`](LLaMA-Factory/local_scripts/run_colt.sh)
- [`colt_qwen3_sft.yaml`](LLaMA-Factory/examples/train_full/colt_qwen3_sft.yaml)
- [`tuner.py`](LLaMA-Factory/src/llamafactory/train/tuner.py)
- [`workflow.py`](LLaMA-Factory/src/llamafactory/train/sft/workflow.py)
- [`modeling_qwen3_vl.py`](transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)

训练开始时，作者额外把 YAML、模型实现和启动脚本复制到输出目录，这对追踪实验代码版本是有价值的；但它没有自动保存 Git commit、环境锁、数据哈希或评测协议。

### 2.2 评测入口

预期调用链是：

```text
Evaluation/VLMEvalKit/run.py
  -> supported_VLM[model_name]
  -> Qwen3VLChat wrapper
  -> model.generate()
  -> 自定义 latent_reasoning_generate()
  -> dataset.evaluate()
```

但 `vlmeval/vlm/__init__.py` 导入 `.qwen_vl`、`.qwen2_vl`、`.qwen3_vl`，当前 Git tree 中三者均缺失。`.gitignore` 还有 `Qwen*` 规则，可能导致作者本地 wrapper 未被提交。

这意味着不能简单从最新上游 VLMEvalKit 随便复制一个文件：必须先确定作者使用的 wrapper 版本、prompt 拼装、post-process 和模型加载参数，否则结果协议会漂移。

## 3. 数据链路

### 3.1 配置期望的数据

训练配置为：

```yaml
dataset: onethinker_sft_image
dataset_dir: LLaMA-Factory/data
template: qwen3_vl
cutoff_len: 16384
tokenized_path: LLaMA-Factory/cache/onethinker_sft_tokenized
```

LLaMA-Factory 会读取：

```text
LLaMA-Factory/data/dataset_info.json
```

并在其中查找 `onethinker_sft_image`。当前仓库没有整个 `LLaMA-Factory/data` 目录，因此当前状态必然无法加载该数据集。README 本来就要求额外下载并软链接数据，所以“Git 仓库不含训练数据”本身不是缺陷；真正的复现缺口是官方数据发布包也没有直接提供 LLaMA-Factory 所需的 `dataset_info.json`，用户还需自行把 `colt_sft_image.json` 注册为配置所引用的 `onethinker_sft_image`。

### 3.2 数据格式的隐式硬约束

模型没有使用 Trainer 传入的 `labels` 来切分 CoT 与答案，而是重新从完整 `input_ids` 中解析：

1. 从后向前找最后一个 `<think>` token；
2. 向后找第一个 `</think>` token；
3. 从后向前找 token id 等于字符串 `answer` 的最后位置；
4. 用固定偏移 `think_end + 2` 和 `answer_end + 4` 截取答案。

这带来以下风险：

- 标签必须严格是 `<think>...</think><answer>...</answer>` 的特定 tokenization；
- 正文中出现单词 `answer` 可能改变结束位置；
- chat template、tokenizer 或空白变化可能使固定偏移失效；
- 解析错误可能不是显式异常，而是静默截错答案范围；
- `cutoff_len=16384` 截断后若丢失标签，会在模型 forward 内报错。

### 3.3 数据到 loss 的实际路径

```text
原始样本
  -> LLaMA-Factory dataset converter
  -> qwen3_vl chat template
  -> input_ids + pixel_values
  -> 模型内部重新拆为 question / textual CoT / answer
  -> textual CoT 按标点附近动态切成 K=3 段
  -> 三段分别监督三个 latent step
  -> answer 重新构造 teacher-forcing labels
```

首轮数据审计必须输出以下统计：

- 样本总数、图像文件缺失数、重复图像数；
- `<think>`、`</think>`、`<answer>`、`</answer>` 格式通过率；
- tokenized 长度分布和 16384 截断率；
- 每个 CoT split 的 token 数分布；
- 空 CoT、极短 CoT、正文含 `answer` 的比例；
- parser 重建答案与原始答案逐 token 一致率。

## 4. 训练范围与资源

### 4.1 不是 LoRA

配置使用：

```yaml
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
```

因此它是语言侧全参数 SFT，而不是 LoRA。LLaMA-Factory 的 full tuning 会冻结命中 forbidden module 名称的参数，其余参数保留 `requires_grad=True`。

### 4.2 声称的视觉冻结并不完整

Qwen3-VL 注册的冻结键只有：

```text
visual.patch_embed
visual.blocks
visual.merger
```

但视觉模型还有：

```text
visual.pos_embed
visual.deepstack_merger_list
```

它们不匹配上述键，静态代码显示仍会保持可训练。这一点在复现时要通过参数清单实测确认，并决定：

- code-faithful：保留官方遗漏；
- paper-faithful/fixed-code：完整冻结视觉侧。

### 4.3 两个 0.6B decoder

模型构造时加载两个独立的 `Qwen/Qwen3-0.6B`：

- `decoder`：latent -> textual CoT；
- `backward_decoder`：textual CoT -> latent alignment。

代码对 backward decoder 的最后 hidden state 执行 `detach()`，所以 backward loss 不会训练该 decoder；但在 full tuning 下，其参数仍可能被优化器/ZeRO 收录、分片和保存。必须通过运行时检查确认：

```text
parameter.requires_grad
parameter.grad is None
optimizer param group membership
checkpoint state_dict membership
```

如果确认全程零梯度，冻结或共享 backward decoder 是直接的资源改进，但应先归类为实现修正/效率优化，而非新的算法贡献。

### 4.4 batch size 口径不一致

论文写 batch size 8；YAML 为每卡 batch 1、梯度累积 8。若使用 README 声明的 8 张 GPU，则有效 global batch 通常是：

```text
1 × 8 grad accumulation × 8 GPU = 64
```

论文可能指 microbatch、每卡累计 batch 或全局 batch，但当前材料无法消除歧义。复现日志必须明确记录 `world_size`、microbatch、accumulation 和 effective global batch。

## 5. 模型与四项损失

### 5.1 训练时 latent 递推

训练先编码 question 与图像，取序列最后 hidden 作为初始 latent：

```text
z0 = last_hidden(question, image)
```

每一步：

```text
Ht = LM(inputs_embeds=zt, shared_KV_cache)
z(t+1) = Ht + alpha * prj(Ht)
```

默认 `K=3`，`alpha` 是可学习标量，初始化为 0.1。

### 5.2 最终答案损失

最后一个 latent 与答案 token embedding 拼接，继续使用主干 LM teacher forcing：

```text
L_answer = CE(answer logits, answer labels)
```

它向主干语言模型、`lm_head` 和生成 latent 的历史路径回传梯度。

### 5.3 forward decoder loss

每个 latent 经 `pj_in` 和可学习 scale 后，作为 0.6B decoder 的前缀 embedding，要求恢复对应 CoT 段：

```text
latent -> pj_in -> scale -> decoder -> pj_out -> CoT CE
```

高风险点是代码先手工执行：

```python
shift_ref_logits = ref_logits[..., :-1, :]
shift_ref_labels = ref_labels[..., 1:]
```

然后调用 `self.loss_function`。Transformers 的 `ForCausalLMLoss` 在未提供 `shift_labels` 时还会再 shift 一次。因此当前实现很可能监督到错误的 token 位置。

必须用一个人工序列构造单元测试，打印每个 logit 位置实际对应的 target token，而不能只看 loss 是否下降。

### 5.4 backward alignment loss

代码实际执行：

```text
previous CoT step -> backward_decoder -> detach hidden
detach hidden -> pj_back -> cosine with current latent
```

梯度会流向 `pj_back` 和 latent 主干路径，不会流向 backward decoder。这与论文公式中对 latent target 使用 stop-gradient 的写法不同。

### 5.5 internal prediction loss

代码实际执行：

```text
pred = latent_predictor(current_latent)
target = stopgrad(next_latent)
L_prediction = MSE(pred, target)
```

论文式（9）写的是 cosine similarity loss，而不是 MSE。这不是等价替换：MSE同时约束方向和范数，cosine主要约束方向。

### 5.6 总损失

代码总损失是：

```text
L = L_answer + 0.2 L_forward + 0.2 L_backward + 0.2 L_prediction
```

三个辅助项分别按有效 step 数平均。当前 forward 每次还直接 `print` 四个 loss，多卡每步打印会显著污染日志并影响性能测量。

## 6. 训练与推理不一致

这是当前最重要的实现风险。

### 6.1 推理时 latent 递推

推理采用：

```text
z0 = prj(last_hidden(question, image))
z(t+1) = prj(LM(zt, shared_KV_cache))
```

与训练相比：

- 初始 latent 训练时不经过 `prj`，推理时经过；
- 训练时是 `H + alpha*prj(H)`，推理时是 `prj(H)`；
- 推理完全没有 residual 和 `alpha`。

因此两条路径不是同一递推系统。论文式（3）只描述 `LM^last` 得到的 hidden state 直接回灌，没有明确写 residual、`prj` 或 `alpha`；所以严谨结论是“训练和推理代码彼此不一致，且两者都增加了论文未说明的状态变换”，而不是论文明确要求其中某一种代码公式。优先实验应比较：

1. 官方 train dynamics + 官方 eval dynamics；
2. train/eval 都用 residual dynamics；
3. train/eval 都用 pure projection dynamics；
4. checkpoint 分别在两种推理 dynamics 下的零样本表现。

### 6.2 K=3 的计算口径

代码会进行 3 次 latent loop，但答案生成前还会把最终 latent 再输入主干一次。静态执行路径因此显示强烈的 off-by-one/计数口径风险；不过论文和代码可能对“latent state”与“作为输入的位置”采用不同边界定义，最终结论应由 KV-cache 增量和 backbone forward profiler 锁定。因此速度复现应统计：

- question/image encoding 的 backbone forward；
- latent loop 的 backbone forward 次数；
- 答案前缀和答案 token decoding 的 forward 次数；
- KV-cache 每阶段的长度变化。

不要只以 `num_latent=3` 代替真实计算量。

### 6.3 batch 大于 1 的 padding 风险

question 使用右 padding，却统一取 `hidden_states[:, -1, :]`。当 batch 内长度不同时，短样本取得的是 pad 位置 hidden，而不是最后一个有效 question token。

官方 `per_device_train_batch_size=1` 通常掩盖该问题；后续若提高 batch、做 sequence packing 或批量推理，必须先修正或验证。

## 7. 生成与评测协议

### 7.1 配置参数被模型覆盖

VLMEvalKit 为 CoLT 配置：

```text
do_sample=False
max_new_tokens=8192
```

但模型的 `generate()` 会丢弃这两个传入值，并强制：

```text
do_sample=True
max_new_tokens=256
temperature=传入值或默认 0.6
top_k=传入值或默认 20
```

所以“配置文件显示 greedy”并不等于实际 greedy。首轮复现必须记录最终生效参数，并至少做 3 个 seed；论文若只报告单次随机采样结果，其方差也应补测。

### 7.2 `--reuse` 不适合首轮复现

README 命令带 `--reuse`。VLMEvalKit 会复用以前日期目录中的预测或中间文件。首轮干净复现应去掉它，避免旧 checkpoint、旧 prompt 或旧生成参数污染结果。

### 7.3 论文八基准的候选数据键

依据当前 VLMEvalKit 注册，建议先锁定为：

```text
SEEDBench_IMG
MMBench_DEV_EN
ChartQA_TEST
TextVQA_VAL
ScienceQA_TEST
MMStar
AI2D_TEST
MMT-Bench_VAL
```

但论文没有披露 MMBench 的精确版本、judge 模型/API、VLMEvalKit commit。MMBench 在 judge 不可用时可能退化为 exact match，分数口径会变化。

### 7.4 当前无法独立核算论文数字

仓库没有发布：

- Table 1 的逐样本 predictions；
- score/metric 原始文件；
- 评测 commit 和完整配置快照；
- Table 7 的计时脚本；
- 输入编码与生成计时边界实现。

所以当前只能重新跑实验，不能从仓库已有 artifact 验算 `79.1`、`10.1×` 或 `22.6×`。

## 8. 公开 checkpoint 的额外风险

GitHub [issue #1](https://github.com/hulianyuyy/CoLT/issues/1) 在 2026-07-08 报告：发布 checkpoint 中 latent 相关模块的权重与 PyTorch 默认随机初始化非常接近，怀疑上传了错误权重。截至二次复核，该 issue 仍为 open；作者已于 2026-07-12 回复会稍后检查，但尚未确认或否认 checkpoint 问题。

这只是**外部报告**，不能直接断言 checkpoint 错误。下载大权重前后应执行独立审计：

1. 列出 checkpoint 所有 key、shape、dtype 和 shard；
2. 确认 `prj`、`latent_predictor`、`pj_in/pj_back/pj_out`、`alpha`、scale 是否存在；
3. 与同 seed 新初始化模型逐 tensor 比较；
4. 检查其范数、均值、标准差、最大差异；
5. 检查 decoder/backward decoder 是否随 checkpoint 发布；
6. 用公开 checkpoint 对同一样本分别跑官方 dynamics 和训练 dynamics；
7. 保存审计 JSON，不能只凭四位小数目测。

在通过此闸门前，不应把公开 checkpoint 的失败直接归因于 CoLT 方法本身。

## 9. 分阶段复现路线

### Phase 0：冻结基线

- 保留官方提交 `331cc78d` 不改；
- 从该提交创建 `repro/original-331cc78` 分支或 tag；
- 记录 OS、CUDA、driver、GPU、Python、PyTorch、Transformers、FlashAttention、DeepSpeed；
- 对数据 manifest、checkpoint 和配置计算 SHA256；
- 所有修复单独 commit，不覆盖原实现。

通过标准：任何实验都能追溯到代码、数据、权重、环境和生成配置。

### Phase 1：补齐发布物并做静态审计

- 下载训练数据，但先不训练；
- 确认 `dataset_info.json` 和 `onethinker_sft_image` 注册；
- 补齐作者实际使用的 Qwen3-VL VLMEvalKit wrapper；
- 锁定八基准数据键与 judge；
- 下载公开 checkpoint 并执行第 8 节权重审计。

通过标准：数据可解析、模型可加载、评测入口可 import，且没有隐式在线下载。

### Phase 2：CPU/单卡单元测试

必须新增以下测试：

1. `test_extract_think_content_roundtrip`；
2. `test_dynamic_cot_split_coverage`；
3. `test_forward_loss_token_alignment`；
4. `test_backward_gradient_ownership`；
5. `test_prediction_loss_matches_selected_formula`；
6. `test_train_eval_latent_transition_equivalence`；
7. `test_last_valid_token_with_right_padding`；
8. `test_effective_generation_config`；
9. `test_checkpoint_contains_all_latent_modules`。

通过标准：每个 target token、每个可训练模块和每项生效配置都可被断言。

### Phase 3：小样本闭环

- 先用 8-32 个样本过拟合；
- 记录四项 loss、各模块 grad norm、latent norm/cosine、显存峰值；
- 每个 checkpoint 固定一组可视化样本；
- 对 code-faithful、paper-faithful、fixed-code 三版跑同样输入；
- 验证保存后重载输出一致。

通过标准：答案 loss 可下降、latent 模块确实更新、保存/重载不丢参数、评测输出稳定。

### Phase 4：公开 checkpoint 基准

- 首轮去掉 `--reuse`；
- 先跑 1、8、32 个样本；
- 再跑完整八基准；
- 同时报告 greedy 与作者代码强制 sampling 两种结果；
- 对 sampling 至少报告 3 seed 的均值和标准差；
- 保存逐样本 prediction，不只保存最终分数。

通过标准：能明确说明与论文差异来自 checkpoint、wrapper、judge、生成配置还是指标版本。

### Phase 5：完整训练

先完成短跑资源估算，再决定 8×80GB 或缩小配置。完整训练至少记录：

- effective global batch；
- trainable/frozen parameter 清单；
- optimizer 中实际参数量；
- backward decoder 是否始终无梯度；
- checkpoint shard 和新增模块 key；
- 每项 loss 与 grad norm；
- 数据吞吐、token 吞吐、峰值显存；
- resume 后的数值连续性。

## 10. 改进方向的优先级

### A. 先做的实现纠偏

这些应先作为 correctness ablation，而不是“新方法”：

1. 统一训练与推理 latent dynamics；
2. 修复 forward CE 双重 shift；
3. 对齐论文的 backward stop-gradient 方向；
4. 对齐 internal cosine loss，或明确把 MSE 作为独立变体；
5. 完整冻结视觉模块；
6. 冻结/共享零梯度 backward decoder；
7. 用 attention mask 取最后有效 question hidden；
8. 让 `generate()` 尊重调用者参数；
9. 把 parser 从 token 偏移改成显式、可测试的结构化边界；
10. 删除 hot path 中逐步 `print`，改用 rank-0 结构化日志。

### B. 在正确基线上做的算法改进

1. **动态 K**：根据问题难度或 latent 收敛程度早停；
2. **transition consistency**：显式约束训练和推理使用相同状态转移；
3. **multi-scale CoT alignment**：避免仅按等比例附近标点切三段；
4. **decoder sharing/distillation**：一个冻结语义 encoder/decoder 是否足够；
5. **方向与范数解耦**：比较 cosine、MSE、whitened MSE 和 VICReg 类目标；
6. **latent intervention**：交换、遮蔽、扰动某一步，验证其是否具有因果作用；
7. **hybrid latent-text routing**：仅在置信度低时生成少量可见 CoT；
8. **token-budget matched evaluation**：与相同 FLOPs、相同延迟、相同输出预算基线比较。

### C. 论文主张需要补强的实验

- 多路径推理不能只由鲁棒性结果推断，需要显式多样性或因果实验；
- latent 可解释性不能只靠 decoder 重建文本，需要干预后答案变化证据；
- 速度不能只报 wall time，需要拆分预处理、视觉编码、latent loop、答案 decoding；
- 公平比较需要同 checkpoint、同 prompt、同 sampling、同 judge、同数据版本；
- ChartQA/TextVQA 的大增益要排除输出格式和 post-process 带来的指标收益。

## 11. 实验矩阵建议

最小但有判别力的矩阵：

| ID | dynamics | forward CE | backward stop-grad | internal loss | generation |
|---|---|---|---|---|---|
| C0 | 官方不一致 | 官方双 shift | detach decoder | MSE | 强制 sampling |
| C1 | 官方不一致 | 修复 | detach decoder | MSE | greedy |
| P0 | 论文一致 | 修复 | stopgrad latent | cosine | greedy |
| F0 | 统一 residual | 修复 | stopgrad latent | cosine | greedy |
| F1 | 统一 projection | 修复 | stopgrad latent | cosine | greedy |
| A0 | 最佳统一 dynamics | 修复 | stopgrad latent | cosine | adaptive K |

先在小数据和 2-3 个代表基准上筛选，再扩展八基准。否则组合爆炸会耗尽训练预算。

建议代表任务：

- ChartQA：结构化视觉与数值推理；
- TextVQA：OCR 和开放式答案；
- MMStar：视觉不可缺失的多选推理；
- MMT-Bench：多任务综合能力。

## 12. 复现时禁止混用的口径

后续报告必须始终区分：

- 论文公式 vs 官方 GitHub 代码；
- 官方公开 checkpoint vs 我们重新训练 checkpoint；
- code-faithful vs paper-faithful vs fixed-code；
- greedy vs sampling；
- 单次分数 vs 多 seed 均值；
- microbatch vs effective global batch；
- 3 个 latent 状态的概念计数 vs 实际 backbone forward 次数；
- 模型生成时间 vs 完整端到端时间；
- exact match vs LLM judge/circular evaluation。

## 13. 当前状态清单

| 项目 | 状态 |
|---|---|
| Git 对象完整性 | 已通过 `git fsck --full --strict` |
| 本地官方提交 | `331cc78df2d4ab542b9a83822a5a69766e194042` |
| Python 静态编译 | 关键模型文件与 LLaMA-Factory 已通过 `compileall` |
| 训练数据 | 未在本地仓库中，尚不可训练 |
| 数据注册 | `dataset_info.json` 缺失 |
| VLMEvalKit Qwen3 wrapper | 缺失，当前不可评测 |
| 公开 checkpoint | 尚未下载和独立审计；issue #1 尚待作者核查 |
| 论文主结果 | 尚未复现 |
| 论文速度结果 | 缺计时脚本，尚不可严格复现 |
| 原有论文中文讲解 | 保留，未覆盖 |

## 14. 下一步唯一合理顺序

```text
补齐数据 manifest 和评测 wrapper
  -> 审计公开 checkpoint
  -> 写 parser/loss/dynamics/gradient 单元测试
  -> 8-32 样本闭环
  -> 公开 checkpoint 小规模评测
  -> 固定协议的八基准评测
  -> 再决定是否完整训练
  -> 在 fixed-code 基线上做算法改进
```

当前最重要的不是马上改网络结构，而是先建立一个“结果可信”的实验地基。只有当数据、梯度、递推、checkpoint 和评测协议都能被逐项验证时，后续改进带来的提升才有研究意义。

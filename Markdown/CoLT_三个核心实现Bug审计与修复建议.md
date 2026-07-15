# CoLT 三个核心实现 Bug：证据、影响与修复建议

> 文档目的：为 CoLT 的官方代码复现、论文一致性核查和后续改进实验提供一个聚焦、可执行的审计基线。
>
> 审计对象：本仓库当前 `transformers-4.57.0` 与 `LLaMA-Factory` 实现。
>
> 结论日期：2026-07-14。

## 1. 总结

当前 CoLT 训练可以正常启动，8 卡分布式、BF16、DeepSpeed ZeRO-3、数据加载和总损失聚合均已实际运行。但源码中存在三个会影响论文复现解释的问题：

| 编号 | 问题 | 是否已由静态代码确认 | 主要影响 | 严重度 |
|---|---|---:|---|---:|
| B1 | forward CoT loss 发生双重 causal shift | 是 | CoT token 监督整体错位，forward supervision 不再对应论文公式 | 高 |
| B2 | `freeze_vision_tower` 没有冻结完整视觉塔 | 是 | 约 1.23 亿视觉侧参数仍训练，形成实验混杂 | 中 |
| B3 | backward stop-gradient 方向与论文相反，且 `backward_decoder` 被无效纳入优化器 | 是 | backward supervision 的梯度所有权改变，同时浪费训练资源 | 高 |

这三个问题不代表训练一定报错，也不代表最终 benchmark 一定很差。模型仍可能让总 loss 快速下降，甚至得到不错的最终准确率。但是：

1. loss 下降只能证明模型正在拟合当前代码定义的目标；
2. 不能据此证明论文中的三项监督机制被正确实现；
3. 当前训练结果应标记为 **official-code-faithful baseline**，不能直接称为 **paper-faithful reproduction**。

---

## 2. B1：forward CoT supervision 双重 causal shift

### 2.1 代码证据

文件：

```text
transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
```

当前 forward loss 路径先手工裁剪 logits 和 labels：

```python
shift_ref_logits = ref_logits[..., :-1, :].contiguous()
shift_ref_labels = ref_labels[..., 1:].contiguous()

forward_loss = self.loss_function(
    shift_ref_logits,
    shift_ref_labels,
    vocab_size=self.config.text_config.vocab_size,
)
```

对应位置约为 `1820-1826` 行。

但 `self.loss_function` 最终使用：

```text
transformers-4.57.0/src/transformers/loss/loss_utils.py
```

其中 `ForCausalLMLoss` 在没有显式传入 `shift_labels` 时还会再次执行：

```python
if shift_labels is None:
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()
```

因此默认训练路径确定发生了两次 shift。

### 2.2 正确目标与当前目标的区别

假设某一段 textual CoT 的 token 是：

```text
[A, B, C]
```

latent embedding 作为这段文本前面的一个 prefix position。标准 causal LM 监督应当是：

```text
latent -> A
A      -> B
B      -> C
```

论文对应的目标是：

```text
P(r_t | r_<t, h)
```

而当前代码先手工 shift，再由 loss 内部 shift，实际有效监督近似变成：

```text
latent -> B
A      -> C
B      -> ignore
```

直接结果是：

- 第一个 CoT token `A` 没有成为预测目标；
- latent position 被要求预测第二个 token `B`；
- 后续 target 相对标准自回归位置错后一位；
- 最后一个位置被忽略；
- 训练时的 token 对齐与正常 autoregressive generation 不一致。

### 2.3 为什么错误目标的 loss 仍然会下降

双重 shift 不会强制产生 NaN，也不保证 loss 停留在随机水平。模型仍可以通过以下方式降低错误目标：

- 利用 CoT 数据中的强语言模式；
- 记忆高频 token 转移；
- 让 latent 编码更多未来 token 信息；
- 依靠全参数语言主干和 decoder 的适应能力；
- 由最终答案 CE 和其他辅助损失共同优化主干。

所以“总 loss 从约 4.9 降到约 1.3”不能反证双重 shift。它只能说明当前错误目标是可学习的。

### 2.4 对实验结论的影响

该问题会削弱以下主张：

1. latent state 可以按论文定义恢复对应的下一步 CoT；
2. forward decoder 的定性文本是标准条件生成读出；
3. forward supervision 的消融结果严格对应论文公式；
4. forward loss 的收益可归因于正确的 `latent -> next reasoning step` 监督。

最终答案 CE 分支没有手工预 shift：

```python
ce_loss = self.loss_function(
    logits=logits,
    labels=labels,
    vocab_size=self.config.text_config.vocab_size,
)
```

因此最终答案生成目标本身是标准的一次 causal shift。这意味着模型仍可能获得较好的最终任务指标，但不能把全部收益归因于正确实现的 forward CoT supervision。

### 2.5 建议修复

最小修复是不要手工预 shift，把完整序列直接交给 `ForCausalLMLoss`：

```python
forward_loss = self.loss_function(
    logits=ref_logits,
    labels=ref_labels,
    vocab_size=self.config.text_config.vocab_size,
)
```

另一种等价方案是保留手工 shift，但不要再调用会自动 shift 的 causal LM loss，而是直接调用一次普通 cross entropy。前一种方案更符合 Transformers 的既有接口，也更不容易再次发生位置错误。

### 2.6 必须添加的验证测试

构造一个很短的人工序列，例如：

```text
[101, 102, 103]
```

逐位置打印有效 target，必须验证：

```text
prefix logit -> 101
token 101 logit -> 102
token 102 logit -> 103
```

仅比较 loss 数值是否下降不足以验证修复。

---

## 3. B2：视觉塔没有彻底冻结

### 3.1 配置意图

A100 训练配置使用：

```yaml
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
```

这通常会被理解为：视觉编码器和多模态 projector 均冻结，只全量训练语言侧及 CoLT 新增模块。

### 3.2 实际冻结键

文件：

```text
LLaMA-Factory/src/llamafactory/model/model_utils/visual.py
```

Qwen3-VL 当前注册为：

```python
_register_composite_model(
    model_type="qwen3_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)
```

因此实际被冻结的是：

```text
visual.patch_embed
visual.blocks
visual.merger
```

### 3.3 遗漏的视觉模块

Qwen3-VL 视觉模型还包含：

```python
self.pos_embed = nn.Embedding(...)
self.deepstack_merger_list = nn.ModuleList(...)
```

对应文件：

```text
transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
```

约在 `580` 和 `592-600` 行。

这些模块不匹配当前 forbidden module keys，因此在 full tuning 下仍保持 `requires_grad=True`。它们不是闲置参数：

- `visual.pos_embed` 在视觉位置编码计算中使用；
- `visual.deepstack_merger_list` 在多个视觉层输出向语言模型注入特征时使用。

根据当前 8B 配置，遗漏的视觉侧可训练参数约为：

```text
123,032,064
```

当前日志中的可训练参数为：

```text
9,564,633,602
```

若把上述遗漏视觉模块也冻结，可训练参数应相应降至约：

```text
9,441,601,538
```

### 3.4 可能产生的影响

可能的正面影响：

- deepstack visual features 可以适应当前训练数据；
- Chart、OCR、Spatial 等任务可能因此受益；
- 保持该行为更有可能复现公开代码本身的结果。

可能的负面影响：

- 与“视觉塔冻结”的通常实验口径不一致；
- 视觉表示可能在 122K 特定 SFT 数据上漂移；
- 通用视觉能力可能下降；
- benchmark 提升可能来自额外视觉调参，而非 CoLT latent reasoning；
- code-faithful 与 paper-faithful/fixed-code 的训练对象不再相同。

这不是一个必然降低准确率的 bug。完整冻结后，结果可能上升，也可能下降。它的核心问题是实验变量发生混杂，以及配置名称不能准确描述实际行为。

### 3.5 资源影响

约 1.23 亿额外可训练参数会增加：

- 梯度存储；
- Adam optimizer state；
- ZeRO-3 分片和通信；
- checkpoint 大小；
- 少量反向计算。

相对于约 100 亿总参数，这不是最大的资源来源，但不应忽略。

### 3.6 建议修复

对于 fixed-code 或 paper-faithful 版本，视觉冻结键至少应覆盖：

```text
visual.patch_embed
visual.pos_embed
visual.blocks
visual.merger
visual.deepstack_merger_list
```

更稳妥的方案不是持续枚举子模块，而是在 `freeze_vision_tower=true` 时直接冻结完整 `visual` 模块，再根据是否需要训练 projector 明确解冻对应部分。这样可以降低未来模型结构更新带来的遗漏风险。

### 3.7 运行时验证

训练前应输出所有视觉参数的状态，并断言 paper-faithful 配置下不存在可训练视觉参数：

```python
visual_trainable = [
    (name, param.numel())
    for name, param in model.named_parameters()
    if "visual." in name and param.requires_grad
]

assert not visual_trainable, visual_trainable
```

如果需要保留特定视觉投影器，则应使用精确白名单，而不是模糊地称为“视觉塔冻结”。

---

## 4. B3：backward stop-gradient 方向相反，并浪费 decoder 资源

### 4.1 论文定义

论文的 backward alignment 可概括为：

```text
textual CoT -> decoder hidden z
z -> 对齐 stopgrad(latent h)
```

也就是说：

```text
decoder/projection 分支获得梯度
latent target 分支停止梯度
```

其目的不是让 latent 主干追逐一个固定 decoder 表示，而是让由文本得到的表示追随当前 latent target。

### 4.2 当前代码

当前实现约在 `modeling_qwen3_vl.py:1869-1889`：

```python
cot_outputs = decoder_backbone(...)
cot_hidden = cot_outputs[0]
cot_last_hidden = cot_hidden[:, -1, :].detach()

cot_to_latent = self.pj_back(cot_last_hidden).unsqueeze(1).to(latent_embd.dtype)
backward_loss = 1 - F.cosine_similarity(
    cot_to_latent.float(),
    latent_embd.float(),
    dim=-1,
).mean()
```

这里 `detach()` 施加在 decoder hidden 上，而不是 latent target 上。

实际梯度路径变为：

```text
backward_decoder: 无梯度
pj_back: 有梯度
latent/main trunk: 有梯度
```

这与论文公式的梯度所有权相反。

### 4.3 算法影响

当前目标可以被解释成另一种方法：

> 使用固定的预训练文本表示作为锚点，让 latent 和 `pj_back` 向该文本语义空间靠近。

这并非完全没有意义，甚至可能有效。但是它已经不是论文所描述的 backward optimization：

- 论文：text decoder representation 追随固定 latent target；
- 当前代码：latent 主干追随固定 decoder representation；
- 当前 `pj_back` 还可以吸收大量对齐压力，形成一条灵活捷径；
- 论文关于 stop-gradient 防止双方共同坍塌的解释不能直接套用；
- backward supervision 的消融结果无法无歧义映射回论文机制。

因此，这一问题同时包含：

1. 论文一致性问题；
2. 梯度方向问题；
3. 机制归因问题。

### 4.4 为什么 `backward_decoder` 仍然浪费资源

虽然 decoder hidden 被 `detach()`，但 `backward_decoder` 参数在 full tuning 下没有显式冻结，因此仍可能：

- 被计入 trainable parameter 数量；
- 被加入 optimizer parameter groups；
- 被 DeepSpeed ZeRO-3 分片；
- 分配 optimizer state 或相关元数据；
- 被保存进 checkpoint；
- 在 forward 时建立一段最终被丢弃的 autograd graph。

它本身不会从 backward alignment loss 获得梯度，因此这些资源开销不能转化为参数更新。

需要强调：`pj_back` 和 latent 主干仍有梯度，所以 backward loss 并不是整体失效；失效的是论文定义的 decoder 学习方向。

### 4.5 两种不同目标下的修复策略

#### 方案 A：严格对齐论文

保留 decoder hidden 的计算图，把 latent 作为 stop-gradient target：

```python
cot_last_hidden = cot_hidden[:, -1, :]
cot_to_latent = self.pj_back(cot_last_hidden).unsqueeze(1).to(latent_embd.dtype)

target_latent = latent_embd.detach()
backward_loss = 1 - F.cosine_similarity(
    cot_to_latent.float(),
    target_latent.float(),
    dim=-1,
).mean()
```

此时 backward decoder 与 `pj_back` 获得梯度，latent target 不从该损失获得梯度。

#### 方案 B：保留当前“固定文本锚点”算法

如果研究上有意让 latent 追随固定文本 encoder，则应明确冻结 backward decoder，并使用无梯度前向：

```python
for param in self.backward_decoder.parameters():
    param.requires_grad_(False)

with torch.no_grad():
    cot_outputs = decoder_backbone(...)
```

这样可以让代码、参数状态和算法意图一致，减少无效优化器与 autograd 开销。但该版本应被命名为 frozen-text-anchor variant，而不是 paper-faithful backward supervision。

### 4.6 必须执行的运行时验证

仅看 `requires_grad=True` 不足以证明参数真的更新。至少需要在一个完整 optimizer step 后记录：

```text
backward_decoder 参数是否在 optimizer param groups 中
backward_decoder 每个参数的 grad 是否全为 None
pj_back 的 grad norm
latent 主干相关参数的 grad norm
optimizer step 前后 backward_decoder 参数 checksum 是否变化
```

当前静态计算图已经足以证明 `detach()` 会切断 backward alignment loss 到 decoder 的梯度；运行时检查用于确认没有其他隐藏损失路径更新该 decoder，并量化资源浪费。

---

## 5. 三个问题之间的联合作用

这三个问题并不是完全独立的。

### 5.1 forward supervision 被削弱或改变

双重 shift 让 latent-to-text 的 token 监督错位，降低了 forward loss 对“下一步推理可读出性”的直接约束。

### 5.2 backward supervision 改为反向拉动 latent

由于 stop-gradient 方向相反，backward loss 会直接推动 latent 主干向 decoder 表示靠近。这可能在一定程度上补偿 forward supervision 的错误，但补偿机制已经不同于论文设计。

### 5.3 未冻结视觉参数提供额外适配能力

额外训练的 `pos_embed` 和 `deepstack_merger_list` 可能帮助模型在特定多模态任务上降低最终答案 loss。于是即使 latent supervision 存在问题，benchmark 仍可能上升。

因此，官方代码最终取得不错结果并不能单独证明三个实现细节正确。可能存在如下混合路径：

```text
最终答案 SFT
  + 语言主干全参数训练
  + 少量视觉侧适配
  + 错位但可学习的 forward loss
  + 方向相反但仍有效的 backward anchor
  + internal prediction loss
  -> 总体 benchmark 改善
```

这也是后续必须做逐项修复消融，而不能只比较训练 loss 的原因。

---

## 6. 对当前 A100 训练的判断

当前运行已经显示：

- 训练进程与 8 个 rank 正常存活；
- 约 72 个 optimizer steps 已完成；
- 总 loss 从 `4.9104` 降到约 `1.3`；
- warmup 后近期 grad norm 多数回落到约 `2-7`；
- 没有发现 NaN、CUDA OOM 或致命 NCCL error。

这表明当前 official-code-faithful 训练在工程意义上可以继续。但它只能承担以下角色：

```text
C0: official-code-faithful baseline
```

它不能单独承担：

```text
P0: paper-faithful reproduction
```

如果继续当前运行，应完整保存：

- 当前源码；
- 实际 A100 YAML 和 DeepSpeed JSON；
- Git diff；
- 训练日志；
- checkpoint；
- 最终八个 benchmark 结果。

这些结果非常有价值，因为后续修复版本可以直接与其比较，量化官方实现偏差究竟是有害、无害还是意外有益。

---

## 7. 建议的最小实验矩阵

不要一次把所有问题都修掉后只跑一个结果，否则无法判断收益来自哪里。建议至少建立：

| 实验 | forward shift | 视觉冻结 | backward stop-gradient | 目的 |
|---|---|---|---|---|
| C0 | 官方双 shift | 官方不完整冻结 | detach decoder hidden | 公开代码忠实基线 |
| C1 | 修复为一次 shift | 官方不完整冻结 | detach decoder hidden | 单独验证 B1 |
| C2 | 修复为一次 shift | 完整冻结视觉侧 | detach decoder hidden | 验证 B2 |
| P0 | 修复为一次 shift | 完整冻结视觉侧 | stopgrad latent target | 论文一致版本 |
| E0 | 修复为一次 shift | 完整冻结视觉侧 | 冻结 text-anchor decoder | 高效固定文本锚点变体 |

推荐流程：

1. 先在相同数据子集上运行 100-200 optimizer steps；
2. 比较四项 loss，而不只比较 total loss；
3. 比较显存、step time 和 grad norm；
4. 用 ChartQA、TextVQA、MathVista 做小规模验证；
5. 确认方向后再投入完整 1910-step 训练；
6. 最终对 C0、P0 和最优 fixed-code 版本做完整八基准评测。

---

## 8. 修复后的验收标准

### 8.1 forward loss

- 人工 token 序列逐位置 target 完全正确；
- 只有一次 causal shift；
- 第一个 CoT token 由 latent prefix position 预测；
- 最后一个 CoT token 没有被意外丢弃。

### 8.2 visual freezing

- paper-faithful 配置下所有 `visual.*` 参数均为 `requires_grad=False`；
- optimizer 中不存在视觉参数；
- 一个 step 后所有视觉参数 checksum 不变。

### 8.3 backward alignment

- paper-faithful 版本中 latent target 对 backward loss 无梯度；
- decoder/`pj_back` 对 backward loss 有非零梯度；
- frozen-text-anchor 版本中 backward decoder 不进入 optimizer；
- 不存在“requires_grad=True 但长期 grad=None”的大参数模块。

### 8.4 实验追溯

- output 目录保存的是实际使用的 A100 YAML，而不是官方默认 YAML；
- 每个版本使用独立 output directory；
- 不允许在 C0 checkpoint 上修改目标函数后继续训练并称为 P0；
- 目标函数或冻结集合发生变化时，从原始预训练模型重新开始。

---

## 9. 最终结论

三个问题的准确分类是：

1. **双重 causal shift**：确定的 token-level 监督错误，优先级最高；
2. **视觉塔冻结不完整**：确定的配置实现缺口，会造成实验混杂；
3. **backward detach 方向相反**：确定的论文算法偏差，同时使一个约 0.6B decoder 产生无效优化器与计算开销。

当前训练仍可作为公开代码基线继续完成，但后续研究至少要同时保留 code-faithful 和 paper-faithful 两条基线。否则，即使最终复现出论文接近的分数，也无法严谨回答：性能究竟来自 CoLT 论文提出的机制，还是来自当前代码中这些不同的训练行为。

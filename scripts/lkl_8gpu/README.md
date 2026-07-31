# CoLT LKL 8-GPU 统一入口

该目录同时服务 LKL A100 和 A800 机器。正式操作只使用一个入口：

```bash
bash scripts/lkl_8gpu/colt.sh help
```

根目录不再按执行时间编号。实现按职责放在：

```text
lkl_8gpu/
├── colt.sh           # 唯一正式入口
├── commands/         # setup / train / eval 实现
├── lib/              # 无副作用公共函数
├── profiles/         # A100 / A800 默认值
├── requirements/     # 评测依赖约束
└── tools/            # Python 校验、汇总和可视化工具
```

## 首次准备

```bash
cd /data/nvme0/lkl/CoLT-reproduction
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

bash scripts/lkl_8gpu/colt.sh profile generic
bash scripts/lkl_8gpu/colt.sh setup all
bash scripts/lkl_8gpu/colt.sh verify ready
bash scripts/lkl_8gpu/colt.sh verify nccl
```

代理只在下载或安装时需要，不写入脚本和仓库。

## 训练

```bash
bash scripts/lkl_8gpu/colt.sh train codefaithful
bash scripts/lkl_8gpu/colt.sh train paper-faithful
bash scripts/lkl_8gpu/colt.sh train oracle-k
```

`paper-faithful` and `oracle-k` share the conservative repaired training semantics: the forward CoT
loss uses one causal shift, while the official partial visual adaptation, backward semantic-anchor
gradient direction, and backward decoder behavior are preserved.

断点恢复必须显式声明，且脚本会检查完整 Trainer checkpoint：

```bash
bash scripts/lkl_8gpu/colt.sh train paper-faithful --resume
```

辅助 decoder 合批默认关闭，可通过命令参数显式启用：

```bash
bash scripts/lkl_8gpu/colt.sh train paper-faithful --batch-aux
bash scripts/lkl_8gpu/colt.sh train oracle-k --batch-aux
```

也可以使用等价的环境变量形式：

```bash
COLT_BATCH_AUX_DECODERS=1 bash scripts/lkl_8gpu/colt.sh train paper-faithful
```

合批调用按 `batch_size x padded_length` 自适应分块，默认每次最多 4096 tokens。
可通过 `COLT_AUX_MAX_BATCH_TOKENS` 调整；单条超过预算的长样本会独立执行，不截断。

## 评测

A100 默认使用 0-7，A800 默认使用 4-7；均可通过 `--gpus` 显式覆盖。默认每卡 3 个
模型 worker，启用 CPU 图像预取，关闭逐样本 `empty_cache()`，使用 Gloo 同步。

```bash
# 本地 code-faithful checkpoint，A100 8 卡，8 个数据集
bash scripts/lkl_8gpu/colt.sh eval codefaithful all8 --gpus 0,1,2,3,4,5,6,7

# 官方 CoLT-8B，sampling + 256，A800 后 4 卡
bash scripts/lkl_8gpu/colt.sh eval official chartqa --gpus 4,5,6,7 --generation official

# 本地 checkpoint，真正采用 greedy + 8192
bash scripts/lkl_8gpu/colt.sh eval codefaithful chartqa --generation respect-args

# Qwen3-VL textual-CoT baseline
bash scripts/lkl_8gpu/colt.sh eval baseline all8 --gpus 0,1,2,3,4,5,6,7
```

模型路径优先级固定为：

```text
--model-path > COLT_EVAL_MODEL_PATH > target 默认路径
```

启动日志会打印 `Model source` 和 `Resolved model`，并同时记录请求/实际的
`do_sample` 与 `max_new_tokens`。结果目录指纹基于实际模型、代码和推理设置，防止错误
复用其他 checkpoint 的预测。

评测日志按 target 分目录保存，并在文件名中直接标明实际解码配置：

```text
logs/eval/official/official_chartqa_sampling_max256_YYYYMMDD_HHMMSS.log
logs/eval/codefaithful/codefaithful_chartqa_greedy_max8192_YYYYMMDD_HHMMSS.log
```

`--generation official` 对应 `sampling_max256`，`--generation respect-args` 对应
`greedy_max8192`。

模型在尚未生成可见文本时可能直接输出 EOS。严格复现默认保留该行为并将空响应计为错误；
诊断性评测可显式阻止“空文本前的 EOS”，且该设置会进入日志名、结果目录和 fingerprint：

```bash
bash scripts/lkl_8gpu/colt.sh eval codefaithful chartqa \
  --generation respect-args \
  --empty-response-policy prevent
```

完整性检查也可单独执行：

```bash
bash scripts/lkl_8gpu/colt.sh verify model codefaithful
bash scripts/lkl_8gpu/colt.sh verify model official --model-path /absolute/model/path
```

## 实验脚本

吞吐 A/B、预处理 A/B 等诊断入口位于 `tests/integration/lkl_8gpu/`，不作为正式训练或
评测入口。正式流程不要从该目录启动。

## Paper-faithful + Oracle-K 串行复现

仓库级流水线会依次完成 paper-faithful 训练、完整性校验与 8 数据集评测，再从同一个
Qwen3-VL 基座独立完成 Oracle-K 训练、完整性校验与评测：

```bash
bash scripts/run_paper_oracle_pipeline.sh
```

换机器时先激活已经准备好的 Python 环境，再运行 pipeline；脚本直接复用当前环境，不需要填写
Conda 安装目录或环境路径：

```bash
conda activate YOUR_ENV_NAME
bash scripts/run_paper_oracle_pipeline.sh
```

只需修改脚本开头的模型和数据路径，或通过同名 `COLT_*` 环境变量覆盖。GPU 型号无需配置，
默认 generic profile 不限制型号。GPU 空闲显存检查和严格磁盘余量检查默认关闭；需要共享服务器
上的严格启动保护时设置 `COLT_STRICT_PREFLIGHT=1`，也可只设置 `COLT_CHECK_GPU_FREE=1`。
辅助 decoder 合批固定关闭。评测默认每卡 3 个 worker、开启 CPU 预取、关闭逐样本
`empty_cache()`，无需显式传 `--workers` 或 `--prefetch`。

先检查全部路径并打印命令，或执行完整的 1-step 串行冒烟测试：

```bash
bash scripts/run_paper_oracle_pipeline.sh --dry-run
bash scripts/run_paper_oracle_pipeline.sh --smoke
```

流水线为每次运行创建独立目录并记录阶段完成标记。中断后使用日志开头打印的绝对路径恢复：

```bash
bash scripts/run_paper_oracle_pipeline.sh --run-dir /absolute/pipeline/run
```

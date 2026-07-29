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

bash scripts/lkl_8gpu/colt.sh profile a100  # A800 改为 a800
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

断点恢复必须显式声明，且脚本会检查完整 Trainer checkpoint：

```bash
bash scripts/lkl_8gpu/colt.sh train paper-faithful --resume
```

辅助 decoder 合批仍默认关闭；完成 loss/gradient 对齐后才启用：

```bash
bash scripts/lkl_8gpu/colt.sh train paper-faithful --batch-aux
```

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

完整性检查也可单独执行：

```bash
bash scripts/lkl_8gpu/colt.sh verify model codefaithful
bash scripts/lkl_8gpu/colt.sh verify model official --model-path /absolute/model/path
```

## 实验脚本

吞吐 A/B、预处理 A/B 等诊断入口位于 `tests/integration/lkl_8gpu/`，不作为正式训练或
评测入口。正式流程不要从该目录启动。

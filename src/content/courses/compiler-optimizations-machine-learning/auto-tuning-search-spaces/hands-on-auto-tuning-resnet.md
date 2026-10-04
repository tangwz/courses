---
course: "compiler-optimizations-machine-learning"
chapter: "auto-tuning-search-spaces"
lesson: "hands-on-auto-tuning-resnet"
sourceId: 7968
sourceUrl: "https://apxml.com/zh/courses/compiler-optimizations-machine-learning/chapter-6-auto-tuning-search-spaces/hands-on-auto-tuning-resnet"
title: "动手实践：ResNet块的自动调优"
description: "使用自动调度器查找ResNet中卷积层的最佳配置。"
order: 5
plots: ["plots/7968-0.json"]
sourceHash: "32f5487b219daf1a48bbd70716468b657e19f87754266b68c7b95558ad72dcbe"
sourceCorrections: []
---

优化深度神经网络 (neural network)需要从理论搜索空间转到具体实现。自动化搜索和成本建模的原理应用于标准ResNet瓶颈块。我们使用现代的基于生成的搜索方法，通常称为自动调度（特别是TVM中的Ansor系统），而不是基于模板的自动调优。此方法使编译器能够自动生成涉及算子融合的复杂子图的搜索空间，而不是依赖手动编写的调度模板。

### 定义工作负载

一个ResNet块包含多个卷积层、批归一化 (normalization)和激活函数 (activation function)，以及一个残差加法。当将其降级到硬件目标时，编译器将其视为一个子图。我们的目标是为整个子图确定最佳循环调度，让编译器能够进行积极的算子融合和分块。

我们首先使用高级中间表示（Relay）定义网络布局。虽然我们着重于ResNet-50的典型形状（输入：$1 \times 64 \times 56 \times 56$），但此方法适用于任何张量计算。

```python
import tvm
from tvm import relay, auto_scheduler
import numpy as np

def get_resnet_block(batch_size=1, dtype="float32"):
    data_shape = (batch_size, 64, 56, 56)
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight1 = relay.var("weight1", shape=(64, 64, 3, 3), dtype=dtype)

    # 卷积 -> 批归一化 -> ReLU
    conv1 = relay.nn.conv2d(data, weight1, padding=(1, 1), kernel_size=(3, 3))
    bn1 = relay.nn.batch_norm(conv1, relay.var("gamma"), relay.var("beta"), 
                              relay.var("mean"), relay.var("var"))[0]
    relu1 = relay.nn.relu(bn1)

    # 模拟残差加法（为演示简化）
    out = relay.add(relu1, data)

    return relay.Function(relay.analysis.free_vars(out), out)
```

### 配置搜索策略

搜索过程包含一个迭代反馈循环。搜索策略从上一节定义的搜索空间生成一批候选程序（草图）。这些候选程序在硬件上进行测量，结果用于更新统计成本模型（XGBoost或LightGBM）。

这个调优循环的架构推动优化。`TaskScheduler`协调整个过程，如果我们要调优完整网络，它会为不同的子图分配时间资源。对于单个块，它严格专注于最大化我们定义函数的吞吐量 (throughput)。

> 自动调度流程。搜索策略生成候选程序，这些程序在进行实际基准测试之前会通过成本模型进行筛选。

### 实现调优循环

为了执行调优，我们从Relay函数中提取搜索任务。编译器分析图以识别可优化的子图。然后我们配置 `Tuner`。

在生产环境中，`measure_ctx` 通常会使用RPC跟踪器指向远程设备。对于这个本地示例，我们假设主机是目标。

```python
# 1. 从网络中提取任务
target = tvm.target.Target("llvm -mcpu=skylake")
mod = tvm.IRModule.from_expr(get_resnet_block())
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], [], target)

# 2. 配置搜索选项
log_file = "resnet_block_tuning.json"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)

tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000,  # 要测试的配置总数
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=1  # 设置为 0 表示静默运行
)

# 3. 执行搜索
# 此过程可能需要几分钟到几小时，耗时视硬件而定
tuner.tune(tune_option)
```

执行期间，调优器搜索由循环分块大小 $T$、向量 (vector)化因子和展开深度定义的空间 $S$。最初，由于演化搜索从低效的默认调度中移开，性能改进会很快。随着过程持续，增益变得渐进，表现为汇编生成中的微优化。

成本模型在此很重要。没有它，调优器将依赖随机搜索。该模型学习特定循环结构（特征）与执行时间（标签）之间的相关性，使其能够有效地修剪搜索空间，而无需在硬件上运行每个候选程序。

### 分析收敛性和性能

调优完成后，我们分析改进情况。下图显示了卷积核的典型收敛模式。X轴表示测量试验次数，Y轴表示GFLOPS的吞吐量 (throughput)。



![搜索收敛：吞吐量 vs. 试验次数](plots/7968-0.json)



> 自动调优会话期间的性能收敛。 “最佳结果”线记录了截至该试验所识别的最高吞吐量配置。

在最初的100次试验中，调优器通常会发现最佳分块因子（例如，拆分循环以适应L1/L2缓存）。后续的改进通常源于向量 (vector)化宽度调整和解决线程绑定冲突。

### 使用优化调度进行编译

调优过程的输出是一个JSON日志文件，包含找到的最佳调度的参数 (parameter)。要在生产推理 (inference)流程中使用此文件，我们使用 `ApplyHistoryBest` 编译模型。此通道用我们在搜索期间找到的特定调度替换默认的降级机制。

```python
# 使用历史最佳上下文编译
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=None)

# 创建运行时模块
dev = tvm.device(str(target), 0)
module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

# 基准测试
print("优化后的推理已就绪。")
```

当 `ApplyHistoryBest` 激活时，编译器会将Relay图中的工作负载签名与日志文件中的条目进行匹配。如果找到匹配项，则应用相应的低级IR (TIR) 转换。如果没有找到匹配项（例如，如果输入形状更改），编译器将回退到默认的、未优化的调度，这表明针对部署中遇到的特定形状进行调优很有用。

### 处理搜索不稳定性

工程师经常遇到调优器未能找到比基线更好的调度，或搜索崩溃的情况。常见原因包括：

1. **搜索预算不足：** 具有高维循环嵌套的复杂算子（如3D卷积）需要更多试验才能遍历有效空间。
2. **无效硬件约束：** 如果搜索空间包含超过硬件SIMD宽度的向量 (vector)化长度，或共享内存使用量超过存储区大小，候选程序将在测量期间失败。
3. **成本模型过拟合 (overfitting)：** 如果模型训练的样本过少，它可能会对无效配置产生过高的性能预测。

为了减少这些问题，请确保 `target` 字符串准确反映处理器能力（例如，包含AVX-512等特定向量扩展或张量核心版本）。此外，使用 `TransferLearning` 会有帮助；您可以从类似的硬件目标加载日志文件来初始化成本模型，从而给搜索一个“热启动”。

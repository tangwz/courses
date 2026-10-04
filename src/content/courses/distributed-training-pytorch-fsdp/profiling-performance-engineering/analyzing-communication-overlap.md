---
course: "distributed-training-pytorch-fsdp"
chapter: "profiling-performance-engineering"
lesson: "analyzing-communication-overlap"
sourceId: 7908
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-6-profiling-performance-engineering/analyzing-communication-overlap"
title: "分析通信重叠"
description: "衡量分布式训练中反向预取和流重叠的效率。"
order: 2
plots: ["plots/7908-0.json"]
sourceHash: "6dfbcd50f6fccbd83053ae45f422c0ce91bd3501cf770d6cf4f6fe24b2f62fa4"
sourceCorrections: []
---

网络互联通常决定着高性能分布式训练的吞吐量 (throughput)上限。尽管现代GPU具备强大的计算能力，但如果设备将大量周期用于等待数据同步，这种潜力就无法发挥。减少“暴露通信”是全分片数据并行（FSDP）训练中性能优化的首要目标。

当GPU的计算引擎在等待网络完成诸如 `AllGather` 或 `ReduceScatter` 等集体操作时处于空闲状态，就会发生暴露通信。理想情况下，这些网络操作应在后台运行，完全被前一层或后一层的计算所掩盖。实现这一点需要精确编排CUDA流以及细致调整FSDP的预取策略。

### 流重叠的机制

PyTorch 使用 CUDA 流管理 NVIDIA GPU 上的并发。流是按顺序执行的操作序列，但不同流中的操作可以并发运行。在标准FSDP配置中，至少有两个核心流处于活动状态：

1. **计算流：** 处理矩阵乘法 (GEMM)、激活以及其他层计算。
2. **通信 (NCCL) 流：** 处理分片和反分片参数 (parameter)及梯度所需的集体操作。

为了实现有效重叠，调度器必须在该操作的依赖项解决后立即在 NCCL 流上分派网络内核，而不阻塞计算流。

以下图表说明了反向传播 (backpropagation)过程中顺序执行与优化重叠之间的差异。在优化流程中，层 $N-1$ 的通信与层 $N$ 的计算同时发生。

> 反向传播中顺序执行与流水线执行流程的比较。优化流程允许计算流继续运行而无需等待网络，前提是通信在数据被严格需要之前完成。

### 在追踪中识别暴露通信

在 PyTorch Profiler 中分析追踪时，通常使用“GPU 内核”视图查看时间线。为了识别通信瓶颈，请找到 NCCL 内核（通常标记 (token)为 `ncclDevKernel_AllGather_...` 或 `ncclDevKernel_ReduceScatter_...`）。

您必须验证与这些 NCCL 内核垂直对齐 (alignment)的计算流上正在发生什么。

1. **完全重叠：** 您会看到计算流上直接位于 NCCL 内核上方有一个密集的 `GEMM` 或逐元素内核块。GPU 利用率保持在 100%，因为算术单元在数据通过互联移动时处于繁忙状态。
2. **暴露通信：** 您会看到计算流上与 NCCL 操作对齐的位置有一个间隙或“气泡”。这表明计算提前完成，GPU 现在因等待网络而停滞。

这种低效率的程度可以通过这些间隙的持续时间总和来计算。有效步长 $T_{	ext{步长}}$ 定义为：

$T_{	ext{步长}} = T_{	ext{计算}} + T_{	ext{暴露通信}}$

其中 $T_{	ext{暴露通信}}$ 是未被计算隐藏的网络时间部分。您的目标是将 $T_{	ext{暴露通信}}$ 降至零。

### 调整反向预取

FSDP 中控制重叠的主要调节项是 `backward_prefetch` 策略。此设置决定了系统何时请求下一层（在反向序列中）的参数 (parameter)。

- **BACKWARD\_POST：** 这是默认的更安全选项。它仅在层 $N$ 完成计算后才请求层 $N-1$ 的参数。这通常会导致与上图左侧类似的序列化。
- **BACKWARD\_PRE：** 此策略在层 $N$ 的计算开始时立即发出层 $N-1$ 的 `AllGather`。这可以最大化重叠窗口。

然而，`BACKWARD_PRE` 会增加峰值内存压力，因为两组分片参数（当前和下一组）必须同时在 VRAM 中实例化。如果您发现 OOM 错误，可能需要恢复到 `BACKWARD_POST` 或启用 CPU 卸载，并接受性能损失。

### 算术强度与通信隐藏

隐藏通信的能力严格受限于给定层计算与通信的比率。这种关系依赖于模型架构和批次大小。

对于 Transformer 块，通信量与参数 (parameter)大小 $P$ 成比例，而计算量与 $P 	imes B$ 成比例（其中 $B$ 是批次大小）。随着批次大小的增加，计算时间呈线性增长，但通信时间大致保持不变（假设带宽固定）。

$\text{计算时间} \propto \frac{B \cdot P}{\text{FLOPS}}$
$\text{通信时间} \propto \frac{P}{\text{带宽}}$

因此，更大的批次大小使得隐藏通信变得更容易。如果您的性能分析显示存在明显的暴露通信，并且无法进一步优化网络，那么增加批次大小（或梯度累积步数）通常是最有效的软件层面修正。

以下图表演示了随着本地批次大小的增加，暴露通信的比例如何减少，最终在网络完全隐藏时趋于平稳。



![批次大小对通信隐藏的影响](plots/7908-0.json)



> 随着批次大小的增加，计算时间最终会超过通信时间，从而消除了暴露通信停滞（黄色条）。

### 同步屏障与 CPU 抖动

有时，性能分析显示存在暴露通信，即使理论计算表明可以重叠。这通常是由于 CPU 端的低效率造成的。

FSDP 依赖 CPU 分派内核。如果 CPU 落后（受 CPU 限制），它无法足够快地发出 `AllGather` 指令来填充流水线。这在追踪中表现为 GPU 时间线上的内核之间出现“间隙”，通常伴随着 CPU 线程上长时间的 `cudaStreamSynchronize` 调用。

为了减轻这种情况：

1. 确保您的 `DataLoader` 没有成为主进程的瓶颈。
2. 检查自定义 `nn.Module` 代码中是否存在过多的 Python 开销。
3. 使用 `torch.compile`（在 PyTorch 2.x 中）来合并内核并减少 Python 解释器开销，从而腾出 CPU 周期以更有效地管理 NCCL 流。

分析重叠是一个迭代过程。您找出最大的间隙，调整预取深度或批次大小，然后重新进行性能分析，直到 GPU 计算流显示出连续、不间断的执行块。

---
course: "llm-compression-acceleration"
chapter: "hardware-acceleration-systems-optimization"
lesson: "optimized-kernels-llm-layers"
sourceId: 4409
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-6-hardware-acceleration-systems-optimization/optimized-kernels-llm-layers"
title: "LLM层的优化算子"
description: "将高度优化的算子（如FlashAttention、Triton算子）用于主要的LLM运算。"
order: 3
plots: []
sourceHash: "c45d7c758b57bd58245b57af6b4503675b91b8db2bdb15028177278d0580d3cb"
sourceCorrections: []
---

尽管量化 (quantization)和剪枝等方法可以降低LLM的理论计算负荷，但要实现显著的速度提升，往往取决于基本运算在目标硬件上的执行效率。即使在优化的深度学习 (deep learning)框架中，核心层的标准实现也可能因内存带宽限制或次优的硬件使用而成为瓶颈。在这种情况下，为特定硬件和计算模式定制的优化算子（即专门的底层软件例程）发挥着主要作用。

算子本质上是经过高度调整的函数，旨在特定架构（如某个GPU代次）上极快地执行特定计算（如矩阵乘法或注意力机制 (attention mechanism)的部分）。它们通常绕过更高级的框架抽象，直接与硬件资源交互，以减少开销并充分利用并行性和数据局部性。

### 计算核心：GEMM和注意力机制 (attention mechanism)

在基于Transformer的LLM中，有两种操作占据了大部分计算时间：通用矩阵乘法（GEMM）和注意力机制。

#### 优化GEMM

密集矩阵乘法出现在前馈网络层以及注意力机制本身（用于投影查询、键、值和输出）。尽管看似简单，但在GPU等现代并行硬件上高效执行GEMM是复杂的。NVIDIA的cuBLAS（用于CUDA）或AMD的rocBLAS（用于ROCm）等库中的优化GEMM算子都经过精心设计。它们采用的技术包括：

- **分块（Tiling）：** 将大型矩阵分解成更小的块，这些块可以放入内存层次结构中速度更快的级别（例如，L2缓存，GPU上的共享内存）。
- **优化内存访问：** 安排数据访问模式，以最大限度地聚合（分组内存请求）并减少缓存未命中。
- **指令级并行：** 使用专用硬件指令（如NVIDIA GPU上的Tensor Cores或AMD GPU上的Matrix Cores），快速执行小矩阵上的融合乘加运算。
- **寄存器分块：** 最大程度地复用加载到最快内存（寄存器）中的数据。

使用这些供应商提供的高度优化的GEMM库，几乎总是加速矩阵乘法的第一步。PyTorch和TensorFlow等框架在可用时通常会自动链接这些库。

#### 注意力瓶颈

标准的自注意力 (self-attention)机制虽然功能强大，但带来了明显的性能挑战，特别是对于长序列。计算过程包括：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


这里，$Q$、$K$和$V$分别是查询、键和值矩阵，$d_k$是键的维度。主要问题源于中间的$QK^T$矩阵（注意力分数）以及随后的softmax操作。

- **内存带宽：** 对于序列长度$N$和隐藏维度$d$， $QK^T$矩阵的大小为$N \times N$。对于大的$N$，这个矩阵可能无法完全放入GPU上快速的片上内存（SRAM）中。计算它需要从较慢的高带宽内存（HBM）读取$Q$和$K$（大小为$N \times d$），计算$N \times N$矩阵，将其写回HBM，再读取它进行softmax，将结果写入HBM，再次读取它与$V$相乘，最后写入输出。这涉及到对HBM中存储的大量数据进行多次传输，使得该操作严重受限于内存带宽。
- **计算冗余：** 标准实现会计算并存储整个$N \times N$的注意力分数矩阵，即使它只是临时需要。

### 融合算子：降低内存流量

加速许多LLM运算（尤其是注意力机制 (attention mechanism)）的思路是减少计算单元和主内存（HBM）之间的数据移动。融合算子通过将多个独立运算组合到一次算子启动中来实现这一点。这使得中间结果可以保留在更快的片上内存（如寄存器或SRAM）中，而无需反复写入和读取HBM。

#### 融合原理

考虑一个简单的序列：先乘以一个标量，然后应用激活函数 (activation function)。

```python
# 标准实现（简化版）
intermediate = x * scale # 读取x，写入中间结果
output = activation(intermediate) # 读取中间结果，写入输出
```

融合算子会一次性执行这两个操作：

```python
# 融合算子理念（简化版）
output = activation(x * scale) # 读取x，写入输出（中间结果保留在寄存器/SRAM中）
```

这避免了`intermediate`变量的内存往返，节省了带宽和延迟。编译器有时可以自动执行简单的融合，但注意力机制等复杂序列通常需要明确设计的融合算子。

#### FlashAttention：注意力计算的新方式

FlashAttention及其后续版本（如FlashAttention v2）是高度优化的融合算子的典型代表，专门用于解决注意力瓶颈。FlashAttention不计算完整的$N \times N$分数矩阵，而是使用**分块**和**重新计算**。

1. **分块：** $Q$、$K$和$V$矩阵从HBM分块（tiles）加载到SRAM中。
2. **SRAM内的融合计算：** 最终输出的一个完整块直接在SRAM内计算。这包括获取对应的$Q$、$K$和$V$块，计算该块的注意力分数（$Q_{block}K_{block}^T$），正确应用softmax缩放和掩码，*而无需*具体化完整的中间分数矩阵，并乘以对应的$V_{block}$。
3. **在线Softmax：** 当处理给定$Q$块的更多$K$和$V$块时，softmax归一化 (normalization)是迭代计算的，同时维护运行统计数据以确保正确的最终归一化，而不需要完整的得分矩阵。
4. **HBM流量降低：** 唯一重要的HBM读取是针对初始的$Q$、$K$、$V$块，唯一重要的HBM写入是针对最终的输出注意力块。大的$N \times N$中间矩阵从未写入或从HBM读取。

> 标准注意力机制与FlashAttention的数据流比较。FlashAttention通过在快速片上SRAM中以块为单位执行计算，避免了大型N x N中间矩阵的具体化，从而大幅减少了对较慢的高带宽内存（HBM）的读写。

FlashAttention可以带来明显的性能提升（仅注意力计算通常可达2-4倍或更多），并大幅降低内存使用，使得在训练和推理 (inference)期间能够支持更长的上下文 (context)长度。它在计算与内存带宽比高的GPU上表现尤为出色。

### 算子开发的领域特定语言

尽管cuBLAS等库和FlashAttention等算子提供了重要的组成部分，但有时您需要为特定模型变体或新运算定制融合算子。编写原始的CUDA或ROCm代码既复杂又耗时。Triton等领域特定语言（DSL）提供了一种更高级的方式来编写高性能GPU代码。

#### Triton简介

Triton是由OpenAI开发的一种基于Python的语言和编译器。它允许开发者使用Python语法编写算子，然后Triton编译器将其转换为高效的底层代码（如NVIDIA GPU的PTX）。其主要特点包括：

- **简化并行编程：** 抽象化了CUDA/ROCm线程管理和同步的许多复杂性。
- **自动优化：** 编译器处理内存合并、指令调度和分块等优化，通常能达到与手动调优算子相当的性能。
- **易于融合：** 使得将多个操作融合到一个算子中变得相对简单。

#### 编写定制融合算子

使用Triton，例如可以将层归一化 (normalization)操作直接与GeLU激活融合。Triton算子可以一次性加载输入数据，执行归一化计算，应用GeLU，并将最终结果写回，所有这些都在一个算子中完成，中间结果保留在寄存器或共享内存中，而不是两次独立的算子启动并伴随中间HBM的写入/读取。

```python
# 用于融合层归一化+GeLU的Triton风格算子
import triton
import triton.language as tl

@triton.jit
def fused_layer_norm_gelu_kernel(
    input_ptr, output_ptr, weight_ptr, bias_ptr,
    n_elements, eps,
    BLOCK_SIZE: tl.constexpr, # 算子参数
):
    # --- 简化概念 ---
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载该块的输入数据
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # --- 层归一化部分（在寄存器/SRAM内）---
    # 实际中需要均值/方差计算
    mean = tl.sum(x) / n_elements # 简化均值
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered) / n_elements # 简化方差
    x_norm = x_centered / tl.sqrt(var + eps)

    # 加载归一化权重/偏置
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    x_scaled = x_norm * weight + bias

    # --- GeLU激活部分（在寄存器/SRAM内）---
    # 使用近似GeLU计算
    cdf = 0.5 * (1.0 + tl.tanh( (0.79788456 * (x_scaled + 0.044715 * x_scaled * x_scaled * x_scaled)) ) )
    y = x_scaled * cdf

    # 写入最终输出
    tl.store(output_ptr + offsets, y, mask=mask)
```

> 这是一个示例，说明如何将层归一化和GeLU激活等操作融合到单个Triton算子中，以减少内存移动。实际实现需要妥善处理数值稳定性和并行归约模式。

尽管Triton降低了门槛，但编写高效的定制算子仍然需要了解硬件架构和并行编程思想。

### 使用优化库和框架支持

幸运的是，通常您无需从头开始编写算子。深度学习 (deep learning)框架和专用推理 (inference)引擎正在不断加入预构建的优化算子。

- **PyTorch：**
  - 对许多操作自动使用cuDNN/cuBLAS/rocDNN/rocBLAS。
  - `torch.backends.cudnn.benchmark = True`可以自动调优特定输入尺寸的卷积算子（对典型LLM层不那么适用，但在视觉任务中有用）。
  - `torch.compile()`使用Triton和TorchInductor等后端，自动融合操作并为模型图的部分生成优化算子。
  - `torch.nn.functional.scaled_dot_product_attention`提供了一个高级接口，根据硬件、输入和可用性，自动选择最有效的注意力实现（包括FlashAttention、内存高效注意力或C++算子）。
- **TensorFlow：** 使用XLA（加速线性代数）编译器，该编译器执行积极的算子融合并针对优化的硬件库。运行由`@tf.function(jit_compile=True)`装饰的TensorFlow函数可以启用XLA编译。
- **推理引擎（TensorRT, vLLM, ONNX Runtime）：** 这些工具专注于优化模型以进行推理。它们执行图优化、层融合，并为目标硬件选择最快的可用算子（通常包括定制或高度调优的算子）。例如，TensorRT大量使用其优化的算子库（通常称为“策略”）。vLLM实现了定制算子，尤其是PagedAttention，用于文本生成过程中键值缓存的高效内存管理。

### 算子的选择与评估

最佳算子选择取决于具体运算、模型架构、硬件平台、序列长度、批处理大小和精度。

- **供应商库：** 从cuBLAS/cuDNN或rocBLAS/rocDNN等标准库开始。它们经过充分测试，为GEMM等常见操作提供了出色的性能。
- **融合注意力：** 对于Transformer模型，使用FlashAttention等优化注意力算子（如果通过框架支持或`xformers`等库可用）几乎总是有利于性能和内存，尤其是在序列长度较长和硬件性能较好的情况下。
- **框架编译器：** 在可能的情况下，使用`torch.compile`或TensorFlow的XLA，因为它们可以自动发现并实现融合机会，无需手动编写算子。
- **推理 (inference)引擎：** 对于部署，专用推理引擎通常通过将算子优化与其他技术（如量化 (quantization)、图重写和专用内存管理（例如vLLM的KV缓存处理））结合，提供最显著的速度提升。
- **基准测试：** 始终在您的目标硬件和工作负载上对不同选项进行基准测试。在一个场景中最快的算子在另一个场景中可能并非如此。严格测量延迟、吞吐量 (throughput)和内存使用，以便做出明智的决定。

优化的算子是LLM加速堆栈中非常重要的一层，它们通过最大限度地利用硬件并减少数据移动瓶颈，将算法效率的提升转化为实际的性能增益。了解它们的作用以及如何通过库、编译器和专用引擎来使用它们，对于部署快速高效的大型语言模型来说非常重要。

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  介绍了FlashAttention算法，这是一种用于注意力机制计算的高度优化的融合内核。
- [Triton Language Documentation](https://triton-lang.org/) — OpenAI (2024)
  Publisher: OpenAI
  Triton官方文档，一种用于编写高性能GPU内核的基于Python的DSL。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  NVIDIA GPU编程和优化指南，涵盖内存层次结构和并行性。
- [Accelerated Linear Algebra (XLA)](https://www.tensorflow.org/xla) — Google (2024)
  Publisher: Google
  官方文档描述了XLA通过运算符融合和编译优化TensorFlow计算的作用。

---
course: "compiler-optimizations-machine-learning"
chapter: "hardware-aware-code-generation"
lesson: "hands-on-writing-triton-kernels"
sourceId: 7959
sourceUrl: "https://apxml.com/zh/courses/compiler-optimizations-machine-learning/chapter-5-hardware-aware-code-generation/hands-on-writing-triton-kernels"
title: "动手实践：编写 Triton 核函数"
description: "使用 Triton 语言开发高性能 GPU 核函数，以避免底层的 CUDA 复杂之处。"
order: 5
plots: ["plots/7959-0.json"]
sourceHash: "5d1ffbd61a768aa00c5b12d21f70bf592acf49f5cf2111fe2c59a3172486e33f"
sourceCorrections: []
---

直接使用 CUDA C++ 编写高性能 GPU 核函数，通常需要处理共享内存冲突、warp 分歧以及张量核心操作数布局等细致问题。尽管这种级别的控制很强大，但它会带来可观的工程开销。Triton 作为一种语言和编译器栈，能够抽象化这些复杂之处，同时保持与供应商优化库相当的性能。它通过公开一种基于块的编程模型，而非 CUDA 中常见的单指令多线程 (SIMT) 模型来实现这一目标。

在本实践部分，我们将实现一个适用于深度学习 (deep learning)推理 (inference)的瓦片式矩阵乘法核函数。此实现说明了如何将之前讨论的内存层级思想，特别是合并访问和共享内存管理，映射到 Triton 基于 Python 的语法中。

### 基于块的编程模型

与 CUDA 定义单个线程行为不同，Triton 核函数定义对数据块的操作。编译器自动处理这些数据块到硬件线程的划分，并管理数据在内存层级（全局内存 $\rightarrow$ 共享内存 $\rightarrow$ 寄存器）中的传输。

编写核函数时，必须首先定义一个网格。网格是核函数并行运行的多个实例的集合。对于大小为 $(M, N, K)$ 的矩阵乘法，我们通常沿着 $M$ 和 $N$ 维度进行并行化。每个核函数实例计算一个大小为 `BLOCK_SIZE_M` $\times$ `BLOCK_SIZE_N` 的结果块。

以下 Graphviz 图展示了单个 Triton 程序实例（一个数据块）内的数据流。请注意操作是如何在瓦片而非标量上定义的。

> 单个 Triton 程序实例处理数据块的数据流执行模型。

### 定义核函数签名

Triton 核函数的入口点是一个用 `@triton.jit` 装饰的 Python 函数。函数签名通常接受输入张量的指针、张量维度和步长。步长对于计算扁平化一维内存空间中特定元素 $(i, j)$ 的内存地址非常重要，其由以下公式定义：

$\text{地址}(i, j) = \text{基址} + i \times \text{步长}_0 + j \times \text{步长}_1$

下方是一个矩阵乘法核函数的初始化。我们计算了矩阵 A 和 B 指针的二维偏移量。

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (memory layout)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters (constants at compile time)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 1. Map the program ID to the block in the output grid
    pid = tl.program_id(axis=0)

    # Calculate the number of blocks along the M and N axes
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Determine the row and column index for this specific block
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # 2. Create pointers for the first block of A and B
    # Range of offsets for the M dimension
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # Range of offsets for the N dimension
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # Range for the K dimension (reduction axis)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Calculate the actual memory addresses
    # a_ptrs is a matrix of pointers [BLOCK_SIZE_M, BLOCK_SIZE_K]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # b_ptrs is a matrix of pointers [BLOCK_SIZE_K, BLOCK_SIZE_N]
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize the accumulator with zeros
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
```

在上面的代码片段中，`tl.arange` 生成一个整数序列。通过广播这些序列（使用 `[:, None]` 和 `[None, :]`），我们创建了偏移量网格。当我们将这些偏移量添加到基指针 `a_ptr` 时，Triton 创建了一个指针块。这种抽象使编译器能够发出适用于目标架构的向量 (vector)加载指令。

### 归约循环

核心计算在一个遍历 $K$ 维度的循环中进行。在每次迭代中，我们将指针前进 `BLOCK_SIZE_K`。这与第 3 章讨论的瓦片技术相似，但这里的“瓦片”是由向量 (vector)类型隐式处理的。

在循环内部，我们将 A 和 B 的数据块从全局内存加载到 SRAM（共享内存）中。`tl.dot` 操作随后执行矩阵乘法。如果硬件支持（例如 NVIDIA 张量核心），Triton 会自动将此高级操作降级为适当的内在函数（例如 `mma.sync`）。

```python
    # 3. Iterate to compute a block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B
        # mask is used to handle boundary conditions if K is not a multiple of BLOCK_SIZE_K
        # other=0.0 ensures padding with zeros does not affect the sum
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Accumulate partial dot products
        accumulator += tl.dot(a, b)

        # Advance pointers to the next K-block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
```

`tl.load` 中的 `mask` 参数 (parameter)对于正确性来说非常重要。它防止核函数读取越界内存，从而避免非法内存访问错误或静默数据损坏。`other=0.0` 参数确保被掩盖的值不会影响点积结果。

### 存储结果和合并访问

归约循环完成后，`accumulator` 存储着 $C$ 的瓦片的最终值。我们必须将其存回全局内存。高效的存储需要内存合并访问，即连续的线程访问连续的内存地址。Triton 管理寄存器文件的布局以对此进行优化，前提是块大小和步长对齐 (alignment)。

```python
    # 4. Store the result
    # Create pointers for C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    # Write to global memory, applying a mask to avoid out-of-bounds writes
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
```

### 性能与块大小选择

`BLOCK_SIZE_M`、`BLOCK_SIZE_N` 和 `BLOCK_SIZE_K` 的选择对性能影响很大。大块能最大限度地提高 L1 缓存/共享内存中的数据重用，减少对全局内存带宽的需求。然而，过大的数据块可能超出可用的共享内存或寄存器文件大小，从而降低占用率（活动 warp 的数量）。

下图展示了标准 GPU 架构上不同块配置的吞吐量 (throughput) (TFLOPS) 比较。请注意，特定配置由于与硬件特性（例如 128 字节缓存行）对齐 (alignment)，从而显著提升了性能。



![吞吐量与块配置对比 (Float16)](plots/7959-0.json)



> 不同块大小配置下的吞吐量变化，突显了调优的重要性。

配置 `128x128x32` 通常表现良好，因为它在瓦片大小和寄存器压力之间取得了平衡。然而，手动选择效率不高。Triton 提供了一种使用 `@triton.autotune` 装饰器的自动调优机制。此功能会用不同的元参数 (parameter)编译核函数的多个版本，并在运行时根据实际输入形状选择最快的版本。

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    # Kernel implementation...
```

通过定义 `num_stages`，您可以控制软件流水线。这种优化（在循环调度章节中有所讨论）允许 GPU 在计算第 $i$ 次迭代的同时加载第 $i+1$ 次迭代的数据，从而隐藏全局内存延迟。

### 核函数启动

为了执行核函数，我们根据输入矩阵大小和选定的块大小计算网格维度。主机端的 Python 代码调用 JIT 编译的核函数。

```python
def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Grid definition function
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c
```

此实现为硬件感知代码生成提供了基础。通过操作块指针和运用 `tl.dot`，您可以有效地控制 GPU 的内存层级和计算单元，而无需手动管理同步屏障或银行冲突，这些问题在降级到 LLVM IR 和 PTX 阶段由编译器解决。

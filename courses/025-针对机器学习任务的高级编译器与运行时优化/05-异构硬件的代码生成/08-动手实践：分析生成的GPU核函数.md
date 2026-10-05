# 动手实践：分析生成的GPU核函数

来源：[原文](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-5-heterogeneous-hardware-code-generation/practice-analyzing-gpu-kernels)

[返回章节目录](README.md) · [返回课程目录](../README.md)

理解编译器为GPU核函数生成的低级代码，对于验证优化效果以及查找源代码甚至高级性能分析中可能不明显的性能瓶颈，是十分重要的。本实践练习将引导你分析生成的GPU汇编代码，以NVIDIA的PTX（并行线程执行）作为主要示例，尽管这些原理也适用于AMD的GCN/RDNA ISA。

### 前提条件与设置

为了跟随操作，你将需要能够生成和检查GPU汇编代码的工具。这通常包含：

1. **一个机器学习 (machine learning)编译器：** 像TVM、XLA（通过TensorFlow）或自定义的能够针对GPU的编译器。
2. **GPU供应商工具包：**
   - 对于NVIDIA：CUDA工具包，包含`nvcc`（编译器驱动程序）和`nvdisasm`（反汇编器）。
   - 对于AMD：ROCm栈，包含其编译器和`rocobjdump`等工具。
3. **一个示例核函数：** 我们将使用一个简单的矩阵乘法（MatMul）或卷积层作为示例。假设你已经编译了这样一个针对你的GPU的操作。

你通常可以直接指示机器学习编译器框架或`nvcc`来输出中间PTX汇编或最终机器码（NVIDIA为SASS，AMD为ISA）。对于`nvcc`，像`nvcc my_kernel.cu -ptx -o kernel.ptx`或`nvcc my_kernel.cu -cubin -o kernel.cubin`后跟`nvdisasm kernel.cubin`这样的命令是常见的。机器学习框架通常有配置选项或环境变量来导出生成的代码。

### 生成目标代码

让我们假设你已使用机器学习 (machine learning)编译器工具链编译了一个简单的$C = A \times B$矩阵乘法核函数，针对具备CUDA能力的NVIDIA GPU。编译器可能已应用了平铺等优化来利用共享内存，并且如果可用且已启用，可能已针对Tensor Cores。你通常会获得一个`.ptx`文件，或者对已编译的`.cubin`或可执行文件使用`nvdisasm`来查看SASS。本次分析我们将主要关注PTX，因为它是一种稳定的中间汇编语言。

### 分析PTX汇编

PTX提供了一个虚拟指令集和寄存器架构。分析它有助于理解编译器在最终映射到特定硬件微架构（SASS）之前的高层策略。

考虑一个用于平铺MatMul核函数的简化PTX片段：

```ptx
.version 7.5
.target sm_80 // 目标架构（例如，Ampere）
.address_size 64

.visible .entry _Z10matmul_kerPfS_S_ii(
    .param .u64 .ptr .align 8 .global _Z10matmul_kerPfS_S_ii_param_0, // 指针 C
    .param .u64 .ptr .align 8 .global _Z10matmul_kerPfS_S_ii_param_1, // 指针 A
    .param .u64 .ptr .align 8 .global _Z10matmul_kerPfS_S_ii_param_2, // 指针 B
    .param .u32 _Z10matmul_kerPfS_S_ii_param_3, // 维度 M
    .param .u32 _Z10matmul_kerPfS_S_ii_param_4  // 维度 N
)
{
    .reg .pred      %p<3>;
    .reg .b32       %r<65>; // 示例：声明了64个通用32位寄存器
    .reg .f32       %f<33>; // 示例：声明了32个浮点32位寄存器
    .reg .b64       %rd<10>;

    .shared .align 16 .b8    __shared_mem_A[4096]; // 用于瓦片A的共享内存
    .shared .align 16 .b8    __shared_mem_B[4096]; // 用于瓦片B的共享内存

    // --- 核函数体开始 ---
    ld.param.u64    %rd1, [_Z10matmul_kerPfS_S_ii_param_0]; // 加载C指针
    ld.param.u64    %rd2, [_Z10matmul_kerPfS_S_ii_param_1]; // 加载A指针
    // ... 使用%tid、%ctaid计算线程/块索引 ...

    // --- 示例：从全局内存加载瓦片到共享内存 ---
    mov.u32         %r1, %tid.x;
    // ... 基于%r1、%ctaid等计算全局内存地址 ...
    ld.global.cs.f32 %f1, [%rd_A_global_addr]; // A的合并全局加载
    st.shared.f32   [%shared_mem_A_addr], %f1; // 存储到共享内存A

    // ... 瓦片B的类似加载 ...

    // 同步屏障
    bar.sync 0;

    // --- 示例：使用共享内存瓦片进行计算 ---
    // 假设寄存器%f10-%f17保存C瓦片的累加器

    // 循环遍历瓦片维度K
.L_K_Loop:
    ld.shared.f32   %f2, [%shared_mem_A_addr_k];
    ld.shared.f32   %f3, [%shared_mem_B_addr_k];
    fma.rn.f32      %f10, %f2, %f3, %f10; // 乘加指令

    // ... 更多输出瓦片的FMA指令 ...
    // ... 增加共享内存指针，循环计数器 ...
    setp.lt.u32     %p1, %r_k_counter, %r_K_dim;
    @%p1 bra        .L_K_Loop;

    // --- 下一次瓦片加载前的同步（如果适用） ---
    bar.sync 0;

    // --- 将寄存器中的结果存储到全局内存C ---
    // ... 计算C的全局内存地址 ...
    st.global.cs.f32 [%rd_C_global_addr], %f10;

    ret;
}
```

**分析要点：**

1. **资源使用：**

   - `.reg .b32 %r<65>; .reg .f32 %f<33>;`：统计声明的寄存器。每个线程的寄存器使用量高（例如，根据架构不同，超过64或128个）会直接影响*占用率*。占用率是活跃warp数量与每个多处理器（SM）上最大可能warp数量之比。较低的占用率有时可以隐藏延迟，但如果指令级并行不足，通常会导致SM计算资源利用不足。
   - `.shared .b8 __shared_mem_A[4096];`：注意分配的共享内存量。与寄存器一样，共享内存是每个SM的有限资源。每个块的寄存器和共享内存的总需求决定了可在SM上同时运行的最大块数。
2. **指令：**

   - `ld.global`、`st.global`：这些指令访问主GPU设备内存（DRAM）。查找模式。它们是否标记 (token)为`.cs`（缓存流）或`.cg`（缓存全局）？访问是否可能为*合并的*（一个warp中的线程访问连续的内存位置）？非合并访问会大幅降低有效带宽。频繁的全局访问表明数据复用不佳。
   - `ld.shared`、`st.shared`：访问快速的片上共享内存。它们的存在通常表明编译器为提高数据复用而实现了平铺策略。
   - `fma.rn.f32`（FP32 FMA）、`mul.f32`、`add.f32`：标准浮点运算。计算指令与内存指令的比例可以让我们了解核函数的算术密度。
   - `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`：此指令（或类似变体）表明使用了Tensor Cores（在NVIDIA Volta及更高版本上）。它的出现确认编译器成功地将操作向量 (vector)化以使用这些专用单元，为特定矩阵操作提供了大幅加速。形状（`m16n8k8`）、数据类型（`f16`、`f32`）和布局（`row`、`col`）是重要的细节。
   - `bar.sync`：同步屏障。当一个块内的线程通过共享内存共享数据时，它们对于正确性是必不可少的。过度使用或放置不当的屏障会使执行串行化并产生停顿。
   - `bra`、`setp`、`@%p1`：控制流指令。复杂的分支，尤其是在一个warp内发散（同一warp中的线程采取不同路径）时，会大幅降低性能。查找循环（`bra`返回到标签）和条件执行（`@%p`谓词守卫）。
3. **推断的优化：**

   - **平铺：** `ld.global` -> `st.shared` -> `bar.sync` -> 包含`ld.shared`和计算的循环 -> `bar.sync`的模式强烈表明是平铺优化。
   - **循环展开：** 你可能会看到一系列相似指令重复出现，并伴随着调整后的立即数偏移量或寄存器索引，而不是紧凑的循环结构，这表明编译器展开了循环以减少开销并改进指令调度。
   - **指令调度：** 内存访问指令和计算指令的交错试图隐藏内存延迟。编译器会尝试足够早地发出加载指令，以便在计算指令需要时数据可用。

### 连接分析与性能

汇编代码的静态分析提供了关于性能的假设：

- **高寄存器/共享内存使用量：** 可能导致低占用率。虽然并非总是坏事，但如果计算单元处于饥饿状态，这可能成为瓶颈。你可以使用供应商文档或在线计算器，根据每个块的这些资源和目标SM的限制来估算理论占用率。

  
  
  [交互图表：理论占用率与资源使用（示例）](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-5-heterogeneous-hardware-code-generation/practice-analyzing-gpu-kernels#plot-1eo9zat)
  
  <details>
  <summary>图表数据（JSON）</summary>
  
  ```json
  {
    "data": [
      {
        "x": [
          32,
          64,
          96,
          128,
          160,
          192,
          224,
          256
        ],
        "y": [
          1.0,
          1.0,
          0.66,
          0.5,
          0.4,
          0.33,
          0.28,
          0.25
        ],
        "type": "scatter",
        "mode": "lines+markers",
        "name": "0KB 共享内存"
      },
      {
        "x": [
          32,
          64,
          96,
          128,
          160,
          192,
          224,
          256
        ],
        "y": [
          0.75,
          0.75,
          0.5,
          0.375,
          0.3,
          0.25,
          0.21,
          0.18
        ],
        "type": "scatter",
        "mode": "lines+markers",
        "name": "24KB 共享内存"
      },
      {
        "x": [
          32,
          64,
          96,
          128,
          160,
          192,
          224,
          256
        ],
        "y": [
          0.5,
          0.5,
          0.33,
          0.25,
          0.2,
          0.16,
          0.14,
          0.125
        ],
        "type": "scatter",
        "mode": "lines+markers",
        "name": "48KB 共享内存"
      }
    ],
    "layout": {
      "title": "理论占用率与资源使用（示例）",
      "xaxis": {
        "title": "每线程寄存器数"
      },
      "yaxis": {
        "title": "每SM最大占用率",
        "range": [
          0,
          1.1
        ]
      },
      "legend": {
        "title": "每块共享内存"
      }
    }
  }
  ```
  
  </details>
  
  

  > 基于GPU SM的每线程寄存器数和每块共享内存，计算理论占用率的示例。实际占用率取决于启动配置和运行时行为。
- **全局内存访问占主导：** 表明核函数受限于内存带宽。检查合并情况以及是否有更好的缓存或平铺潜力。
- **缺少预期的专用指令：** 如果你在兼容硬件上对矩阵乘法预期使用Tensor Core（`mma.sync`）但只看到`fma.f32`，则编译器可能未识别出模式，或者未满足Tensor Core代码生成所需的对齐 (alignment)/维度限制。
- **明显的分支：** 尤其是在循环内的发散分支可能代价高昂。

### 在PTX中：SASS与其他架构

虽然PTX提供了有价值的信息，但实际机器码（NVIDIA为SASS，AMD为ISA）显示了最终的指令调度、寄存器分配选择以及潜在的硬件特有瓶颈，如共享内存中的bank冲突。像`nvdisasm`（用于SASS）或`rocobjdump`（用于GCN/RDNA ISA）这样的工具允许在此级别进行检查。分析原理保持一致：检查资源使用、指令组合、内存访问模式和控制流，但将它们映射到特定硬件架构的能力和限制。分析AMD的ISA包含理解VLIW（超长指令字）特性、标量与向量 (vector)指令以及波前执行。

### 结论

分析生成的GPU核函数汇编代码对于高级编译器开发者和性能工程师而言是一种强大的技术。它超越了抽象，清楚地显示了编译器如何转换高级操作并应用优化。通过检查PTX或SASS/ISA中的寄存器使用、共享内存分配、指令序列、内存访问模式和控制流，你可以充分了解代码生成过程的效率，验证专用硬件单元的使用，并形成关于性能瓶颈的具体想法，这些想法随后可以使用性能分析工具（如第九章所述）确认。这种低级可见性通常是异构硬件上复杂机器学习 (machine learning)工作负载实现最大性能所必需的。

## 参考资料

- [NVIDIA CUDA C++ PTX ISA](https://docs.nvidia.com/cuda/pdf/ptx-isa-12.4.pdf) — NVIDIA Corporation (2025)
  Publisher: NVIDIA Corporation
  理解并行线程执行（PTX）虚拟指令集架构的权威参考，有助于分析生成的NVIDIA GPU内核。
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  提供CUDA编程、GPU架构细节、内存模型和性能优化技术的全面指导，与解释GPU汇编代码直接相关。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-32-391231-0) — Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj (2022)
  Publisher: Elsevier Morgan Kaufmann
  一本关于GPU硬件、CUDA编程模型和优化策略的综合性教材，为分析编译器生成的GPU代码提供了背景。
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594
  介绍了TVM，一种为包括GPU在内的各种硬件生成高性能代码的优化编译器，提供了对导致所分析汇编代码的编译策略的见解。
- [AMD RDNA 3 Instruction Set Architecture Reference Guide](https://developer.amd.com/wp-content/resources/RDNA3_ISA_v1.0.pdf) — Advanced Micro Devices (AMD) (2023)
  Publisher: Advanced Micro Devices (AMD)
  AMD GPU指令集架构的官方文档，为分析AMD GPU生成的汇编代码提供了背景信息。

---

[上一节](07-%E4%BE%9B%E5%BA%94%E5%95%86%E4%B8%93%E7%94%A8%E7%BC%96%E8%AF%91%E5%99%A8%E5%B7%A5%E5%85%B7%E9%93%BE%E5%92%8C%E5%BA%93%20%28cuDNN%2C%20MIOpen%29.md) · [下一节](../06-%E9%AB%98%E7%BA%A7%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%BF%90%E8%A1%8C%E6%97%B6%E7%B3%BB%E7%BB%9F/01-%E8%BF%90%E8%A1%8C%E6%97%B6%E6%9E%B6%E6%9E%84%E6%A6%82%E8%A7%88.md)

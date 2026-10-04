# GPU 架构 (NVIDIA Ampere, Hopper)

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-18-hardware-considerations-llm-training/gpu-architectures-nvidia)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练大型语言模型推动了计算的极限，使得专用硬件不仅有益，而且通常必不可少。尽管存在各种加速器，但 NVIDIA GPU 因其成熟的软件生态系统（CUDA）和专门为深度学习 (deep learning)工作负载设计的硬件，已成为大多数大规模 AI 训练工作的主力。理解这些 GPU 的架构特点，尤其是 Ampere 和 Hopper 等近期代系，对任何计划或执行 LLM 训练的人都很有帮助。

NVIDIA GPU 的核心是数千个称为 CUDA 核心的处理单元，它们实现大规模并行，适用于神经网络 (neural network)中常见的矩阵乘法及其他操作。然而，深度学习的真正加速来自名为张量核心 (Tensor Cores) 的专用单元。张量核心首次出现在 Volta 架构中，并在后续代系中大幅增强，它们以比标准 CUDA 核心在 FP32 精度下操作高得多的吞吐量 (throughput)执行混合精度矩阵乘累加运算。

### NVIDIA Ampere 架构（例如 A100）

NVIDIA Ampere 架构以 A100 GPU 为代表，在推出时代表了 AI 训练的显著进步。它带来了几项直接有利于大型模型训练的改进：

1. \*\*第三代张量核心：\*\*Ampere 张量核心扩展了支持的精度范围。重要的一点是，它们引入了 TensorFloat-32 (TF32)。TF32 在乘法精度上使用与 FP16 相同的 10 位尾数，但保留了 FP32 的 8 位指数，在较低精度的速度/内存优势与 FP32 的数值范围之间取得了良好平衡。这通常可以在许多框架中实现接近 FP32 的精度，并带来明显更高的吞吐量 (throughput)（相比上一代在 FP32 上的矩阵运算，理论速度提升高达 8 倍），同时无需明确的代码修改。Ampere 还提升了 FP16、BF16（BFloat16，提供比 FP16 更宽的范围，有利于训练稳定性）以及用于推理 (inference)加速的 INT8/INT4 的性能。此外，这些张量核心引入了细粒度结构化稀疏性，如果权重 (weight)矩阵的部分可以按特定模式剪枝，则可以实现加速，有可能使吞吐量翻倍。

   ```python
   import torch

   # 检查 TF32 是否在 Ampere 或更高版本 GPU 上可用并已启用
   # PyTorch 在 Ampere+ 默认启用 TF32 矩阵乘法操作
   print(f"TF32 enabled on matmul: {torch.backends.cuda.matmul.allow_tf32}")
   print(f"TF32 enabled on cuDNN: {torch.backends.cudnn.allow_tf32}")

   # 显式启用或禁用 TF32 用于矩阵乘法
   # torch.backends.cuda.matmul.allow_tf32 = True # 默认
   # torch.backends.cuda.matmul.allow_tf32 = False
   ```
2. \*\*增加的 HBM2e 内存：\*\*大型模型需要大量内存来存储参数 (parameter)、激活值、优化器状态和中间梯度。A100 提供 40GB 或 80GB 高带宽内存 (HBM2e) 配置，相比前几代提供了明显更大的容量和带宽（最高约 2 TB/s）。这种更大的内存占用直接支持训练更大的模型或使用更大的批次大小，而不会遇到内存不足错误。
3. \*\*第三代 NVLink：\*\*训练单个 GPU 内存无法容纳的模型，需要将模型和计算分布到多个 GPU 上。这些 GPU 之间的通信速度成为瓶颈。Ampere 采用第三代 NVLink，相比标准 PCIe 通道，提供高得多的 GPU 到 GPU 直接带宽（例如，每块 A100 总带宽 600 GB/s）。这对于高效实现模型并行（张量并行、流水线并行）和减少数据并行中的通信开销非常重要。

digraph G {
layout=neato;
node [shape=box, style=filled, fillcolor="#a5d8ff", fontsize=12];
edge [color="#495057", fontsize=12];
GPU1 [pos="0,1!"];
GPU2 [pos="2,1!"];
GPU3 [pos="0,-1!"];
GPU4 [pos="2,-1!"];

subgraph cluster\_nvlink {
label = "NVLink 连接";
style=dashed;
color="#adb5bd";
GPU1 -- GPU2 [len=2];
GPU3 -- GPU4 [len=2];
GPU1 -- GPU3 [len=2];
GPU2 -- GPU4 [len=2];
GPU1 -- GPU4 [len=2.8]; // 4-GPU 设置中常见的对角连接
GPU2 -- GPU3 [len=2.8];
}
HostCPU [pos="1,-2.5!", shape=ellipse, fillcolor="#ced4da"];
HostCPU -- GPU1 [style=dashed, color="#868e96", label="PCIe"]; // 代表 PCIe 连接
HostCPU -- GPU2 [style=dashed, color="#868e96"];
HostCPU -- GPU3 [style=dashed, color="#868e96"];
HostCPU -- GPU4 [style=dashed, color="#868e96"];
}

````
```
> NVLink 在服务器节点中提供 GPU 之间的高带宽直接连接的简化视图，相比之下，连接到主 CPU 的 PCIe 连接速度较慢。
````

4. \*\*多实例 GPU (MIG)：\*\*虽然较少直接用于单一、大规模训练运行，但 MIG 允许将单个 A100 分区为最多七个独立的 GPU 实例，每个实例都有自己的内存、缓存和计算核心。这有助于通过并行运行较小的推理工作负载或开发任务来最大限度地提高利用率。

### NVIDIA Hopper 架构（例如 H100）

Hopper 架构以 H100 GPU 为典型代表，在 Ampere 的基础上构建，其功能明确针对 GPT-4 级别 Transformer 等大型模型的需求。

1. \*\*第四代张量核心和 Transformer 引擎：\*\*Hopper 引入了对新 8 位浮点格式 (FP8) 的支持，该格式由两种变体组成：E4M3（4 位指数，3 位尾数）和 E5M2（5 位指数，2 位尾数）。FP8 相比 FP16/BF16 提供两倍的吞吐量 (throughput)和一半的内存占用。重要的是，Hopper 包含 Transformer 引擎。该引擎使用软件和硬件启发式方法，在训练期间动态分析层统计数据，并决定在 Transformer 层内的特定矩阵乘法中使用 FP8 还是 FP16/BF16，同时保持更高精度的累加和高保真结果。这旨在提供 FP8 的速度和内存优势，同时无需大量手动调整或牺牲模型精度，这对于万亿级参数 (parameter)模型尤其有利。

digraph G {
rankdir=LR;
node [shape=record, style=filled, color="#adb5bd", fontsize=24];
Input [label="{输入 (FP16/BF16)}", fillcolor="#e9ecef"];
TransformerEngine [label="{Transformer 引擎|{分析统计|选择精度}}", shape=record, fillcolor="#96f2d7"];
TensorCoreOp [label="{第四代张量核心|{FP8 或 FP16/BF16 矩阵乘法}}", fillcolor="#a5d8ff"];
Output [label="{输出 (FP16/BF16)}", fillcolor="#e9ecef"];

Input -> TransformerEngine:f0;
TransformerEngine:f1 -> TensorCoreOp;
TensorCoreOp -> Output;
}

````
```
````

> 该流程图显示 Transformer 引擎分析层统计数据，以为 Hopper 的张量核心选择最佳精度（FP8 或更高）。

2. \*\*HBM3 内存：\*\*H100 使用 HBM3 内存，相比 A100 的 HBM2e，提升了容量（通常为 80GB）并大幅增加了带宽（最高约 3.35 TB/s）。这进一步缓解了内存瓶颈，允许更大的模型、激活值或训练数据批次直接驻留在 GPU 上。
3. \*\*第四代 NVLink 和 NVLink 交换系统：\*\*Hopper 增加了其 GPU 到 GPU NVLink 直接连接的带宽（每块 H100 总带宽最高达 900 GB/s）。更重要的是，NVIDIA 引入了 NVLink 交换系统。这允许在专用“NVLink 域”内连接多达 256 个 H100 GPU，直接在 GPU 之间提供全到全的高带宽通信，对于域内的某些通信模式，无需通过 InfiniBand 等较慢的网络结构。这旨在大幅提高跨越多个节点的超大型模型训练运行的扩展效率。
4. \*\*DPX 指令：\*\*Hopper 包含旨在加速涉及动态规划算法的新指令。尽管可能适用于序列比对或生物信息学等领域，但它们对标准 Transformer 训练的直接影响可能不如张量核心或内存带宽的改进那么明显。

### 代系比较总结

从 Ampere (A100) 到 Hopper (H100) 为 LLM 训练带来了实在的好处：

| 特性 | Ampere (A100) | Hopper (H100) | 对 LLM 的重要性 |
| --- | --- | --- | --- |
| **张量核心代系** | 第三代 | 第四代 | 更高吞吐量 (throughput)，支持新精度。 |
| **精度** | TF32, FP16, BF16 | FP8 (通过 Transformer 引擎), FP16, BF16 | FP8 大幅提升速度并减少内存。 |
| **内存类型** | HBM2e | HBM3 | 更高带宽和容量支持更大模型。 |
| **最大内存** | 80 GB | 80 GB (SXM 变体) | 容纳更大的状态和激活。 |
| **内存带宽** | 约 2.0 TB/s | 约 3.35 TB/s | 更快的数据访问，更少内存限制的执行。 |
| **NVLink 代系** | 第三代 (600 GB/s) | 第四代 (900 GB/s) + 交换系统 | 更快的 GPU 间通信，更好的大规模扩展。 |



[交互图表：张量核心吞吐量峰值估算 (稀疏性关闭)](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-18-hardware-considerations-llm-training/gpu-architectures-nvidia#plot-ekrpb7)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "A100 (BF16/FP16)",
        "H100 (FP16/BF16)",
        "H100 (FP8)"
      ],
      "y": [
        312,
        1000,
        2000
      ],
      "type": "bar",
      "name": "峰值 TFLOPS (大约)",
      "marker": {
        "color": "#228be6"
      }
    }
  ],
  "layout": {
    "title": "张量核心吞吐量峰值估算 (稀疏性关闭)",
    "yaxis": {
      "title": "TFLOPS"
    },
    "xaxis": {
      "title": "GPU 架构和精度"
    },
    "template": "plotly_white",
    "bargap": 0.3
  }
}
```

</details>



> 张量核心运算的大约理论峰值吞吐量凸显了明显的性能提升，尤其是在 Hopper 上使用 FP8 时。实际性能因工作负载而异。

尽管 Hopper 提供优越性能，但 Ampere GPU 仍然强大且广泛使用。在它们之间或与其他潜在硬件的选择，涉及在性能需求、预算限制和硬件可用性之间取得平衡。理解这些架构差异有助于选择合适的硬件，并配置训练任务以有效发挥其特定优势。

## 参考资料

- [NVIDIA A100 Tensor Core GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf) — NVIDIA Corporation (2020)
  Journal: NVIDIA Whitepaper; Publisher: NVIDIA Corporation
  提供了NVIDIA Ampere A100 GPU的全面技术规格和架构细节，包括Tensor Cores、HBM2e和NVLink。
- [NVIDIA H100 Tensor Core GPU Architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — NVIDIA Corporation (2022)
  Journal: NVIDIA Whitepaper; Publisher: NVIDIA Corporation
  介绍了NVIDIA Hopper H100 GPU的详细技术规格和架构改进，涵盖了FP8、Transformer Engine、HBM3和NVLink Switch System。
- [NVIDIA H100 Transformer Engine: The New Standard for AI Model Acceleration](https://developer.nvidia.com/blog/nvidia-h100-transformer-engine-new-standard-ai-model-acceleration/) — Kari Pulli, Hongxiang Li, Sharat Singh, and Cliff Woolley (2022)
  Journal: NVIDIA Developer Blog; Publisher: NVIDIA
  阐述了Hopper架构中Transformer Engine和FP8精度的实现及其优势，加速了大型语言模型训练。

---

[上一节](../17-LLMs%E7%9A%84%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/05-%E9%80%89%E6%8B%A9%E4%BC%98%E5%8C%96%E5%99%A8%E8%B6%85%E5%8F%82%E6%95%B0%20%28lr%2C%20betas%2C%20eps%2C%20weight_decay%29.md) · [下一节](02-TPU%20%E6%9E%B6%E6%9E%84%EF%BC%88Google%20TPU%EF%BC%89.md)

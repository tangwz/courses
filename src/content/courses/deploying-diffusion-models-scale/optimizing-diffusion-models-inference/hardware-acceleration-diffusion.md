---
course: "deploying-diffusion-models-scale"
chapter: "optimizing-diffusion-models-inference"
lesson: "hardware-acceleration-diffusion"
sourceId: 5390
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-2-optimizing-diffusion-models-inference/hardware-acceleration-diffusion"
title: "硬件加速（GPU、TPU）"
description: "有效使用专用硬件以加快扩散计算。"
order: 5
plots: ["plots/5390-0.json"]
sourceHash: "3562495cce13b9eab5450fad606bfc161242533c1499966d875808ad6cda44dc"
sourceCorrections: []
---

扩散模型，特别是每一步处理带噪声隐变量的核心UNet，计算密集，主要由卷积和矩阵乘法等操作组成。硬件加速侧重于更快地执行这些计算。这些操作表现出高数据并行性，使其成为图形处理单元（GPU）和张量处理单元（TPU）等专用硬件加速器的理想选择。单独依靠传统CPU进行扩散模型推理 (inference)，通常会导致生成时间过长而不切实际，特别是对于高分辨率图像或有严格吞吐量 (throughput)要求的场景。

### 图形处理单元（GPU）

GPU最初用于图形渲染，拥有数千个核心，这些核心经过优化，可以同时对多个数据点执行相同的操作（单指令多数据 - SIMD）。这种架构非常适合深度学习 (deep learning)中的基本张量操作。

- **并行处理能力：** GPU的大规模并行性使其能够比CPU更快地执行扩散模型去噪网络中的大量矩阵乘法和卷积，而CPU通常只有少数为顺序任务设计的强大核心。
- **CUDA与库：** NVIDIA GPU通过CUDA编程模型和cuDNN（CUDA深度神经网络 (neural network)库）等库，提供了一个成熟的环境来加速深度学习工作负载。PyTorch和TensorFlow等框架大量使用了这些库，通常只需少量代码修改即可在兼容GPU上运行模型。您只需确保将张量和模型移动到GPU设备（例如，在PyTorch中使用`.to('cuda')`）。
- **GPU内存（VRAM）：** 扩散模型，特别是最先进的版本，可能拥有数十亿参数 (parameter)，并且需要大量内存来存储模型权重 (weight)、中间激活和正在处理的隐变量。GPU上可用的显存（VRAM）是一个重要的限制。显存不足会导致折衷，例如使用较小的批处理大小（影响吞吐量 (throughput)）或需要更复杂的模型并行技术。NVIDIA的A100或H100等高端GPU提供大容量内存（40GB、80GB或更多），专门用于大型模型推理 (inference)。
- **混合精度：** 现代GPU通常具有专门的张量核心，旨在显著加速混合精度（例如FP16）矩阵乘法，与FP32精度相比，在减少内存使用的情况下提供大幅提速。这与前面讨论的量化 (quantization)技术配合良好。

> 此图说明了CPU（用于顺序任务的少数强大核心）和GPU（用于并行任务的众多核心）之间处理方式的根本区别，这使得GPU适用于扩散模型中的张量操作。

### 张量处理单元（TPU）

TPU是谷歌定制设计的专用集成电路（ASIC），专门用于加速机器学习 (machine learning)工作负载，特别是那些使用TensorFlow开发的（尽管通过XLA对PyTorch的支持也很完善）。

- **矩阵乘法重点：** TPU采用脉动阵列，这是一种专门的硬件设计，经过优化以高速和高能效执行大型矩阵乘法。这使得它们在大型神经网络 (neural network)的核心计算中非常有效。
- **高带宽内存（HBM）：** 与高端GPU类似，TPU配备了HBM，可以快速访问存储在内存中的模型参数 (parameter)和激活。
- **互连：** TPU通常以“Pod”的形式使用，包含多个通过高速互连连接的TPU芯片，方便大规模分布式训练和推理 (inference)场景。
- **云集成：** TPU主要通过谷歌云平台（GCP）提供。虽然对于某些工作负载，TPU可能比GPU提供更高的每美元性能，但使用TPU通常会使您的部署策略与GCP生态系统更紧密地联系在一起。
- **软件栈：** 运用TPU通常需要使用支持XLA（加速线性代数）编译器的框架，该编译器会优化并编译模型图，以便在TPU硬件上执行。



![每张图片的扩散推理时间](plots/5390-0.json)



> 示例扩散任务在不同硬件类型上的相对推理速度比较。实际时间根据模型、优化和具体硬件代次的不同而有较大差异。

### 选择合适的加速器

选择合适的硬件需要平衡几个方面：

1. **性能要求：** 目标延迟（每张图片的时间）和吞吐量 (throughput)（每秒图片数量）是多少？高吞吐量或低延迟要求通常需要强大的GPU或TPU。
2. **模型大小与复杂度：** 更大的模型需要更多的显存 (VRAM)和计算能力，这通常会排除低端GPU或CPU。
3. **预算：** 高端GPU和TPU在采购/租用和功耗方面都代表着较高的成本。评估性价比。云服务提供商提供不同层级的GPU/TPU，价格也不同（包括可能节省费用的竞价实例，稍后讨论）。
4. **软件生态与框架：** 确保您选择的模型框架（PyTorch、TensorFlow、JAX）和优化库（如TensorRT，稍后讨论）与目标硬件兼容。
5. **可用性与基础设施：** 考虑您选择的云平台或本地基础设施上特定硬件类型的可用性。将专用硬件集成到您的部署工作流中（例如，Kubernetes设备插件）也是一个考量因素。

有效使用GPU或TPU对于大规模部署扩散模型非常重要。虽然框架处理了大部分底层交互，但了解底层硬件的能力和限制有助于您在优化和基础设施设计过程中做出明智的决定，确保您的部署达到性能和成本目标。下一节将说明编译器优化如何进一步提升这种专用硬件的性能。

## 参考资料

- [A Domain-Specific Architecture for Deep Neural Networks](https://dl.acm.org/doi/10.1145/3154484) — Norman P. Jouppi, Cliff Young, Nishant Patil, David Patterson (2018)
  Journal: Communications of the ACM; Publisher: ACM; Volume: 61; Pages: 50-59; DOI: [10.1145/3154484](https://doi.org/10.1145/3154484)
  介绍谷歌第一代张量处理单元（TPU）的架构和设计原则。
- [Mixed-Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: ICLR 2018; DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍了使用混合精度训练深度神经网络的方法，显著提升了配备Tensor Cores的GPU训练速度并减少内存占用。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA
  NVIDIA GPU CUDA编程官方指南，内容包括架构、编程模型和API。
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) — Ronny Krashinsky, Olivier Giroux, Stephen Jones, Nick Stam, Sridhar Ramaswamy (2020)
  Journal: NVIDIA Technical Blog; Publisher: NVIDIA
  NVIDIA Ampere GPU架构的详细技术说明，强调了用于AI任务的Tensor Cores和优化的内存带宽等特性。

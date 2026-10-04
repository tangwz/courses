---
course: "how-to-build-a-large-language-model"
chapter: "hardware-considerations-llm-training"
lesson: "tpu-architectures-google"
sourceId: 6039
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-18-hardware-considerations-llm-training/tpu-architectures-google"
title: "TPU 架构（Google TPU）"
description: "Google 张量处理单元及其机器学习工作负载设计概览。"
order: 2
plots: ["plots/6039-0.json"]
sourceHash: "ce0912002d65c5bac4313a5a911cde0f1e8cfc0b9590b4752acc8c8e6222d31a"
sourceCorrections: []
---

虽然 GPU 提供了广泛用于深度学习 (deep learning)的通用并行计算能力，但 Google 开发了张量处理单元（TPU），专门用于加速神经网络 (neural network)工作负载，特别是 Transformer 模型中占主导地位的密集矩阵乘法和向量 (vector)运算。TPU 是应用专用集成电路（ASIC），从头开始设计，旨在提高机器学习 (machine learning)的性能和效率，尤其是在大规模应用时。

### 核心设计理念：矩阵乘法加速

多数 TPU 版本的核心是矩阵乘法单元（MXU）。与 GPU 核心中成千上万个简单的算术逻辑单元（ALU）不同，MXU 是一个专用硬件模块，旨在极快地执行矩阵乘法。它通常以*脉动阵列*的形式运行。

想象数据流经一个处理单元网格。在脉动阵列中，输入从边缘进入，在每个处理单元进行运算（执行乘法和加法），部分结果系统地流向相邻单元，最终结果从另一边缘输出。这种设计最大限度地减少了芯片上的数据移动，这是主要瓶颈，从而为矩阵乘法的特定任务实现了高吞吐量 (throughput)和高能效。

> 脉动阵列中数据流（权重 (weight) `w` 和激活值 `x`）的示意图，并得到输出 `y`。处理单元（ALU）执行乘加运算。

这种专用硬件意味着 TPU 擅长处理 Transformer 中普遍存在的密集矩阵运算，但对于需要更通用并行计算的工作，其灵活性可能不如 GPU。

### TPU 代次和 Pod 架构

Google 迭代了多个 TPU 代次（v2、v3、v4、v5e、v5p），每个代次都在计算能力（以每秒拍次运算 - $10^{15}$ 次运算衡量）、内存容量（高带宽内存 - HBM）以及重要的互连速度方面提供了大幅改进。

- **TPU v2/v3:** 确立了核心架构并引入了 TPU Pod 的设计。Pod 通过专用的、高速、低延迟的二维环面芯片间互连（ICI）连接数百或数千个 TPU 芯片。该网络不同于标准以太网或 InfiniBand，并针对大规模分布式训练中常见的集体通信模式（如 `all-reduce`）进行了调整。
- **TPU v4:** 相较于 v3，性能有了显著提升，改进了 ICI 带宽和拓扑结构，并成为训练许多大型模型的主力。每个芯片都拥有专用的 ICI 链接。
- **TPU v5e/v5p:** 进一步提高了性能和效率。例如，TPU v5p 与 v4 相比，显著提升了每秒浮点运算次数（FLOPS）和 HBM 带宽，对训练规模日益增大的 LLM 非常重要。这些新代次通常具有改进的 ICI 拓扑结构，以实现更好的扩展性。



![每个芯片的峰值性能（TeraFLOPS - bfloat16）](plots/6039-0.json)



> TPU 各代次每个芯片近似 bfloat16 峰值性能对比。请注意，实际性能会因工作负载和系统配置而异。

Pod 架构对大型模型来说是一个明显优势。训练最先进的 LLM 通常需要数百或数千个加速器共同工作。TPU Pod 内的高带宽、低延迟 ICI 使这些芯片能够高效通信，使得数据、张量和流水线并行等分布式训练方法比仅仅依靠 GPU 节点之间标准数据中心网络更为有效。

### 软件生态系统和 PyTorch 集成

虽然最初与 TensorFlow 紧密结合，但 TPU 现在拥有更广泛的框架支持。对于 PyTorch 用户来说，一个重要方面是 `torch_xla` 库。PyTorch/XLA 充当桥梁，将 PyTorch 操作编译为 XLA（加速线性代数）表示，然后可以在 TPU 硬件上高效执行。

使用 TPUs 与 PyTorch 通常涉及：

1. **安装：** 设置与目标 TPU 环境（通常在 Google Cloud 上）兼容的特定 `torch` 和 `torch_xla` 版本。
2. **设备指定：** 明确将模型和数据移动到 TPU 设备。

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

# 检查 XLA 设备（TPU）是否可用
if xm.xla_available():
    # 获取 XLA 设备（例如，第一个 TPU 核心）
    device = xm.xla_device()
    print(f"Using XLA device: {device}")
else:
    print("XLA device not found. Using CPU/GPU instead.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 示例：将模型和张量移动到 TPU 设备
# model = YourTransformerModel().to(device)
# input_tensor = torch.randn(batch_size, seq_len, embedding_dim).to(device)

# 训练循环逻辑大致相似，但使用 xm 函数
# 用于分布式设置中的梯度归约（如 xm.optimizer_step）
```

虽然核心 PyTorch 模型定义通常保持不变，但使用 `torch_xla` 需要理解 XLA 编译原理，并使用特定函数进行分布式训练编排（`xm.optimizer_step`、`xm.all_reduce` 等），这与原生的 PyTorch 分布式（`torch.distributed`）不同。

### TPU 与 GPU 在 LLM 训练中的权衡

- **性能：** 对于以密集矩阵乘法为主的工作负载（如标准 Transformer），TPU 因其专用 MXU，可以在每美元或每瓦特方面提供更优的性能。其高速 ICI 对于大规模分布式训练具有明显优势。
- **灵活性：** GPU 通常对于各种计算任务更灵活。CUDA 为通用 GPU 计算提供了成熟且广泛使用的编程环境。
- **精度：** TPU 强烈偏好 `bfloat16` 格式，该格式提供与 `fp32` 相似的范围但精度较低，从而平衡了深度学习 (deep learning)的稳定性和性能。现代 GPU 也有效地支持 `bfloat16` 和 `fp16`。
- **内存：** 高端 GPU 和 TPU 都使用 HBM，但容量和带宽因代次和模型而异。内存仍然是两者的一个重要限制。
- **生态系统与可用性：** GPU 生态系统（尤其是 NVIDIA 的）可以说更广泛，在云提供商和本地部署中都有更广的硬件可用性。TPU 主要通过 Google Cloud Platform 提供。
- **软件：** 虽然 `torch_xla` 使 PyTorch 能够在 TPU 上运行，但 JAX 和 TensorFlow 生态系统在 TPU 支持方面有更长的历史，因此其原生开发体验和工具可能感觉更完善。调试 XLA 编译问题有时可能具有挑战性。

选择 TPU 还是 GPU 取决于训练任务的具体规模、模型架构、预算限制、框架偏好和平台可用性。对于需要大规模并行计算的超大型模型，TPU Pod 的一体化设计和高速互连提供了一个值得考虑的选项。

## 参考资料

- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://dl.acm.org/doi/10.1145/3143890.3140600) — Jouppi, Norman P., Cliff Young, Nishant Agrawal, Miachael Broomhall, Raymond Chou, Kaijie Dai, Manoj Gelb, Al Gleason, Chris Horton, Veri Jones, Gerard Jourdan, Samuel Knag, Mike Larson, George Ma, Andy Newman, H. Fred Pugsley, Brian R. Riley, David Ross, Alan Smith, Kourosh Taraporewalla, Valentine Turner, Norman Underwood, Chunqiang Xu, Bert Van Zee, and Wolfgang Wang (2017)
  Journal: ACM SIGARCH Computer Architecture News; Publisher: ACM; Volume: 45; Pages: 1-13; DOI: [10.1145/3143890.3140600](https://doi.org/10.1145/3143890.3140600)
  介绍了第一代谷歌TPU的架构和性能，详细说明了用于机器学习加速的矩阵乘法单元及其脉动阵列设计。
- [Google's TPU v4: A Domain-Specific Architecture for Modern Machine Learning](https://ieeexplore.ieee.org/document/10207390) — Andrew W. Norrie, Derek Bruening, Scott P. Callaway, Patrick W. D. Chi, Nicholas R. Johnson, Alex K. K. Lee, Yanzhi Wang, Jason H. Yoon, Cliff Young, Norman P. Jouppi (2023)
  Journal: Proceedings of the 50th Annual International Symposium on Computer Architecture (ISCA); Publisher: IEEE; Pages: 818-831; DOI: [10.1109/ISCA55941.2023.00063](https://doi.org/10.1109/ISCA55941.2023.00063)
  描述了TPU v4的架构改进，包括其Pod架构和高带宽互连，支持大规模分布式机器学习训练。
- [PyTorch/XLA Documentation](https://pytorch.org/xla/) — PyTorch Contributors (2024)
  使用PyTorch通过XLA编译器在谷歌TPU上运行的官方指南，涵盖了安装、配置和编程实践。
- [Tensor Processing Units (TPUs)](https://cloud.google.com/tpu/docs) — Google Cloud Documentation (2024)
  Publisher: Google Cloud
  谷歌云官方资源，提供了TPU代次、功能以及如何访问和利用TPU进行机器学习工作负载的概览。

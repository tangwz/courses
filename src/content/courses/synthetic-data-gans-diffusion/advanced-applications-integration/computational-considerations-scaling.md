---
course: "synthetic-data-gans-diffusion"
chapter: "advanced-applications-integration"
lesson: "computational-considerations-scaling"
sourceId: 5734
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-6-advanced-applications-integration/computational-considerations-scaling"
title: "计算考量与扩展"
description: "管理训练大型生成模型所需的计算资源。"
order: 6
plots: ["plots/5734-0.json"]
sourceHash: "a238613248ef4f0c95ce93f599732bfc9c893041b447d65f3afd675b762b69a3"
sourceCorrections: []
---

训练复杂GAN和扩散模型，特别是用于高分辨率输出或文本到图像合成等复杂条件任务时，对现有计算资源构成挑战。有效扩展这些模型需要仔细考量硬件、训练时间、内存管理和分布式计算策略。妥善处理这些方面对于取得高质量成果，通常与模型架构本身同等重要。

### 硬件要求：生成的动力

现代生成模型计算量大，高度依赖于专用硬件，主要是图形处理单元（GPU）。

- **GPU：** 高端GPU必不可少。主要考量因素是显存（VRAM）容量、Tensor Core可用性（用于混合精度加速）以及原始计算能力（以FLOPS衡量）。像StyleGAN或大型扩散模型通常需要充足的显存（例如24GB、48GB，甚至80GB以上）来支持大批量大小、高分辨率和复杂的网络架构。NVIDIA的Ampere和Hopper架构因其性能和内存容量而被普遍使用。
- **系统内存（RAM）：** 尽管显存通常是主要瓶颈，但数据加载、预处理以及在分布式训练设置中可能存储模型状态或中间数据都需要足够的系统内存。大型数据集可能需要数十或数百吉字节的RAM。
- **存储：** 快速存储（例如NVMe SSD）对于快速访问大型数据集很重要。训练涉及频繁读取数据批次，保存模型检查点也可能密集占用I/O。需保证足够的存储空间用于数据集（可能达数百吉字节或数太字节）以及在长时间训练过程中生成的众多检查点。

### 理解训练时间

训练最先进的生成模型通常不是一个快速过程。根据几个因素，训练时间可能以天、周甚至月为单位计算：

- **数据集大小和分辨率：** 更大的数据集和更高的目标分辨率自然会增加训练时间。
- **模型复杂度：** 更深的网络或具有更多参数 (parameter)的模型（例如扩散模型中的大型U-Net或复杂的StyleGAN生成器）在每次迭代中需要更多计算。
- **硬件：** 所用GPU的数量和类型直接影响训练速度。将GPU数量加倍（通过高效并行化）大致可以将训练时间减半。
- **训练迭代次数：** 实现收敛通常需要数十万或数百万次训练迭代。

举例来说，在FFHQ数据集（1024x1024图像）上训练StyleGAN2-ADA模型，即使在多个高端GPU上，也可能需要一周以上。用于文本到图像合成的大型扩散模型可能需要数百或数千个GPU日。

### 分布式训练策略

为了管理这些漫长的训练时间并应对超出单GPU内存的模型，分布式训练必不可少。

- **数据并行：** 这是最常用的策略。模型在多个GPU上复制，数据批次被分配到这些GPU，每个GPU处理其分配的部分。梯度在本地计算，然后聚合（例如求平均）到所有GPU上，以保持所有副本上的模型权重 (weight)一致更新。PyTorch的`DistributedDataParallel` (DDP) 是实现此目的的标准工具。尽管数据并行对于加速训练是高效的，但它不减少每个GPU存储模型、激活和优化器状态所需的内存。
- **模型并行：** 当模型过大无法适应单个GPU的显存 (VRAM)时，模型并行是必需的。这需要将模型本身在多个GPU上进行分区。

  - *张量并行*：将单个层或张量在GPU之间划分。操作需要持有张量不同部分的GPU之间进行通信。
  - *流水线并行*：将层顺序地在GPU之间划分。数据流经流水线阶段，不同的GPU同时处理不同的迷你批次以提升利用率。通信发生在相邻阶段之间。
- **混合方法：** 复杂的场景经常结合使用数据并行和模型并行。比如，一个大型模型可能使用模型并行在多个GPU上划分，然后这个完整的多GPU单元再使用数据并行进行复制，以更快地处理批次。DeepSpeed或Megatron-LM等库为这些策略提供了高级的实现。

> 显示基本数据并行（模型复制，数据拆分）与流水线模型并行（模型顺序拆分）的图示。

### 优化内存使用

显存 (VRAM)通常是最大的约束。有几种技术有助于减少内存占用：

- **梯度检查点（激活重计算）：** 这种技术不存储正向传播期间的所有中间激活（这会消耗大量内存），而是只存储一个子集。在反向传播 (backpropagation)期间，缺失的激活会即时重新计算。这以增加计算时间为代价来减少内存使用，通常允许训练更大的模型或使用更大的批次大小。
- **混合精度训练：** 利用较低精度的浮点数（如16位浮点数FP16或bfloat16 BF16）来存储权重 (weight)、激活和计算梯度，而不是标准的32位浮点数（FP32）。这可以将这些组件所需的内存减半，并大幅加速在具有专用核心（如NVIDIA Tensor Cores）的硬件上的计算。它需要仔细实现，通常涉及*梯度缩放*，以防止由于低精度格式的动态范围减小而导致的数值下溢或上溢问题。PyTorch（通过`torch.cuda.amp`）和TensorFlow等框架提供了内置支持。
- **优化器状态分片：** Adam等优化器为每个模型参数 (parameter)维护状态（例如，动量、方差估计），通常会消耗与梯度和参数总量相当（或更多）的内存。像DeepSpeed等库中实现的零冗余优化器（ZeRO）之类的技术，将这些优化器状态在数据并行工作器之间分区。每个GPU只持有优化器状态的一小部分，大幅减少了每个GPU的内存需求，使得训练更大的模型成为可能。



![大致显存使用细分（示意图）](plots/5734-0.json)



> 在标准FP32训练、混合精度训练以及混合精度结合优化器状态分片（如ZeRO阶段1）下，不同组件的估计相对显存使用量。请注意，使用分片后，优化器状态内存大幅减少。

### 扩散模型的特定扩展考量

尽管许多扩展原则适用于GAN和扩散模型，但扩散模型有其独有的方面：

- **采样成本：** 生成一个样本一般需要通过去噪网络进行多次正向传播（通常是数十或数百次，对应于扩散步骤），这使得采样比GAN中的单次正向传播慢。像去噪扩散隐式模型（DDIM）或减少采样步骤数量等技术可以加快推理 (inference)速度，但可能影响样本质量。
- **训练成本：** 训练涉及为每个数据样本在多个时间步上预测噪声或得分。计算成本与模型大小（通常基于U-Net）、每次训练迭代模拟的扩散时间步数量以及批次大小成比例。训练期间的内存使用主要取决于网络架构和批次大小，这与其他深度学习 (deep learning)模型类似。像梯度检查点和混合精度等内存优化措施普遍使用。

### 基础设施和实践考量

高效管理大规模训练需关注周围的基础设施：

- **云端 vs. 本地：** 云平台（AWS、GCP、Azure）提供灵活的强大硬件和托管服务访问权限，但对于长时间的训练运行可能变得昂贵。本地集群提供更多控制权，并可能降低长期成本，但需要大量前期投资和维护开销。
- **实验追踪：** 训练生成模型包含许多不同超参数 (parameter) (hyperparameter)、架构和数据集的实验。像Weights & Biases、MLflow或TensorBoard这样的工具对于记录指标、可视化结果、比较运行、追踪资源使用和管理模型检查点非常重要。
- **成本管理：** 无论是使用云资源还是本地资源，都要密切关注成本。采取诸如使用竞价实例（云端）、优化资源分配和终止闲置资源等策略。
- **数据管道：** 高效的数据加载和预处理很重要。此处的瓶颈可能导致昂贵的GPU闲置。使用优化的库，并尽可能考虑离线预处理数据。

在启动全面训练运行之前，在较小数据集或较低分辨率上进行原型设计，可以通过发现错误并有助于初始超参数调整来节省大量时间和资源。定期监控GPU利用率（`nvidia-smi`）、内存使用和训练损失，以保证效率并及早发现问题。

归根结底，扩展生成模型是一项工程实践，它结合了对模型计算特性的了解，以及硬件、软件优化和基础设施管理的高效使用。掌握这些方面是推进合成数据生成前沿的必需条件。

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models](https://ieeexplore.ieee.org/document/9355792) — Samyam Rajbhandari, Cong Guo, Jeff Rasley, Shaden Smith, Yuxiong He (2020)
  Journal: SC '20: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: IEEE; Pages: 636-649; DOI: [10.1109/SC41405.2020.00078](https://doi.org/10.1109/SC41405.2020.00078)
  介绍了ZeRO内存优化策略，通过在设备间分片优化器状态、梯度和参数，对训练大型模型至关重要。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2019)
  Journal: arXiv; DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  提出了用于训练超大型神经网络的模型并行技术，解决了模型无法适应单个设备内存的挑战。
- [Automatic Mixed Precision for Deep Learning](https://pytorch.org/docs/stable/notes/amp_examples.html) — PyTorch Documentation (2024)
  Publisher: PyTorch Foundation
  提供了在PyTorch中使用自动混合精度训练的指南和示例，有助于减少内存占用并加速计算。
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) — Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin (2016)
  Journal: arXiv preprint arXiv:1604.06174; DOI: [10.48550/arXiv.1604.06174](https://doi.org/10.48550/arXiv.1604.06174)
  描述了通过在反向传播中重新计算中间激活来减少训练期间内存消耗的方法，即梯度检查点技术。
- [Training Generative Adversarial Networks with Limited Data](https://proceedings.neurips.cc/paper/2020/file/0c949d21c43063f0301a2bb46497f901-Paper.pdf) — Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, Timo Aila (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 33
  介绍了一种自适应判别器增强方法，显著改进了有限数据下的GAN训练，其训练需要大量的计算资源。

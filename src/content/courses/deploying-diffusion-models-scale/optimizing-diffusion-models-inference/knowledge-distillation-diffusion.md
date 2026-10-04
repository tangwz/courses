---
course: "deploying-diffusion-models-scale"
chapter: "optimizing-diffusion-models-inference"
lesson: "knowledge-distillation-diffusion"
sourceId: 5383
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-2-optimizing-diffusion-models-inference/knowledge-distillation-diffusion"
title: "扩散模型的知识蒸馏"
description: "训练更小、更快的学生模型以模仿大型教师扩散模型的输出。"
order: 3
plots: []
sourceHash: "c590da82c926e6d5a7a93b94e09fe783ac6ebce95a281900db9215dd266af27e"
sourceCorrections: []
---

扩散模型虽然功能强大，但通常伴随着显著的计算开销。它们的迭代特性和庞大的参数 (parameter)量通常会导致推理 (inference)延迟较长。解决此问题的一种有效方法是**知识蒸馏 (knowledge distillation) (KD)**。其核心思想是训练一个更小、更快的模型（“学生模型”）来模仿一个更大、预训练 (pre-training)的高性能模型（“教师模型”）的行为。

### 扩散模型中的教师-学生方法

在标准分类任务中，知识蒸馏 (knowledge distillation)通常涉及将学生模型的输出对数（logits）与教师模型的软化对数（通过温度缩放）进行匹配，或对齐 (alignment)中间特征表示。将知识蒸馏应用于扩散模型需要将此原理调整到生成过程中。

主要目标是训练一个学生扩散模型（通常具有更浅或更窄的架构，例如具有更少残差块或通道的U-Net），表示为 $\epsilon_{\theta_S}(x_t, t)$，以逼近大型教师模型 $\epsilon_{\theta_T}(x_t, t)$ 的输出。

> 图示说明了扩散模型的知识蒸馏过程。学生模型通过最小化其输出与教师模型在相同输入时间步 $t$ 和噪声数据 $x_t$ 下的输出之间的差异来学习。

### 蒸馏目标

定义蒸馏目标有以下几种方法：

1. **输出匹配：** 最直接的方法是最小化学生模型和教师模型预测的噪声之间的差异。一个常用的损失函数 (loss function)是它们输出之间的均方误差（MSE）：

   
   $$
   \mathcal{L}_{KD} = \mathbb{E}_{x_0, t, \epsilon \sim \mathcal{N}(0, I)} \left[ || \epsilon_{\theta_T}(x_t, t) - \epsilon_{\theta_S}(x_t, t) ||^2 \right]
   $$
   

   其中，$x_t$ 是在时间步 $t$ 从干净数据 $x_0$ 生成的带噪声输入，$\epsilon$ 是采样噪声。期望是在数据集、时间步和噪声样本上进行的。在此过程中，教师模型的权重 (weight) $\theta_T$ 保持不变。
2. **特征图匹配：** 与其他方面的知识蒸馏 (knowledge distillation)类似，可以通过鼓励学生模型的中间特征图（例如U-Net块内的激活）类似于教师模型的特征图来迁移知识。这需要定义教师和学生架构之间的对齐 (alignment)层，并向整体目标添加一个合适的特征损失项（如L1或L2距离）。这有时可以提供更强的训练信号，特别是当学生架构与教师架构存在显著差异时。
3. **轨迹蒸馏：** 一些先进技术涉及蒸馏整个采样*轨迹*。学生模型不再仅仅匹配单个步骤的预测，而是学习生成一个状态序列 $(x_{T}, x_{T-1}, ..., x_0)$，该序列紧密匹配教师模型使用特定采样器（如DDIM）生成的序列。这旨在更好地保留教师模型的生成动态。

### 学生模型的训练

训练过程通常包含以下步骤：

1. **教师推理 (inference)：** 对于一批训练数据 $x_0$，采样时间步 $t$ 和噪声 $\epsilon \sim \mathcal{N}(0, I)$。计算带噪声的输入 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$。将 $x_t$ 和 $t$ 输入冻结的教师模型 $\epsilon_{\theta_T}$，以获得目标噪声预测 $\epsilon_{\theta_T}(x_t, t)$（如果使用特征匹配，则为中间特征图）。
2. **学生推理：** 将相同的 $x_t$ 和 $t$ 输入学生模型 $\epsilon_{\theta_S}$，以获得其预测 $\epsilon_{\theta_S}(x_t, t)$。
3. **损失计算：** 基于选定的目标计算蒸馏损失（例如，教师和学生噪声预测之间的均方误差，可能与特征损失结合）。
4. **优化：** 使用优化器（如Adam）基于计算出的损失梯度更新学生模型的权重 (weight) $\theta_S$。

此过程需要访问预训练 (pre-training)的教师模型和具代表性的数据集 $x_0$。虽然训练学生模型需要计算，但通常比从头开始训练大型教师模型所需的计算量小得多，因为它通常收敛更快，并在较小的网络上运行。

### 学生模型的架构选择

学生模型需要显著更小、更快于教师模型，以实现预期的优化益处。常见的架构修改包括：

- 减少U-Net结构中的残差块或Transformer块的数量。
- 减少卷积层或线性层中的通道数（模型宽度）。
- 使用更高效的注意力机制 (attention mechanism)，或者在某些块中完全替换注意力层。
- 简化时间嵌入 (embedding)投影网络。

具体的架构选择很大程度上取决于目标硬件、所需推理 (inference)速度、可接受的质量下降以及原始教师模型的架构。通常需要进行实验以找到最佳平衡点。

### 优点与注意事项

**益处：**

- **降低延迟：** 更小的模型推理 (inference)步骤执行速度更快，直接缩短生成图像所需的时间。
- **内存占用更少：** 需要更少的显存 (VRAM)，使其能够部署在性能较低或成本更优的硬件上（例如，更小的GPU，甚至边缘设备）。
- **计算成本降低：** 每个推理步骤的浮点运算（FLOPs）更少，这意味着更低的能耗和每生成一张图像可能更低的操作成本。

**需要考虑的方面：**

- **质量下降：** 模型大小/速度与生成质量之间几乎总是一种权衡。蒸馏后的学生模型可能无法达到与大型教师模型完全相同的图像保真度、多样性或对复杂提示的遵循程度。缩小这一差距需要仔细调整蒸馏过程、损失函数 (loss function)和学生架构。
- **训练复杂性：** 构建蒸馏训练流程需要仔细实施。超参数 (parameter) (hyperparameter)调整（学习率、结合目标时的损失权重 (weight)、学生架构细节）通常是必要的。
- **对教师模型的依赖：** 学生模型的质量上限由教师模型决定。一个普通的教师模型很可能会产生一个普通的学生模型。

### 蒸馏与其他技术的结合

知识蒸馏 (knowledge distillation)本身就是一种强大的技术，但当它与本章讨论的其他优化方法结合时，它的益处可以被放大。一个典型的多阶段优化流程可能如下所示：

1. **蒸馏：** 使用知识蒸馏从大型、高质量教师模型中训练一个更小的学生模型。
2. **量化 (quantization)：** 对蒸馏后的学生模型应用训练后量化或量化感知训练，将权重 (weight)和/或激活转换为FP16或INT8等较低精度格式。
3. **采样器优化：** 在推理 (inference)过程中使用步骤更少的更快采样方法（例如DDIM、DPM-Solver++），得益于蒸馏模型可能提升的步效率。
4. **编译器优化：** 使用TensorRT或OpenVINO等工具，针对特定目标硬件（GPU、CPU）优化最终蒸馏且可能量化的学生模型的计算图。

通过策略性地应用知识蒸馏，通常作为其他技术之前的初始步骤，可以创建显著更高效的扩散模型。这些优化后的模型在推理速度、资源利用率和成本是重要考量因素的大规模部署场景中变得更具实用性。尽管它引入了一系列自身的实现细节并需要管理质量权衡，但知识蒸馏是面向生成式AI的MLOps工具包中一个有价值的工具。

## 参考资料

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Geoffrey Hinton, Oriol Vinyals, Jeff Dean (2015)
  Journal: arXiv:1503.02531; DOI: [10.48550/arXiv.1503.02531](https://doi.org/10.48550/arXiv.1503.02531)
  介绍了将知识从大型教师模型迁移到小型学生模型的知识蒸馏概念。
- [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) — Tim Salimans, Jonathan Ho (2022)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2202.00512](https://doi.org/10.48550/arXiv.2202.00512)
  提出了一种逐步蒸馏扩散模型的方法，使学生模型需要更少的采样步数，从而显著加快推理速度。
- [Consistency Models](https://arxiv.org/abs/2303.01469) — Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever (2023)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.2303.01469](https://doi.org/10.48550/arXiv.2303.01469)
  提出了一类新的生成模型，通过蒸馏评分函数实现一步生成，有效地学习了快速采样的直接映射。

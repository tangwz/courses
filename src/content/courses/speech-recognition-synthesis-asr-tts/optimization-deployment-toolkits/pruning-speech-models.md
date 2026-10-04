---
course: "speech-recognition-synthesis-asr-tts"
chapter: "optimization-deployment-toolkits"
lesson: "pruning-speech-models"
sourceId: 3219
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-6-optimization-deployment-toolkits/pruning-speech-models"
title: "模型剪枝与稀疏化"
description: "通过权重剪枝减少模型尺寸和计算成本。"
order: 2
plots: ["plots/3219-0.json"]
sourceHash: "a8250026509240c547ee78f021d7c4df83d4a72db128fb459477959cb94d1dc3"
sourceCorrections: []
---

如章节引言所述，部署用于语音处理的复杂深度学习 (deep learning)模型，经常会遇到计算成本、内存占用和推理 (inference)延迟等方面的实际限制。当量化 (quantization)降低了权重 (weight)和激活的精度时，模型剪枝和稀疏化提供了一种补充方法，通过移除被认为不那么重要的参数 (parameter)或结构，从而创建出更小且可能更快的模型。

深度神经网络 (neural network)，特别是ASR和TTS中常见的大量Transformer和基于RNN的模型，常常包含相当多的冗余参数。这些参数对模型整体性能贡献不大，但会占用内存、带宽和计算资源。剪枝旨在识别并消除这些冗余，形成一个稀疏模型，其中许多连接或参数的值为零。

### 稀疏性和剪枝类型的认识

稀疏性是指模型中零值参数 (parameter)的比例。剪枝是引入稀疏性的过程。我们大致可以将剪枝技术分为两种主要类型：

1. **非结构化剪枝：** 这涉及移除网络中的单个权重 (weight)，通常基于它们的大小。非结构化剪枝后的权重矩阵将包含分散的零条目。尽管这可以大幅减少非零参数的数量，但要实现实际的推理 (inference)加速，通常需要专门的硬件加速器（如支持稀疏张量核的NVIDIA Ampere GPU）或能够高效处理稀疏矩阵操作的软件库。如果没有此类支持，零的不规则模式可能无法在标准CPU或GPU上转换为更快的计算。
2. **结构化剪枝：** 这种技术移除网络的整个结构元素，例如神经元（权重矩阵中的行/列）、卷积层中的滤波器或通道，甚至是Transformer中的注意力头。这会产生一个更小、更密集的模型，其在标准硬件上本身就更高效。例如，在CNN中移除一个滤波器会减少输出通道的数量，直接减少后续层的计算负载（$FLOPs$）和参数数量。在目标硬件不具备专门稀疏计算能力时，通常优先采用结构化剪枝。

### 常见剪枝策略

存在多种算法，用于确定要移除哪些参数 (parameter)或结构：

- **基于幅度的剪枝：** 这是最直接的方法。具有最小绝对值的参数（权重 (weight)）被认为最不重要并被设置为零。选择一个稀疏性目标（例如，80%稀疏度），并确定一个阈值，使得将所有低于此阈值的权重设置为零即可达到目标。这可应用于整个模型或逐层应用。通常，基于幅度的剪枝是迭代进行的：剪掉一小部分权重，对剩余模型进行几个epoch的微调 (fine-tuning)以恢复精度，然后重复直到达到期望的稀疏度。这种迭代过程通常比一次性剪掉所有权重效果更好。
- **基于重要性的剪枝：** 这些方法不单依赖幅度，而是尝试评估每个参数或结构对模型性能的贡献，通常通过对损失函数 (loss function)的影响来衡量。方法可能包括：

  - 在训练期间使用梯度信息，以确定移除后对损失增加影响最小的参数。
  - 近似二阶导数（Hessian）来估计敏感度。
  - 使用激活统计数据来识别激活频率较低的神经元。
    这些方法可能比简单的基于幅度的剪枝计算量更大，但可能识别出不那么明显的冗余。
- **训练期间剪枝（稀疏训练）：** 不再是剪枝预训练 (pre-training)的密集模型，稀疏性可以在训练过程中本身产生。像L1正则化 (regularization)这样的技术会在损失函数中添加一个惩罚项，与权重绝对值之和成比例（$\lambda \sum |w|$）。这促使权重在优化过程中趋向于零。其他方法在训练期间动态识别并移除低幅度权重，有时允许已剪枝的权重在后续变得重要时重新生长。

### 剪枝在ASR和TTS模型中的应用

剪枝可应用于语音模型的各个组成部分：

- **ASR声学模型：** 端到端ASR系统中的大型RNN（LSTM/GRU）以及特别是Transformer编码器，因其参数 (parameter)量大，是剪枝的主要候选对象。非结构化剪枝和结构化剪枝（例如，移除注意力头或前馈网络神经元）都已显示出成功。
- **TTS声学模型：** 像Tacotron 2或Transformer TTS这样的模型依赖于可剪枝的编码器和解码器。像FastSpeech这样的模型，尽管已较快，但仍可剪枝以减小尺寸。
- **神经声码器：** 尽管通常比主要声学模型小，但像WaveNet或HiFi-GAN这样的声码器仍可从剪枝中受益，尤其适用于资源受限设备的部署。卷积滤波器的结构化剪枝在此处很常见。

主要挑战是保持性能。过度激进的剪枝会显著降低ASR精度（增加词错误率，WER）或TTS自然度（降低平均意见得分，MOS）。找到稀疏度和性能之间的最佳平衡点通常需要仔细的实验和迭代微调 (fine-tuning)。

### 实际实施和权衡

实施剪枝时，考虑以下几点：

- **硬件目标：** 非结构化稀疏性提供了最高的潜在压缩率，但需要硬件/软件支持以实现加速。结构化剪枝在常规硬件上提供更可靠的加速。
- **工具包：** 像TensorFlow模型优化工具包和PyTorch的`torch.nn.utils.prune`模块这样的库，为各种剪枝技术（基于幅度、结构化、非结构化）提供了便捷的API。像ESPnet或NeMo这样的框架，可能会为其预构建的模型集成剪枝方案。
- **微调 (fine-tuning)：** 剪枝后的微调几乎总是需要，以恢复参数 (parameter)移除期间损失的精度。学习率和微调持续时间是重要的超参数 (hyperparameter)。
- **精度与效率：** 剪枝本身涉及一种权衡。你获得效率（更小的尺寸，可能更快的推理 (inference)）是以牺牲一些潜在的精度下降为代价的。目标是在这条曲线上找到满足应用要求的一个点。



![剪枝权衡（ASR示例）](plots/3219-0.json)



> 通过剪枝实现的模型稀疏度与ASR性能（WER）之间的典型关系。稀疏度增加会减小模型尺寸，但最终会降低精度。目标是找到一个在显著减小尺寸的同时，不导致WER出现不可接受的增长的稀疏度水平。

模型剪枝和稀疏化是有效的技术，用于减少大型ASR和TTS模型的资源需求。通过谨慎地移除冗余参数或结构，并常常结合微调，我们可以创建显著更小、可能更快的模型，适用于在各种环境中部署，补充量化 (quantization)等其他优化方法。

## 参考资料

- [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) — Song Han, Jeff Pool, John Tran, William J. Dally (2015)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1506.02626](https://doi.org/10.48550/arXiv.1506.02626)
  本文介绍了基于幅度的权重剪枝，随后结合量化和霍夫曼编码，展示了在保持最小精度损失的情况下显著减小模型尺寸。这是模型压缩领域的一项基础性贡献。
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Subnetworks](https://arxiv.org/abs/1803.03635) — Jonathan Frankle, Michael Carbin (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1803.03635](https://doi.org/10.48550/arXiv.1803.03635)
  提出密集神经网络包含稀疏子网络，这些子网络经过训练可以达到与原始密集网络相似的准确性，这是稀疏训练中的一个显著发现。
- [Pruning in PyTorch](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) — Michela Paganini (2024)
  Publisher: PyTorch
  PyTorch `torch.nn.utils.prune` 模块的官方文档，提供了应用各种剪枝技术的实用指导和API。

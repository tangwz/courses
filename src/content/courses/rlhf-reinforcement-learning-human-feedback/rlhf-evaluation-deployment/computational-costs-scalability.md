---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "rlhf-evaluation-deployment"
lesson: "computational-costs-scalability"
sourceId: 5209
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-7-rlhf-evaluation-deployment/computational-costs-scalability"
title: "计算成本与可扩展性"
description: "理解RLHF模型的训练和部署资源需求。"
order: 6
plots: ["plots/5209-0.json"]
sourceHash: "64e6a3a274cb6f8635d05a93c5c0b22a95f58c003a39f07c37b4341d88e61e5f"
sourceCorrections: []
---

实施和部署通过人类反馈强化学习 (reinforcement learning)（RLHF）对齐 (alignment)的模型，需要大量的计算资源。了解每个阶段的成本及其随规模增长的变化，对于实验规划、项目预算编制和优化整体工作流程很要紧。与标准的监督微调 (fine-tuning)不同，RLHF带来了复杂性，大幅增加了资源需求。

### 分析RLHF资源消耗

完整的RLHF流程，包括监督微调 (fine-tuning)（SFT）、奖励模型（RM）训练和近端策略优化（PPO）微调，展现出不同的计算特性：

1. **监督微调（SFT）：** 这一初始阶段类似于标准的语言模型微调。成本取决于基础模型大小、数据集大小、序列长度和训练轮次。尽管要求较高，特别是对于大型模型，但其计算模式是在因果语言建模目标上进行相对直接的梯度下降 (gradient descent)。资源需求主要由GPU内存（用于存储模型参数 (parameter)、梯度和优化器状态）和GPU计算（用于前向/后向传播）驱动。
2. **奖励模型（RM）训练：** 训练奖励模型涉及在成对偏好数据上微调语言模型（通常从SFT模型或基础预训练 (pre-training)模型初始化）。其主要操作是处理给定提示$x$的响应对$(y_w, y_l)$，使用奖励模型计算它们各自的标量奖励分数，并优化一个损失函数 (loss function)（如Bradley-Terry似然），使偏好响应$y_w$的分数高于较差响应$y_l$。

   - **内存：** 类似于SFT，但在未仔细优化的情况下，每个训练步骤需要容纳两次前向传播（每对响应各一次）。
   - **计算：** 每个数据点与SFT相当，但数据集结构（对）会影响批处理和训练动态。
3. **近端策略优化（PPO）微调：** 这通常是计算成本最高的阶段。它包括一个迭代循环，其中包含几个组成部分：

   - **策略生成（生成）：** 当前策略模型（行动者）根据数据集中的提示$x$生成响应$y$。这需要从语言模型中采样，其速度可能取决于生成长度和解码策略。
   - **奖励计算：** 每个生成的响应$(x, y)$由训练好的奖励模型评分。这需要通过奖励模型进行一次前向传播。
   - **价值估计：** 价值模型（评论者，通常从奖励模型或SFT模型初始化）估计当前状态（提示）的预期回报（累积奖励）。这需要通过价值模型进行另一次前向传播。
   - **优势计算：** 使用奖励和价值估计，计算优势（例如，通过广义优势估计，GAE）。
   - **策略优化：** 使用PPO目标函数更新策略模型。这包括计算当前策略与参考策略（通常是初始SFT模型）之间的概率比，计算策略损失、价值损失，并执行梯度更新。一个要点是，此步骤需要通过*策略*（行动者）和*价值*（评论者）模型进行前向和后向传播。
   - **KL散度惩罚：** 计算当前策略的输出分布与参考策略的输出分布之间每个token的Kullback-Leibler（KL）散度，在PPO更新期间增加了一个计算步骤，对于规范策略偏移是必要的。

### PPO的多模型负担

PPO阶段通常需要同时在GPU内存中保存多个大型模型：

- **策略模型（行动者）：** 正在积极训练的模型。需要内存来存储参数 (parameter)、梯度和优化器状态。
- **参考模型（SFT模型）：** 用于计算KL散度惩罚。通常保持冻结，仅在进行KL计算所需的前向传播期间需要参数内存。
- **奖励模型（RM）：** 用于对生成的响应进行评分。通常也保持冻结，在其前向传播期间需要参数内存。
- **价值模型（评论者）：** 用于估计状态值以进行优势计算。与策略一同积极训练，需要内存来存储参数、梯度和优化器状态。

这种同时的内存需求对于大型模型（例如70亿参数及以上）很容易超出单个GPU的容量，因此需要分布式训练设置或参数高效微调 (fine-tuning)（PEFT）等方法。



![各阶段相对计算成本（示例）](plots/5209-0.json)



> 示例性分解显示，由于其迭代生成、评分和多模型更新循环，PPO通常比SFT或奖励模型训练消耗更多的计算资源。确切的比例因实现和超参数 (hyperparameter)而异。

### 影响规模的因素

RLHF中的计算成本受多种因素影响：

- **模型大小：** 在涉及梯度计算的训练阶段，变压器 (Transformer)模型的成本（内存和计算）通常与参数 (parameter)数量呈二次或更差的关系。更大的模型需要更多显存 (VRAM)、每步骤更多浮点运算，并且常需要更复杂的分布式训练策略。
- **序列长度：** 更长的提示和生成的响应会增加前向/后向传播期间的激活内存和计算负载。
- **批处理大小：** 更大的批处理大小能提高计算效率（浮点运算利用率），但会增加内存需求。在PPO中，有效批处理大小是每批次提示数量和每个提示生成的响应数量的组合。
- **数据集大小：** 更大的SFT和偏好数据集需要更多的训练步骤才能收敛，直接增加了总计算时间。PPO生成过程中使用的提示数据集大小也会影响运行时间。
- **PPO步数：** PPO微调 (fine-tuning)可能需要多次生成和优化迭代，这大大增加了总成本。

### 成本降低的优化策略

鉴于高资源需求，可以采用多种策略来使RLHF更可行：

- **参数 (parameter)高效微调 (fine-tuning)（PEFT）：** 像低秩适应（LoRA）或QLoRA这样的方法大幅减少了可训练参数的数量。这大大降低了梯度和优化器状态的内存需求，可能使得多个PEFT适应模型（策略、价值）可以在更少的GPU上运行。然而，前向传播仍需要完整的原始模型参数。
- **高效PPO实现：** Hugging Face TRL (`trl`)等库针对RLHF进行了优化，包含行动者和评论者之间共享层、高效生成和评分批处理以及优化KL估计等方法。
- **混合精度训练：** 使用`bfloat16`或`float16`等格式可以减少内存使用，并在兼容硬件（如NVIDIA Tensor Cores）上加速计算，通常对最终模型质量影响很小。
- **梯度累积：** 减少模型权重 (weight)更新频率，通过累积多个小批次的梯度来实现。这允许使用比一次性内存容量更大的有效批处理大小进行训练。
- **ZeRO/FSDP：** 像DeepSpeed ZeRO（零冗余优化器）或PyTorch的完全分片数据并行（FSDP）这样的分布式训练方法，可以将模型状态（参数、梯度、优化器状态）分布到多个GPU/节点上，从而训练极大型模型。
- **模型蒸馏：** 训练一个更小、更快的学生模型来模仿经过RLHF微调的大型教师模型的行为，可以降低推理 (inference)成本。
- **替代算法：** 像直接偏好优化（DPO）这样的方法绕过了显式奖励建模步骤，与PPO相比，可能降低了复杂性和多模型负担，尽管它们有自己的计算取舍。

### 推理 (inference)成本与部署

虽然训练成本高昂，但部署RLHF微调 (fine-tuning)模型也涉及成本。推理需要加载最终的策略模型（通常较大）。

- **延迟：** 对于长输出，逐个token生成响应可能很慢。
- **吞吐量 (throughput)：** 同时服务多个用户需要大量的GPU资源。
- **硬件：** 大型模型通常需要GPU才能获得可接受的推理速度。
- **优化：** 量化 (quantization)、优化内核（例如FlashAttention）和连续批处理等方法对于高效部署很要紧。

与大小相似的SFT模型相比，RLHF在本质上并未改变推理成本情况，但对更高质量和更安全交互的预期，通常促使部署最大可行模型，从而产生可观的服务成本。规划这些成本，包括硬件配置和推理优化，是部署流程中必要的一部分。

了解这些计算需求和可扩展性特点对于有效规划、执行和部署RLHF项目很要紧。需要仔细考量模型大小、批处理大小、硬件可用性和优化方法，以便有效管理资源。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://doi.org/10.48550/arXiv.2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适配（LoRA），这是一种主流的参数高效微调技术，能够显著降低大型语言模型训练的内存和计算成本。
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://dl.acm.org/doi/10.1109/SC20-501.2020.00018) — Samyam Rajbhandari, Cong Li, Zhun Liu, Kshitij Kumar, and Yuxiong He (2020)
  Journal: SC20: International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: IEEE; Pages: 1-16; DOI: [10.1109/SC20-501.2020.00018](https://doi.org/10.1109/SC20-501.2020.00018)
  介绍了ZeRO（零冗余优化器），这是一系列内存优化技术，通过在设备间分片模型状态，对训练巨型深度学习模型至关重要。
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (2023)
  Journal: arXiv preprint arXiv:2305.18290; DOI: [10.48550/arXiv.2305.18290](https://doi.org/10.48550/arXiv.2305.18290)
  介绍了直接偏好优化（DPO），这是一种替代PPO的算法，用于根据人类偏好微调语言模型，通过移除显式奖励模型来简化训练过程。

# 优化强化学习训练循环 (PPO 效能)

来源：[原文](https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-8-optimization-scalability-cai-rlaif/optimizing-ppo-efficiency)

[返回章节目录](README.md) · [返回课程目录](../README.md)

强化学习 (reinforcement learning)阶段，特别是使用近端策略优化（PPO）时，在RLAIF管线中通常代表着大量的计算工作。训练涉及反复生成响应（rollouts），使用AI偏好或奖励模型评估它们，计算优势估计，并对策略和价值模型执行多次梯度更新。优化此循环对于使RLAIF实用化不可或缺，尤其是在大型语言模型（LLM）的应用中。

### 分析PPO循环的组成部分

RLAIF中的PPO核心循环通常包含这些计算量大的步骤：

1. **Rollout 阶段：** 当前LLM策略对一批提示生成响应。这需要对可能非常大的策略模型进行多次正向传递。
2. **奖励计算：** 每个生成的响应（通常与提示配对或与替代响应比较）由AI奖励模型（RM）评估。这涉及对另一个可能很大的模型进行正向传递。
3. **价值估计：** 一个价值模型（通常与策略模型共享参数 (parameter)）估计在rollout期间遇到的状态的预期未来奖励。
4. **优势计算：** 使用广义优势估计（GAE）或类似方法，这需要奖励信号和价值估计。
5. **梯度更新阶段：** 对从收集到的rollout数据中采样的迷你批次执行多个随机梯度上升周期，以优化PPO替代目标函数，更新策略和价值模型的参数。

优化PPO循环需要处理这些阶段中的效率不足之处。

### 减少Rollout和奖励计算延迟

尽管第8章“反馈生成的有效实现”部分介绍了奖励模型本身的优化，但在PPO循环范围内，请考虑：

- **异步执行：** 如果系统架构允许，可以使rollout生成、奖励计算和梯度更新重叠进行。一批数据可以进行梯度更新，同时下一批数据正在生成和评估。这需要对模型版本和数据管线进行细致管理。
- **微批处理：** 在rollout和奖励计算阶段，以更小的微批处理提示并计算奖励。这可以通过让更小的计算块填充管线来提高GPU利用率。
- **奖励模型蒸馏：** 如前所述，在强化学习 (reinforcement learning)阶段使用一个明显更小、经过蒸馏的奖励模型可以大幅减少奖励计算的延迟，如果蒸馏效果好，通常对对齐 (alignment)性能的影响很小。

### 优化梯度更新阶段

梯度更新阶段，涉及对收集到的经验进行多次遍历，在初始rollout之后通常是计算量最大的部分。主要的优化策略有：

#### 内存优化技巧

使用PPO训练大型LLM需要大量内存，因为涉及到模型参数 (parameter)、优化器状态、梯度和激活内存。

- **混合精度训练：** 采用16位浮点格式（FP16或BF16）可以将参数、梯度和优化器状态的内存使用量几乎减半，同时在兼容硬件（如NVIDIA Tensor Cores）上加速计算。需要谨慎实施，通常使用梯度缩放来保持数值稳定性。BF16通常比FP16在大型模型训练中提供更佳的稳定性，但需要较新的硬件。
- **梯度累积：** 不必为每个迷你批次计算梯度并更新权重 (weight)，而是将多个迷你批次的梯度累积起来，然后执行一个优化器步骤。这实际模拟了更大的批次大小，而不会相应增加内存，尽管会增加每个有效更新步骤所需的时间。
- **强化学习 (reinforcement learning)中的参数高效微调 (fine-tuning)（PEFT）：** 像低秩适应（LoRA）这样的技术可以适用于PPO更新阶段。不更新所有LLM参数，而只训练小得多的LoRA适配器参数。这大幅减少了：
  - **优化器状态内存：** Adam等优化器存储每个可训练参数的动量。优化数百万而非数十亿参数可节省大量内存（优化器内存通常减少70%以上）。
  - **梯度内存：** 存储梯度所需的内存更少。
  - **计算：** 由于只计算适配器参数的梯度，反向传播 (backpropagation)速度更快。
    当基础LLM在强化学习期间保持冻结，并且只有适配器学习对齐 (alignment)策略调整时，此方法效果尤其好。

> 在PPO更新期间使用LoRA，可以使强化学习微调在VRAM远低于完全微调的硬件上变为可能。权衡之处可能是收敛动态或最终性能略有不同，这需要经验验证。

#### 计算速度提升

- **优化后的核：** 使用像`cuDNN`这样的库以及深度学习 (deep learning)框架（PyTorch、TensorFlow）提供的专门核，这些核对GPU架构进行了高度优化。请确保您使用的是最新版本的框架。
- **融合操作：** 框架通常提供“融合”操作（例如，融合Adam优化器，融合层归一化 (normalization)），它们将多个计算合并为一次核启动，从而减少内存带宽需求并提高计算吞吐量 (throughput)。
- **高效的GAE实现：** 确保您的广义优势估计计算是向量 (vector)化的并得到了高效实现，避免对轨迹数据使用Python循环。
- **优化器选择：** 尽管Adam/AdamW很常见，但如果内存或计算是瓶颈，可以考虑其他选择。例如，AdaFactor可以提供内存节省，特别是对于大型模型，尽管它可能需要不同的超参数 (hyperparameter)调优。

#### 分布式训练

对于大规模RLAIF，分布式PPO计算通常是必需的。

- **分布式数据并行（DDP）：** 这是常见方法。Rollout数据被分片到多个工作器（GPU/节点）上。每个工作器在其分片上计算梯度，并在优化器步骤在所有工作器上同步更新模型参数之前，对梯度进行同步（例如，使用all-reduce）。像PyTorch DDP或`deepspeed`这样的框架提供了实现。
- **ZeRO 优化：** 像`deepspeed`这样的库实现了ZeRO（零冗余优化器）技术，该技术将优化器状态、梯度甚至参数分区到各个工作器上，从而减少每个GPU的内存占用，使得训练更大的模型成为可能。

> PPO循环的概览，显示了计算阶段（蓝色/黄色）以及相应的优化策略（绿色）。虚线表示优化应用的地方。

### PPO超参数 (parameter) (hyperparameter)调优以提高效率

某些PPO超参数直接影响计算开销：

- **PPO周期数：** 减少每个rollout批次的更新周期数（`num_epochs`）直接降低计算量，但这会使学习的样本效率降低，可能总体需要更多rollout。找到一个适合您设置的平衡点。
- **迷你批次大小：** 更大的迷你批次可以在更新阶段产生更稳定的梯度和更好的硬件利用率，但会增加内存需求。更小的迷你批次会减少内存，但由于并行度较低和梯度可能不稳定，训练时间可能更长。梯度累积可以缓解有效批次大小增加带来的内存限制。
- **Rollout批次大小：** 在执行更新之前收集的经验数量（`num_rollout_steps`或总批次大小）需要在数据收集成本（推理 (inference)）与更新成本（训练）之间进行权衡。更大的rollout可以分摊更新的固定成本，但会增加存储经验的内存，并且如果策略在rollout期间发生较大变化，可能会导致数据过时。

优化RLAIF中的PPO循环是一个多方面的过程，它涉及算法选择（PEFT）、实现细节（混合精度、融合操作）、基础设施（分布式训练）以及细致的超参数调优。通过系统地解决rollout生成、奖励计算和梯度更新中的瓶颈，您可以大幅减少使用这些有效的强化学习 (reinforcement learning)技术对齐 (alignment)LLM所需的时间和资源。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了PPO算法，一种广泛用于策略优化的强化学习方法。
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (2015)
  Journal: arXiv preprint arXiv:1506.02438; DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  详细介绍了广义优势估计 (GAE)，一种减少策略梯度方法方差的技术，对高效PPO很有用。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  提出了低秩适应 (LoRA)，一种参数高效的微调技术，显著降低内存和计算需求，尤其适用于基于LLM的PPO。
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://doi.org/10.1145/3429388.3444838) — Samyam Rajbhandari, Cong Xu, Yuxiong He, Jeff Rasley, Shaden Smith, Olatunji Ruwase, Dean Macy (2020)
  Journal: SC '20: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: Association for Computing Machinery (ACM); DOI: [10.1145/3429388.3444838](https://doi.org/10.1145/3429388.3444838)
  介绍了ZeRO（零冗余优化器），一套用于分布式训练的内存优化技术，可支持训练具有数十亿到数万亿参数的模型。
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  描述了使用FP16的混合精度训练，这是一种减少内存占用并加速兼容硬件上计算的技术。

---

[上一节](02-%E9%AB%98%E6%95%88%E7%9A%84%E5%8F%8D%E9%A6%88%E7%94%9F%E6%88%90.md) · [下一节](04-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E7%AD%96%E7%95%A5.md)

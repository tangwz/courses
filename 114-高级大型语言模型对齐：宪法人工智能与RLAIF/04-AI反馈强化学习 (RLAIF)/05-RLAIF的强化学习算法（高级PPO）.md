# RLAIF的强化学习算法（高级PPO）

来源：[原文](https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-4-rlaif-concepts/rlaif-rl-algorithms)

[返回章节目录](README.md) · [返回课程目录](../README.md)

AI偏好模型 $p_\theta(y_w \succ y_l | x)$ 能够根据AI生成的标签预测给定提示哪个响应更好。该模型用于指导语言模型策略（记作$\pi_\phi$）的优化。强化学习 (reinforcement learning)（RL）为此优化提供了框架。近端策略优化（PPO）已成为RLHF和RLAIF中微调 (fine-tuning)大型语言模型的实际标准算法，主要因为它相对于其他RL算法具有较好的稳定性和样本效率。

本节详细说明了PPO如何在RLAIF背景下进行调整和应用，强调了当奖励信号源自AI模型而非直接人类反馈时所需的具体考量和高级方法。

### RLAIF中的PPO目标

基本目标与RLHF中保持一致：训练策略$\pi_\phi$以生成提示$x$的响应$y$，使其最大化奖励信号，同时避免策略与参考策略$\pi_{ref}$发生过大偏离。参考策略通常是RL微调 (fine-tuning)*之前*的模型，例如CAI阶段产生的监督微调（SFT）模型，甚至是基础预训练 (pre-training)模型。这种约束对于维持模型的通用能力以及避免灾难性遗忘或向狭隘的、高奖励但低质量的生成策略崩溃非常重要。

在RLAIF中使用PPO优化的标准目标函数，结合了来自AI偏好模型的预期奖励和库尔巴克-莱布勒（KL）散度惩罚项：


$$
L(\phi) = \mathbb{E}_{x \sim D, y \sim \pi_\phi(\cdot|x)} [r_\theta(x, y) - \beta (\log \pi_\phi(y|x) - \log \pi_{ref}(y|x))]
$$


此处：

- $x \sim D$ 表示从数据集 $D$ 中采样的提示。
- $y \sim \pi_\phi(\cdot|x)$ 是当前正在优化的策略生成的响应。
- $r_\theta(x, y)$ 是分配给提示-响应对 $(x, y)$ 的标量奖励。此奖励来自AI偏好模型 $p_\theta$。一种常见的推导方式是 $r_\theta(x, y) = \sigma(f_\theta(x, y))$，其中 $f_\theta$ 是一个学习到的标量函数，代表偏好分数（通常是与 $p_\theta$ 对应的logit），可能经过归一化 (normalization)或缩放。
- $\pi_{ref}(y|x)$ 是在参考策略下给定提示 $x$ 生成响应 $y$ 的概率。
- $\beta$ 是KL散度系数，控制偏离$\pi_{ref}$的惩罚力度。

PPO不直接优化此目标。而是使用基于优势估计 $A(x, y)$ 的截断代理目标函数来执行更新。优势通常衡量生成响应 $y$ 相对于提示 $x$ 的预期基准值好多少，该基准值由学习到的价值函数 $V_\psi(x)$ 估计。使用广义优势估计（GAE）是平衡这些估计中偏差和方差的常见做法。

### 为大型语言模型和AI反馈调整PPO组件

尽管PPO算法的核心结构不变，但其在RLAIF环境下应用于大型语言模型需要具体考量：

1. **策略和价值函数架构**：策略$\pi_\phi$和价值函数$V_\psi$通常都源自大型Transformer模型。它们通常共享大部分参数 (parameter)（核心Transformer主体），并使用单独的线性头部来生成下一个标记 (token)的概率（策略）和预测标量值（价值函数）。这种参数共享提升了计算和内存效率。使用$\pi_{ref}$的权重 (weight)初始化$\pi_\phi$是标准做法。价值函数$V_\psi$可以随机初始化或共享相同的初始权重。
2. **AI生成的奖励信号**：与人类反馈不同，AI生成的奖励$r_\theta(x, y)$可以为*任何*生成的响应$y$计算。这使得在RL优化过程中能够获得密集的反馈。然而，此信号会继承AI偏好模型$p_\theta$中存在的任何偏差、不一致或可利用的漏洞。$r_\theta$的尺度和分布也可能与人类获得的奖励显著不同，这通常需要归一化 (normalization)技术，如奖励白化（减去批次内奖励的均值并除以标准差），以稳定训练。
3. **KL散度实现**：计算KL项 $\log \pi_\phi(y|x) - \log \pi_{ref}(y|x)$ 需要计算在当前策略$\pi_\phi$和固定参考策略$\pi_{ref}$下生成序列$y$的对数概率。这增加了计算开销，因为它需要对批次中每个生成的序列进行一次通过$\pi_{ref}$的前向传播。一些实现按每个标记近似KL散度，而另一些则为整个序列计算。在批次内按每个提示应用KL惩罚有时能比在整个批次上平均提供更好的策略偏离控制。精心调整$\beta$系数很重要；如果太低，策略可能会“欺骗”奖励模型或遗忘通用能力；如果太高，学习会停滞。
4. **数据生成与流程**：RLAIF中典型的PPO循环包括：

   - 从数据集 $D$ 中采样一批提示 $x$。
   - 使用当前策略$\pi_\phi$为每个提示生成响应 $y$。
   - 使用AI偏好模型 $p_\theta$ 计算每个响应的奖励 $r_\theta(x, y)$。
   - 使用$\pi_\phi$和$\pi_{ref}$计算KL惩罚项。
   - 使用价值函数估计每个提示的价值 $V_\psi(x)$。
   - 计算优势（例如，使用GAE）。
   - 对收集到的经验批次（提示、响应、奖励、价值、对数概率）执行多次PPO优化周期，以更新$\pi_\phi$和$V_\psi$。

> RLAIF PPO典型步骤中的简化数据流。策略生成响应，这些响应由AI偏好模型评估以产生奖励。这些奖励，连同价值估计和KL惩罚，驱动PPO对策略和价值函数的更新。

### 高级考量与挑战

在RLAIF中应用PPO，在标准RL任务中带来独特的挑战：

- **优化稳定性**：大型语言模型对优化超参数 (parameter) (hyperparameter)很敏感。梯度裁剪、价值函数裁剪（PPO目标的一部分）和仔细的学习率调度等技术很重要。策略更新与价值函数学习之间的彼此作用需要管理，以防止发散。
- **奖励缩放与裁剪**：如前所述，AI奖励$r_\theta$可能具有任意尺度。奖励归一化 (normalization)（例如，批次内白化）是常见做法。一些实践者还会将奖励裁剪到特定范围（例如，[-10, 10]），以防止极端值破坏更新的稳定性，尽管这可能会丢弃有用的信号。
- **小批次大小与周期数**：PPO通常对收集到的经验批次执行多次优化周期。对于大型模型，批次大小常受限于GPU内存。较小的批次大小可能导致更新噪声较大。每个批次的PPO周期数涉及一个权衡：更多的周期能提高样本效率，但存在策略发散和违反PPO代理目标背后假设的风险。常见值范围为1到4个周期。
- **奖励模型的利用**：最重要的挑战是确保最大化AI奖励$r_\theta$真正对应于提升所需的对齐 (alignment)属性。策略可能会通过利用漏洞或偏差，变得擅长生成根据$p_\theta$得分很高的输出，这种现象被称为“奖励作弊”或“规范博弈”。这可能表现为重复输出、过于冗长或奉承的语言（对偏好模型隐含标准的谄媚），或者找到$p_\theta$不恰当地给出高分的边缘情况。缓解此问题需要精心设计偏好模型，可能在其训练期间引入正则化 (regularization)，或在RL阶段集成检查（例如，密切监控KL散度，使用多个奖励头部）。

在RLAIF中有效应用PPO，不仅需要对算法本身有扎实的认识，还需要对大型语言模型训练的细节以及针对AI生成目标优化可能带来的风险有深入的认识。精心的实现、超参数调整和监控对于实现稳定且有意义的对齐改进是必要的。稳定性和收敛性问题将在下一节中进一步讨论。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍PPO的奠基性论文，PPO是一种广泛使用且稳定的深度强化学习算法，对LLM对齐应用尤为重要。
- [Learning to Align Language Models from Human Feedback](https://arxiv.org/abs/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: arXiv preprint arXiv:2203.02155; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  阐述了使用PPO和RLHF微调大型语言模型以对齐人类偏好的开创性工作，为许多现代对齐技术提供了技术基础。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  介绍了“宪法式AI”，一种通过自我改进和AI反馈训练有益且无害AI助手的技术，与RLAIF和AI偏好模型概念直接相关。

---

[上一节](04-%E4%BB%8EAI%E5%81%8F%E5%A5%BD%E6%9E%84%E5%BB%BA%E5%A5%96%E5%8A%B1%E5%87%BD%E6%95%B0.md) · [下一节](06-%E5%BA%94%E5%AF%B9RLAIF%E4%B8%AD%E7%9A%84%E7%A8%B3%E5%AE%9A%E6%80%A7%E4%B8%8E%E6%94%B6%E6%95%9B%E9%97%AE%E9%A2%98.md)

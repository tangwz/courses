# RLAIF 中 PPO 循环的实施

来源：[原文](https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-5-advanced-rlaif-implementation/implementing-rlaif-ppo)

[返回章节目录](README.md) · [返回课程目录](../README.md)

AI 偏好模型（PM）提供了一种机制，可以根据学到的偏好对潜在的 LLM 响应进行评分，有效地近似 $P(y_1 \succ y_2 | x)$。这个 PM 用作强化学习 (reinforcement learning)框架中的奖励信号，对语言模型进行微调 (fine-tuning)。近端策略优化（PPO）是 RLAIF 中用于微调语言模型的标准算法，这与它在 RLHF 中的用途相似。为 RLAIF 定制的 PPO 训练循环的实施细节在此介绍。

目标是优化 LLM 策略，记作 $\pi_{\theta}(y|x)$，以为提示 $x$ 生成响应 $y$，使其能最大化从 AI 偏好模型获得的预期奖励，同时惩罚与初始参考策略 $\pi_{\text{ref}}(y|x)$ 的大幅偏离。这个参考策略通常是经过监督微调（SFT）或初始 CAI 阶段后获得的模型，确保模型不会灾难性地忘记其核心能力，也不会过度偏离到奖励作弊行为。

### RLAIF PPO 循环的核心组成部分

在 LLM 微调 (fine-tuning)中实施 PPO 需要协调多个模型组件和数据流：

1. **策略模型 ($\pi_{\theta}$):** 正在积极训练的 LLM。它生成响应，其参数 (parameter) $\theta$ 由 PPO 算法更新。
2. **参考模型 ($\pi_{\text{ref}}$):** 初始 LLM 的一个冻结副本（例如，SFT 后或 CAI-SL 后）。用于计算 KL 散度惩罚，防止策略模型偏离过远。
3. **奖励模型 ($r_{\phi}$):** 训练好的 AI 偏好模型。它接收提示 $x$ 和生成的响应 $y$，并输出一个标量奖励分数。在 PPO 阶段，此模型通常也处于冻结状态。
4. **值模型 ($V_{\psi}$):** 一个模型，用于估计从给定状态（提示 $x$）开始的预期未来奖励（回报）。通常，它被实现为 LLM 主干上的一个回归头，可能从参考模型或奖励模型权重 (weight)进行初始化。其参数 $\psi$ 与策略模型一同更新。

> RLAIF PPO 训练循环的流程图。提示被送入策略模型以生成响应。奖励使用奖励模型和 KL 惩罚计算。优势使用值模型估计。最后，策略模型和值模型分别使用 PPO 目标和值损失进行更新。

### PPO 训练循环的步骤解析

PPO 循环的一次迭代通常包括以下步骤：

1. **样本生成（Rollout）：**

   - 从数据集中采样一批提示 $x$（通常与 SFT 或 CAI 使用的数据集相同）。
   - 对于每个提示 $x$，当前策略模型 $\pi_{\theta}$ 生成一个响应 $y$。这涉及自回归 (autoregressive)解码。存储完整的序列 $y$ 和相应的动作概率（对数概率） $\log \pi_{\theta}(y|x)$。
   - 同时或随后，计算在冻结参考模型 $\pi_{\text{ref}}$ 下生成的序列 $y$ 的对数概率，得到 $\log \pi_{\text{ref}}(y|x)$。
   - 查询值模型 $V_{\psi}$ 以获取初始状态（提示 $x$）的值估计，记作 $V_{\psi}(x)$。对于序列模型，根据优势计算方法，可能需要每个 token 的值估计，尽管如果奖励是最终奖励，通常只使用初始值 $V_{\psi}(x)$。
2. **奖励计算：**

   - 对于每个生成的对 $(x, y)$，从冻结的 AI 偏好模型获取标量奖励分数：$r_{\text{PM}} = r_{\phi}(x, y)$。
   - 计算策略模型和参考模型在序列上的 KL 散度：$KL(x, y) = \log \pi_{\theta}(y|x) - \log \pi_{\text{ref}}(y|x)$。注意：这通常是按 token 计算，然后求和或平均，或者直接计算整个序列的。
   - 结合偏好奖励和 KL 惩罚。一种常见的公式是将 KL 惩罚应用于每个 token，并将 PM 奖励应用于序列末尾。用于 PPO 更新的最终奖励信号通常如下所示：
     
     $$
     R(x, y) = r_{\text{PM}}(x, y) - \beta \cdot KL(x, y)
     $$
     
     此处，$\beta$ 是一个超参数 (parameter) (hyperparameter)，控制 KL 惩罚的强度。
3. **优势与回报估计：**

   - 使用计算出的奖励 $R(x, y)$ 和值估计 $V_{\psi}(x)$ 来计算优势。广义优势估计（GAE）常用于更好的方差减少：
     
     $$
     \hat{A}_t = \sum_{k=t}^{T-1} (\gamma \lambda)^{k-t} \delta_k, \quad \text{其中} \quad \delta_k = r_k + \gamma V_{\psi}(s_{k+1}) - V_{\psi}(s_k)
     $$
     
     在具有最终奖励的 LLM 情境中，这会简化。如果奖励 $R(x, y)$ 仅应用于最终 token $T$，那么优势可能主要反映这个经过值估计和 KL 惩罚部分调整并折回的最终奖励。这里的 $r_k$ 将代表每个 token 的 KL 惩罚，而最终奖励 $r_{\text{PM}}$ 将被加到 $\delta_T$ 中。
   - 计算回报 $G_t = \hat{A}_t + V_{\psi}(s_t)$，它们作为训练值函数的目标。
4. **优化（策略和值更新）：**

   - 对收集到的经验批次（提示、响应、奖励、优势、对数概率）执行多次更新。
   - **策略更新：** 使用 PPO 裁剪代理目标更新策略模型 $\pi_{\theta}$：
     
     $$
     L^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min \left( \rho_t(\theta) \hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
     $$
     
     此处的 $\rho_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是当前策略与采样期间使用的策略（$\theta_{\text{old}}$）之间的概率比，$\hat{A}_t$ 是估计的优势，$\epsilon$ 是裁剪超参数（例如 0.2）。期望 $\mathbb{E}$ 取自样本批次和时间步（token）。
   - **值更新：** 通过最小化其预测 $V_{\psi}(s_t)$ 与计算出的回报 $G_t$ 之间的均方误差来更新值模型 $V_{\psi}$：
     
     $$
     L^{\text{VF}}(\psi) = \mathbb{E} \left[ (V_{\psi}(s_t) - G_t)^2 \right]
     $$
     
   - 这些更新通常使用随机梯度下降 (gradient descent)或 Adam 等变体进行。

### 实施考量

- **同步前向传播：** PPO 过程需要对批次中的每个样本进行策略、参考、奖励和值模型的前向传播。有效地管理潜在大型模型之间的计算和内存是重要的。DeepSpeed 或 Accelerate 等框架会有帮助。
- **批次构建：** 批次包含提示、生成的序列、来自策略模型和参考模型的对数概率、计算出的奖励和值估计。需要仔细的数据处理。
- **值函数输入：** 值函数 $V_\psi(s_t)$ 可以接受不同的输入。有时它接受提示嵌入 (embedding)，有时是 LLM 在特定 token $t$ 的隐藏状态。仅使用提示 $x$ 假定值主要依赖于输入，从而简化架构。
- **奖励归一化 (normalization)：** 归一化奖励信号（例如，使用运行平均值和标准差）对于稳定的 PPO 训练通常是不可或缺的。
- **KL 系数 ($\beta$)：** 调整 $\beta$ 很重要。过低时，策略可能偏离过远，潜在地导致生成质量下降或找到奖励漏洞。过高时，训练会停滞，因为策略受到过度约束。这通常需要反复试验。
- **梯度累积：** 为了在 GPU 内存有限的情况下处理大型有效批次大小，跨多个小批次的梯度累积是一种常用技术。
- **混合精度训练：** 使用 bfloat16 或 float16 等技术可以显著加速训练并减少内存占用，尽管需要监测数值稳定性。

实施 PPO 循环可以说是 RLAIF 流程中最复杂的部分，需要仔细整合多个模型并处理强化学习 (reinforcement learning)优化过程。成功取决于稳定的训练动态，这通过仔细调整超参数 (parameter) (hyperparameter)（学习率、$\beta$、$\epsilon$、$\gamma$、$\lambda$）和实施方法来达成。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; Publisher: arXiv; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍PPO算法的原始论文，强化学习中策略优化的核心方法。
- [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (2022)
  Journal: Advances in Neural Information Processing Systems 36 (NeurIPS); Volume: 36; Pages: 27735-27756; DOI: [10.48550/arXiv.2203.02155](https://doi.org/10.48550/arXiv.2203.02155)
  描述了使用人类反馈（RLHF）奖励模型通过PPO微调大型语言模型的应用，与RLAIF的PPO阶段直接相关。
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  介绍了广义优势估计（GAE），一种用于降低策略梯度方法方差的技术，广泛应用于PPO。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  一篇描述使用AI反馈（RLAIF）将语言模型与原则对齐的论文，包括奖励建模和基于PPO的微调方法。

---

[上一节](03-%E5%81%8F%E5%A5%BD%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83.md) · [下一节](05-RLAIF%20%E7%B3%BB%E7%BB%9F%E7%9A%84%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E6%95%B4.md)

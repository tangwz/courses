---
course: "llm-constitutional-ai-rlaif"
chapter: "rlaif-concepts"
lesson: "rlaif-reward-function-design"
sourceId: 4495
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-4-rlaif-concepts/rlaif-reward-function-design"
title: "从AI偏好构建奖励函数"
description: "将AI偏好得分转化为强化学习标量奖励信号的技术"
order: 4
plots: []
sourceHash: "97143b1e0046535ef3bbb741a58543b228d1402d96c91c0c11d7230ba300dceb"
sourceCorrections: []
---

设计奖励函数涉及将成对的AI偏好（通常由像 $p_\theta(y_w \succ y_l | x)$ 这样的模型表示，该模型能够根据AI生成的标签识别受偏好的回复）转化为可用的标量奖励信号 $r(x, y)$。这个奖励函数是后续强化学习 (reinforcement learning)阶段的根本组成部分，它指引LLM策略生成与所学偏好一致的输出。

### 从成对偏好到标量奖励

从RLHF继承的基本假设是，偏好模型 $p_\theta$ 估算了一个潜在的奖励函数 $r_\phi(x, y)$。一个普遍的建模方式是布拉德利-特里模型，它假定在给定提示 $x$ 的情况下，偏好回复 $y_w$ 优于 $y_l$ 的概率可以表示为：


$$
p(y_w \succ y_l | x) = \frac{\exp(r_\phi(x, y_w))}{\exp(r_\phi(x, y_w)) + \exp(r_\phi(x, y_l))} = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$


这里，$\sigma(\cdot)$ 是 Sigmoid 函数。在偏好模型训练期间，我们用AI生成的偏好数据 $(x, y_w, y_l)$ 来拟合 $p_\theta$，实际上学习到成对回复之间奖励差异的估算值。

我们现在的目的是，基于已训练的偏好模型 $p_\theta$，提取一个标量奖励函数 $r(x, y)$，以反映给定提示 $x$ 时 *单个* 回复 $y$ 的质量。由于 $p_\theta$ 对奖励的*差异*进行建模，所提取奖励函数 $r(x, y)$ 的绝对尺度有些随意，但其相对值应反映 $p_\theta$ 所学习到的偏好。

### 提取奖励信号

一种普遍定义奖励信号 $r(x, y)$ 的方法涉及利用已训练偏好模型 $p_\theta$ 的内部计算。许多偏好模型架构会为每个输入回复 $y$ 计算一个内部标量得分，我们称之为 $s_\theta(x, y)$。偏好概率随后根据胜出回复和落败回复之间分数的差异计算：


$$
p_\theta(y_w \succ y_l | x) = \sigma(s_\theta(x, y_w) - s_\theta(x, y_l))
$$


鉴于这种结构，奖励函数的一个自然选择是直接使用这个内部得分：


$$
r(x, y) = s_\theta(x, y)
$$


这个得分 $s_\theta(x, y)$ 表示根据AI偏好标注器学习到的回复 $y$ 的质量。它直接获取了偏好模型训练期间优化的信息。

**替代表述：**

在一些实现中，奖励函数可以使用分数经过 Sigmoid 函数后再取对数的方式来定义：


$$
r(x, y) = \beta \log(\sigma(s_\theta(x, y)))
$$


缩放因子 $\beta$ 常作为一个超参数 (parameter) (hyperparameter)引入，以控制奖励的量级，这会显著影响RL训练的动态。此表述强调与中性分数（通常为零）的偏离。

### 奖励函数设计的实际考量

仅仅提取分数往往不足以实现稳定有效的RL训练。会涉及一些实际要考虑的方面：

1. **奖励缩放与归一化 (normalization)：** 原始分数 $s_\theta(x, y)$ 可能具有任意尺度或偏移。大的奖励值可能导致过大的策略更新以及PPO等算法的不稳定。反之，极小的奖励可能导致学习缓慢。常用技术包括：

   - **白化处理：** 将每个批次的奖励归一化为零均值和单位方差。这有助于稳定PPO更新，通过确保奖励尺度在不同批次和提示之间保持一致。
   - **裁剪：** 将奖励值限制在预设范围，以防止极端更新。
   - **缩放因子 ($\beta$)：** 明确缩放奖励信号，如替代表述所示，可直接控制其相对于RL目标其他组成部分（例如KL散度惩罚）的作用。
2. **KL散度惩罚：** 为防止RL策略 $\pi_{RL}$ 过度偏离原始监督微调 (fine-tuning) (SFT) 策略 $\pi_{SFT}$（或CAI监督阶段后训练的策略），PPO通常会纳入KL散度惩罚。PPO更新中使用的最终奖励信号通常形式为：

   
   $$
   r_{final}(x, y) = r(x, y) - \beta_{KL} \text{KL}(\pi_{RL}(\cdot|x) || \pi_{SFT}(\cdot|x))
   $$
   

   这里，$r(x, y)$ 是源自AI偏好模型的奖励，$\beta_{KL}$ 控制偏离参考策略的惩罚强度。构建奖励函数 $r(x, y)$ 必须考虑其与此KL项的关系。 $r(x, y)$ 相对于KL惩罚的尺度是一个敏感的超参数 (parameter) (hyperparameter)。
3. **奖励模型漂移：** 如果AI偏好模型本身在RL训练过程中被更新或改变（较不常见，但在高级配置中可能发生），奖励函数会变得非平稳。这会使RL收敛变得复杂并需要谨慎处理。
4. **偏见放大：** AI偏好模型可能继承甚至放大用于生成标签的模型中存在的偏见（例如，基于宪法的评价器或其他LLM）。形成的奖励函数将包含这些偏见。需留意的是，奖励函数优化的是AI所*认为*好的东西，这可能与预期的宪法原则或更广泛的安全目标不完全一致。

### 将奖励整合到RL算法中

构建的奖励函数 $r(x,y)$ 作为主要的信号，指引LLM策略 $\pi_{RL}$ 使用PPO（或其他适合的RL算法）进行优化。在PPO展开阶段，对于从数据集中抽样的每个提示 $x$，当前策略 $\pi_{RL}$ 生成一个回复 $y$。这个回复 $y$ 随后输入到已冻结、已训练的AI偏好模型中，以计算奖励 $r(x,y)$。这个奖励，可能与KL惩罚结合，用于计算优势并更新策略参数 (parameter)。

> 流程图说明了RLAIF PPO循环中奖励信号的生成与使用。已冻结的AI偏好模型计算生成回复的分数，然后将其转化为用于策略更新的奖励信号。

从AI偏好构建有效的奖励函数是一个严谨的过程。它不仅需要知晓偏好概率与标量奖励之间的理论关系，还需要在缩放、归一化 (normalization)以及与所选RL算法目标函数的结合方面进行细致的实现，以确保稳定高效的策略优化。奖励函数的质量直接取决于上游AI偏好标注器的质量和一致性，以及基于其输出训练的偏好模型。

## 参考资料

- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei (2017)
  Journal: arXiv preprint arXiv:1706.03741; DOI: [10.48550/arXiv.1706.03741](https://doi.org/10.48550/arXiv.1706.03741)
  介绍人类偏好强化学习（RLHF）的开创性论文，详细说明了从成对人类偏好中学习奖励函数的过程，这与RLAIF的奖励函数设计直接类似。
- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267) — Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, Sushant Prakash (2024)
  Journal: Proceedings of the 41st International Conference on Machine Learning; Publisher: PMLR; Volume: 235; Pages: 26874-26901; DOI: [10.48550/arXiv.2309.00267](https://doi.org/10.48550/arXiv.2309.00267)
  专注AI反馈强化学习（RLAIF），将其作为一种扩展基于偏好学习的方法，详细说明了AI偏好的生成及其在训练奖励模型中用于改进大语言模型校准的应用。

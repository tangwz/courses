---
course: "rlhf-reinforcement-learning-human-feedback"
chapter: "advanced-rlhf-techniques"
lesson: "direct-preference-optimization-dpo"
sourceId: 5186
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-6-advanced-rlhf-techniques/direct-preference-optimization-dpo"
title: "直接偏好优化 (DPO)"
description: "了解DPO作为显式奖励模型构建的一种替代方法。"
order: 1
plots: []
sourceHash: "3e92a6a5438648c304130bb286cf47f8fd6cf7d3ab23bca427bd4c5e2283da04"
sourceCorrections: []
---

三阶段RLHF流程（SFT -> 奖励模型 -> PPO）是一种常见且有效的大型语言模型对齐 (alignment)方法。然而，此过程包含多个复杂组成部分。训练独立的奖励模型（RM）会带来校准和潜在不准确性方面的问题，而随后的PPO微调 (fine-tuning)需要仔细实施和超参数 (parameter) (hyperparameter)调整以保持稳定。直接偏好优化（DPO）作为一种有吸引力的替代方案出现，它显著简化了此过程。

DPO重新定义了对齐问题，绕过了显式奖励模型训练和PPO等在线强化学习 (reinforcement learning)优化的复杂性。相反，它通过人类偏好与策略更新之间的直接对应关系，直接在偏好数据上优化语言模型。

### 核心理念：从偏好到策略

回顾RLHF的目标是找到一个能生成人类偏好响应的策略 $\pi_\theta$，通常形式化为最大化从人类反馈中得出的预期奖励信号，同时保持与初始参考策略 $\pi_{ref}$（通常是SFT模型）接近，以维持能力并避免模式崩溃。这通常表示为：


$$
\max_{\pi_\theta} E_{x \sim D, y \sim \pi_\theta(\cdot|x)}[r^*(x, y)] - \beta D_{KL}(\pi_\theta || \pi_{ref})
$$


在此，$r^*(x, y)$ 表示反映人类偏好的（未知）真实奖励函数，$D$ 是提示的分布，$\beta$ 是一个控制KL散度惩罚的参数 (parameter)，$D_{KL}$ 衡量优化策略 $\pi_\theta$ 和参考策略 $\pi_{ref}$ 之间的差异。

标准方法通过在偏好对 $(x, y_w, y_l)$ 上训练奖励模型 $r_\phi(x, y)$ 来估计 $r^*$，对于提示 $x$， $y_w$ 优于 $y_l$。这通常使用基于Bradley-Terry模型的损失，假设偏好 $y_w$ 优于 $y_l$ 的概率与其奖励差成比例：


$$
P(y_w \succ y_l | x) = \sigma(r^*(x, y_w) - r^*(x, y_l))
$$


在此，$\sigma$ 是S型函数。训练 $r_\phi$ 后，PPO被用于以 $r_\phi$ 作为奖励信号来优化 $\pi_\theta$。

DPO巧妙地表明，RLHF目标的最佳解可以直接与偏好概率关联。它推导出一个损失函数 (loss function)，允许直接使用偏好数据 $(x, y_w, y_l)$ 优化 $\pi_\theta$，而无需拟合中间奖励模型 $r_\phi$。

### DPO损失函数 (loss function)

DPO的主要成果是其损失函数。通过将KL约束奖励最大化问题的最优解代入Bradley-Terry偏好模型，我们可以纯粹以策略 $\pi_\theta$、参考策略 $\pi_{ref}$ 和偏好数据 $D$ 的形式推导出损失：


$$
L_{DPO}(\pi_\theta; \pi_{ref}) = - E_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$


我们来拆解这个表达式：

- $(x, y_w, y_l) \sim D$: 我们从人类偏好数据集中采样一个提示 $x$、一个偏好的完成 $y_w$ 和一个不偏好的完成 $y_l$。
- $\pi_\theta(y|x)$: 在我们当前优化的策略下，给定提示 $x$ 生成完成 $y$ 的概率。
- $\pi_{ref}(y|x)$: 在固定参考策略（例如SFT模型）下，给定提示 $x$ 生成完成 $y$ 的概率。
- $\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$: 在优化策略和参考策略下，给定完成的对数概率比。此项隐含地表示奖励。
- $\beta$: 温度参数 (parameter)，类似于PPO中的KL系数。它控制优化策略 $\pi_\theta$ 允许偏离参考策略 $\pi_{ref}$ 的程度。较高的 $\beta$ 允许更大的偏离。
- $\log \sigma(...)$: 应用于偏好和不偏好响应的隐含奖励之间缩放差异的逻辑损失。

最小化 $L_{DPO}$ 能有效地促使策略 $\pi_\theta$ 为偏好完成 $y_w$ 分配更高的相对概率（与 $\pi_{ref}$ 相比），并为不偏好完成 $y_l$ 分配更低的相对概率。

### 隐含奖励建模

尽管DPO避免了训练*显式*奖励模型，但它仍然基于一个*隐含*模型运行。损失函数 (loss function)中的项可以被解释为与奖励相关：


$$
\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$


DPO损失旨在根据逻辑损失最大化差异 $\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$。本质上，DPO直接优化策略，使其对数概率比（按 $\beta$ 缩放）与观察到的人类偏好对齐 (alignment)，从而有效地同时学习策略和隐含奖励。

### DPO与PPO-RLHF工作流程比较

DPO的主要优点在于与传统的基于PPO的RLHF流程相比，其流程更为简化。

> 基于PPO的RLHF和DPO工作流程的比较。DPO将奖励建模和策略优化步骤整合为一个阶段。

如图所示：

1. **PPO-RLHF**：需要三个不同阶段：SFT以获得 $\pi_{ref}$，在偏好数据上训练奖励模型 $r_\phi$，最后，使用PPO以 $r_\phi$ 作为奖励信号并结合对 $\pi_{ref}$ 的KL惩罚来优化 $\pi_\theta$。这包括PPO循环中的在线采样、奖励计算、优势估计和策略更新。
2. **DPO**：仅需要两个阶段：SFT以获得 $\pi_{ref}$，然后使用DPO损失直接微调 (fine-tuning) $\pi_\theta$，其中将 $\pi_{ref}$ 和偏好数据作为输入。优化过程更接近于标准的监督学习 (supervised learning)设置（尽管使用了自定义损失函数 (loss function)），并且在训练期间不需要在线采样或优势估计。

### 实现与考量

实现DPO包括以下步骤：

1. 从参考模型 $\pi_{ref}$（通常是SFT模型）开始。
2. 加载偏好数据集 $D = \{(x, y_w, y_l)\}$。
3. 设置待优化的模型 $\pi_\theta$（从 $\pi_{ref}$ 初始化）。
4. 在训练循环中，对于每批偏好三元组：
   - 计算对数概率 $\log \pi_\theta(y_w|x)$ 和 $\log \pi_\theta(y_l|x)$。
   - 计算对数概率 $\log \pi_{ref}(y_w|x)$ 和 $\log \pi_{ref}(y_l|x)$（需要通过冻结的参考模型进行前向传播）。
   - 使用这些对数概率和温度 $\beta$ 计算DPO损失。
   - 执行反向传播 (backpropagation)并更新 $\pi_\theta$ 的参数 (parameter)。

像Hugging Face的TRL (`DPOTrainer`) 这样的库提供了便捷的实现，抽象掉了大量样板代码。

**潜在优势：**

- **简洁性：** 消除了训练、存储和加载独立奖励模型的必要。
- **稳定性：** 避免了显式奖励模型与强化学习 (reinforcement learning)算法（PPO）之间相互作用可能产生的不稳定。优化过程通常更稳定，也更易于调整。
- **效率：** 与PPO相比，计算量可能更小，因为它避免了PPO循环中的采样和奖励计算步骤。

**潜在缺点：**

- **隐含奖励：** 由于没有显式奖励模型可供检查，调试有时会更困难。性能直接取决于偏好数据集的质量和构成。
- **优化动态：** 优化行为与PPO不同。超参数 (hyperparameter)调整，特别是对于 $\beta$，仍然很重要。
- **数据要求：** 像所有基于偏好的方法一样，DPO非常依赖高质量和足够大的偏好数据集。

DPO代表了LLM对齐 (alignment)技术方面的重要进展。通过基于偏好数据直接优化策略，它提供了一种更简单、通常更稳定且可能更高效的替代方案，取代了涉及显式奖励建模和PPO等复杂强化学习算法的传统多阶段RLHF流程。它正迅速成为LLM对齐工具包中的一个标准工具。

## 参考资料

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (2023)
  Journal: arXiv preprint arXiv:2305.18290; DOI: [10.48550/arXiv.2305.18290](https://doi.org/10.48550/arXiv.2305.18290)
  这篇基础论文介绍了DPO，详细阐述了其理论推导和实验结果，并展示了它作为传统RLHF的更简单替代方案的有效性。
- [DPOTrainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face TRL库的官方文档，提供了DPO的实际实现细节和使用示例。
- [A Survey on Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2307.09051) — Xiufeng Huang, Sheng Zhou (2023)
  Journal: arXiv preprint arXiv:2307.09051; DOI: [10.48550/arXiv.2307.09051](https://doi.org/10.48550/arXiv.2307.09051)
  一项全面的调查，概述了包括DPO在内的RLHF方法，并讨论了它们的理论基础、应用和挑战。

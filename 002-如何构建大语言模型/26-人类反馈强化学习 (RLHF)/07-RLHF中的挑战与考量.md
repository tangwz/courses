# RLHF中的挑战与考量

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-26-reinforcement-learning-human-feedback-rlhf/rlhf-challenges-considerations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

强化学习 (reinforcement learning)从人类反馈 (RLHF) 流程为使大型语言模型与人类偏好保持一致提供了一个强大的框架。然而，其实现通常很复杂，并带来一些实际困难。要成功应对这些，需要仔细的设计选择、工程实践和持续监督。

### 数据质量与可扩展性

RLHF的根本在于高质量的人类偏好数据。收集这些数据需要大量资源，并带来一些障碍：

1. **主观性与分歧：** 人类偏好本身就具有主观性。不同的标注者可能在哪个响应更好上存在分歧，特别是对于复杂的提示。制定清晰的标注指南并进行标注者校准是重要步骤，但一定程度的噪声和分歧是无法避免的。这种噪声会影响学到的奖励模型的质量和一致性。
2. **标注者偏见：** 标注者带有自己的偏见，可能影响数据集中编码的偏好。这些偏见可能与人口统计学、文化背景，甚至给出的特定指令有关。确保标注人员的多样性并设计任务以减少偏见，是重要但有难度的操作考量。
3. **成本与规模：** 生成成对比较需要人工标注者阅读并评估大量提示的多个模型输出。将此过程扩展到覆盖大型语言模型广泛的潜在输入和输出范围，既昂贵又耗时。训练一个能够良好泛化的可靠奖励模型所需的数据量常常是瓶颈。
4. **数据多样性：** 偏好数据集必须足够多样，以涵盖各种情景、主题和LLM潜在的失败模式。如果数据只涵盖狭窄的交互范围，那么由此产生的奖励模型和对齐 (alignment)策略可能无法很好地泛化到未见过的情况。

### 奖励模型局限

奖励模型（RM）在强化学习 (reinforcement learning)（RL）阶段充当人类偏好的代表。然而，它是一个不完美的代表，可能导致一些问题：

1. **规格博弈（奖励作弊）：** 强化学习策略专门针对奖励模型的分数进行优化。如果奖励模型存在缺陷或可利用的模式，策略可能学会以不对应于在有用性、诚实性或无害性方面真实改进的方式来最大化分数。例如，策略可能学到奖励模型偏爱更长、更冗余的回答，导致不必要的冗长输出，或者它可能发现特定的短语，欺骗奖励模型给出高分。
2. **分布漂移：** 奖励模型是在一个静态的偏好数据集上训练的。在强化学习训练期间，策略模型（$\pi_{\text{RL}}$）会演变，生成可能与奖励模型训练期间所见输出明显不同的输出。当评估这些分布外输出时，奖励模型的准确性会下降，导致不可靠的奖励信号。
3. **难以捕捉细节：** 对齐 (alignment)的复杂方面，如不明显的偏见、响应的长期后果或严格的事实正确性，可能很难被一个简单的基于偏好的奖励模型准确捕捉。成对比较格式简化了判断任务，但也可能过度简化了期望的行为。
4. **校准：** 确保奖励模型分数良好校准（即分数差异准确反映偏好强度）是困难的。差的校准会影响强化学习优化过程的稳定性和有效性。

### 训练不稳定性与超参数 (parameter) (hyperparameter)敏感性

强化学习 (reinforcement learning)，特别是应用于大型Transformer模型的PPO，以其对超参数的敏感性以及潜在的不稳定性而闻名：

1. **PPO的复杂性：** PPO涉及多个组成部分：值函数估计、优势计算（通常使用广义优势估计GAE）、带有裁剪的策略更新以及KL散度惩罚。每个组成部分都引入了需要仔细调整的超参数（GAE的$\lambda$、裁剪的$\epsilon$、KL系数$\beta$）。
2. **奖励与值估计的方差：** 奖励模型的回报可能存在噪声，并且为高维状态空间（由模型激活或输入/输出表示）准确估计值函数具有挑战性。高方差会减缓或破坏学习的稳定性。
3. **KL散度约束：** KL惩罚（$\beta \cdot D_{\text{KL}}(\pi_{\text{RL}} || \pi_{\text{SFT}})$）对于防止强化学习策略偏离初始SFT模型太远非常重要，从而减轻奖励作弊并保留通用语言能力。然而，选择正确的系数$\beta$非常重要。太小，策略可能对奖励模型过度优化；太大，学习就会受阻。这通常需要在训练期间进行动态调整。

考虑一个包含KL惩罚的简化PPO目标函数：


$$
L_{\text{PPO+KL}}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right] - \beta \cdot \mathbb{E}_t \left[ D_{\text{KL}}(\pi_{\theta}(\cdot|s_t) || \pi_{\text{SFT}}(\cdot|s_t)) \right]
$$


这里，$r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$是概率比，$A_t$是优势估计，$\epsilon$是裁剪参数，$\beta$控制对原始SFT策略$\pi_{\text{SFT}}$的KL惩罚。调整$\epsilon$和$\beta$会明显影响稳定性和性能。

```python
# PyTorch中包含KL惩罚的简化PPO更新
import torch
import torch.nn.functional as F

def compute_ppo_loss(
    policy_log_probs,
    old_policy_log_probs,
    advantages,
    rewards_from_rm,
    sft_policy_log_probs,
    clip_param,
    kl_beta
):
    """
计算带KL惩罚的PPO损失。
    假定输入是适当形状的张量
    （例如，[batch_size]）。
    """
    # 策略比率
    ratio = torch.exp(policy_log_probs - old_policy_log_probs)

    # 裁剪的替代目标
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    # 负号，因为我们是最小化
    policy_loss = -torch.min(surr1, surr2).mean()

    # 值损失（通常是预测值与实际回报之间的MSE）
    # value_loss = F.mse_loss(predicted_values, returns) # 占位符

    # KL散度惩罚项
    # 注意：KL散度需要概率分布，而不仅仅是对
    # 所采取行动的对数概率
    # 一个正确的实现将计算完整分布之间的KL
    # pi_RL 和 pi_SFT
    # 这里，为简化起见，我们使用对数概率作为代理，但这并非严格意义上的
    # 正确KL散度
    # 简化代理
    kl_div = (policy_log_probs - sft_policy_log_probs).mean()

    # 总损失
    # total_loss = policy_loss + value_loss_coeff * value_loss \
    #              - kl_beta * kl_div # 值损失的占位符
    # 简化版，不包含值损失
    total_loss = policy_loss - kl_beta * kl_div

    return total_loss

# --- 用于说明的模拟输入 ---
# 当前策略、旧策略和SFT策略下采取行动的对数概率
policy_log_probs = torch.randn(4, requires_grad=True)
old_policy_log_probs = torch.randn(4)
sft_policy_log_probs = torch.randn(4)

# 优势估计和奖励
advantages = torch.randn(4)
# 在此简化策略损失部分中未直接使用
rewards_from_rm = torch.randn(4)

# 超参数
clip_param = 0.2
kl_beta = 0.1

loss = compute_ppo_loss(
    policy_log_probs,
    old_policy_log_probs,
    advantages,
    rewards_from_rm,
    sft_policy_log_probs,
    clip_param,
    kl_beta
)

print(f"计算出的PPO+KL损失: {loss.item()}")
# 示例输出：计算出的PPO+KL损失：0.1234...
# (值取决于随机输入)
```

> 一个PyTorch代码片段，说明了PPO损失计算的组成部分，特别突出显示了替代目标和简化的KL惩罚项。请注意，一个完整的实现需要仔细处理用于KL计算和值函数训练的分布。

4. **梯度方差与优化：** 使用强化学习优化大型语言模型通常涉及大批量和分布式训练设置。管理梯度同步、通信开销以及确保跨多个工作节点的数值稳定性增加了复杂性。梯度累积和仔细选择优化器（例如AdamW）等技术是标准做法，但强化学习增加了另一层优化挑战。

### 计算成本

RLHF在计算上要求很高：

1. **多模型训练：** 这需要训练至少三个大型模型：初始SFT模型、奖励模型和最终的强化学习 (reinforcement learning)策略。每个阶段都需要大量GPU资源。
2. **推理 (inference)开销：** 在强化学习期间，需要频繁进行推理：强化学习策略生成响应，SFT策略通常用于KL惩罚计算，并且奖励模型评估生成的响应。这个推理循环在许多优化步骤中重复进行，构成了主要的计算负担。
3. **内存需求：** 存储多个模型状态（策略、值函数、奖励模型、可能的SFT参考策略）及其梯度、激活和优化器状态需要大量的GPU内存，通常需要模型并行和ZeRO等内存优化技术。

### 评估困难

衡量RLHF对齐 (alignment)的成功具有挑战性：

1. **标准度量：** 标准的自然语言处理度量（如BLEU或ROUGE），甚至困惑度，都不能充分捕捉有用性或无害性等对齐目标。
2. **人工评估：** 可靠的评估通常需要对最终模型输出进行进一步的人工评估，这既昂贵又耗时。设计评估协议具有挑战性。
3. **对齐税：** 通过RLHF在对齐指标上有所改进，有时可能会以牺牲某些能力或基准测试上的性能下降为代价（即“对齐税”）。量化 (quantization)和平衡这些权衡非常重要。
4. **基准局限：** 尽管存在针对安全性或真实性的特定基准（例如TruthfulQA、ToxiGen），但它们可能无法涵盖期望行为的所有方面，而且模型有时可能会对这些基准过拟合 (overfitting)。

### 伦理考量

定义“人类偏好”的过程引发了伦理问题：

1. **谁的偏好？** 标注者的选择和偏好提示的设计隐含地编码了特定的价值观。确保公平性、代表性，并避免放大数据中或标注者之间存在的社会偏见，是一项重要的伦理责任。
2. **透明度：** RLHF过程的复杂性可能使其不透明。理解模型在RLHF后为何以某种方式行动可能很困难，这使得确保问责制和可信度的努力变得复杂。
3. **潜在滥用：** 像任何强大的技术一样，对齐 (alignment)后的模型也可能被滥用。持续考量安全措施和负责任的部署实践是必要的。

成功实施RLHF需要认识到这些挑战，并在仔细的数据整理、工程实践、全面评估和持续的伦理反思方面投入。像直接偏好优化（DPO）这样的技术，旨在通过绕过显式奖励建模来简化过程，也作为标准RLHF流程的潜在替代方案或补充方案而受到关注。

## 参考资料

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  本文介绍了使用RLHF训练有益且无害AI助手的概念，讨论了与人类数据收集、奖励模型设计和对齐相关的挑战。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了PPO算法，这是一种广泛使用且稳定的强化学习算法，构成了RLHF中强化学习阶段的基础。

---

[上一节](06-KL%E6%95%A3%E5%BA%A6%E6%83%A9%E7%BD%9A%E7%9A%84%E4%BD%9C%E7%94%A8.md) · [下一节](08-%E6%9B%BF%E4%BB%A3%E6%96%B9%E6%B3%95%EF%BC%9A%E7%9B%B4%E6%8E%A5%E5%81%8F%E5%A5%BD%E4%BC%98%E5%8C%96%20%28DPO%29.md)

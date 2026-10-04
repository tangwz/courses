# 替代方法：直接偏好优化 (DPO)

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-26-reinforcement-learning-human-feedback-rlhf/direct-preference-optimization-dpo)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管使用近端策略优化 (PPO) 的人类反馈强化学习 (reinforcement learning) (RLHF) 在使语言模型与人类偏好保持一致方面已被证明有效，但此过程可能复杂且有时不稳定。它通常包含多个阶段：监督微调 (fine-tuning) (SFT)、基于人类偏好数据训练一个独立的奖励模型 (RM)，然后使用由 RM 指导的强化学习对 SFT 模型进行微调。这种多阶段流程引入了多个超参数 (parameter) (hyperparameter)和潜在的失败点，特别是在强化学习阶段，该阶段对实现细节敏感，且容易出现奖励作弊等问题。

直接偏好优化 (DPO) 提供了一种更简化的偏好对齐 (alignment)方法，绕过了显式奖励模型训练的需求，并彻底避免了强化学习的复杂性。DPO 使用一个简单的分类目标，直接基于偏好数据优化语言模型。

DPO 的核心思路源于 RLHF 寻求的最优策略与潜在（隐式）奖励函数之间的数学关系。回想一下，标准奖励建模通常使用 Bradley-Terry 模型，将提示 $x$ 的成对偏好 $(y_w, y_l)$（其中 $y_w$ 优于 $y_l$）与潜在奖励函数 $r$ 关联起来：

$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$

此处，$\sigma$ 是 logistic 函数。RLHF 的目标是找到一个策略 $\pi_{RL}$，在使预期奖励 $E[r(x, y)]$ 最大化的同时，通过 KL 散度惩罚，保持与参考策略 $\pi_{ref}$（通常是 SFT 模型）的接近度：

$\max_{\pi_{RL}} E_{(x,y) \sim \pi_{RL}}[r(x, y)] - \beta D_{KL}(\pi_{RL}(y|x) || \pi_{ref}(y|x))$

DPO 使用此约束优化问题的解析解。可以证明，最优策略 $\pi_{RL}$ 具有以下形式：

$\pi_{RL}(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$

其中 $Z(x)$ 是一个配分函数，确保概率和为一。此方程连接了最优策略、参考策略和奖励函数。通过将此关系代回 Bradley-Terry 偏好模型，可以消除奖励函数 $r(x, y)$，直接用策略表示偏好概率：

$P(y_w \succ y_l | x) = \sigma\left( \beta \log \frac{\pi_{RL}(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_{RL}(y_l|x)}{\pi_{ref}(y_l|x)} \right)$

DPO 直接训练语言模型 $\pi_\theta$ 来满足此偏好模型，其中 $\pi_\theta$ 代替了未知的最优策略 $\pi_{RL}$。其目标是最大化在此模型下观测到的人类偏好的对数似然。这从而得到 DPO 损失函数 (loss function)：

$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = - E_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$

此处，$\mathcal{D}$ 是偏好三元组 $(x, y_w, y_l)$ 的数据集，$\pi_\theta$ 是正在优化的语言模型，$\pi_{ref}$ 是冻结的参考 SFT 模型，$\beta$ 是一个超参数，它隐式控制优化策略 $\pi_\theta$ 与参考策略 $\pi_{ref}$ 的偏离程度。更高的 $\beta$ 会赋予偏好数据更大的权重 (weight)，可能导致更大的偏离。

### 实现和使用

实现 DPO 比 RLHF 显著更简单。它需要：

1. 一个偏好数据集 $\mathcal{D} = \{(x_i, y_{w,i}, y_{l,i})\}$。
2. 一个基础 SFT 模型作为参考策略 $\pi_{ref}$。此模型的权重 (weight)在 DPO 训练期间保持冻结。
3. SFT 模型的一个副本 $\pi_\theta$，其权重将被更新。

训练循环涉及标准监督学习 (supervised learning)：

- 从 $\mathcal{D}$ 中采样一批偏好三元组 $(x, y_w, y_l)$。
- 计算当前模型 $\pi_\theta$ 和参考模型 $\pi_{ref}$ 下，选中 ($y_w$) 和拒绝 ($y_l$) 响应的对数概率。
- 使用这些对数概率计算 DPO 损失 $\mathcal{L}_{DPO}$。
- 执行反向传播 (backpropagation)并使用 AdamW 等优化器更新 $\pi_\theta$ 的权重。

下面是一个 PyTorch 代码片段，用于在训练步骤中计算 DPO 损失的核心部分，假设您已获得对数概率：

```python
import torch
import torch.nn.functional as F

# 假设 log_probs_policy 和 log_probs_ref 包含对数概率
# 形状: (batch_size,)，适用于选中 (w) 和拒绝 (l) 响应
# log_probs_policy_w, log_probs_policy_l: 来自正在
#                                         训练的模型 (pi_theta) 的对数概率
# log_probs_ref_w, log_probs_ref_l: 来自冻结的参考
#                                   模型 (pi_ref) 的对数概率
# beta: 超参数（例如 0.1）

def dpo_loss(log_probs_policy_w, log_probs_policy_l,
             log_probs_ref_w, log_probs_ref_l, beta):
    """计算一批偏好的 DPO 损失。"""

    # 计算对数比率
    log_ratio_w = log_probs_policy_w - log_probs_ref_w
    log_ratio_l = log_probs_policy_l - log_probs_ref_l

    # 对数比率差值乘以 beta
    diff_scaled = beta * (log_ratio_w - log_ratio_l)

    # 使用 logistic sigmoid 计算损失
    loss = -F.logsigmoid(diff_scaled)

    # 对批次内的损失取平均
    return loss.mean()

# 在训练循环中的示例用法
# batch = get_preference_batch()
# # (提示, 选中响应, 拒绝响应)
# outputs_policy = policy_model(prompts,
#                               chosen_responses,
#                               rejected_responses)
# with torch.no_grad():
#     outputs_ref = ref_model(prompts,
#                             chosen_responses,
#                             rejected_responses)
#
# # 提取对数概率（细节取决于模型实现）
# log_probs_policy_w = get_log_probs(outputs_policy, chosen_responses)
# log_probs_policy_l = get_log_probs(outputs_policy, rejected_responses)
# log_probs_ref_w = get_log_probs(outputs_ref, chosen_responses)
# log_probs_ref_l = get_log_probs(outputs_ref, rejected_responses)
#
# loss = dpo_loss(log_probs_policy_w, log_probs_policy_l,
#                 log_probs_ref_w, log_probs_ref_l, beta=0.1)
# loss.backward()
# optimizer.step()
```

### 优点和缺点

**优点：**

- **简洁性：** DPO 在 SFT 之后只需要一个训练阶段，无需进行奖励模型拟合和复杂的强化学习 (reinforcement learning)算法。
- **稳定性：** 它避免了 RLHF 中 PPO 经常遇到的优化不稳定问题，例如对 KL 散度系数和奖励缩放的敏感性。
- **效率：** 它通常比基于 PPO 的 RLHF 计算成本更低，因为它在训练期间不需要从策略中采样生成内容，也不需要训练独立的奖励模型。

**缺点：**

- **性能：** 尽管 DPO 通常能达到与 RLHF 可比的结果，但与精心调优的 PPO 流程相比，有时可能表现不佳，特别是在潜在的奖励信号复杂或有噪声的情况下。
- **超参数 (parameter) (hyperparameter)调优：** 超参数 $\beta$ 仍然需要仔细调优，类似于 PPO 中的 KL 系数。
- **数据要求：** 类似于 RLHF，DPO 依赖于高质量的偏好数据集。

总之，DPO 为标准 RLHF 流程提供了一种更简单的替代方案。其稳定性及易于实现使其成为一个有吸引力的选择，尤其是在计算资源或强化学习专业知识有限的情况下，它能有效地使语言模型与人类偏好保持一致。这代表了使大型语言模型更具帮助性、诚实性和无害性的过程中的显著简化。

## 参考资料

- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS 2017); DOI: [10.48550/arXiv.1706.03741](https://doi.org/10.48550/arXiv.1706.03741)
  提出从人类偏好中学习奖励函数以训练强化学习智能体的基础论文。
- [DPO (Direct Preference Optimization)](https://huggingface.co/docs/trl/main/en/dpo_trainer) — Hugging Face (2024)
  Publisher: Hugging Face
  提供使用Hugging Face TRL库实现DPO的实用信息和代码示例。

---

[上一节](07-RLHF%E4%B8%AD%E7%9A%84%E6%8C%91%E6%88%98%E4%B8%8E%E8%80%83%E9%87%8F.md) · [下一节](../27-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E6%8A%80%E6%9C%AF/01-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E7%9A%84%E5%8A%A8%E5%9B%A0.md)

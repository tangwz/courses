---
course: "how-to-build-a-large-language-model"
chapter: "reinforcement-learning-human-feedback-rlhf"
lesson: "rl-fine-tuning-ppo"
sourceId: 6087
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-26-reinforcement-learning-human-feedback-rlhf/rl-fine-tuning-ppo"
title: "RL PPO 微调"
description: "使用 PPO 微调 SFT 模型，以奖励模型为指导。"
order: 5
plots: []
sourceHash: "f3a3f44c6dd45674238d82e262c102b020d11e3c0ff9c6093afae7b3ed17bfe3"
sourceCorrections: []
---

强化学习 (reinforcement learning) (RL) 结合了监督微调 (fine-tuning) (SFT) 模型和奖励模型 (RM)。SFT 模型为遵循指令提供了坚实基础，RM 经过训练可以根据人类偏好对响应进行评分。这种结合旨在改进 SFT 模型，优化其生成能最大化 RM 评分的输出。目标是有效地引导模型产生人类偏好的行为。近端策略优化 (PPO) 是大多数强化学习人类反馈 (RLHF) 流程中此过程的主要算法，因为它相较于其他 RL 算法具有相对稳定性和样本效率。

### 大语言模型 (LLM)微调 (fine-tuning)中的强化学习 (reinforcement learning)框架

让我们将标准强化学习术语映射到大语言模型语境：

1. **智能体/策略 ($\pi_{\theta}$):** 这是我们正在积极微调的语言模型。它最初是 SFT 模型的副本，其参数 (parameter) $\theta$ 在 PPO 过程中更新。策略 $\pi_{\theta}(a|s)$ 定义了在给定当前词元 (token)序列 $s$ 的情况下生成下一个词元 $a$ 的概率。
2. **动作 ($a$):** 动作对应于从模型的词表中选择下一个词元并将其附加到序列中。
3. **状态 ($s$):** 状态表示到目前为止已生成的词元序列，起始于初始提示。
4. **环境:** 环境是隐式定义的。它接收模型响应提示（初始状态）生成的序列（动作序列），并返回奖励。核心组成部分是提示分布和奖励函数（我们的 RM + KL 惩罚）。
5. **奖励函数 ($R(s, a)$ 或 $R(sequence)$):** 这是引导优化的重要信号。在 RLHF 中，一个完整生成序列（提示 + 响应）的奖励主要由训练好的奖励模型 ($RM$) 的评分决定。为了保持语言连贯性并防止策略与可靠的 SFT 模型偏离过远，RM 评分与基于当前策略 $\pi_{\theta}$ 和初始 SFT 策略 $\pi_{ref}$ 之间 Kullback-Leibler (KL) 散度的惩罚项相结合。

### PPO 迭代周期

PPO 微调 (fine-tuning)过程是迭代进行的，通常在每次迭代中包含以下步骤：

1. **数据生成 (Rollout):** 采样一批提示（例如，来自 SFT 训练分布或单独的提示数据集）。对于每个提示，使用*当前*策略模型 $\pi_{\theta}$ 生成响应。这涉及自回归 (autoregressive)地采样词元 (token)，直到生成序列结束词元或达到最大长度。在此生成过程中，我们需要为每个词元步长 $t$ 存储几项信息：

   - 状态 $s_t$（提示 + 截至步长 $t$ 生成的词元）。
   - 动作 $a_t$（在步长 $t$ 生成的词元）。
   - 在当前策略下生成该词元的对数概率：$\log \pi_{\theta}(a_t|s_t)$。
   - 在*参考* SFT 策略下生成该词元的对数概率：$\log \pi_{ref}(a_t|s_t)$。参考策略 $\pi_{ref}$ 在整个 RL 阶段保持固定，通常是初始 SFT 模型状态。
2. **奖励计算:** 一旦生成完整序列（提示 + 响应），计算奖励。这通常包含：

   - 使用训练好的奖励模型 ($RM$) 对最终序列评分。我们称之为 $R_{RM}$。此评分反映了基于人类偏好的对齐 (alignment)质量。
   - 为每个步长 $t$ 计算每词元 KL 散度惩罚：$R_{KL_t} = -\beta (\log \pi_{\theta}(a_t|s_t) - \log \pi_{ref}(a_t|s_t))$。超参数 (parameter) (hyperparameter) $\beta$ 控制此惩罚的强度。较高的 $\beta$ 值会阻止模型偏离 SFT 模型。
   - 将这些组合成最终的奖励信号。一种常见做法是仅在最后一个词元步长分配 $R_{RM}$ 评分，并在每个步长添加每词元 $R_{KL}$。因此，一个序列的总奖励可能看起来像一个每词元 KL 惩罚序列，其中 RM 评分被添加到最后一个词元的奖励中。

   让我们在 PyTorch 中演示对数概率和 KL 惩罚部分的计算：

   ```python
   import torch
   import torch.nn.functional as F

   # 假设 'policy_model' 是当前正在训练的模型 (pi_theta)
   # 假设 'ref_model' 是冻结的 SFT 模型 (pi_ref)
   # 假设 'prompt_tokens' 是提示的输入张量
   # 假设 'generated_tokens' 是 policy_model 生成的词元张量

   # 连接提示和生成的词元作为模型输入
   input_ids = torch.cat([prompt_tokens, generated_tokens], dim=-1)
   attention_mask = (input_ids != pad_token_id).long() # 假设 pad_token_id

   # 从两个模型获取 logits
   with torch.no_grad(): # ref_model 或计算 KL 惩罚输入不需要梯度
       policy_outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
       ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)

   policy_logits = policy_outputs.logits
   ref_logits = ref_outputs.logits

   # 移动 logits 和标签以符合下一个词元预测格式
   # 我们只关注分配给*生成的*词元的概率
   gen_len = generated_tokens.size(1)
   policy_log_probs = F.log_softmax(policy_logits[:, -gen_len-1:-1, :], dim=-1)
   ref_log_probs = F.log_softmax(ref_logits[:, -gen_len-1:-1, :], dim=-1)

   # 收集实际生成词元的对数概率
   # generated_tokens 需要正确调整形状以进行 gather 操作
   # 形状: (批量大小, 生成长度, 1)
   gathered_policy_log_probs = policy_log_probs.gather(dim=-1, index=generated_tokens.unsqueeze(-1)).squeeze(-1)
   gathered_ref_log_probs = ref_log_probs.gather(dim=-1, index=generated_tokens.unsqueeze(-1)).squeeze(-1)

   # 计算每词元 KL 惩罚部分（对数比率）
   # 形状: (批量大小, 生成长度)
   log_ratio = gathered_policy_log_probs - gathered_ref_log_probs
   kl_penalty_per_token = -beta * log_ratio # beta 是 KL 系数

   # 现在，获取完整序列的 RM 评分（实现取决于 RM 架构）
   # full_sequences = torch.cat([prompt_tokens, generated_tokens], dim=-1)
   # rm_scores = reward_model(full_sequences).score # 示例 RM 输出

   # 组合奖励（简化示例：RM 评分在末尾添加）
   rewards = kl_penalty_per_token.clone()
   rewards[:, -1] += rm_scores
   ```

   > PyTorch 代码片段，演示从策略模型和参考模型计算对数概率以及由此产生的 KL 惩罚项。`beta` 是一个重要的超参数。
3. **优化 (PPO 更新):** 使用收集到的轨迹（状态、动作、对数概率、奖励）来更新策略模型 $\pi_{\theta}$。PPO 通过优化一个替代目标函数来实现这一点。核心思路是最大化预期优势 $A_t = Q(s_t, a_t) - V(s_t)$，它表示动作 $a_t$ 相较于状态 $s_t$ 的平均动作有多大改进。PPO 使用带截断目标的的重要性采样来确保更新的稳定性：

   - **优势估计:** 计算每个步长的优势估计值 $\hat{A}_t$。虽然朴素策略梯度使用回报 $G_t$（未来奖励的总和），但 PPO 常使用广义优势估计 (GAE) 来降低方差。在许多 RLHF 实现中，RM 评分本身（可能经过归一化 (normalization)或与学习到的价值函数 $V(s_t)$ 结合）用作 $Q(s_t, a_t)$ 或优势的代理。一个单独的价值模型，通常从 RM 或 SFT 模型初始化并训练以预测预期累积奖励（包括 KL 惩罚），常用于计算 $V(s_t)$ 和改进优势估计。
   - **策略更新:** 计算概率比率 $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} = \exp(\log \pi_{\theta}(a_t|s_t) - \log \pi_{\theta_{old}}(a_t|s_t))$，其中 $\pi_{\theta_{old}}$ 是更新前的策略（来自数据生成阶段）。截断替代目标函数为：
     $L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{截断}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$
     $\text{截断}$ 函数将比率 $r_t(\theta)$ 限制在 $[1-\epsilon, 1+\epsilon]$ 区间内，防止过大的更新导致训练不稳定。$\epsilon$ 是一个小的超参数（例如 0.2）。
   - 使用收集到的批次数据对该目标进行多轮随机梯度上升。更新策略模型的参数 $\theta$。如果使用单独的价值模型，它通常通过最小化其预测与观察到的回报之间的均方误差进行同步更新。

这种数据生成、奖励计算和 PPO 优化的循环重复进行，逐步调整策略 $\pi_{\theta}$ 以生成从 RM 获得更高评分的响应，同时 KL 惩罚使其保持在 SFT 阶段学到的流畅性和能力上。

### 实际考量

- **价值函数:** 如前所述，在 PPO 中，与策略同时训练一个单独的价值函数是标准做法，以降低优势估计的方差。这个价值函数预测给定状态（词元 (token)序列）的预期折扣未来奖励。它通常通过最小化其预测与数据生成中观察到的回报之间的平方误差进行训练。
- **超参数 (parameter) (hyperparameter)调整:** RLHF，特别是 PPO，包含几个敏感的超参数：KL 系数 $\beta$、PPO 截断参数 $\epsilon$、策略模型和价值模型的学习率、每次数据生成的 PPO 迭代次数、批次大小以及 GAE 参数 ($\lambda$, $\gamma$)。找到正确的平衡通常是经验性的且计算密集。
- **计算成本:** RLHF 对计算要求高。它需要维护多个模型（策略、参考、RM，可能还有一个价值模型）、进行数据生成（推理 (inference)）、计算奖励（推理）以及执行 PPO 更新（训练）。像 DeepSpeed 这样的框架和像 `trl` (Transformer Reinforcement Learning) 这样的库提供了优化的实现来管理这种复杂性。

通过精心实现这种基于 PPO 的微调 (fine-tuning)循环，我们能够有效地运用奖励模型中捕获的信号，使语言模型的行为更贴近人类期望的特性，例如有益性和无害性，这建立在监督微调奠定的基础上。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  介绍了近端策略优化（PPO）算法，这是一种流行且有效的强化学习方法，以其稳定性和样本效率而闻名。
- [Aligning Language Models to Follow Instructions](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf) — Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Karen Simonyan, Jonathan Kaplan, Hendrik Strobelt, Raymond Burdo, Robert Long, Jeremy Nelson, Sam Stewart, Anand Sharma, Amber Kuo, Adam Gough, Ilya Sutskever, Jacob Hilton, Noah Fiedel, Harish Khandelia, Anna Chen, Christine McGrew, Lora Gordon, Michael Petrov, Hyung Won Chung, Susanne Frandsen, Andy Li, Thom Lane, Françoise Chollet, Elizabeth L. Huang, Ariel Herbert-Voss, Alexis Gray, Houman Shadab, Eric Tang, Laura Gao, Amanda Askell, Brian Chen, Anna Goldie, Azalia Mirhoseini, Chris Hallacy, Monika Zimowski, Brandon Houghton, Girish Sastry, Anna Pavlick, Geoffrey Irving, Owain Evans, Josh Achiam, John Schulman, Da Yan, Alex Passos, Charles Foster, David Amodei (2022)
  Journal: Advances in Neural Information Processing Systems; Volume: 35; Pages: 27730-27744
  详细介绍了人类反馈强化学习（RLHF）流程，包括监督微调、奖励模型训练和基于PPO的策略优化，以使语言模型与人类偏好保持一致。
- [Hugging Face TRL Library Documentation](https://huggingface.co/docs/trl/en/index) — Hugging Face (2024)
  为人类反馈强化学习（RLHF），包括PPO，提供了实用的指导和实现，专门用于微调基于Transformer的语言模型。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本全面的教科书，涵盖了强化学习的基本概念、算法和理论，对于理解PPO的基础原理至关重要。

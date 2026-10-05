# 实践：实现PPO更新步骤

来源：[原文](https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning/ppo-update-step-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

PPO更新步骤是主要的学习阶段。此步骤根据收集到的经验数据调整策略网络和价值网络。这些经验数据包含通过当前策略根据提示生成的响应、从奖励模型接收的奖励（经KL惩罚调整），以及评估的状态价值。此调整旨在遵循PPO限制的同时，使预期奖励最大化。

本节侧重于计算PPO损失组成部分和执行梯度更新的实际实现细节。我们假设您已经为经验批次中的每个时间步 $t$ 计算了优势值 ($A_t$) 和回报值 ($R_t$)，这通常使用之前讨论过的广义优势估计（GAE）。

### PPO目标函数的组成部分

回想一下，PPO的目标是优化一个裁剪代理目标函数。整体损失函数 (loss function)通常结合了策略（actor）和价值函数（critic）的项，并且通常包含一个熵奖励项以鼓励策略多样性。

1. **策略损失 ($L^{CLIP}$)**: 这是PPO的核心。它使用一个比率 r\_t(\theta) = \frac{\pi\_\theta(a\_t|s\_t)}{\pi\_{\theta\_{old}}(a\_t|s\_t)，衡量自数据收集以来策略改变了多少。为了防止过大、导致不稳定的更新，这个比率被裁剪。单个时间步的目标项是：

   
   $$
   L^{CLIP}_t(\theta) = \min\left( r_t(\theta) A_t, \, \text{裁剪}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right)
   $$
   

   这里，$\epsilon$ 是裁剪超参数 (parameter) (hyperparameter)（例如0.2），$A_t$ 是时间步 $t$ 的优势估计值，$\pi_{\theta_{old}}$ 代表生成数据的策略。由于优化器通常 *最小化* 损失，我们通常取该项在批次上的期望（平均值）的负值。
2. **价值损失 ($L^{VF}$)**: 价值网络 $V_\phi(s_t)$ 被训练来预测从状态 $s_t$ 开始的预期回报（未来奖励的总和）。它通常使用均方误差（MSE）损失函数，以计算出的目标回报 $R_t$ 为目标进行训练：

   
   $$
   L^{VF}_t(\phi) = (V_\phi(s_t) - R_t)^2
   $$
   

   一些PPO实现也会裁剪价值函数更新，以进一步提高稳定性，类似于策略损失裁剪，尽管这比策略裁剪不那么常见。
3. **熵奖励 ($S$)**: 为了鼓励策略多样性，并防止其过快地收敛到确定性输出，目标函数中通常会添加一个熵奖励项（或从损失中减去）。对于分类策略（如预测词元 (token)的语言模型），熵 $H$ 的计算公式是：

   
   $$
   H(\pi_\theta(\cdot|s_t)) = - \sum_{a'} \pi_\theta(a'|s_t) \log \pi_\theta(a'|s_t)
   $$
   

   该项通常在批次上取平均值，并乘以一个系数 $c_2$。

优化器最小化的最终损失结合了这些组成部分：


$$
L_{PPO}(\theta, \phi) = \mathbb{E}_t \left[ -L^{CLIP}_t(\theta) + c_1 L^{VF}_t(\phi) - c_2 H(\pi_\theta(\cdot|s_t)) \right]
$$


其中 $c_1$ 和 $c_2$ 是超参数，用于平衡价值损失和熵奖励的贡献。请注意，前面讨论过的KL散度惩罚通常被纳入用于计算优势值 ($A_t$) 和回报值 ($R_t$) 的 *奖励* 信号中，而不是作为此PPO损失函数中的一个单独项。这简化了更新步骤，同时仍能确保策略不会与参考（SFT）模型偏差过大。

### 实现更新步骤

实际操作中，更新步骤涉及到使用小批量数据对收集到的经验批次进行多次迭代（epoch）。以下是单个PPO更新周期中，一个迷你批次的处理流程细分：

1. **加载小批量数据**: 获取收集到的经验数据子集（提示、生成的序列、在 $\pi_{\theta_{old}}$ 下的对数概率、奖励、回报 $R_t$、优势值 $A_t$、注意力掩码等）。
2. **前向传播**: 将小批量中的序列通过 *当前* 策略（$\pi_\theta$）和价值（$\ V_\phi$）网络进行前向计算。

   - 获得生成序列中动作（词元 (token)）的当前对数概率 $\log \pi_\theta(a_t|s_t)$。
   - 获得当前价值估计 $V_\phi(s_t)$。
3. **计算比率**: 使用新计算的对数概率和存储在小批量中的旧对数概率来计算概率比率 $r_t(\theta)$：

   
   $$
   r_t(\theta) = \exp(\log \pi_\theta(a_t|s_t) - \log \pi_{\theta_{old}}(a_t|s_t))
   $$
   
4. **计算策略损失**: 使用计算出的比率 $r_t(\theta)$、小批量中的优势值 $A_t$ 和裁剪参数 (parameter) $\epsilon$ 计算裁剪代理目标。将该项的负值在小批量中取平均。

   ```python
   # 伪代码，使用类PyTorch语法
   import torch

   # 假设优势值和旧对数概率是小批量的一部分
   # 当前对数概率由当前策略模型计算

   ratio = torch.exp(current_log_probs - batch['old_log_probs'])

   surr1 = ratio * batch['advantages']
   surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch['advantages']

   # 负号，因为优化器最小化损失
   policy_loss = -torch.min(surr1, surr2).mean()
   ```
5. **计算价值损失**: 计算当前价值估计 $V_\phi(s_t)$ 与小批量中的目标回报 $R_t$ 之间的均方误差。

   ```python
   # 伪代码
   # 当前价值由当前价值模型计算
   # 回报是小批量的一部分

   value_loss = ((current_values - batch['returns'])**2).mean()
   ```

   *可选：如果需要，实现价值裁剪。*
6. **计算熵奖励（可选）**: 计算策略输出分布的熵。这通常涉及到从策略网络的前向传播中获取词汇表 (vocabulary)上的完整概率分布。对该熵取平均值并取负数（因为我们最小化损失）。

   ```python
   # 伪代码（简化版）
   # 策略模型前向传播的logits

   probs = torch.softmax(logits, dim=-1)
   log_probs = torch.log_softmax(logits, dim=-1)
   entropy = -(probs * log_probs).sum(dim=-1).mean()

   # 将 -熵 存储为要从损失中减去的奖励项
   entropy_bonus = -entropy # 稍后记得系数 c2
   ```
7. **组合损失**: 使用系数 $c_1$ 和 $c_2$ 计算总损失。

   ```python
   # 伪代码
   vf_coef = 0.5 # 示例系数 c1
   entropy_coef = 0.01 # 示例系数 c2

   total_loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy_bonus
   # 注意：entropy_bonus 已经是负熵，所以减去它会加上熵项
   ```
8. **反向传播 (backpropagation)和优化**: 对 `total_loss` 执行反向传播，计算策略和价值网络参数的梯度。使用 Adam 或 AdamW 等优化器更新参数。

   ```python
   # 伪代码
   optimizer.zero_grad()
   total_loss.backward()
   # 可选：可以在此处应用梯度裁剪
   # torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
   # torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_grad_norm)
   optimizer.step()
   ```

对所有小批量重复步骤1-8，直到达到指定数量的PPO周期。

### 实际考量

- **共享网络与独立网络**: 策略（$\pi_\theta$）和价值（$V_\phi$）函数可以共享一些层（例如，基础LLM transformer），也可以是完全独立的网络。共享参数 (parameter)可以提高效率，但可能会导致策略和价值目标之间的干扰。
- **梯度累积**: 对于大型模型和批次大小，在执行优化器步骤之前，跨多个小批量累积梯度是常见的做法，以适应内存限制。
- **混合精度训练**: 使用 `float16` 或 `bfloat16` 等技术可以显著加快训练速度并减少内存使用，这对于大型LLM尤为重要。
- **初始化**: 策略网络从SFT模型进行初始化。价值网络也常从SFT模型的权重 (weight)进行初始化，可能带有不同的输出头。仔细的初始化对于稳定性很重要。
- **调试**: 在训练过程中，监控不同的损失组成部分（策略、价值、熵）、策略与参考模型之间的KL散度、更新的幅度以及奖励分数，对于调试和确保稳定性是必不可少的。Weights & Biases 或 TensorBoard 等工具非常宝贵。

这一实践步骤构成了PPO微调 (fine-tuning)阶段的计算核心。通过重复采样经验数据，并使用这个精心构建的目标来更新策略和价值网络，语言模型逐渐学会生成更符合奖励模型中编码偏好的响应，同时KL惩罚和PPO裁剪机制防止其变化过于剧烈或不稳定。掌握这一更新步骤对于成功实现RLHF必不可少。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017)
  Journal: arXiv preprint arXiv:1707.06347; DOI: [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
  Proximal Policy Optimization (PPO) 算法的原始文献，详细阐述了其目标函数和剪裁机制。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本基础强化学习教科书，涵盖了价值函数、回报、优势估计和演员-评论家方法等理论。
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  详细介绍了在大型语言模型中，PPO 在人类反馈强化学习 (RLHF) 背景下的应用。

---

[上一节](07-PPO%E8%AE%AD%E7%BB%83%E4%B8%8D%E7%A8%B3%E5%AE%9A%E6%80%A7%E6%95%85%E9%9A%9C%E6%8E%92%E9%99%A4.md) · [下一节](../05-%E6%95%B4%E5%90%88%E5%AE%8C%E6%95%B4%E7%9A%84RLHF%E6%B5%81%E7%A8%8B/01-%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%BC%96%E6%8E%92.md)

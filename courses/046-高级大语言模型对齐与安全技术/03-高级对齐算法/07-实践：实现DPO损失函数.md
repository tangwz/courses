# 实践：实现DPO损失函数

来源：[原文](https://apxml.com/zh/courses/llm-alignment-safety/chapter-3-advanced-alignment-algorithms/practice-implementing-dpo-loss)

[返回章节目录](README.md) · [返回课程目录](../README.md)

直接偏好优化（DPO）为多阶段RLHF流程提供了一个很好的替代方案。DPO不训练单独的奖励模型，而是直接使用偏好数据优化语言模型策略。这里提供了一份实践指南，帮助您理解和实现DPO损失函数 (loss function)，这是此方法的主要组成部分。

### DPO目标回顾

回想一下，DPO旨在提高给定提示（$x$）下首选回复（$y_w$）相对于拒绝回复（$y_l$）的相对对数概率。它通过基于策略模型（$\pi_\theta$）的概率与参考模型（$\pi_{ref}$）的概率对于给定补全的比率，隐式定义奖励函数来实现这一点。由此公式推导出的目标函数是：


$$
L_{DPO}(\pi_\theta; \pi_{ref}) = - \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$


让我们分析各项：

- $D$: Preference triplets $(x, y_w, y_l)$ 数据集。
- $\pi_\theta$: 我们正在微调 (fine-tuning)的语言模型（即策略）。
- $\pi_{ref}$: 一个固定的参考语言模型（通常是DPO之前的初始SFT模型）。
- $\beta$: 一个超参数 (parameter) (hyperparameter)，用于控制偏好数据的重要性，同时平衡与参考模型的一致性。$\beta$ 值越高，表示对参考模型的依从性越强。
- $\log \pi(y|x)$: 给定提示 $x$，序列 $y$ 中所有token对数概率的总和。
- $\sigma$: 逻辑Sigmoid函数，$\sigma(z) = 1 / (1 + e^{-z})$。
- $\log \sigma(\cdot)$: 对数Sigmoid函数。此损失旨在最大化 $\log \sigma$ 内部的参数，该参数表示首选回复和拒绝回复之间隐式奖励的缩放差异。最大化 $\log \sigma(z)$ 等同于最小化 $-\log \sigma(z)$。

$\log \sigma$ 函数内部的项可以改写为：


$$
\beta \left( (\log \pi_\theta(y_w|x) - \log \pi_{ref}(y_w|x)) - (\log \pi_\theta(y_l|x) - \log \pi_{ref}(y_l|x)) \right)
$$


这表明我们正在最大化首选回复与拒绝回复的对数概率比之间的差异，并按$\beta$进行缩放。

### DPO损失计算的实现

要在PyTorch或TensorFlow等常见深度学习 (deep learning)框架中实现此损失，您需要获取在训练中的策略模型（$\pi_\theta$）和冻结的参考模型（$\pi_{ref}$）下，首选（$y_w$）和拒绝（$y_l$）序列的对数概率。

以下是计算一批偏好数据损失的步骤：

1. **获取对数概率**：对提示（$x$）以及首选（$y_w$）和拒绝（$y_l$）补全，分别通过策略模型和参考模型执行前向传播。这会为批次中的每个样本生成四组对数概率：

   - `policy_chosen_logps`: $\log \pi_\theta(y_w|x)$
   - `policy_rejected_logps`: $\log \pi_\theta(y_l|x)$
   - `ref_chosen_logps`: $\log \pi_{ref}(y_w|x)$
   - `ref_rejected_logps`: $\log \pi_{ref}(y_l|x)$
     请记住，参考模型$\pi_{ref}$在训练期间不更新；其参数 (parameter)保持固定，并且不为其计算梯度。策略模型$\pi_\theta$是其参数正在被优化的模型。
2. **计算对数比**：计算首选和拒绝回复相对于参考模型的对数概率比：

   - `log_ratio_w = policy_chosen_logps - ref_chosen_logps`
   - `log_ratio_l = policy_rejected_logps - ref_rejected_logps`
3. **计算差异**：找到这些对数比之间的差异并按$\beta$缩放：

   - `diff = beta * (log_ratio_w - log_ratio_l)`
4. **应用逻辑损失**：计算差异的负对数Sigmoid。这是每个样本的核心DPO损失。使用 `logsigmoid` 等标准库函数有助于保持数值稳定性。

   - `loss_per_sample = -torch.nn.functional.logsigmoid(diff)` (在PyTorch中) 或等效函数。请注意，$-\log \sigma(z)$ 等同于 $\text{softplus}(-z)$。
5. **对损失求平均值**：计算批次中 `loss_per_sample` 的平均值，以获得训练步骤的最终损失值。

### 代码示例 (PyTorch)

下面是一个使用PyTorch的Python函数，它说明了DPO损失的计算，假设您已经计算出所需的对数概率。

```python
import torch
import torch.nn.functional as F

def compute_dpo_loss(policy_chosen_logps: torch.Tensor,
                     policy_rejected_logps: torch.Tensor,
                     ref_chosen_logps: torch.Tensor,
                     ref_rejected_logps: torch.Tensor,
                     beta: float) -> torch.Tensor:
    """
    计算直接偏好优化（DPO）损失。

    参数：
        policy_chosen_logps: 策略模型下首选回复的对数概率
                             。形状: (batch_size,)
        policy_rejected_logps: 策略模型下拒绝回复的对数概率
                               。形状: (batch_size,)
        ref_chosen_logps: 参考模型下首选回复的对数概率
                          。形状: (batch_size,)
        ref_rejected_logps: 参考模型下拒绝回复的对数概率
                            。形状: (batch_size,)
        beta: 控制偏离参考模型的温度参数。

    返回：
        批次上的平均DPO损失。
    """

    # 计算首选和拒绝回复的对数概率比
    # pi_logratios = policy_chosen_logps - policy_rejected_logps # 不直接按此公式使用
    # ref_logratios = ref_chosen_logps - ref_rejected_logps # 不直接按此公式使用

    # 计算相对于基础模型 (pi_ref) 的对数比
    log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
    log_ratio_rejected = policy_rejected_logps - ref_rejected_logps

    # 计算差异，并按beta缩放
    # 该项表示 beta * (reward_chosen - reward_rejected)
    # 其中奖励隐式定义为 log(pi_policy / pi_ref)
    diff = beta * (log_ratio_chosen - log_ratio_rejected)

    # 使用负对数Sigmoid函数计算损失
    # loss = -log(sigmoid(diff)) = softplus(-diff)
    # 使用log_sigmoid以提高数值稳定性: log_sigmoid(x) = -softplus(-x)
    # 因此，损失 = -log_sigmoid(diff)
    loss = -F.logsigmoid(diff)

    # 对批次中的损失求平均
    average_loss = loss.mean()

    return average_loss

# --- 用法 ---
# 假设这些张量来自模型的正向传播
# (例如，使用 model.forward(input_ids, labels=labels).logits)
# 通常，对数概率是为每个回复在序列长度上求和的。
batch_size = 8
# 示例对数概率（请确保在实际中正确计算）
policy_chosen_logps = torch.tensor([-10.5, -12.1, -9.8, -11.0, -13.5, -10.1, -11.8, -12.5], requires_grad=True)
policy_rejected_logps = torch.tensor([-11.2, -11.9, -10.5, -11.5, -12.8, -10.9, -12.3, -13.0], requires_grad=True)
ref_chosen_logps = torch.tensor([-10.2, -11.8, -9.5, -10.7, -13.0, -9.8, -11.5, -12.1]) # 不需要梯度
ref_rejected_logps = torch.tensor([-11.0, -11.5, -10.1, -11.1, -12.2, -10.5, -11.9, -12.5]) # 不需要梯度

beta_value = 0.1

# 计算DPO损失
dpo_loss_value = compute_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                                  ref_chosen_logps, ref_rejected_logps, beta_value)

print(f"计算得到的DPO损失: {dpo_loss_value.item():.4f}")

# --- 在训练循环中 ---
# optimizer.zero_grad()
# dpo_loss_value.backward() # 计算策略模型参数的梯度
# optimizer.step()
```

### 实践考量

- **参考模型管理**：确保参考模型的权重 (weight)被冻结，并设置为评估模式（例如，PyTorch中的`ref_model.eval()`），以便在计算对数概率的前向传播期间禁用dropout或其他训练特定行为。
- **对数概率计算**：正确计算$\log \pi(y|x)$涉及自回归 (autoregressive)运行模型或使用教师强制（teacher-forcing）将目标序列 $y$ 作为标签。您需要序列 $y$ 中实际token的对数概率总和。Hugging Face的`transformers`等框架通常提供获取序列似然的方法。
- **超参数 (parameter) (hyperparameter)$\beta$**：调整$\beta$很重要。一个常见的起始点是0.1左右。需要通过实验找到一个能够有效整合偏好，同时不会导致策略模型严重偏离参考模型的流畅性和知识的值，否则可能导致在其他任务上的性能下降或生成质量问题。
- **数值稳定性**：如所示，在对数空间中进行操作非常重要。使用`F.logsigmoid`等库函数可以防止直接计算`log(1 / (1 + exp(-diff)))`可能导致的潜在下溢/上溢问题。
- **集成到训练中**：此损失函数 (loss function)取代了监督微调 (fine-tuning)期间使用的标准交叉熵损失。训练循环包括采样$(x, y_w, y_l)$批次，执行四次前向传播（两次使用$\pi_\theta$，两次使用$\pi_{ref}$），计算DPO损失，然后*仅*通过策略模型$\pi_\theta$反向传播 (backpropagation)梯度。

### 练习

考虑修改一个标准语言模型微调 (fine-tuning)脚本（例如使用Hugging Face `transformers`的脚本）。修改训练循环以整合DPO损失计算。您将需要：

1. 一个格式化为偏好三元组的数据集。
2. 您的基础语言模型的两个实例：一个用于策略（`policy_model`），一个用于参考（`ref_model`）。确保`ref_model`被冻结。
3. 一个函数，用于计算在两个模型下首选和拒绝回复的序列对数概率。
4. 将`compute_dpo_loss`函数（或类似函数）集成到您的训练步骤中。

Hugging Face的`trl`等库提供了预构建的`DPOTrainer`类，这些类抽象化了大部分复杂性，但自己实现核心损失能让您更透彻地理解其内部工作原理。

## 参考资料

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (2023)
  Journal: arXiv; DOI: [10.48550/arXiv.2305.18290](https://doi.org/10.48550/arXiv.2305.18290)
  介绍了直接偏好优化（DPO）算法、其理论基础和实证结果。
- [DPOTrainer - trl Documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer) — Hugging Face (2024)
  Publisher: Hugging Face
  为使用 trl 库实现 DPO 提供了实践指导和代码示例，trl 是 LLM 对齐的常用框架。
- [A Survey of Large Language Model Alignment](https://arxiv.org/abs/2402.16438) — Tianyi Tang, Wenyang Luo, Haoyang Huang, Dongdong Zhang, Xiaolei Wang, Xin Zhao, Furu Wei, Ji-Rong Wen (2024)
  Journal: ACL 2024; DOI: [10.48550/arXiv.2402.16438](https://doi.org/10.48550/arXiv.2402.16438)
  全面概述了大型语言模型对齐技术，其中包含对直接偏好优化（DPO）的讨论。

---

[上一节](06-%E5%AF%B9%E9%BD%90%E6%96%B9%E6%B3%95%E5%AF%B9%E6%AF%94%E5%88%86%E6%9E%90.md) · [下一节](../04-%E8%AF%84%E4%BC%B0LLM%E7%9A%84%E5%AE%89%E5%85%A8%E6%80%A7%E5%92%8C%E5%AF%B9%E9%BD%90%E5%BA%A6/01-%E5%AE%9A%E4%B9%89%E5%AE%89%E5%85%A8%E6%96%B9%E9%9D%A2%EF%BC%9A%E6%97%A0%E5%AE%B3%E6%80%A7%E3%80%81%E8%AF%9A%E5%AE%9E%E6%80%A7%E3%80%81%E6%9C%89%E7%94%A8%E6%80%A7.md)

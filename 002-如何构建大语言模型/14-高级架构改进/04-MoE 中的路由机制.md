# MoE 中的路由机制

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-14-advanced-architectural-modifications/routing-mechanisms-moe)

[返回章节目录](README.md) · [返回课程目录](../README.md)

MoE（专家混合）层的核心思想是在不按比例增加处理每个令牌的计算成本的情况下，显著增加模型的参数 (parameter)数量。这是通过在MoE层内设置多个“专家”网络（通常是简单的前馈网络），但对于每个输入令牌，只激活其中一小部分（通常是一个或两个）来实现的。实现这种条件计算的重要组成部分是**路由机制**，通常称为**门控网络**。

### 门控网络

门控网络充当MoE层的交通管制员。它的职责是查看每个传入的令牌表示，并决定哪个专家应处理它。

通常，门控网络本身是一个相对简单的神经网络 (neural network)。常见的设计包括：

1. **输入：** 来自前一层的令牌隐藏状态表示 $x$（例如，自注意力 (self-attention)子层的输出）。
2. **变换：** 将带有权重 (weight) $W_g$ 的线性层应用于输入令牌表示：$h = x W_g$。
3. **输出分数（Logits）：** 结果 $h$ 是一个维度等于专家数量 $N$ 的向量 (vector)。这些是每个专家的初始分数或“logits”。
4. **概率（可选但常见）：** 通常对这些 logits 应用 softmax 函数，以生成专家上的概率分布 $p$：$p = \text{softmax}(h)$。此向量 $p$ 包含概率 $p_i$，使得 $\sum_{i=1}^{N} p_i = 1$，表示门将令牌分配给每个专家 $i$ 的置信度。

以下是一个简单门控网络的 PyTorch 代码片段：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGatingNetwork(nn.Module):
    def __init__(self, model_dim: int, num_experts: int):
        super().__init__()
        # 用于计算每个专家 logits 的线性层
        self.layer = nn.Linear(model_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状：(batch_size, sequence_length, model_dim)
        # 计算 logits
        logits = self.layer(x)
        # logits 的形状：(batch_size, sequence_length, num_experts)

        # 应用 softmax 获取概率
        gate_probabilities = F.softmax(logits, dim=-1)
        # gate_probabilities 的形状：(batch_size, sequence_length, num_experts)

        return gate_probabilities, logits # 返回概率和原始 logits
```

### Top-k 门控

尽管门控网络会生成所有专家的分数或概率，但激活所有专家将违背条件计算的目的。最普遍的路由策略是 **Top-k 门控**，其中只选择得分最高（根据门 logits）的 $k$ 个专家来处理令牌。在实践中，$k$ 通常非常小，通常只有1或2。

- **k=1：** 每个令牌仅由单个最佳专家处理，这最大化了稀疏性，但可能限制了模型组合来自多个专业化路径的信息的能力。
- **k=2：** 每个令牌由排名前两位的专家处理。这已成为一种受欢迎的选择（例如，在 Google 的 Switch Transformer 变体和 Mixtral 模型中），因为它提供了良好的平衡。它允许令牌受益于两位专家的“知识”，可能提高表示质量，同时将计算开销保持在远低于密集模型的水平。

**计算输出：**
如果令牌 $x$ 被路由到排名前 $k$ 的专家 $E_{i_1}, E_{i_2}, ..., E_{i_k}$，并带有相应的门控概率（或归一化 (normalization)分数）$p_{i_1}, p_{i_2}, ..., p_{i_k}$，则该令牌在 MoE 层的最终输出 $y$ 计算为这些选定专家输出的加权和：


$$
y = \sum_{j=1}^{k} p_{i_j} \cdot E_{i_j}(x)
$$


此和中使用的概率 $p_{i_j}$ 通常来自门的 softmax 输出，但仅对选定的 Top-k 专家进行重新归一化。

我们来展示一下 PyTorch 中的 Top-k 选择逻辑：

```python
# 假设 gate_logits 形状：(batch_size * sequence_length, num_experts)
# 假设 experts 是一个专家网络模块列表
# 此示例中 k = 2

# 如果需要，展平输入令牌
# x_flat 形状：(batch_size * sequence_length, model_dim)

num_experts = len(experts) # 假设 experts 已定义
k = 2

# 从门控网络获取 logits 和概率
gate_probabilities, gate_logits = gating_network(x_flat)
# gate_probabilities 形状：(num_tokens, num_experts)

# 找出 Top-k 专家（索引和值）
# top_k_weights 是所选专家的门控概率
# top_k_indices 包含所选专家的索引
top_k_weights, top_k_indices = torch.topk(gate_probabilities, k, dim=-1)
# top_k_weights 形状：(num_tokens, k)
# top_k_indices 形状：(num_tokens, k)

# 对 Top-k 专家之间的权重进行归一化（可选但常见）
# 确保它们加起来为1，以便进行加权平均
normalized_weights = top_k_weights / torch.sum(
    top_k_weights, dim=-1, keepdim=True
)
# normalized_weights 形状：(num_tokens, k)

# 初始化最终输出张量
final_output = torch.zeros_like(x_flat)

# 这部分在实践中通常经过高度优化
# 使用 scatter/gather 操作
# 循环仅为便于理解：
for i in range(num_experts):
    # 找出哪些令牌将专家 'i' 选为它们的 Top-k 之一
    # 创建一个掩码，其中 top_k_indices 等于当前专家索引 'i'
    expert_mask = (top_k_indices == i) # 形状：(num_tokens, k)

    # 获取选择专家 'i' 的令牌索引
    # 使用 torch.nonzero 获取 expert_mask 为 True 的索引
    token_indices, _ = torch.nonzero(expert_mask, as_tuple=True)

    if token_indices.numel() > 0:
        # 获取这些令牌分配给专家 'i' 的特定权重
        # 收集与专家索引 'i' 对应的权重
        weights_for_expert = normalized_weights[expert_mask]
        # 形状：(num_tokens_for_this_expert,)

        # 选择路由到专家 'i' 的输入令牌
        inputs_for_expert = x_flat[token_indices]

        # 通过专家 'i' 处理这些令牌
        expert_output = experts[i](inputs_for_expert)
        # 形状：(num_tokens_for_this_expert, model_dim)

        # 用相应的门控权重对专家输出进行加权
        weighted_output = expert_output * weights_for_expert.unsqueeze(-1)
        # 确保权重正确广播

        # 将加权输出添加到最终输出张量
        # 对于正确的令牌
        # 使用 index_add_ 或 scatter_add_ 进行高效更新
        final_output.index_add_(0, token_indices, weighted_output)

# final_output 形状：(num_tokens, model_dim)
# 如果需要，重新塑形回 (batch_size, sequence_length, model_dim)
```

“*注意：* 上述循环效率非常低。实际实现会使用优化的 scatter/gather 操作或专用核来路由令牌和聚合输出，而无需显式循环，特别是在分布式环境中。”

### 带噪声的 Top-k 门控

为了可能改进负载平衡并引入一种正则化 (regularization)形式，一些 MoE 实现使用了**带噪声的 Top-k 门控**。思路很简单：在应用 softmax 并选择 Top-k 专家*之前*，向门 logits 添加随机噪声（通常是高斯噪声）。


$$
h_{\text{noisy}} = h + \text{噪声}
$$


$$
p = \text{softmax}(h_{\text{noisy}})
$$


噪声通常由可学习权重 (weight)或固定超参数 (parameter) (hyperparameter)进行缩放。这种噪声注入可以防止门始终依赖于少数几个相同的专家，鼓励训练期间的尝试，有时会带来更好的泛化能力和更均衡的专家利用。

```python
# 在 Top-k 选择前添加噪声的例子
if self.training: # 仅在训练期间应用噪声
    noise = torch.randn_like(gate_logits) * noise_std_dev
    # noise_std_dev 是一个超参数
    noisy_logits = gate_logits + noise
else:
    noisy_logits = gate_logits # 推理期间不加噪声

# 使用 noisy_logits 继续进行 softmax 和 Top-k 选择
gate_probabilities = F.softmax(noisy_logits, dim=-1)
top_k_weights, top_k_indices = torch.topk(gate_probabilities, k, dim=-1)
# ... 其他逻辑 ...
```

### 路由考量

实现有效的路由机制涉及处理几个实际挑战：

1. **专家容量：** 在并行处理设置（如 GPU）中，当工作负载均衡时，计算效率最高。如果门控网络在处理批次中将明显更多的令牌路由到一个专家，而不是其他专家，那么该专家将成为瓶颈。为缓解这种情况，通常会引入**专家容量**这一思想。它定义了专家每批次可以处理的最大令牌数量，该数量根据令牌总数和专家数量计算，并加上一个缓冲（容量因子）。

   - **容量因子：** 大于 1.0 的值（例如 1.25）允许一定程度的不平衡。容量 = `ceil( (num_tokens / num_experts) * capacity_factor )`。
   - **令牌丢弃：** 如果分配给某个专家的令牌数量超出其容量，多余的令牌可能会被“丢弃”，这意味着它们在 MoE 层的输出变为零（或者输入表示未变地通过）。这是不希望的，但有时为了系统效率是必要的。
2. **负载平衡损失：** 为了明确鼓励门控网络在专家之间均匀分配令牌，训练期间通常会在主模型损失中添加一个**辅助负载平衡损失**。一种常见的表述旨在最小化分配给每个专家的令牌比例以及分配给每个专家的路由概率质量比例的变化。

   - 令 $N$ 为专家数量，$B$ 为批次中的令牌数量。
   - 将 $f_i$ 定义为分配给专家 $i$ 的令牌比例。
   - 将 $P_i$ 定义为门控网络在批次中为所有令牌分配给专家 $i$ 的平均概率。
   - 一个典型的负载平衡损失（简化形式）可能如下所示：
     
     $$
     L_{\text{平衡}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
     $$
     
     其中 $\alpha$ 是控制此损失项强度的超参数 (parameter) (hyperparameter)。最小化此损失鼓励 $f_i$ 和 $P_i$ 都接近 $1/N$，从而促进负载均衡。

   ```python
   # 负载平衡损失组件的计算

   # gate_probabilities 形状：(num_tokens, num_experts)
   # top_k_indices 形状：(num_tokens, k)

   num_tokens, num_experts = gate_probabilities.shape

   # 计算 Fi：分配给专家 i 的令牌比例
   # 统计 top_k_indices 中每个专家索引的出现次数
   # 这需要对 k > 1 的情况进行仔细处理
   # 简化 k=1 的情况：
   if k == 1:
     expert_counts = torch.bincount(top_k_indices.squeeze(), minlength=num_experts)
     f_i = expert_counts.float() / num_tokens
   else:
     # 对于 k > 1，需要更复杂的计数，通常通过独热编码和求和完成
     # 示例占位符：假设 calculate_fraction_dispatched 处理 k>1
     f_i = calculate_fraction_dispatched(top_k_indices, num_experts, num_tokens)

   # 计算 Pi：专家 i 的平均路由概率
   p_i = torch.mean(gate_probabilities, dim=0) # 跨令牌的平均概率

   # 计算损失
   load_balancing_loss = alpha * num_experts * torch.sum(f_i * p_i)

   # 将此损失添加到主任务损失（例如，交叉熵）
   total_loss = main_task_loss + load_balancing_loss
   ```
3. **稀疏路由与软路由：** Top-k 门控是一种稀疏路由形式。存在其他方案（“软路由”），其中每个专家处理每个令牌，但输出由来自门控的完整概率分布加权。虽然可能更简单，但软路由失去了 MoE 的计算优势，因为所有专家都对所有令牌处于活动状态，这使得它在大规模效率提升方面较不常见。

路由机制的设计和调整，包括 $k$ 的选择、噪声的使用、容量因子以及负载平衡损失系数，是构建和训练有效 MoE 模型的重要方面。这些选择直接影响模型性能、训练稳定性和计算效率。像 DeepSpeed 这样的框架提供了抽象和优化来管理这些复杂性，尤其是在专家可能位于不同硬件设备上的分布式训练场景中。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean (2017)
  Journal: arXiv preprint arXiv:1701.06538; DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控专家混合（MoE）层的基本概念，详细阐述了Top-k门控、噪声Top-k门控以及负载平衡损失的初步概念。
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) — Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen (2020)
  Journal: arXiv preprint arXiv:2006.16668; DOI: [10.48550/arXiv.2006.16668](https://doi.org/10.48550/arXiv.2006.16668)
  改进了MoE的路由机制，为超大模型引入了增强的负载平衡策略和专家容量的概念。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Volume: 23; Pages: 1-36; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  介绍了Switch Transformers，这是一种使用k=1路由的高效MoE架构，强调了专家容量和负载平衡等实用考虑因素，用于扩展模型。

---

[上一节](03-%E4%B8%93%E5%AE%B6%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B%20%28MoE%29%20%E7%AE%80%E4%BB%8B.md) · [下一节](05-MoE%20%E5%B1%82%E4%B8%AD%E7%9A%84%E8%B4%9F%E8%BD%BD%E5%9D%87%E8%A1%A1.md)

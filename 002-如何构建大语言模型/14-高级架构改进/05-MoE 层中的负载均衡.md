# MoE 层中的负载均衡

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-14-advanced-architectural-modifications/load-balancing-moe-layers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然路由机制（例如 top-k 门控）会将输入令牌导向专家混合 (MoE) 层中的特定专家，但它们本身不保证计算负载在所有专家之间均匀分配。通常，如果没有明确干预，门控网络可能会偏好一小部分专家，从而导致严重的负载不均。这种不均带来了以下几个问题：

1. **计算效率低下：** 如果只有少数专家持续被选择，与其余专家相关的参数 (parameter)和计算资源就会未充分利用。这会抵消 MoE 旨在提供的一些效率提升。
2. **训练不稳定：** 通过大量使用的专家回传的梯度可能会主导训练过程，可能导致不稳定或收敛缓慢。
3. **模型质量下降：** 未充分利用的专家可能无法接收到足够的训练信号以变得专业或有效，从而限制了模型的整体能力和表现。

因此，在训练 MoE 模型时，实施促进负载均衡的机制是一种标准做法。最常见的方法是向主要任务损失（例如，语言建模的交叉熵损失）添加一个辅助损失项。此辅助损失会惩罚不平衡，指导门控网络更均匀地分配令牌。

### 辅助负载均衡损失

辅助损失的目标是激励路由器为每个专家分配大致相同数量的令牌。Switch Transformer 论文和后续工作中介绍的一种被广泛采用的公式，旨在最小化每个专家处理的令牌数量的变异。

令 $N$ 为专家数量， $B$ 为当前批次（或微批次）中的令牌数量。对于每个专家 $i \in \{1, \dots, N\}$，我们可以定义两个量：

- $f_i$: 路由到专家 $i$ 的批次内令牌分数。这通过对批次中所有令牌的专家 $i$ 路由决策（硬路由如 top-k 为 0 或 1）求和，再除以 $B$ 来计算。
  
  $$
  f_i = \frac{1}{B} \sum_{x \in \text{Batch}} \mathbb{I}(\text{为令牌 } x \text{ 选择的专家 } i)
  $$
  
- $P_i$: 门控网络分配给专家 $i$ 的总路由概率质量的分数，对批次中的令牌进行平均。如果 $g(x)_i$ 是给定令牌 $x$ 时门控网络输出的专家 $i$ 概率，那么：
  
  $$
  P_i = \frac{1}{B} \sum_{x \in \text{Batch}} g(x)_i
  $$
  

辅助负载均衡损失 $L_{balance}$ 通常计算为这两个向量 (vector)的点积，并乘以专家数量 $N$ 和一个可调超参数 (parameter) (hyperparameter) $\alpha$：


$$
L_{balance} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$


用于反向传播 (backpropagation)的总损失是主要任务损失 $L_{task}$ 和均衡损失之和：


$$
L_{total} = L_{task} + L_{balance}
$$


**直观理解：** 最小化 $L_{balance}$ 促使所有专家的 $f_i$ 和 $P_i$ 都接近 $1/N$。如果某个专家接收了大量令牌份额（$f_i$ 较高），损失就会增加。类似地，如果门控网络分配给某个专家高概率（$P_i$ 较高），损失也会增加。当实际分配 ($f_i$) 和路由器的置信度 ($P_i$) 均匀分布时，损失最小化。超参数 $\alpha$ 控制这种均衡激励相对于主要任务目标的强度；典型值通常很小（例如 0.01）。

下面是一个 PyTorch 代码片段，说明了计算方法，假设 `gating_outputs` 包含来自门控网络的概率，而 `indices` 包含每个令牌选择的专家索引：

```python
import torch
import torch.nn.functional as F

# 示例输入（替换为实际模型输出）
# gating_outputs: 形状 [num_tokens, num_experts] - softmax 概率
# indices: 形状 [num_tokens, k] - 每个令牌选择的 top-k 专家索引
num_experts = 8
num_tokens = 1024
k = 2
gating_outputs = torch.randn(num_tokens, num_experts).softmax(dim=-1)
# 模拟 top-k 路由索引（实际中这些来自路由器）
indices = torch.topk(gating_outputs, k, dim=-1).indices

# --- 辅助损失计算 ---
# 计算 f_i: 路由到专家 i 的令牌分数
expert_mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)
# 形状 [num_tokens, num_experts]，如果选择了专家则为 1，否则为 0
tokens_per_expert = expert_mask.sum(dim=0) # 形状 [num_experts]
f_i = tokens_per_expert / num_tokens # 每个专家的令牌分数

# 计算 P_i: 专家 i 的平均路由概率
P_i = gating_outputs.mean(dim=0) # 形状 [num_experts]

# 计算损失
# 注意: N = num_experts
load_balance_loss = num_experts * torch.sum(f_i * P_i)

# 示例: 添加到主要任务损失（假设 alpha = 0.01）
alpha = 0.01
# task_loss = ... （在其他地方计算）
# total_loss = task_loss + alpha * load_balance_loss
# total_loss.backward()

print(f"负载均衡损失项: {load_balance_loss.item():.4f}")
print(f"每个专家的令牌分布: {f_i.detach().numpy()}")
print(f"每个专家的平均概率: {P_i.detach().numpy()}")
```



[交互图表：专家负载分布（每专家令牌数）](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-14-advanced-architectural-modifications/load-balancing-moe-layers#plot-3jxuoi)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "Expert 1",
        "Expert 2",
        "Expert 3",
        "Expert 4",
        "Expert 5",
        "Expert 6",
        "Expert 7",
        "Expert 8"
      ],
      "y": [
        0.11,
        0.15,
        0.09,
        0.13,
        0.18,
        0.1,
        0.14,
        0.1
      ],
      "name": "不均衡负载",
      "marker": {
        "color": "#ff8787"
      }
    },
    {
      "type": "bar",
      "x": [
        "Expert 1",
        "Expert 2",
        "Expert 3",
        "Expert 4",
        "Expert 5",
        "Expert 6",
        "Expert 7",
        "Expert 8"
      ],
      "y": [
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.125
      ],
      "name": "理想均衡负载",
      "marker": {
        "color": "#69db7c"
      }
    }
  ],
  "layout": {
    "title": "专家负载分布（每专家令牌数）",
    "xaxis": {
      "title": "专家"
    },
    "yaxis": {
      "title": "令牌分数",
      "range": [
        0,
        0.25
      ]
    },
    "barmode": "group",
    "bargap": 0.2
  }
}
```

</details>



> 示例可视化，比较专家间不均衡的令牌分布与理想的完全均衡状态。辅助损失旨在将分布推向均衡状态。

### 容量因子

另一个常与辅助损失结合使用的机制是**容量因子** ($C$)。这限制了任何单个专家在一个批次内可以处理的令牌数量。每个专家的容量通常设置为：


$$
\text{容量} = C \times \frac{\text{每批次令牌数}}{\text{专家数量}}
$$


容量因子 $C$ 通常略大于 1（例如 1.25 或 1.5）。如果路由机制分配给某个专家的令牌数量超过其允许的容量，则超额令牌被视为“丢弃”或“溢出”。这些丢弃的令牌不参与该 MoE 层的计算（包括前向和反向传播 (backpropagation)），实际上像通过了一个恒等函数一样被处理。

虽然丢弃令牌可能看起来有害，但使用容量因子提供了对抗严重不平衡的硬性约束。它防止单个专家过载，即使辅助损失尚未完全纠正路由器的偏好。然而，将 $C$ 设置得过低可能导致过多的令牌丢弃，从而阻碍学习。在强制均衡和保留所有信息之间存在权衡。在训练期间监控丢弃令牌的百分比对于调整 $C$ 很重要。

### 与分布式训练的交互

负载均衡在分布式设置中尤为重要，尤其是在使用**专家并行**时，即不同专家位于不同的计算设备（例如 GPU）上。如果负载不均衡，承载受偏好专家的设备会成为瓶颈，而拥有未充分利用专家的设备则处于空闲状态，导致扩展性差和资源浪费。辅助损失和容量因子共同作用，以确保计算在分布式硬件上更均匀地分布。

总之，实现专家间负载的均衡分配对于高效且稳定的 MoE 模型训练来说，是必要条件。辅助负载均衡损失与仔细调整的容量因子相结合，提供了有效机制来鼓励门控网络更均匀地使用所有专家，从而最大化条件计算的益处。调整辅助损失系数 $\alpha$ 和容量因子 $C$ 是成功训练大型 MoE 模型的重要方面。

## 参考资料

- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2021)
  Journal: Journal of Machine Learning Research; Pages: 1-22; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  本文介绍了Switch Transformer架构、辅助负载均衡损失公式以及用于专家混合模型的容量因子。
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  这篇基础性论文介绍了稀疏门控专家混合层，并讨论了为实现专家有效利用而进行负载均衡的挑战。
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) — Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen (2020)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2006.16668](https://doi.org/10.48550/arXiv.2006.16668)
  本文描述了一个用于大型模型的条件计算框架，详细介绍了专家混合层的有效负载均衡策略和分布式训练。

---

[上一节](04-MoE%20%E4%B8%AD%E7%9A%84%E8%B7%AF%E7%94%B1%E6%9C%BA%E5%88%B6.md) · [下一节](../15-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E7%AD%96%E7%95%A5/01-%E5%8A%A8%E6%9C%BA%EF%BC%9A%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E8%BF%9B%E8%A1%8C%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%EF%BC%9F.md)

# 噪声Top-k门控实现负载均衡

来源：[原文](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-2-advanced-routing-mechanisms/noisy-top-k-gating)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准Top-k门控机制的路由决策，对于给定的输入和模型权重 (weight)集，是确定的。尽管这听起来不错，但它可能导致在训练期间迅速形成固定的路由模式。某些专家可能持续从门控网络获得最高分数，吸引绝大多数令牌，而其他专家则缺乏数据。这种不平衡阻碍了模型形成多样化专业专家的能力，甚至可能导致一些专家失效，对最终输出毫无贡献。

噪声Top-k门控提供了一种简单且计算成本低廉的解决方案来应对此问题：在选择Top-k专家之前，向门控网络的输出logits中加入少量随机噪声。此方法作为一种正则化 (regularization)形式，促使模型考量不同的路由路径，并避免其过度依赖一小部分专家。

### 噪声门控的运行方式

主要思路是扰动门控网络生成的确定性分数。对于每个令牌，在门控网络计算出每个专家的logits（原始、未归一化 (normalization)分数）后，我们添加一个从噪声分布中抽取的随机值，通常是高斯分布。

数学表示直接基于标准门控过程。设 $h(x)$ 表示门控网络为输入令牌 $x$ 生成的logits向量 (vector)，其中 $h(x) = x \cdot W_g$，且 $W_g$ 是门的权重 (weight)矩阵。在噪声门控中，我们计算一组修改后的logits，$h_{noisy}(x)$：


$$
h_{noisy}(x) = h(x) + \text{噪声}
$$


噪声项通常从均值为零的正态分布中抽取。该噪声的标准差是一个可调超参数 (parameter) (hyperparameter)。一种常见实现，由原始稀疏门控MoE论文提出，使用一个单独的可训练权重矩阵 $W_{noise}$ 来缩放噪声：


$$
\text{噪声} = \text{标准正态}() \cdot \text{softplus}(x \cdot W_{noise})
$$


这里，`StandardNormal()` 从 $\mathcal{N}(0, 1)$ 生成随机值，并且 `softplus` 函数确保缩放因子始终为正。随后，基于这些含噪声的logits进行Top-k选择。此过程仅在训练期间生效。在推理 (inference)期间，噪声被禁用以保证确定性和稳定的输出。

下图说明了添加噪声如何改变路由决策。

> 在标准门控中，专家1和专家3被选中。加入噪声后，专家3的 logit 被扰动得恰好使得专家4成为次高选择，从而改变了路由路径。

### 对负载均衡和训练的影响

添加噪声的主要益处是改进了专家间的负载分配。通过“搅动”分数，那些处于被路由到热门专家与不太热门专家边缘的令牌，有时会被发送到后者。这避免了任何单个专家成为瓶颈，并确保所有专家获得足够多样的训练样本，以习得有价值的专业化。

此机制直接补充了上一章讨论的辅助负载均衡损失。尽管损失函数 (loss function)惩罚不平衡，噪声门控则主动抑制其最初的形成。结果通常是更稳定的训练过程，具有更平滑的损失曲线，并降低了专家失效的可能性。

### 实现和实际考量

实际中，实现噪声门控非常直接。它只需在MoE层的门控模块中添加几行代码。

这是一个PyTorch风格的简化示例：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyTopkRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate_linear = nn.Linear(d_model, num_experts)
        self.noise_linear = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x 形状: (批次大小 * 序列长度, d_model)
        logits = self.gate_linear(x)

        # 仅在训练期间添加噪声
        if self.training:
            noise = self.noise_linear(x)
            noise_std = F.softplus(noise)
            noisy_logits = logits + (torch.randn_like(logits) * noise_std)
        else:
            noisy_logits = logits

        # 选择Top-k专家
        top_k_logits, indices = torch.topk(noisy_logits, self.top_k, dim=-1)

        # 创建稀疏路由掩码
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices
```

一个重要的考量是注入噪声的量级。

- **噪声过小：** 对负载均衡的影响将微不足道，模型仍可能面临路由不平衡问题。
- **噪声过大：** 路由决策可能变得过于随机，干扰学习过程。如果门控网络的决策持续被噪声掩盖，它可能难以学习有意义的路由模式。

使用可训练的 `W_{noise}` 矩阵使得模型能够根据每个令牌调整噪声水平，这通常比在整个训练运行中使用单一、固定的噪声超参数 (parameter) (hyperparameter)更有效。

最终，噪声Top-k门控是一种简单而有效的方法，用于提升MoE模型的稳定性和表现。它鼓励在路由空间中进行试用，从而带来更好的负载分配和更专业的专家表现，且不增加显著的计算开销。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控专家混合 (MoE) 层和噪声top-k门控机制，这是所描述内容的奠基性工作。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/v23/23-120.html) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 23; Pages: 1-39; DOI: [10.5598/jmlr.v23i1.21-0990](https://doi.org/10.5598/jmlr.v23i1.21-0990)
  介绍了Switch Transformers，一个重要的大规模MoE模型，它基于稀疏门控并讨论了实际的负载均衡策略。

---

[上一节](01-Top-k%E9%97%A8%E6%8E%A7%E5%8F%8A%E5%85%B6%E5%8F%98%E4%BD%93%E7%9A%84%E5%88%86%E6%9E%90.md) · [下一节](03-%E5%9F%BA%E4%BA%8E%E5%93%88%E5%B8%8C%E7%9A%84%E7%A1%AE%E5%AE%9A%E6%80%A7%E9%80%89%E6%8B%A9%E8%B7%AF%E7%94%B1.md)

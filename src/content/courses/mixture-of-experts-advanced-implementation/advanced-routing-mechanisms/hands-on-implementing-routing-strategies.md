---
course: "mixture-of-experts-advanced-implementation"
chapter: "advanced-routing-mechanisms"
lesson: "hands-on-implementing-routing-strategies"
sourceId: 7058
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-2-advanced-routing-mechanisms/hands-on-implementing-routing-strategies"
title: "动手实践：实现不同的路由策略"
description: "对多种高级路由算法的实际实现，并比较其性能特点。"
order: 7
plots: ["plots/7058-0.json"]
sourceHash: "34496518a044d5697360c06ecf27cc14fb6f556ee01aeb0647121da8f9530b0b"
sourceCorrections: []
---

此实践练习将实现几种高级路由机制。它旨在巩固您对每个路由器如何运作、其具体的实现细节以及所涉权衡的理解。每个路由器都将构建为模块化的 PyTorch `nn.Module`，使其可以在更大的 MoE 层中互换使用。

对于此练习，我们假设处理的是一批 token 嵌入 (embedding)。我们将定义一套标准维度以确保示例的一致性。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 配置 ---
NUM_EXPERTS = 8
D_MODEL = 512
BATCH_SIZE = 4
SEQ_LEN = 1024
TOKENS_PER_BATCH = BATCH_SIZE * SEQ_LEN
```

我们的目标是创建接收形状为 `(TOKENS_PER_BATCH, D_MODEL)` 的张量的路由器，并为 MoE 层生成必要的分配和辅助损失。

### 实现一个带噪声的 Top-k 路由器

带噪声的 Top-k 路由器是一种常用且实用的策略，用于在训练期间改善负载均衡。它的工作方式是在选择 Top-k 专家之前，向路由器的 logits 添加随机噪声。这种随机性有助于避免路由器持续偏爱相同的少数专家。

噪声通常从正态分布中抽取，并通过学习到的权重 (weight)矩阵进行缩放。此噪声仅在训练期间应用。


$$
\text{对数几率} = \text{线性}(x)
$$


$$
\text{带噪声的对数几率} = \text{对数几率} + \text{torch.randn\_like(对数几率)} \cdot \text{Softplus}(\text{噪声权重})
$$


我们来实现它。我们将为每个 token 选择 `k=2` 个专家。

```python
class NoisyTopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts

        # 生成对数几率的层
        self.gate = nn.Linear(d_model, num_experts)
        # 生成噪声缩放因子的层
        self.noise_net = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x 的形状：[TOKENS_PER_BATCH, d_model]
        logits = self.gate(x)

        # 在训练期间添加噪声
        if self.training:
            noise_logits = self.noise_net(x)
            # 使用 softplus 确保噪声缩放因子为正
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            logits = logits + noise

        # 获取 Top-k 对数几率及其索引
        # top_k_logits 的形状：[TOKENS_PER_BATCH, top_k]
        # top_k_indices 的形状：[TOKENS_PER_BATCH, top_k]
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)

        # 对 Top-k 对数几率应用 softmax 以获取权重
        gating_scores = F.softmax(top_k_logits, dim=-1)

        # 我们还需要一个用于负载均衡损失的掩码
        # 此掩码对于每个被选中的 token-专家对为 1
        # B = TOKENS_PER_BATCH, E = NUM_EXPERTS
        # zero_mask 的形状：[B, E]
        zero_mask = torch.zeros(x.size(0), self.num_experts, device=x.device)
        # router_mask 的形状：[B, E]
        router_mask = zero_mask.scatter(1, top_k_indices, 1)

        return top_k_indices, gating_scores, router_mask

# --- 使用示例 ---
noisy_router = NoisyTopKRouter(D_MODEL, NUM_EXPERTS, top_k=2)
noisy_router.train() # 设置为训练模式以启用噪声

input_tokens = torch.randn(TOKENS_PER_BATCH, D_MODEL)
indices, scores, mask = noisy_router(input_tokens)

print("带噪声的 Top-k 路由器输出：")
print("索引形状：", indices.shape) # [TOKENS_PER_BATCH, 2]
print("得分形状：", scores.shape)   # [TOKENS_PER_BATCH, 2]
print("掩码形状：", mask.shape)       # [TOKENS_PER_BATCH, 8]
```

输出精确地提供了后续 MoE 层所需的信息：应该路由到哪些专家 (`indices`)、如何加权它们的输出 (`scores`)，以及一个用于计算负载均衡损失的 `mask`。

### 实现一个 Switch 路由器

Switch Transformer 架构通过设置 `k=1` 简化了路由。这种设计选择显著减少了通信成本和计算复杂度，因为每个 token 仅由单个专家处理。在这种设置下，负载均衡损失变得尤为重要，以防止专家未被充分利用。

此实现是对我们的 `NoisyTopKRouter` 的直接简化，其中 `top_k` 固定为 1，并且为清晰起见移除了噪声机制，尽管它也可以被包含在内。

```python
class SwitchRouter(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x 的形状：[TOKENS_PER_BATCH, d_model]
        logits = self.gate(x) # [B, E]

        # 应用 softmax 以获取用于负载均衡损失的概率
        router_probs = F.softmax(logits, dim=-1)

        # 选择单个最佳专家
        # top_1_scores 等同于最大对数几率
        # top_1_indices 是所选专家的索引
        top_1_scores, top_1_indices = torch.max(router_probs, dim=-1)

        # 为选定的专家创建独热掩码
        # 此掩码既用于路由也用于计算损失
        # one_hot_mask 的形状：[B, E]
        one_hot_mask = F.one_hot(top_1_indices, num_classes=self.num_experts)

        # 对于所选专家，门控得分仅为 1.0。
        # 我们以与 Top-k 路由器一致的形状返回它。
        gating_scores = top_1_scores.unsqueeze(-1)

        return top_1_indices.unsqueeze(-1), gating_scores, one_hot_mask

# --- 使用示例 ---
switch_router = SwitchRouter(D_MODEL, NUM_EXPERTS)
switch_router.eval() # 在此简化版本中，训练/评估模式无差异

input_tokens = torch.randn(TOKENS_PER_BATCH, D_MODEL)
indices, scores, mask = switch_router(input_tokens)

print("\nSwitch 路由器 (Top-1) 输出：")
print("索引形状：", indices.shape) # [TOKENS_PER_BATCH, 1]
print("得分形状：", scores.shape)   # [TOKENS_PER_BATCH, 1]
print("掩码形状：", mask.shape)       # [TOKENS_PER_BATCH, 8]
```

注意输出形状与我们之前的路由器一致，保证了模块化。`mask` 现在是每个 token 的独热向量 (vector)，反映了 `k=1` 的路由决策。

### 路由决策可视化

路由策略的基本区别在于它们如何将 token 分配给专家。像 Switch 路由器这样的“硬”路由机制做出离散选择，而“软”机制则创建加权混合。下面的图表展示了这种区别。

> 硬路由将 token 发送给一组离散的专家。软路由计算所有专家的加权平均值，创建混合输出。

### 比较负载分布

使用带噪声路由的一个主要原因是改善负载均衡。没有它，标准路由器可能会持续将大多数 token 发送给少数“热门”专家，导致其他专家训练不足。噪声鼓励分散，使负载更均匀地分布。

下表展示了标准 Top-2 路由器与带噪声 Top-2 路由器在 8 个专家之间 token 的分布。



![噪声门控对专家负载分布的影响](plots/7058-0.json)



> 与标准路由器相比，带噪声的 Top-k 路由器通常会带来更均匀的负载分布，而标准路由器可能会出现严重的不平衡问题，即一些专家接收了不成比例的 token 数量。

### 在 MoE 层中整合

这些模块化路由器可以轻松地嵌入 (embedding)到完整的 MoE 层中。MoE 层的职责是使用路由器的输出进行稀疏计算。这里是我们如何使用 `NoisyTopKRouter` 的一个框架。

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.router = NoisyTopKRouter(d_model, num_experts, top_k)
        # self.experts = nn.ModuleList([...]) # 专家网络列表
        # ...

    def forward(self, x):
        # 1. 获取路由分配
        # 索引, 分数, 掩码 = self.router(x)

        # 2. 执行分发/聚合操作
        # 这是一个复杂的步骤，涉及根据 'indices' 对 token 进行排列
        # 以便每个专家接收一批分配给它的 token。

        # 3. 并行计算专家输出
        # 专家输出 = ...

        # 4. 使用 'scores' 组合专家输出
        # 最终输出 = ...

        # 5. 使用 'mask' 计算并返回负载均衡损失
        # 负载均衡损失 = ...

        # 返回最终输出, 负载均衡损失
        pass
```

此实践练习表明，尽管策略不同，但它们的实现可以包含在清晰、模块化的接口中。路由器的选择是一个重要的设计决策，您现在不仅可以在理论上进行分析，还可以清晰地了解底层代码。下一章关于分布式训练将展示这些路由决策如何与系统级并行性相互作用，以实现大规模模型扩展。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控专家混合层和用于负载均衡的噪声门控机制，为噪声Top-k路由器奠定基础。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/v23/21-0990.html) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 23; Pages: 1-39; DOI: [10.5555/3545945.3545958](https://doi.org/10.5555/3545945.3545958)
  介绍了Switch Transformer架构，该架构使用简化的k=1路由策略，与Switch路由器实现直接相关。
- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch所有神经网络模块的基类官方文档，对于实现示例中所示的自定义路由策略至关重要。
- [Designing and Training Sparse Mixture-of-Experts Models for Language](https://ai.googleblog.com/2023/07/designing-and-training-sparse-mixture.html) — Elena Albarran, William Fedus, Andrew M. Dai, Sharan Narang, Noam Shazeer (2023)
  Publisher: Google AI Blog
  提供了专家混合模型的概述，包括路由机制和负载均衡的重要性。

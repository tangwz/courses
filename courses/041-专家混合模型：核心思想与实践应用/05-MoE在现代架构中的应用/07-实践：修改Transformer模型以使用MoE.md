# 实践：修改Transformer模型以使用MoE

来源：[原文](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-5-integrating-moe-into-architectures/practice-modifying-transformer-moe)

[返回章节目录](README.md) · [返回课程目录](../README.md)

你将修改一个标准的Transformer编码器块，用专家混合（MoE）层替换其密集的馈送网络（FFN）。这是将MoE集成到现有架构中的常见做法，直接体现了在增加模型参数 (parameter)的同时控制计算负担的原理。

我们将分四步进行：

1. 查看标准Transformer编码器块以找出FFN部分。
2. 实现一个完整的MoE层，包括专家、门控网络和路由逻辑。
3. 构建一个新的`MoETransformerEncoderLayer`，用我们的MoE层替换FFN。
4. 检查训练循环所需的调整，以加入辅助负载平衡损失。

本次练习中，我们将使用PyTorch。假定你对`torch.nn.Module`有较好的掌握。

### 第一步：标准Transformer馈送网络

首先，我们来看Transformer块中一个典型的馈送网络。它通常由两个线性层组成，中间夹着一个非线性激活函数 (activation function)。该FFN在自注意力 (self-attention)机制 (attention mechanism)之后独立地应用于每个token表示。

以下是`TransformerEncoderLayer`中FFN部分的简化实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return x
```

这个`StandardFFN`模块是我们要替换的目标。它是一个密集操作；每个输入token都通过`linear1`和`linear2`中相同的权重 (weight)集进行处理。

### 第二步：实现专家混合层

现在，我们将构建MoE层，它将作为`StandardFFN`的稀疏替代。我们的实现将采用top-2门控，这意味着每个token将被路由到两个专家。这一设计遵循了第2章中讨论的原理。

MoE层需要三个主要部分：

- **专家模块**：独立FFN的集合。
- **门控网络**：一个小型网络，通常是一个单线性层，它决定哪些专家应该处理每个token。
- **分派/组合逻辑**：一种将token发送到其指定专家并汇集结果的机制。

```python
class Expert(nn.Module):
    """一个简单的馈送网络专家。"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k

        # N个专家网络的列表
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

        # 门控网络
        self.gate = nn.Linear(d_model, n_experts)

        # 辅助损失系数
        self.aux_loss_coef = 0.01

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model) # 重塑为 (batch_size * seq_len, d_model)
        num_tokens = x_flat.shape[0]

        # 1. 获取门控分数并选择top-k专家
        gate_logits = self.gate(x_flat)
        gate_scores = F.softmax(gate_logits, dim=-1)

        # 为每个token找到top-k专家
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.k, dim=-1)

        # 将top-k分数归一化，使其总和为1
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        # 2. 计算辅助负载平衡损失
        # 此损失鼓励门控网络均匀分配token
        tokens_per_expert = F.one_hot(top_k_indices, self.n_experts).sum(0)
        load_balancing_loss = self.aux_loss_coef * (self.n_experts / num_tokens) * torch.sum(tokens_per_expert * gate_scores.mean(0))

        # 3. 将token分派给专家并组合结果
        final_output = torch.zeros_like(x_flat)

        # 创建一个扁平索引以实现高效的专家处理
        flat_top_k_indices = top_k_k_indices.flatten()

        # 创建一个组合批次，供所有专家并行处理
        # 这是一种简化；实际系统使用更复杂的分派方式
        # 这里为了清晰起见使用循环，但可以进行优化。
        for i in range(self.n_experts):
            # 找出哪些token被路由到此专家
            token_indices_for_expert = torch.where(top_k_indices == i)[0]

            if token_indices_for_expert.numel() > 0:
                # 获取这些token的门控分数
                gating_values = gate_scores[token_indices_for_expert, i]

                # 用专家处理token
                expert_output = self.experts[i](x_flat[token_indices_for_expert])

                # 根据其门控分数加权专家输出
                final_output[token_indices_for_expert] += expert_output * gating_values.unsqueeze(-1)

        return final_output.view(batch_size, seq_len, -1), load_balancing_loss
```

> **注意**：上述代码中的分派逻辑为了清晰使用了一个循环。在第3章中讨论的高性能实现中，这种token到专家的分派是一个高度优化的操作，通常由自定义CUDA核或专用库处理，以避免显式循环并尽量减少数据混洗。

### 第三步：将MoE层集成到Transformer块中

定义好`MoELayer`后，我们现在可以创建一个新的`MoETransformerEncoderLayer`。其结构与标准层相似，但它使用`MoELayer`而不是`StandardFFN`。一个重要区别是，它的`forward`方法现在还必须返回MoE层产生的辅助损失。

> 比较标准FFN块与稀疏MoE块的图示。在MoE版本中，门控网络选择一部分专家（在此示例中，专家2处于活跃状态）来处理输入，而其他专家（专家1、专家N）则保持非活跃。

以下是启用MoE的编码器层的实现：

```python
class MoETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, n_experts: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 用我们的MoE层替换标准FFN
        self.moe_layer = MoELayer(d_model=d_model, d_ff=d_ff, n_experts=n_experts, k=2)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None):
        # 自注意力块
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # MoE块
        moe_output, aux_loss = self.moe_layer(src)
        src = src + self.dropout2(moe_output)
        src = self.norm2(src)

        return src, aux_loss
```

请注意，`forward`方法现在返回两个值：最终处理后的张量`src`和来自MoE层的`aux_loss`。这是一个重要的修改。

### 第四步：调整训练循环

使用`MoETransformerEncoderLayer`构建的模型将在每个MoE层产生一个辅助损失。这些损失在训练期间必须收集并加到主要任务损失（例如交叉熵）中。这有助于模型在执行主要任务的同时，学习有效的路由策略。

典型的训练循环需要更新以处理此情况。

```python
# --- 假设模型、数据加载器、优化器和损失函数已定义 ---

# model = nn.TransformerEncoder(
#     MoETransformerEncoderLayer(d_model=512, nhead=8, d_ff=2048, n_experts=8),
#     num_layers=6
# )

for data, targets in dataloader:
    optimizer.zero_grad()

    # 模型的正向传播现在返回输出和辅助损失的列表/元组
    # 我们需要正确处理堆叠编码器层的输出
    # 假设nn.TransformerEncoder有一个收集辅助损失的封装

    # --- 正向传播的简化表示 ---
    # 在实际实现中，你需要迭代各层
    # 或者让模型在内部累积损失。

    # 此训练示例的简化模型定义：
    class MoETransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                MoETransformerEncoderLayer(d_model=512, nhead=8, d_ff=2048, n_experts=8)
                for _ in range(6)
            ])
        def forward(self, x):
            total_aux_loss = 0.0
            for layer in self.layers:
                x, aux_loss = layer(x)
                total_aux_loss += aux_loss
            return x, total_aux_loss

    # model = MoETransformer()
    # output, total_aux_loss = model(data)

    # 假设`output`和`total_aux_loss`被正确获取
    # output, total_aux_loss = model(data) 
    # 对于一个简单的单层示例：
    layer = MoETransformerEncoderLayer(d_model=512, nhead=8, d_ff=2048, n_experts=8)
    output, aux_loss = layer(data)

    # 1. 计算主要任务损失
    main_loss = criterion(output.view(-1, vocab_size), targets.view(-1))

    # 2. 与MoE层的辅助损失合并
    # aux_loss已在我们的MoELayer中按其系数缩放
    total_loss = main_loss + aux_loss

    # 反向传播合并后的损失
    total_loss.backward()
    optimizer.step()
```

通过将`aux_loss`加到`main_loss`中，我们创建了一个梯度信号，它同时优化模型以实现其主要目标并平衡其专家之间的路由。如果没有这个辅助损失，门控网络很可能会收敛到一个总是选择相同少数专家的状态，导致第1章中讨论的“专家崩溃”问题。

这次动手修改完成了从密集FFN到Transformer中功能性稀疏MoE块的过程。你现在已经实现了核心的架构变化，这使得MoE模型能够将其参数 (parameter)数量扩展到远远超出其密集对应模型，这是后续章节中大规模训练和推理 (inference)技术得以发展的基础。

## 参考资料

- [Sparsely-Gated Mixture-of-Experts Layers](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了深度学习中的混合专家（MoE）架构，包括门控机制和用于专家之间负载均衡的辅助损失。
- [\`torch.nn\` - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch Core Team (2024)
  Publisher: PyTorch
  PyTorch `torch.nn` 模块的官方文档，这是在PyTorch中构建神经网络组件的基础，如练习中所示。

---

[上一节](06-%E5%88%86%E6%9E%90%E5%8F%82%E6%95%B0%E4%B8%8EFLOPs%E7%9A%84%E6%9D%83%E8%A1%A1.md)

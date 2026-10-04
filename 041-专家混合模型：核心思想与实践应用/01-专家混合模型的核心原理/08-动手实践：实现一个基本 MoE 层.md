# 动手实践：实现一个基本 MoE 层

来源：[原文](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-1-foundations-of-moe-models/hands-on-basic-moe-layer)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将混合专家层（Mixture of Experts layer）的架构转化为一个实用的 PyTorch 实现，需要构建一个包含门控和专家逻辑的独立 `nn.Module`。对于这个初步的实现，我们将侧重于一种简单而有效的 `top-1` 路由策略，即每个 token 被路由到一个最合适的专家。这为我们在后续章节中处理更复杂的 `top-k` 和开关式机制打下了良好开端。

我们将使用三个主要部分来构建 MoE 层：

1. **专家模块：** 一个标准的、处理 token 的前馈网络。
2. **门控网络：** 一个小型网络，用于决定哪个专家应处理每个 token。
3. **MoE 层：** 负责协调路由并组合输出的主类。

### 专家模块

在大多数基于 Transformer 的 MoE 模型中，"专家"就是简单的前馈网络（FFN）。每个专家在结构上相同，但由于接收到的数据有特殊性，它们在训练过程中会学习到不同的功能。让我们定义一个简单的 `Expert` 模块。它包含两个线性层，中间夹着一个 GELU 激活函数 (activation function)，这是现代 Transformer 中常见的模式。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    一个简单的前馈网络专家。
    它处理形状为 (..., d_model) 的输入张量，并返回相同形状的输出。
    """
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

这个 `Expert` 类是一个标准的构建块。MoE 层有趣的部分并非专家本身，而是模型如何将信息路由到它们那里。

### 完整的 MoE 层

现在我们将组装主 `MoELayer` 类。这个类将包含门控网络和专家模块列表。`forward` 方法的实现包含了 token 分派的核心逻辑。

前向传播的流程如下：

1. 获取 token 嵌入 (embedding)的输入张量，通常形状为 `(batch_size, sequence_length, d_model)`。
2. 将输入重塑为 `(batch_size * sequence_length, d_model)`，以便独立处理每个 token。
3. 将展平的 token 通过一个线性层（门控）以获取每个专家的 logits。
4. 对 logits 应用 `softmax` 函数以得到门控权重 (weight) $g(x)$。
5. 对于我们的 `top-1` 路由器，为每个 token 找出得分最高的专家。
6. 将每个 token 分派到其指定的专家。
7. 收集专家们的输出并将它们组合起来。在 `top-1` 路由中，输出就是所选专家的输出，并按其门控分数加权。
8. 将最终张量重塑回原始输入形状。

> `top-1` MoE 层的数据流程。Token 被传递到门控网络，门控网络选择一个专家进行处理。然后收集结果以形成最终输出。

这是 `MoELayer` 类的实现。请仔细注意 `forward` 方法中的注释，它们说明了分派机制。

```python
class MoELayer(nn.Module):
    """
    混合专家层。

    参数：
        d_model (int): 输入和输出的维度。
        num_experts (int): 专家总数。
        d_hidden (int): 每个专家 FFN 的隐藏层维度。
        top_k (int): 每个 token 路由到的专家数量。目前只支持 top_k=1。
    """
    def __init__(self, d_model: int, num_experts: int, d_hidden: int, top_k: int = 1):
        super().__init__()
        if top_k != 1:
            raise ValueError("This basic implementation only supports top_k=1")

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控网络
        self.gate = nn.Linear(d_model, num_experts)
        # 专家网络
        self.experts = nn.ModuleList([Expert(d_model, d_hidden) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE 层的前向传播。

        参数：
            x (torch.Tensor): 形状为 (batch_size, seq_len, d_model) 的输入张量

        返回：
            torch.Tensor: 形状为 (batch_size, seq_len, d_model) 的输出张量
        """
        batch_size, seq_len, d_model = x.shape

        # 重塑以进行 token 级别的处理
        x = x.view(-1, d_model) # (batch_size * seq_len, d_model)

        # 1. 获取门控 logits 和权重
        router_logits = self.gate(x)
        gating_weights = F.softmax(router_logits, dim=1)

        # 2. 为每个 token 选择 top-1 专家
        # topk 返回一个元组 (值, 索引)
        top_k_weights, top_k_indices = torch.topk(gating_weights, self.top_k, dim=1)

        # 对于 top-1，top_k_indices 形状为 (num_tokens, 1)。我们将其压缩。
        expert_indices = top_k_indices.squeeze(1)

        # 3. 创建一个最终输出张量，初始化为零
        final_output = torch.zeros_like(x)

        # 4. 将 token 分派到其选定的专家
        # 这是一种简单但性能不高的方法。
        # 实际中，分派和组合步骤经过了大量优化。
        for i in range(self.num_experts):
            # 找到所有路由到该专家的 token
            token_mask = (expert_indices == i)

            # 如果没有 token 路由，则继续
            if token_mask.sum() == 0:
                continue

            # 获取当前专家的 token
            selected_tokens = x[token_mask]

            # 通过专家处理 token
            expert_output = self.experts[i](selected_tokens)

            # 获取相应的门控权重以缩放输出
            # 对于 top-1，我们可以直接使用 top_k_weights，因为它形状为 (num_tokens, 1)
            # 我们选择路由到该专家的 token 的权重
            gating_scores = top_k_weights[token_mask]

            # 将缩放后的专家输出放回最终输出张量
            final_output[token_mask] = expert_output * gating_scores

        # 重塑回原始维度
        return final_output.view(batch_size, seq_len, d_model)
```

### 运行前向传播

让我们用一些模拟数据测试我们的实现，以确保它按预期运行。我们将实例化 `MoELayer` 并将一个随机张量传递给它。

```python
# Configuration
batch_size = 4
seq_len = 16
d_model = 128
num_experts = 8
d_hidden = 512 # Hidden dimension of each expert FFN

# Create a random input tensor
input_tensor = torch.randn(batch_size, seq_len, d_model)

# Instantiate the MoE layer
moe_layer = MoELayer(d_model=d_model, num_experts=num_experts, d_hidden=d_hidden, top_k=1)

# Perform a forward pass
output_tensor = moe_layer(input_tensor)

# Print shapes to verify
print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)

# Check that the output shape is correct
assert input_tensor.shape == output_tensor.shape
```

运行这段代码会产生以下输出，确认该层正确处理了输入并返回了相同维度的张量。

```text
Input shape: torch.Size([4, 16, 128])
Output shape: torch.Size([4, 16, 128])
```

### 分析和局限性

这个实现提供了一个可用的 `top-1` MoE 层，并演示了稀疏路由的基本机制。但有必要认识到它的局限性，这些局限性也突出了为生产环境构建此类模型的复杂性：

- **分派效率低：** 遍历每个专家的 `for` 循环易于理解，但效率极低。它不能很好地并行化专家计算，并引入了显著的开销。高性能的 MoE 实现使用优化的内核在 GPU 上高效执行这种分派和收集操作。
- **缺少辅助损失：** 我们尚未实现本章前面讨论的负载均衡辅助损失。`forward` 方法需要返回 `router_logits` 和 `gating_weights`，这样上层训练循环就可以计算此损失并将其添加到主任务损失中。这是稳定训练的一个重要组成部分。
- **仅限于 `top-1`：** 我们的代码是为 `top_k=1` 硬编码的。将其扩展到 `top_k > 1` 将需要一种更复杂的策略来组合多个专家为一个 token 提供的输出。

本次动手实践是一个可靠的起点。您现在拥有了一个可在此基础上构建的 MoE 层工作模型。在后续章节中，我们将通过考察高级路由策略、训练优化技术以及高效大规模部署的方法来解决这些局限性。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: arXiv preprint arXiv:1701.06538; DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控混合专家（MoE）层，详细阐述了其架构、top-k 路由以及负载均衡辅助损失。本文为讨论 MoE 层提供了基础。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Volume: 23; Pages: 1-39; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  讨论了现代 MoE 架构，包括本节中实现的 top-1 路由策略，以及将 MoE 模型扩展到大量参数的方法。它还详细阐述了高效实现和辅助损失的作用。
- [torch.nn - A powerful building block](https://pytorch.org/docs/stable/nn.html) — PyTorch Documentation (2024)
  PyTorch 官方 `torch.nn` 模块文档，该模块用于构建 `Expert` 和 `MoELayer` 等神经网络层和模型。提供了详细的 API 参考和使用示例。

---

[上一节](07-%E4%B8%8E%E5%AF%86%E9%9B%86%E6%A8%A1%E5%9E%8B%E6%89%A9%E5%B1%95%E7%9A%84%E6%AF%94%E8%BE%83.md) · [下一节](../02-%E8%BF%9B%E9%98%B6%E8%B7%AF%E7%94%B1%E6%9C%BA%E5%88%B6/01-Top-k%E9%97%A8%E6%8E%A7%E5%8F%8A%E5%85%B6%E5%8F%98%E4%BD%93%E7%9A%84%E5%88%86%E6%9E%90.md)

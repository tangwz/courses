---
course: "mixture-of-experts"
chapter: "advanced-moe-architectures"
lesson: "practice-custom-gating"
sourceId: 4690
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-2-advanced-moe-architectures/practice-custom-gating"
title: "动手实践：实现自定义门控机制"
description: "不同门控网络架构的代码实现。"
order: 6
plots: []
sourceHash: "bb5809ac3186792bc29afe3eb052bddce7c092d3c7510859c7cfd2eeeff8271c"
sourceCorrections: []
---

探讨各种门控网络设计（例如 Top-k 路由、噪声注入和架构变体）的实际实现。这里提供使用 PyTorch 构建自定义门控机制的实际示例。理解如何将这些门控设计理念转化为代码，对于构建和试验高级专家混合模型非常重要。

我们将实现三种常见类型的门控网络：

1. 使用线性层和 Softmax 的标准 **Top-k 门控** 网络。
2. **带噪声的 Top-k 门控** 网络，在选择前向路由器 logits 添加噪声。
3. 使用小型多层感知器 (MLP) 的简单**非线性门控**网络。

这些示例假设您熟悉 PyTorch 的基本知识。我们将特别关注门控模块本身，展示它如何处理输入 token 并生成路由决定（专家索引和权重 (weight)）。

### 设置

首先，让我们导入必要的库并定义一些在所有示例中都会用到的配置参数 (parameter)。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 配置 ---
model_dim = 512    # 输入 token 表示的维度
num_experts = 8    # 专家总数
top_k = 2          # 每个 token 路由到的专家数量
batch_size = 4     # 示例批次大小
seq_len = 10       # 示例序列长度

# 示例输入张量（批次大小，序列长度，模型维度）
input_tokens = torch.randn(batch_size, seq_len, model_dim)
```

### 1. 标准 Top-k 门控

这是最常见的门控机制。它使用单个线性层将输入 token 维度投影到专家数量。Softmax 用于获取概率，`torch.topk` 选择具有最高概率的专家。

```python
class StandardTopKGating(nn.Module):
    """
    标准 Top-k 门控网络。
    使用线性层和 softmax 计算专家分数，
    然后根据这些分数选择 top-k 专家。
    """
    def __init__(self, model_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # 用于将 token 嵌入映射到专家分数的线性层
        self.gate_proj = nn.Linear(self.model_dim, self.num_experts, bias=False)
        print(f"初始化标准 Top-k 门控：维度={model_dim}，专家数={num_experts}，TopK={top_k}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        门控网络的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为（批次大小，序列长度，模型维度）

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - combined_weights (torch.Tensor): 选定专家的路由权重，
                  形状为（批次大小 * 序列长度，top_k）。
                - expert_indices (torch.Tensor): 选定专家的索引，
                  形状为（批次大小 * 序列长度，top_k）。
                - raw_logits (torch.Tensor): 线性层输出的原始 logits，
                  形状为（批次大小 * 序列长度，专家数）。对辅助损失有用。
        """
        # 为线性层重塑输入：（B * S，D）
        original_shape = x.shape
        x = x.view(-1, self.model_dim) # 展平批次和序列维度

        # 将输入 token 投影到专家分数（logits）
        # 形状：（B * S，专家数）
        raw_logits = self.gate_proj(x)

        # 使用 torch.topk 获取 top-k 分数和索引
        # top_k_logits 形状：（B * S，top_k）
        # top_k_indices 形状：（B * S，top_k）
        top_k_logits, top_k_indices = torch.topk(raw_logits, self.top_k, dim=-1)

        # 对选定的 top-k logits 应用 softmax 以获得权重
        # 形状：（B * S，top_k）
        combined_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float)

        # 返回权重、索引和原始 logits（辅助损失可能需要）
        return combined_weights, top_k_indices, raw_logits

# --- 实例化和测试 ---
standard_gating = StandardTopKGating(model_dim, num_experts, top_k)
weights, indices, logits = standard_gating(input_tokens)

print("\n--- 标准 Top-k 门控输出 ---")
print("输入形状:", input_tokens.shape)
print("组合权重形状:", weights.shape)
print("专家索引形状:", indices.shape)
print("原始 Logits 形状:", logits.shape)
# 单个 token 的示例输出
print("示例权重（Token 0）:", weights[0])
print("示例索引（Token 0）:", indices[0])
```

输出 `combined_weights` 代表每个 token 针对每个选定专家的归一化 (normalization)重要性分数。`expert_indices` 告诉我们选择了*哪些*专家。`raw_logits` 通常用于计算辅助负载平衡损失，这将在下一章中介绍。

流程图展示了标准 Top-k 门控机制。输入 token 被投影，选择 Top-k logits 和索引，然后对选定的 logits 应用 softmax 以生成路由权重 (weight)。

### 2. 带噪声的 Top-k 门控

在 Top-k 选择前向门控 logits 添加噪声是一种在训练期间促进多样性并有时提高负载平衡或稳定性的方法。高斯噪声是常用的。

```python
class NoisyTopKGating(nn.Module):
    """
    带噪声的 Top-k 门控网络。
    在训练期间，在 Top-k 选择前向 logits 添加高斯噪声。
    """
    def __init__(self, model_dim: int, num_experts: int, top_k: int, noise_stddev=1.0):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_stddev = noise_stddev

        self.gate_proj = nn.Linear(self.model_dim, self.num_experts, bias=False)
        # 用于添加噪声的层，仅在训练期间应用
        self.noise_layer = nn.Linear(self.model_dim, self.num_experts, bias=False)

        print(f"初始化带噪声的 Top-k 门控：维度={model_dim}，专家数={num_experts}，TopK={top_k}，噪声标准差={noise_stddev}")

    def forward(self, x: torch.Tensor, is_training: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        带噪声门控网络的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为（批次大小，序列长度，模型维度）
            is_training (bool): 标志，指示模型是否处于训练模式。噪声仅在训练期间添加。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - combined_weights (torch.Tensor): 选定专家的路由权重。
                - expert_indices (torch.Tensor): 选定专家的索引。
                - raw_logits (torch.Tensor): 噪声添加*前*的原始 logits。
        """
        original_shape = x.shape
        x = x.view(-1, self.model_dim)

        # 获取基本 logits
        # 形状：（B * S，专家数）
        clean_logits = self.gate_proj(x)

        if is_training:
            # 计算噪声贡献
            # 我们使用一个单独的线性层来控制噪声幅度，并由标准正态噪声进行缩放
            # 形状：（B * S，专家数）
            noise_magnitude = self.noise_layer(x)
            # Softplus 确保幅度缩放为正
            noise_scale = F.softplus(noise_magnitude)
            # 采样标准高斯噪声
            # 形状：（B * S，专家数）
            sampled_noise = torch.randn_like(clean_logits) * self.noise_stddev
            # 将缩放后的噪声添加到干净的 logits
            noisy_logits = clean_logits + (noise_scale * sampled_noise)
        else:
            # 推理期间无噪声
            noisy_logits = clean_logits

        # 根据（可能带噪声的）logits 选择 top-k
        # top_k_logits 形状：（B * S，top_k）- 如果是训练模式，则来自*带噪声的* logits
        # top_k_indices 形状：（B * S，top_k）- 如果是训练模式，则来自*带噪声的* logits
        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)

        # 对选定的 top-k logits 应用 softmax 以获得权重
        # 形状：（B * S，top_k）
        combined_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float)

        # 返回权重、索引以及原始的*干净* logits，用于辅助损失计算
        return combined_weights, top_k_indices, clean_logits # 注意：返回 clean_logits

# --- 实例化和测试 ---
noisy_gating = NoisyTopKGating(model_dim, num_experts, top_k)

# 在训练模式下测试
weights_train, indices_train, logits_train = noisy_gating(input_tokens, is_training=True)
print("\n--- 带噪声的 Top-k 门控输出（训练）---")
print("输入形状:", input_tokens.shape)
print("权重形状（训练）:", weights_train.shape)
print("索引形状（训练）:", indices_train.shape)
print("Logits 形状（训练 - 干净）:", logits_train.shape)
print("示例索引（训练 - Token 0）:", indices_train[0]) # 可能因噪声而与标准结果不同

# 在推理模式下测试
weights_eval, indices_eval, logits_eval = noisy_gating(input_tokens, is_training=False)
print("\n--- 带噪声的 Top-k 门控输出（评估）---")
print("权重形状（评估）:", weights_eval.shape)
print("索引形状（评估）:", indices_eval.shape) # 如果权重相同，应与标准门控匹配
print("示例索引（评估 - Token 0）:", indices_eval[0])

# 检查评估索引是否与标准门控匹配（假设权重初始化相同）
# 注意：由于浮点精度问题，可能会出现微小差异。
# 在实际场景中，权重初始化将受到控制。
# print("评估索引与标准匹配吗？", torch.allclose(indices_eval, indices)) # 需要相同的权重初始化
```

请注意，噪声仅在训练期间添加（`is_training=True`）。在评估或推理 (inference)期间，行为恢复为基于干净 logits 的标准 Top-k 选择。返回*干净*的 logits 也很重要，以便可能用于辅助损失计算，因为这些 logits 反映了路由器在没有随机训练噪声情况下的潜在偏好。具体的噪声实现（例如，使用单独的可学习层 `noise_layer` 和 `softplus` 进行缩放）遵循了 Switch Transformer 等文献中常见的做法。

### 3. 非线性门控（MLP 路由器）

线性路由器很常见，但有时使用非线性的表达能力更强的路由器可以捕捉更复杂的路由模式。这是一个使用带有 ReLU 激活的两层 MLP 的简单示例。

```python
class NonLinearGating(nn.Module):
    """
    使用简单 MLP 的非线性 Top-k 门控网络。
    """
    def __init__(self, model_dim: int, num_experts: int, top_k: int, hidden_dim_multiplier=2):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = model_dim * hidden_dim_multiplier

        # 简单 MLP：线性 -> ReLU -> 线性
        self.mlp = nn.Sequential(
            nn.Linear(self.model_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        )
        print(f"初始化非线性门控：维度={model_dim}，专家数={num_experts}，TopK={top_k}，隐藏层={self.hidden_dim}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        非线性门控网络的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为（批次大小，序列长度，模型维度）

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - combined_weights (torch.Tensor): 选定专家的路由权重。
                - expert_indices (torch.Tensor): 选定专家的索引。
                - raw_logits (torch.Tensor): MLP 输出的原始 logits。
        """
        original_shape = x.shape
        x = x.view(-1, self.model_dim)

        # 从 MLP 获取 logits
        # 形状：（B * S，专家数）
        raw_logits = self.mlp(x)

        # 获取 top-k 分数和索引
        # top_k_logits 形状：（B * S，top_k）
        # top_k_indices 形状：（B * S，top_k）
        top_k_logits, top_k_indices = torch.topk(raw_logits, self.top_k, dim=-1)

        # 对选定的 top-k logits 应用 softmax
        # 形状：（B * S，top_k）
        combined_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float)

        # 返回权重、索引和原始 logits
        return combined_weights, top_k_indices, raw_logits

# --- 实例化和测试 ---
nonlinear_gating = NonLinearGating(model_dim, num_experts, top_k)
weights_nl, indices_nl, logits_nl = nonlinear_gating(input_tokens)

print("\n--- 非线性（MLP）门控输出 ---")
print("输入形状:", input_tokens.shape)
print("权重形状:", weights_nl.shape)
print("索引形状:", indices_nl.shape)
print("Logits 形状:", logits_nl.shape)
print("示例索引（Token 0）:", indices_nl[0])
```

与简单的线性路由器相比，这种 MLP 路由器引入了更多的参数 (parameter)和计算。线性路由器和非线性路由器之间的选择取决于具体的任务和数据集，通常需要经验验证。更复杂的路由器架构，例如那些包含注意力机制 (attention mechanism)的架构，也可以遵循类似的原则进行实现。

### 使用门控输出

任何门控机制的主要输出是 `combined_weights` 和 `expert_indices`。这些在完整的 MoE 层中使用，以组合选定专家的输出。虽然完整的 MoE 层实现未在本节中介绍，但主要思想是：

1. 使用 `expert_indices` 标识哪些专家需要处理哪些 token。这在分布式设置中通常涉及复杂的调度逻辑（第 4 章介绍）。
2. 将相关 token 通过其分配的专家。
3. 使用 `combined_weights` 对每个 token 的选定专家输出进行加权求和。

例如：

```python
# 用法（简化，假设专家输出已收集）
# 假设 'expert_outputs' 是一个张量，其中 expert_outputs[i] 是
# 第 i 个 token 从其分配的*某个*专家处获得的输出。
# 完整实现需要处理每个 token 的多个专家并收集结果。

# 针对分配给专家 'e1' 和 'e2' 的单个 token 't' 的简化组合
# 权重分别为 'w1' 和 'w2'：
# final_output_t = w1 * expert_output_t_e1 + w2 * expert_output_t_e2
```

### 注意事项

- **计算成本：** 非线性路由器的计算成本高于线性路由器。带噪声的路由器在训练期间增加的开销很小。
- **表达能力与稳定性：** 更复杂的路由器可能提供更好的专业化能力，但有时更难稳定训练。前面讨论过的方法，例如路由器稳定性方法，变得重要。
- **与负载平衡的结合：** `raw_logits` 输出对于实现辅助损失函数 (loss function)（第 3 章）非常重要，这些函数旨在防止专家负载不平衡，这是 MoE 训练中常见的问题。
- **Top-k 值：** `k` 的选择（每个 token 的专家数量）影响计算负载和模型容量。`k=1` 或 `k=2` 是常见的起始点。

本次动手练习说明了如何实现不同的门控策略。通过修改这些模块，您可以试验本章中讨论的各种架构思想，根据 MoE 模型的具体需求调整路由机制。下一章将侧重于训练动态，特别是如何使用 `raw_logits` 等输出来确保训练的稳定和平衡。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了带有top-k路由和可训练门控网络的稀疏门控专家混合层，为许多现代专家混合架构奠定了基础。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://jmlr.org/papers/v23/21-0998.html) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research (JMLR); Volume: 23; Pages: 1-39; DOI: [10.5555/3540209.3540212](https://doi.org/10.5555/3540209.3540212)
  详细阐述了嘈杂Top-k门控机制，包括可学习噪声尺度的应用，这在所提供的实践示例中有所实现。
- [ST-MoE: Designing Stable and Transferable Sparse Mixture-of-Experts Models](https://arxiv.org/abs/2202.08906) — Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, William Fedus (2022)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2202.08906](https://doi.org/10.48550/arXiv.2202.08906)
  探讨了提高专家混合模型稳定性和可迁移性的方法，解决了此类架构训练和部署中的实际考量。

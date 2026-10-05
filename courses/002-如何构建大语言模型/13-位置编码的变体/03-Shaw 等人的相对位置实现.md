# Shaw 等人的相对位置实现

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-13-positional-encoding-variations/implementation-shaw-relative-position)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然绝对位置编码 (positional encoding)，例如正弦编码或学习型嵌入 (embedding)，为 Transformer 提供了必要的序列顺序信息，但它们独立处理每个位置。这可能会限制它们对训练中未见过的序列长度的泛化能力，并且没有明确地根据标记 (token)之间的距离来表示它们的关系。Shaw 等人（2018）在“带相对位置表示的自注意力 (self-attention)”中提出的方法，通过将相对距离直接融入注意力机制 (attention mechanism)本身，提供了一种替代方案。

核心思想是修改注意力分数计算和潜在的价值向量 (vector)聚合方式，使其对交互标记之间的相对偏移量敏感。这种方法不是仅将位置信息添加到初始嵌入中，而是引入了可学习的嵌入，这些嵌入表示不同的相对距离。

### 用相对嵌入 (embedding)修改注意力分数

回顾位置 $i$ 的查询向量 (vector) ($q_i = x_i W^Q$) 和位置 $j$ 的键向量 ($k_j = x_j W^K$) 之间的标准缩放点积注意力分数：


$$
\text{分数}(q_i, k_j) = \frac{q_i k_j^T}{\sqrt{d_k}}
$$


Shaw 等人引入了可学习的键相对位置嵌入，表示为 $a^K$。这些嵌入捕捉了从位置 $i$ 到位置 $j$ 的关系。具体来说，$a^K_{ij}$ 表示相对距离 $j-i$ 的嵌入。然后修改注意力分数计算，使其包含一个纳入了此相对位置信息的项：


$$
e_{ij} = \frac{(x_i W^Q)(x_j W^K)^T + (x_i W^Q)(a^K_{ij})^T}{\sqrt{d_k}}
$$


我们来分解这个修改后的分数 $e_{ij}$：

1. **基于内容的项：** $(x_i W^Q)(x_j W^K)^T$ 是位置 $i$ 的查询和位置 $j$ 的键之间的标准点积。它根据标记 (token)表示本身衡量兼容性。
2. **基于位置的项：** $(x_i W^Q)(a^K_{ij})^T$ 引入了基于相对位置的偏差。查询 $x_i W^Q$ 与对应于偏移量 $j-i$ 的学习型嵌入 $a^K_{ij}$ 发生作用。此项使得模型能够纯粹根据位置 $i$ 和位置 $j$ 的相对距离，而不考虑位置 $j$ 的实际内容，来学习位置 $i$ 应该对位置 $j$ 给予多少注意力。

最终的注意力权重 (weight) $\alpha_{ij}$ 是通过对这些修改后的分数 $e_{ij}$ 应用 softmax 函数获得的。

> Shaw 等人方法中的注意力分数计算，同时包含了内容相似性以及源自相对位置嵌入的偏差。

### 裁剪相对距离

在非常长的序列中，为每个可能的相对距离计算和存储独特的嵌入 (embedding)将是低效且可能不必要的。相距很远的标记 (token)之间的关系可能信息量较少或遵循一般模式。因此，裁剪考虑的最大相对距离是常见做法。

选择一个最大距离 $k$。任何 $|j-i| > k$ 的相对距离 $j-i$ 都将被裁剪到 $-k$ 或 $k$。例如，如果 $k=8$，相对距离 $j-i = 10$ 将被视为 $8$，而 $j-i = -12$ 将被视为 $-8$。这意味着模型只需要学习范围 $[-k, k]$ 内的相对距离嵌入，从而得到 $2k+1$ 个独特的相对位置嵌入。

### 实现考量

在实际的 PyTorch 实现中，你通常会：

1. **定义相对位置嵌入 (embedding)层：** 创建一个 `nn.Embedding` 层来存储可学习的嵌入 $a^K$。其大小将是 `(2 * max_relative_position + 1, head_dim)`。

   ```python
   import torch
   import torch.nn as nn

   max_relative_position = 8 # 示例最大距离
   head_dim = 64 # 示例每个注意力头的维度
   # 我们需要从 -k 到 +k 的距离嵌入，总共 2k+1 个。
   num_relative_embeddings = 2 * max_relative_position + 1

   relative_key_embeddings = nn.Embedding(num_relative_embeddings, head_dim)
   ```
2. **计算相对位置：** 对于给定的序列长度 `seq_len`，计算所有查询 ($i$) 和键 ($j$) 位置之间的相对位置矩阵。

   ```python
   seq_len = 512 # 示例序列长度
   range_vec = torch.arange(seq_len)
   relative_pos_matrix = range_vec[None, :] - range_vec[:, None] # 形状: [seq_len, seq_len]
   ```
3. **裁剪并将位置映射到索引：** 裁剪相对位置并将它们转换为适合嵌入查找的非负索引。

   ```python
   clipped_relative_pos = torch.clamp(relative_pos_matrix,
                                      -max_relative_position,
                                      max_relative_position)
   # 将索引偏移到 0 到 2k
   embedding_indices = clipped_relative_pos + max_relative_position
   ```
4. **查找嵌入：** 检索相应的 $a^K_{ij}$ 嵌入。

   ```python
   # 形状: [seq_len, seq_len, head_dim]
   rel_key_embeds = relative_key_embeddings(embedding_indices)
   ```
5. **计算相对注意力项：** 计算 $(x_i W^Q)(a^K_{ij})^T$ 项。这需要仔细的张量操作，以便对每个查询 $q_i$ 和所有 $j$ 对应的相对键嵌入 $a^K_{ij}$ 执行点积。

   ```python
   # 假设查询 'q' 的形状为 [batch_size, num_heads, seq_len, head_dim]
   # 假设 rel_key_embeds 的形状为 [seq_len, seq_len, head_dim]

   # 我们需要计算 torch.einsum('bhqd, Lqd -> bhqL', q, rel_key_embeds)
   # 或类似的有效计算。这部分需要仔细实现。
   # 使用重塑/置换后的矩阵乘法的示例：
   # queries_r: [batch_size * num_heads, seq_len, head_dim]
   # rel_key_embeds_r: [head_dim, seq_len, seq_len] (置换后) -> 需要仔细索引
   # relative_logits = torch.matmul(queries_r, rel_key_embeds_r) # 简化概念

   # 一种常见的有效方法涉及重塑查询和嵌入
   # 并使用批量矩阵乘法 (bmm)，这在 T5 或 Tensor2Tensor 等实现中常被称为“倾斜”逻辑
   # in implementations like T5 or Tensor2Tensor.

   # 计算出的相对项的占位符 (形状: [batch, heads, seq_len, seq_len])
   relative_logits = torch.zeros(q.shape[0], q.shape[1], seq_len, seq_len, device=q.device)
   # --- 实际计算逻辑在此处 ---
   ```

   *注意：高效计算相对注意力项的具体实现可能很复杂，通常涉及库实现或原始论文中详细说明的特定重塑和矩阵乘法策略。目标是计算每个查询 $q_i$ 与所有相关 $a^K_{ij}$ 向量 (vector)之间的点积。*
6. **组合分数：** 在缩放并应用 softmax 之前，将 `relative_logits` 添加到 `content_logits`。

   ```python
   # 假设 content_logits 的形状为 [batch, heads, seq_len, seq_len]
   # content_logits = torch.matmul(q, k.transpose(-2, -1)) # 标准内容注意力

   combined_logits = content_logits + relative_logits
   attention_scores = combined_logits / (head_dim ** 0.5)
   attention_weights = F.softmax(attention_scores, dim=-1)
   ```

### 修改价值聚合（不常见）

Shaw 等人还提出，在聚合价值向量 (vector) $v_j = x_j W^V$ 时，添加一个类似的相对位置嵌入 (embedding)项 $a^V_{ij}$。位置 $i$ 的输出 $z_i$ 将变为：


$$
z_i = \sum_j \alpha_{ij} (x_j W^V + a^V_{ij})
$$


与重要的嵌入修改相比，这项修改提及较少，但它遵循相同的原则：使输出对被注意力关注标记 (token)的相对位置敏感。它需要一个单独的嵌入层 `relative_value_embeddings`。

### 优点和缺点

**优点：**

- **更好的泛化能力：** 通过关注相对距离，与可能无法很好外推的绝对位置编码 (positional encoding)相比，模型可能对训练中未遇到的序列长度表现出更好的泛化能力。
- **明确的相对信息：** 将成对的位置关系直接编码到注意力机制 (attention mechanism)中。

**缺点：**

- **增加复杂性：** 修改了核心注意力计算，增加了计算步骤，并可能增加延迟。
- **更多参数 (parameter)：** 为相对位置嵌入 (embedding)引入了新的可学习参数 ($a^K$ 和潜在的 $a^V$)。
- **实现细节：** 高效计算相对注意力项需要仔细的张量操作。

此方法是在 Transformer 注意力机制中直接考虑相对位置的一个重要进展。虽然有效，但它是多种方法之一，后续章节将讨论其他方案，例如 Transformer-XL 中使用的相对编码方案和旋转位置嵌入 (RoPE)，它们通过不同的机制实现类似的目标。

## 参考资料

- [Self-Attention with Relative Position Representations](https://aclanthology.org/N18-2074/) — Peter Shaw, Jakob Uszkoreit, Ashish Vaswani (2018)
  Journal: Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers); Publisher: Association for Computational Linguistics; Pages: 464-468; DOI: [10.18653/v1/N18-2074](https://doi.org/10.18653/v1/N18-2074)
  介绍将相对位置表示引入Transformer自注意力机制的开创性论文，详细说明了注意力分数和值聚合的修改。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构的原始论文，为所有后续位置编码变体提供了基础背景。
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu (2019)
  Journal: Journal of Machine Learning Research; Volume: 21; Pages: 1-67; DOI: [10.48550/arXiv.1910.10683](https://doi.org/10.48550/arXiv.1910.10683)
  描述了T5模型，该模型采用了一种不同但相关的相对位置编码方案，为大型模型中的实现提供了实用见解。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  解释Transformer架构的教科书章节，包含对各种位置编码技术（如相对位置嵌入）的讨论。第10章涵盖Transformer和大型语言模型。

---

[上一节](02-%E7%9B%B8%E5%AF%B9%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E7%9A%84%E5%8E%9F%E7%90%86.md) · [下一节](04-Transformer-XL%20%E7%9B%B8%E5%AF%B9%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)

# 预归一化与后归一化 (预LN与后LN)

来源：[原文](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/pre-ln-vs-post-ln)

[返回章节目录](README.md) · [返回课程目录](../README.md)

层归一化 (normalization)（LN）是每个Transformer块中的基本组成部分，与自注意力 (self-attention)子层和前馈子层周围的残差连接一同应用。它的主要作用是通过对每个位置的激活值在特征维度上独立进行归一化，来稳定训练过程中隐藏状态的动态变化。这有助于保持激活尺度的一致性，平滑损失曲面，并通常改善梯度流动，使得训练具有更多层级的网络成为可能。

然而，层归一化步骤相对于残差连接的 *放置位置* 显著影响训练动态和稳定性。两种主要策略是后归一化（Post-LN），用于原始论文“Attention Is All You Need”中，以及预归一化（Pre-LN），因其增强的稳定性而受到欢迎。接下来分别介绍这两种方法。

### 后归一化 (normalization) (Post-LN)

在原始Transformer架构中，层归一化应用于子层（如多头注意力 (multi-head attention)或前馈网络）的输出通过残差连接加回到输入 *之后*。

后归一化块中子层的计算流程如下所示：

1. **子层计算：** $SubLayerOutput = SubLayer(x)$
2. **残差相加：** $Added = x + SubLayerOutput$
3. **层归一化：** $Output = LayerNorm(Added)$

> 后归一化Transformer块中的数据流向。归一化在残差相加之后进行。

**特点：**

- **原始形式：** 这是Vaswani等人（2017）论文中描述的设置。
- **潜在的不稳定性：** 后归一化面临的主要问题是，残差分支的输出（和 $x + SubLayer(x)$）在传递给下一层 *之前* 未经归一化。在深度网络中，激活值的幅度可能在层与层之间显著变化，这可能导致训练初期出现梯度爆炸或梯度消失。
- **预热要求：** 后归一化配置通常需要仔细的学习率预热阶段（从较小的学习率开始并逐渐增加）。如果没有预热，由于未经归一化的加法，初始梯度可能过大，导致训练发散。

### 预归一化 (normalization) (Pre-LN)

为解决后归一化的稳定性问题，预归一化方法被提出。在这里，层归一化应用于输入，在它进入子层模块 *之前*，但 *在* 残差分支内部。残差连接随后将原始的、未经修改的输入 $x$ 添加到子层的输出中。

预归一化块中子层的计算流程是：

1. **层归一化：** $Normalized\_x = LayerNorm(x)$
2. **子层计算：** $SubLayerOutput = SubLayer(Normalized\_x)$
3. **残差相加：** $Output = x + SubLayerOutput$

> 预归一化Transformer块中的数据流向。归一化在子层计算之前进行。

**特点：**

- **提高稳定性：** 通过对每个子层的 *输入* 进行归一化，预归一化防止了传递到这些可能复杂的函数中的激活值发生爆炸。通过残差连接 ($x$) 的输出梯度路径保持清晰，使得梯度能够更顺畅地流经深度网络，而不会被归一化层过度缩放。
- **对预热的敏感性降低：** 预归一化配置通常对学习率调度不那么敏感，即使没有特定的预热阶段，或者预热阶段很短，也能稳定训练。这简化了超参数 (parameter) (hyperparameter)的调整。
- **常见实践：** 由于其稳定性优势，预归一化已成为许多现代大规模Transformer实现（例如GPT-2、GPT-3、ViT）中的事实标准。与后归一化相比，它使得训练更深的模型成为可能。

### 比较与权衡

| 特点 | 后归一化 (normalization) (Post-LN) | 预归一化 (Pre-LN) |
| --- | --- | --- |
| **放置位置** | `LayerNorm(x + SubLayer(x))` | `x + SubLayer(LayerNorm(x))` |
| **稳定性** | 稳定性较差，尤其在深层模型中 | 稳定性更好，有助于训练更深的模型 |
| **预热** | 通常需要仔细的学习率预热 | 对学习率预热不那么敏感，常无需预热也能训练 |
| **梯度流动** | 梯度在相加后通过归一化层 | 梯度通过残差路径绕过归一化层 |
| **原始论文** | 是 | 否 (后续改进) |
| **现代应用** | 在非常大的模型中较少见 | 被广泛采用，尤其对于大型模型 |
| **最佳表现** | 经过大量调整有时能达到略好的最佳结果 | 通常更容易调整以获得良好、稳定的结果 |



[交互图表：训练损失比较](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/pre-ln-vs-post-ln#plot-chk1kx)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        0,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500
      ],
      "y": [
        7.5,
        6.8,
        6.1,
        5.5,
        5.0,
        4.6,
        4.3,
        4.1,
        4.0,
        3.9,
        3.85
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "预归一化 (稳定)",
      "line": {
        "color": "#228be6",
        "width": 2.5
      }
    },
    {
      "x": [
        0,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500
      ],
      "y": [
        12,
        15,
        18,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "后归一化 (无预热 - 发散)",
      "line": {
        "color": "#fa5252",
        "width": 2.5,
        "dash": "dash"
      }
    },
    {
      "x": [
        0,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500
      ],
      "y": [
        9.5,
        7.0,
        6.0,
        5.3,
        4.8,
        4.4,
        4.15,
        3.95,
        3.85,
        3.8,
        3.78
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "后归一化 (有预热)",
      "line": {
        "color": "#fd7e14",
        "width": 2.5
      }
    }
  ],
  "layout": {
    "title": {
      "text": "训练损失比较",
      "font": {
        "size": 16
      }
    },
    "xaxis": {
      "title": {
        "text": "训练步数"
      }
    },
    "yaxis": {
      "title": {
        "text": "训练损失"
      },
      "range": [
        3.5,
        16
      ]
    },
    "legend": {
      "yanchor": "top",
      "y": 0.99,
      "xanchor": "right",
      "x": 0.99,
      "font": {
        "size": 10
      }
    },
    "margin": {
      "l": 50,
      "r": 20,
      "t": 40,
      "b": 40
    },
    "template": "plotly_white"
  }
}
```

</details>



> 训练损失曲线。预归一化通常显示稳定收敛。无预热的后归一化可能会发散，而经过适当预热的后归一化可以良好收敛，有时能达到比预归一化略低的最终损失，但需要仔细调整。

### 总结

尽管原始Transformer使用了后归一化 (normalization)，但预归一化变体在实际应用中具有显著优势，包括训练稳定性提高以及对学习率调度等超参数 (parameter) (hyperparameter)选择的敏感性降低。通过在输入通过复杂的自注意力 (self-attention)和前馈层 *之前* 进行归一化，预归一化确保了更平滑的优化过程，这在将Transformer扩展到数十甚至数百层时尤为重要。基于这些原因，预归一化常是当代Transformer架构中的首选。但对两种配置的了解，能够帮助我们更好地理解这些强大模型的设计选择和训练过程。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了原始的Transformer架构，该架构采用了后置归一化。
- [A Survey of Transformer Architectures and Applications](https://doi.org/10.1109/ACCESS.2022.3195092) — Hafiz Tayyab, Muhammad Umair Khan, Asif Ali Laghari, Abdullah Khan (2022)
  Journal: IEEE Access; Publisher: IEEE; Volume: 10; Pages: 76620-76641; DOI: [10.1109/ACCESS.2022.3195092](https://doi.org/10.1109/ACCESS.2022.3195092)
  广泛概述了Transformer架构变体，包括对归一化位置及其对训练大型模型影响的讨论。

---

[上一节](07-%E7%9B%B8%E5%AF%B9%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md) · [下一节](09-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E7%BC%A9%E6%94%BE%E6%B3%95%E5%88%99.md)

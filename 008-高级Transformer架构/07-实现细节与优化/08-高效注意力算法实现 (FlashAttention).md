# 高效注意力算法实现 (FlashAttention)

来源：[原文](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/efficient-attention-flashattention)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Transformer的计算需求，尤其是自注意力 (self-attention)在序列长度$N$上的二次复杂度($O(N^2)$)，构成了一个主要瓶颈。优化方法可以直接解决这一挑战。尽管稀疏或线性注意力等架构变体旨在以较低的理论复杂度**逼近**注意力机制 (attention mechanism)，但重点在于优化**精确**的缩放点积注意力计算。这使其在实践中，特别是在现代GPU等硬件上，明显更快且内存效率更高。

标准注意力实现中的主要挑战不仅仅是浮点运算（FLOPs）的数量，而是内存带宽的限制。计算涉及几个大型中间矩阵，尤其值得注意的是$N \times N$的注意力得分矩阵$S = QK^T$和概率矩阵$P = \text{softmax}(S)$。这些矩阵必须从GPU的高带宽内存（HBM）中读写，而HBM比片上SRAM慢得多。对于长序列，在不同内存层级之间传输这些矩阵所花费的时间通常会主导实际的计算时间。

## I/O感知型注意力算法

认识到这一内存瓶颈，研究人员发展出了“I/O感知型”注意力算法。这些方法旨在最小化慢速HBM和快速SRAM之间的数据移动。最突出且被广泛采用的例子是**FlashAttention**。

### FlashAttention：融合算子与分块处理

FlashAttention不改变注意力的数学定义；它计算出的输出与标准算法完全相同。其创新之处在于它**如何**执行计算。核心思想包括：

1. **算子融合（Kernel Fusion）**：FlashAttention没有为矩阵乘法（$QK^T$）、缩放、遮掩（如适用）、softmax以及与$V$的最终乘法执行独立的GPU操作（算子），而是将这些操作融合到一个更大的单一算子中。这大幅减少了数据需要从HBM读取和写回的次数。
2. **分块处理（Tiling）**：融合后的算子以更小的块或“瓦片”处理输入矩阵（$Q$, $K$, $V$），这些块可以完全放入GPU的快速SRAM中。它通过遍历键和值的块来计算查询块的注意力输出。中间结果，例如注意力得分矩阵的块和softmax归一化 (normalization)统计数据，会尽可能地保留在SRAM中。
3. **在线Softmax（Online Softmax）**：softmax计算以数值稳定的方式逐块执行。当算法为给定查询块遍历键和值的块时，它会维护softmax所需的运行统计数据（用于减法的最大值，用于归一化的指数和）。这避免了在应用softmax之前计算和存储完整的$N \times N$得分矩阵$S$。

> 内存访问模式对比。标准注意力涉及对较慢的HBM进行多次读/写操作，以处理中间矩阵（$S$, $P$）。FlashAttention对加载到较快SRAM中的数据块执行融合操作，从而最小化HBM访问。

### 优势与影响

这种I/O感知型方法带来了显著的成果：

- **加速：** 与标准的PyTorch或TensorFlow实现相比，FlashAttention可以提供明显的加速（通常是2-4倍或更多），特别是对于内存带宽是主要限制的长序列。
- **内存效率：** 由于大型$N \times N$中间矩阵（$S$和$P$）不会在HBM中完全实例化，注意力的内存使用量相对于序列长度变为线性$O(N)$，而不是二次方$O(N^2)$（不包括$Q, K, V$本身的内存）。这使得在典型GPU内存限制内训练比以前长得多的序列模型成为可能。
- **精确性：** 与近似方法不同，FlashAttention计算出精确的注意力输出，确保不会因优化而导致模型质量下降。
- **易于集成：** 像官方FlashAttention库这样的实现通常被设计为PyTorch等框架中标准注意力模块的直接替代品，只需要很少的代码改动。



[交互图表：性能：注意力实现](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/efficient-attention-flashattention#plot-1z0kw2g)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "性能：注意力实现",
    "xaxis": {
      "title": "序列长度 (N)"
    },
    "yaxis": {
      "title": "执行时间 / 内存",
      "type": "log"
    },
    "legend": {
      "title": "实现方式"
    },
    "template": "plotly_white",
    "font": {
      "family": "Arial"
    }
  },
  "data": [
    {
      "name": "标准注意力 (时间/内存)",
      "x": [
        512,
        1024,
        2048,
        4096,
        8192
      ],
      "y": [
        1,
        4,
        16,
        64,
        256
      ],
      "mode": "lines+markers",
      "line": {
        "color": "#fa5252",
        "width": 3
      },
      "marker": {
        "symbol": "circle",
        "size": 8
      }
    },
    {
      "name": "FlashAttention (时间)",
      "x": [
        512,
        1024,
        2048,
        4096,
        8192
      ],
      "y": [
        0.5,
        1.5,
        5,
        15,
        40
      ],
      "mode": "lines+markers",
      "line": {
        "color": "#1c7ed6",
        "dash": "dash",
        "width": 3
      },
      "marker": {
        "symbol": "square",
        "size": 8
      }
    },
    {
      "name": "FlashAttention (内存)",
      "x": [
        512,
        1024,
        2048,
        4096,
        8192
      ],
      "y": [
        0.8,
        1.6,
        3.2,
        6.4,
        12.8
      ],
      "mode": "lines+markers",
      "line": {
        "color": "#37b24d",
        "dash": "dot",
        "width": 3
      },
      "marker": {
        "symbol": "diamond",
        "size": 8
      }
    }
  ]
}
```

</details>



> 图表说明性能扩展。标准注意力在时间和内存方面呈现二次方扩展($O(N^2)$)，主要由中间矩阵决定。FlashAttention的目标是接近线性的时间扩展（更接近计算限制）和线性的内存扩展($O(N)$)。实际加速效果因硬件和维度而异。

尽管FlashAttention是一个具体的实现，但算子融合和最小化HBM流量的原理对于优化现代硬件上的计算密集型操作非常重要。在训练或部署大型Transformer模型时，特别是处理长上下文 (context)的模型，运用此类优化的注意力实现通常是达到可接受性能和效率的必不可少。许多流行的库和框架正越来越多地直接或通过集成来整合这些技术。

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Aware Operators](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2022); DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  介绍FlashAttention的基础论文，详细阐述了其内核融合、分块和在线softmax等核心技术，以实现精确且I/O感知的注意力优化。
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Tri Dao (2023)
  Journal: arXiv preprint arXiv:2307.08691; DOI: [10.48550/arXiv.2307.08691](https://doi.org/10.48550/arXiv.2307.08691)
  描述了FlashAttention-2的改进，侧重于增强并行度和工作分区，以在现代GPU上实现更快的速度和更高的内存效率。
- [FlashAttention: Faster and More Efficient Attention for Transformers](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)
  Journal: arXiv; Publisher: arXiv; DOI: [2205.14135](https://doi.org/2205.14135)
  原始作者撰写的易于理解的解释，详细阐述了FlashAttention内存和速度优化的实际应用以及底层硬件原理。

---

[上一节](07-%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83.md) · [下一节](09-%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C%E7%AD%96%E7%95%A5.md)

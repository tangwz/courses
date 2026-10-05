# Top-k门控及其变体的分析

来源：[原文](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-2-advanced-routing-mechanisms/analysis-top-k-gating)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Top-k门控是稀疏专家混合模型中用于路由令牌的标准方式。它的设计简单明了且作用良好，构成了许多成功MoE架构的根基。门控网络，一个简单的线性层，为每个专家生成一个对数，表示该专家处理当前输入令牌的偏好。`TopK`函数随后选择对数分数最高的`k`个专家。

### Top-k门控的运作方式

对于给定的输入令牌表示$x$，该流程始于由权重 (weight)矩阵$W_g$参数 (parameter)化的门控网络，该网络计算对数$h(x)$：


$$
h(x) = W_g \cdot x
$$


这里，$h(x)$是一个向量 (vector)，其中每个元素$h(x)_i$对应于专家$i$的对数。`TopK`操作选择$h(x)$中`k`个最高值的索引。我们将这组索引称为$S$。

最终的门控分数$G(x)$通常通过仅对选定对数应用softmax函数来计算。这保证了所选专家的权重总和为一。


$$
G(x)_i = \begin{cases} \frac{e^{h(x)_i}}{\sum_{j \in S} e^{h(x)_j}} & \text{如果 } i \in S \\ 0 & \text{如果 } i \notin S \end{cases}
$$


MoE层的输出$y$是选定专家输出的加权和：


$$
y = \sum_{i \in S} G(x)_i \cdot E_i(x)
$$


超参数 (hyperparameter)$k$是一个重要的设计选择。使用$k=2$使得每个令牌可以由两个专家处理，这提供了比路由到一个专家更丰富的表示。这可以在增加计算成本的情况下提升模型质量。相比之下，像Switch Transformers所推广的那样，将$k=1$可以最大限度地减少计算和通信开销。

> 单个令牌的门控流程。门控网络计算对数，`TopK`函数选择对数分数最高的两个专家（专家2和专家4）来处理该令牌。

### 负载不平衡及其影响

尽管功能简单，Top-k门控存在一个重要的运行不足：它可能导致严重的负载不平衡。在训练期间，门控网络可能偏向选择一小部分“常用”专家，而其他专家则利用不足或很少被选中。这种现象出现的原因是门控网络是一个学习函数，在没有约束的情况下，它会优化预测准确性，这可能涉及重复使用在训练早期被证明最有效的少数专家。

这种不平衡带来两个主要问题：

1. **训练效率低下：** 未被充分选择的专家接收到的训练信号很少，导致它们的参数 (parameter)优化不足。模型实际上未能充分利用其全部能力，因为其大部分权重 (weight)对其性能的贡献很小。
2. **计算浪费：** 在硬件层面，专家通常分布在多个设备（例如GPU）上。如果一个专家接收到过高比例的令牌，其分配到的设备就会成为瓶颈，而承载不常用专家的设备则处于闲置状态。



[交互图表：专家负载不平衡示例](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-2-advanced-routing-mechanisms/analysis-top-k-gating#plot-11whid4)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "专家负载不平衡示例"
    },
    "xaxis": {
      "title": {
        "text": "专家ID"
      },
      "tickmode": "array",
      "tickvals": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
      ],
      "ticktext": [
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "E8"
      ]
    },
    "yaxis": {
      "title": {
        "text": "批次中分配的令牌数"
      }
    },
    "bargap": 0.2,
    "plot_bgcolor": "#f8f9fa",
    "paper_bgcolor": "#ffffff"
  },
  "data": [
    {
      "type": "bar",
      "x": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
      ],
      "y": [
        120,
        550,
        80,
        115,
        490,
        95,
        75,
        105
      ],
      "marker": {
        "color": [
          "#4dabf7",
          "#fa5252",
          "#4dabf7",
          "#4dabf7",
          "#fa5252",
          "#4dabf7",
          "#4dabf7",
          "#4dabf7"
        ],
        "line": {
          "width": 0
        }
      }
    }
  ]
}
```

</details>



> 单个训练批次中八个专家间的令牌分布概览。专家2和专家5负载过重，而其他专家则利用不足，表明负载平衡不佳。

### 容量因子与丢弃令牌

为了管理令牌流并防止任何单个专家超负荷，MoE实现引入了一个`容量因子`。这个超参数 (parameter) (hyperparameter)定义了每个专家在批次中可以处理多少令牌的缓冲区。每个专家的容量$C$定义为：


$$
C = \mathrm{容量因子} \times \frac{\mathrm{批次中令牌总数}}{\mathrm{专家数量}}
$$


大于1.0的`容量因子`为轻微的不平衡提供了缓冲区。例如，1.25的值允许每个专家处理比平均值多达25%的令牌。

但是，如果一个常用的专家接收到的令牌超过其容量$C$，多余的令牌就会被“丢弃”。这些丢弃的令牌不会被任何专家处理。相反，它们会完全绕过MoE层，并通过残差连接。这是一种信息丢失形式，并可能降低模型性能，因为模型无法对这些令牌应用专门的计算。找到合适的`容量因子`是一个权衡：较高的值会减少丢弃令牌的数量，但会增加内存分配，并且如果负载平衡，可能导致更多的计算浪费。

在标准Top-k系统中应对负载不平衡的主要方式是辅助负载平衡损失，我们在第1章中已作介绍。这个损失项惩罚不平衡的路由决策，促使门控网络将令牌更均匀地分配到所有可用专家。它是稳定训练和使Top-k门控在实践中可行的一个重要组成部分。然而，正如我们在后续章节中将看到的，修改路由算法本身可以为此挑战提供更直接和有效的方法。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://doi.org/10.48550/arXiv.1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: arXiv preprint arXiv:1701.06538; DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控混合专家层，定义了Top-k门控机制和辅助负载平衡损失。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, and Noam Shazeer (2021)
  Journal: JMLR; Volume: 23; Pages: 1-36; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  推广了Top-k门控中使用k=1，详细阐述了容量因子、专家丢弃以及大规模MoE模型中负载平衡的实践问题。

---

[上一节](../01-%E4%B8%93%E5%AE%B6%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%A0%B8%E5%BF%83%E5%8E%9F%E7%90%86/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AA%E5%9F%BA%E6%9C%AC%20MoE%20%E5%B1%82.md) · [下一节](02-%E5%99%AA%E5%A3%B0Top-k%E9%97%A8%E6%8E%A7%E5%AE%9E%E7%8E%B0%E8%B4%9F%E8%BD%BD%E5%9D%87%E8%A1%A1.md)

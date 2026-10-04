# 分析参数与FLOPs的权衡

来源：[原文](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-5-integrating-moe-into-architectures/moe-parameter-flops-tradeoffs)

[返回章节目录](README.md) · [返回课程目录](../README.md)

专家混合模型的主要架构优势在于它能够将总参数 (parameter)数量与每token的计算成本分离。密集模型的参数和浮点运算（FLOPs）同步增长，而MoE模型可以显著增加其参数数量（这是模型容量的一种衡量），同时保持训练和推理 (inference)的计算预算大致不变。对这种权衡进行定量分析。

### 缩放的不对称性

如章概述中所述，MoE层的缩放特性由两个不同的方程决定。对于一个拥有 $N$ 个专家的层，其中门控网络为每个token选择排名靠前的 $k$ 个专家，关系如下：


$$
\text{总参数} \propto N \times \text{每个专家的参数}
$$


$$
\text{计算FLOPs} \propto k \times \text{每个专家的FLOPs}
$$


这里的要点在于 $N$ 和 $k$ 之间的差异。由于 $k$ 通常很小（例如1或2），而 $N$ 可以很大（例如64、128或更多），因此总参数 (parameter)数量可以比计算成本增长快得多。

### 一个具体比较

让我们通过一个实际例子进行分析。考虑模型中一个标准的Transformer前馈网络（FFN）块，其中隐藏维度 $d_{model}$ 为4096，FFN的内部维度 $d_{ffn}$ 为16384。

#### 密集型FFN块

一个密集型FFN块通常包含两个线性层。第一个将维度从 $d_{model}$ 扩展到 $d_{ffn}$，第二个将其重新投射回去。

- **参数 (parameter):** 参数数量约为 $2 \times d_{model} \times d_{ffn}$。
  - $2 \times 4096 \times 16384 \approx 134.2$ 百万参数。
- **FLOPs:** 每token的计算成本也与 $2 \times d_{model} \times d_{ffn}$ 成正比。
  - 这大致产生每token 134.2 百万次乘加运算。

#### MoE FFN块

现在，让我们将这个密集块替换为MoE层。我们希望保持每token的计算量大致相同，以确保训练成本的公平比较。我们将使用 $k=2$ 个每token活跃专家。为了匹配密集模型的FLOPs，每个专家的FFN维度 $d_{expert}$ 应该是密集模型 $d_{ffn}$ 的大约一半。

设定:

- 专家总数 $N = 64$。
- 每token活跃专家数 $k = 2$。
- 专家FFN内部维度 $d_{expert} = 8192$。

现在我们可以计算MoE层的属性:

- **总参数:** 总参数数量是所有 $N$ 个专家参数的总和，加上门控网络可忽略不计的参数量。
  - $N \times (2 \times d_{model} \times d_{expert}) = 64 \times (2 \times 4096 \times 8192) \approx 4.3$ 亿参数。
- **FLOPs:** 单个token的计算仅涉及选择的 $k$ 个专家。
  - $k \times (2 \times d_{model} \times d_{expert}) = 2 \times (2 \times 4096 \times 8192) \approx 134.2$ 百万次乘加运算每token。

这一比较清楚地展现了这种权衡：在每次前向传播计算成本相同的情况下，MoE架构在其FFN层中包含超过32倍的参数数量。参数的大幅增长使得模型能够培养高度专业化的专家并存储更多“知识”，而不会按比例增加训练时间或推理 (inference)延迟。

下面的图表展示了这种关系，将密集模型与专家数量不断增加的MoE模型进行比较，同时保持计算FLOPs不变。



[交互图表：参数与计算FLOPs对比](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-5-integrating-moe-into-architectures/moe-parameter-flops-tradeoffs#plot-1hdp3va)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "barmode": "group",
    "title": {
      "text": "参数与计算FLOPs对比"
    },
    "xaxis": {
      "title": {
        "text": "模型配置"
      },
      "tickvals": [
        0,
        1,
        2
      ],
      "ticktext": [
        "密集型FFN",
        "MoE (N=8, k=2)",
        "MoE (N=64, k=2)"
      ]
    },
    "yaxis": {
      "title": {
        "text": "数量（相对比例）"
      },
      "type": "log"
    },
    "legend": {
      "orientation": "h",
      "yanchor": "bottom",
      "y": 1.02,
      "xanchor": "right",
      "x": 1
    },
    "font": {
      "family": "Arial, sans-serif"
    },
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#ffffff"
  },
  "data": [
    {
      "type": "bar",
      "name": "总参数量（十亿）",
      "x": [
        0,
        1,
        2
      ],
      "y": [
        0.134,
        0.536,
        4.29
      ],
      "marker": {
        "color": "#4263eb"
      }
    },
    {
      "type": "bar",
      "name": "相对FLOPs",
      "x": [
        0,
        1,
        2
      ],
      "y": [
        1,
        1,
        1
      ],
      "marker": {
        "color": "#40c057"
      }
    }
  ]
}
```

</details>



> 对数刻度突出了MoE模型参数的指数增长，而计算成本（FLOPs）保持不变。

### 架构调控点

这种权衡为模型架构师提供了两个主要的调整点：专家总数 ($N$) 和活跃专家数 ($k$)。

- **专家数量 (N):** 这是扩展模型容量的主要手段。增加 $N$ 直接增加总参数 (parameter)数量，从而实现更高的专业化。成本主要体现在模型存储和内存上，而非计算量。
- **活跃专家数量 (k):** 这直接控制计算成本。一个 $k=2$ 的模型运行成本大约是 $k=1$ 模型（如Switch Transformer）的两倍，但它允许每个token的表示由多个专家输出的组合形成，这可以提升性能。

下图显示了这些因素如何影响最终的模型特性。

> 这是一个关于架构选择如何影响模型属性的图表。总参数主要受专家数量和大小的影响，而计算FLOPs则受活跃专家数量和它们大小的影响。

### 重要注意事项

尽管参数 (parameter)-FLOPs的权衡很有效，但并非没有代价。理想化的FLOPs计算未考虑几个实际开销：

1. **通信成本:** 在分布式环境下，将token路由到不同设备上分配给它们的专家需要大量的网络通信（一种“all-to-all”操作）。这种开销可能成为瓶颈，简单的FLOPs计算无法体现。
2. **内存需求:** 包含数十亿参数的模型需要大量内存。即使并非所有参数都同时被使用，它们也必须被存储和可访问，这对部署和推理 (inference)构成重大挑战，这些问题可通过专家卸载等技术来解决。
3. **负载不均衡:** 恒定FLOPs的假设依赖于token到专家的完美均衡路由。如果门控网络持续偏向少数专家，这些专家将成为计算热点，而其他专家则处于闲置状态。这种不均衡会降低效率，这就是负载均衡辅助损失非常重要的原因。

理解这一基本权衡对于有效设计、训练和部署MoE模型非常重要。它使得您能够构建容量远超密集型模型的模型，前提是您能管理好相关的通信和内存复杂性。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/pdf/1701.06538.pdf) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR) 2017
  介绍了稀疏门控混合专家（MoE）层，展示了如何在不显著增加每令牌计算成本的情况下大幅提升模型容量。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/v23/21-0998.html) — William Fedus, Barret Zoph, and Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Publisher: JMLR, Inc. and Microtome Publishing; Volume: 23; Pages: 1-40
  介绍了Switch Transformer，将MoE层简化为每令牌一个活跃专家（k=1），展示了其将模型扩展到万亿参数的能力，并解决了通信和负载均衡方面的挑战。
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/pdf/2201.05824.pdf) — Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Quoc V. Le, and Zhifeng Chen (2022)
  Journal: arXiv preprint arXiv:2201.05824; DOI: [10.48550/arXiv.2201.05824](https://doi.org/10.48550/arXiv.2201.05824)
  详细介绍了GLaM，一种MoE架构，与同等质量的密集模型相比，以显著更少的训练FLOP实现了高性能，强调了效率和规模。
- [A Survey of Mixture of Experts](https://arxiv.org/pdf/2308.14073.pdf) — Xufeng Lin, Yiming Qian, Yuanyang Liu, Huadong Liu, Xizhen Sun, Jianyang Li, Guanyu Chen, Qingyu Jin, Meng Zhang, and Bo Xu (2023)
  Journal: arXiv preprint arXiv:2308.14073; Publisher: arXiv; DOI: [10.48550/arXiv.2308.14073](https://doi.org/10.48550/arXiv.2308.14073)
  全面概述了混合专家模型，涵盖其历史、架构变体、训练技术以及效率和负载均衡等实际考虑。

---

[上一节](05-%E6%9E%B6%E6%9E%84%E5%8F%98%E4%BD%93%E5%8F%8A%E5%85%B6%E7%89%B9%E6%80%A7.md) · [下一节](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BF%AE%E6%94%B9Transformer%E6%A8%A1%E5%9E%8B%E4%BB%A5%E4%BD%BF%E7%94%A8MoE.md)

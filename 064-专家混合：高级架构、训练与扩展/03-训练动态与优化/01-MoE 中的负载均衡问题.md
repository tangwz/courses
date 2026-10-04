# MoE 中的负载均衡问题

来源：[原文](https://apxml.com/zh/courses/mixture-of-experts/chapter-3-moe-training-dynamics-optimization/moe-load-balancing-problem)

[返回章节目录](README.md) · [返回课程目录](../README.md)

有效训练专家混合（MoE）模型带来了标准密集架构通常不会遇到的挑战。其中首要的是**负载均衡问题**。此问题直接源于 MoE 的核心机制：由门控网络调节的条件计算。

回想一下，在 MoE 层中，门控网络决定哪个专家处理每个输入令牌。理想情况下，我们希望计算负载在训练批次过程中大致均匀地分配到层内所有可用专家。然而，没有内在保证门控网络（仅由最小化主要任务损失驱动，如交叉熵）会实现这种理想状态。

### 负载不均衡是什么？

当门控网络不成比例地将令牌分配给一部分专家，导致其他专家未被充分使用时，就会出现负载不均衡。对于一个拥有 $N$ 个专家的层，理想的平衡意味着每个专家处理在给定前向传播或一个训练批次中通过该层路由的大约 $1/N$ 的令牌。严重偏离这种均匀分布即构成不均衡。

假设一个 Transformer 块包含一个 MoE 层，有 $E=64$ 个专家和一个 $T$ 个令牌的批次。门控网络 $G$ 为每个令牌 $x$ 生成选择专家 $i$ 的概率 $p_i(x)$。例如，如果使用 top-k 门控，其中 $k=2$，则每个令牌被路由到两个专家。令 $C_i$ 为批次内分配给专家 $i$ 的令牌数量。当 $C_i$ 值在 $i=1, \dots, E$ 上的分布高度倾斜时，负载不均衡就会出现。

### 不均衡为何会造成问题？

专家使用不均导致几个重要问题，这些问题削弱了 MoE 的优势并使训练过程复杂化：

1. **计算效率低下：** 稀疏 MoE 的主要动机是计算节省；我们仅激活模型参数 (parameter)的一小部分来处理每个输入。如果负载不均衡，一些专家（以及在分布式设置中分配给它们的硬件资源）会成为计算瓶颈，而其他专家则闲置。这抵消了潜在的吞吐量 (throughput)优势，因为整体处理时间由负载最重的专家决定。
2. **参数浪费和模型容量下降：** 未被充分使用的专家未获得足够的输入信号来学习有意义的专业分工。它们的参数实际上被浪费了，对模型的整体表示能力贡献甚微。模型实际运行的活跃参数少于预期，限制了其容量。
3. **训练不稳定：** 接收极少令牌的专家可能出现梯度消失，导致学习缓慢或停滞。反之，持续过载的专家可能经历大而嘈杂的梯度，可能导致不稳定或发散，特别是如果未通过梯度裁剪等技术进行仔细管理。
4. **专家专业化程度低：** MoE 的目标是让专家学习针对不同输入类型的专用功能。如果门控网络持续偏向少数专家，模型未能形成这种多样性。这可能导致少数“通才”专家占据主导，而其他专家未能分化，这种现象有时被称为专家崩溃。

### 直观展示不均衡

想象一下在一个训练步骤中令牌在专家间的分配情况。不均衡的情况可能看起来像左侧的分布，而均衡的情况则显示在右侧。



[交互图表：专家负载分布（每专家令牌数）](https://apxml.com/zh/courses/mixture-of-experts/chapter-3-moe-training-dynamics-optimization/moe-load-balancing-problem#plot-gduew7)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "专家负载分布（每专家令牌数）",
    "xaxis": {
      "title": "专家索引"
    },
    "yaxis": {
      "title": "分配的令牌数量"
    },
    "barmode": "group"
  },
  "data": [
    {
      "type": "bar",
      "name": "不均衡",
      "x": [
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "E8"
      ],
      "y": [
        350,
        50,
        20,
        400,
        15,
        10,
        150,
        5
      ],
      "marker": {
        "color": "#fa5252"
      }
    },
    {
      "type": "bar",
      "name": "均衡",
      "x": [
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "E8"
      ],
      "y": [
        120,
        130,
        115,
        125,
        135,
        110,
        128,
        137
      ],
      "marker": {
        "color": "#40c057"
      }
    }
  ]
}
```

</details>



> 批次中8个专家间的令牌分布。不均衡情况显示出显著倾斜，专家1和4处理了大部分令牌，而其他专家几乎闲置。均衡情况显示出更加均匀的分布。

### 不均衡的根本原因

负载不均衡在训练过程中通常自然出现：

- **初始化：** 门控网络的初始随机权重 (weight)可能内在偏向某些专家。
- **早期训练动态：** 初始梯度可能会强化早期建立的路由模式，形成反馈循环，导致某些专家逐渐获得更多令牌，因为它们在初始阶段表现稍好。
- **数据分布：** 某些特征或模式在训练数据中可能更为普遍，自然地促使优化良好的门控网络将它们路由到特定专家，如果这些模式很常见，可能会导致这些专家过载。
- **缺乏明确的压力：** 如果没有专门鼓励平衡的机制，纯粹由任务损失驱动的优化过程没有动力均匀分配负载，如果一个不均衡配置能实现稍低的任务损失。

因此，解决这个负载均衡问题不仅仅是一个优化细节；它对于成功训练大型、高性能的 MoE 模型具有根本意义。接下来的章节将介绍常用技术，特别是辅助损失函数 (loss function)的使用，这些技术明确旨在抵消这些趋势并促进专家得到均衡使用。

## 参考资料

- [Sparsely-Gated Mixture-of-Experts Layers](https://arxiv.org/pdf/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  这篇基础性论文介绍了稀疏门控专家混合模型（MoE）的概念，并提出了用于平衡专家负载的辅助损失函数，直接解决了文中描述的问题。
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/pdf/2006.16668) — Dmitry Lepikhin, Hieu Pham, Orhan Firat, Michele Catasta, Zhifeng Chen, George Tucker, Azade Nova, Andre Barreto, Max Dean, and Jeff Dean (2020)
  Journal: arXiv preprint arXiv:2006.16668; DOI: [10.48550/arXiv.2006.16668](https://doi.org/10.48550/arXiv.2006.16668)
  这项工作展示了MoE模型实际扩展到大规模的案例，强调了高效负载分配和自动化分片策略的必要性，这些策略与负载平衡问题紧密相关。
- [Router Argumentation for Mixture-of-Experts](https://arxiv.org/pdf/2202.04944) — Koustuv Sinha, Michael Noukhovitch, Subhabrata Roy, Karthik Srinivasan, William Fedus, Michael Ryoo, and Yoshua Bengio (2022)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2202.04944](https://doi.org/10.48550/arXiv.2202.04944)
  本文提出了改进门控网络路由决策的方法，通过使路由更加稳健，直接促进了更好的负载平衡和专家特化。

---

[上一节](../02-%E8%BF%9B%E9%98%B6%20MoE%20%E6%9E%B6%E6%9E%84/06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%AE%9A%E4%B9%89%E9%97%A8%E6%8E%A7%E6%9C%BA%E5%88%B6.md) · [下一节](02-%E8%BE%85%E5%8A%A9%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E7%94%A8%E4%BA%8E%E8%B4%9F%E8%BD%BD%E5%9D%87%E8%A1%A1.md)

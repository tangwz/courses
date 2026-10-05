# 归一化折损累计增益 (NDCG)

来源：[原文](https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-5-evaluating-recommendation-systems/ndcg-metric)

[返回章节目录](README.md) · [返回课程目录](../README.md)

评估推荐系统通常需要比单纯判断相关条目是否存在更为细致的指标。许多评估方法在衡量推荐列表中条目位置的相对价值时会遇到挑战。例如，排名第一的条目通常比排名第十的条目更有价值，但某些方法却对它们一视同仁。此外，准确评估相关性意味着要识别不同程度的效用，而不仅仅是二元的“相关”或“不相关”。

在许多实际应用中，相关性并不是二元的。用户可能非常喜欢一部电影，对另一部评价尚可，而觉得第三部仅仅是可以接受。归一化 (normalization)折损累计增益 (NDCG) 是一种专门为这些场景设计的强力排名指标。它通过结合以下两个核心思想来评估推荐列表的质量：

1. 高相关的条目比低相关的条目更有价值。
2. 出现在列表靠前位置的相关条目比出现在靠后位置的更有用。

为了理解 NDCG，我们将从它的组成部分逐步构建：累计增益 (CG)、折损累计增益 (DCG)，以及最后的归一化版本。

### 从累计增益到 DCG

让我们从最简单的形式开始：**累计增益 (CG)**。CG 是推荐列表中直到特定排名 $k$ 的条目相关性得分的总和。它不考虑条目的顺序，只考虑它们的相关性。

在位置 $k$ 处的 CG 公式为：


$$
CG_k = \sum_{i=1}^{k} rel_i
$$


这里，$rel_i$ 是位于位置 $i$ 的条目的相关性得分。例如，如果我们有一个相关性得分为 `[5, 2, 3]` 的前 3 名列表，则 $CG_3$ 仅仅是 $5 + 2 + 3 = 10$。这告诉了我们累积的总相关性，但它没有因为模型将最相关的条目（得分 5）放在顶部而给予奖励。

为了解决这个问题，我们引入了一项惩罚，针对将相关条目放在列表较低位置的情况。这就引出了 **折损累计增益 (DCG)**。DCG 根据条目的排名系统地对相关性得分进行折减。最常用的方法是使用对数折损。

在位置 $k$ 处的 DCG 公式为：


$$
DCG_k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}
$$


分母中的 $\log_2(i+1)$ 随着位置 $i$ 的增加而增大。这意味着较高排名（如位置 1 或 2）的条目其相关性得分除以一个较小的数字，而较低排名的条目则除以一个较大的数字，从而有效地“折抵”了它们对总分的贡献。



[交互图表：DCG 中的位置折损](https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-5-evaluating-recommendation-systems/ndcg-metric#plot-1lqo93l)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "DCG 中的位置折损"
    },
    "xaxis": {
      "title": "排名位置 (i)"
    },
    "yaxis": {
      "title": "折损因子 (1 / log2(i+1))"
    },
    "template": "plotly_white"
  },
  "data": [
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        1.0,
        0.6309,
        0.5,
        0.4307,
        0.3869,
        0.3562,
        0.3333,
        0.3155,
        0.301,
        0.2891
      ],
      "type": "bar",
      "marker": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> 应用于条目相关性得分的折损因子随着其排名位置的增加而减小。位于位置 10 的条目的影响不到位置 1 条目的三分之一。

让我们回顾一下相关性得分为 `[5, 2, 3]` 的示例列表。

- 位置 1：$rel_1 = 5$。折损因子：$\frac{1}{\log_2(1+1)} = 1$。贡献：$5 \times 1 = 5$。
- 位置 2：$rel_2 = 2$。折损因子：$\frac{1}{\log_2(2+1)} \approx 0.631$。贡献：$2 \times 0.631 = 1.262$。
- 位置 3：$rel_3 = 3$。折损因子：$\frac{1}{\log_2(3+1)} = 0.5$。贡献：$3 \times 0.5 = 1.5$。

$DCG_3$ 是这些贡献的总和：$5 + 1.262 + 1.5 = 7.762$。

现在，考虑如果我们的模型产生了一个较差的排名 `[2, 3, 5]` 会发生什么：

- 位置 1：$rel_1 = 2$。贡献：$2 \times 1 = 2$。
- 位置 2：$rel_2 = 3$。贡献：$3 \times 0.631 = 1.893$。
- 位置 3：$rel_3 = 5$。贡献：$5 \times 0.5 = 2.5$。

这个列表的 $DCG_3$ 是 $2 + 1.893 + 2.5 = 6.393$。如你所见，DCG 分数较低，正确地惩罚了模型将最相关的条目放在列表底部的行为。

### 归一化 (normalization) DCG 以获得 NDCG

DCG 是一个很好的关注位置的指标，但它还有一个问题：它的值不容易解释。最大可能的 DCG 取效于特定用户和可用的相关条目。一个拥有十个高相关条目的用户比一个只有三个低相关条目的用户拥有更高的潜在 DCG。这使得跨用户平均分数或比较不同数据集上的表现变得困难。

解决方案是对 DCG 分数进行归一化。我们将模型的 DCG 除以 **理想折损累计增益 (IDCG)**。IDCG 是一个完美（或“理想”）排名的 DCG。要计算它，你需要获取某个用户的所有相关条目，按相关性降序排列，然后计算该完美列表的 DCG。

**归一化折损累计增益 (NDCG)** 的最终公式为：


$$
NDCG_k = \frac{DCG_k}{IDCG_k}
$$


得到的 NDCG 分数始终介于 0.0 和 1.0 之间。NDCG 为 1.0 意味着模型的排名是完美的，而得分为 0.0 则意味着推荐的条目中没有一个是相关的。

让我们完成示例。用户的相关性得分为 `[5, 2, 3]`。

1. **模型的排名：** `[5, 2, 3]`

   - $DCG_3 = 7.762$（如前所述）。
2. **理想排名：** 首先，按降序排列相关性得分：`[5, 3, 2]`。这是完美的排名。

   - 计算此理想列表的 $IDCG_3$：
     - 位置 1 ($rel=5$)：$5 / \log_2(2) = 5.0$
     - 位置 2 ($rel=3$)：$3 / \log_2(3) \approx 1.893$
     - 位置 3 ($rel=2$)：$2 / \log_2(4) = 1.0$
   - $IDCG_3 = 5.0 + 1.893 + 1.0 = 7.893$。
3. **计算 NDCG：**

   - $NDCG_3 = \frac{DCG_3}{IDCG_3} = \frac{7.762}{7.893} \approx 0.983$。

这个分数非常接近 1.0，表明模型的排名几乎是完美的。

### 何时使用 NDCG

NDCG 是评估排名列表最有参考价值的离线指标之一。在以下情况下你应该使用它：

- **推荐的顺序非常重要。** 如果确保前 3 个条目准确比第 8 到 10 个条目准确更有意义，那么 NDCG 是个极佳的选择。
- **可以为条目分配不同的相关级别。** 这在使用显式反馈（如 1-5 星评分）时非常直观。对于隐式反馈，你可以将“购买”事件分配比“点击”事件更高的相关性。

通过同时捕捉相关性和位置，NDCG 比更简单的指标能更全面地反映推荐系统的表现。它已成为信息检索中的标准指标，也是任何致力于优化推荐系统的人员必备的工具。

## 参考资料

- [Cumulated gain-based evaluation of IR techniques](https://doi.org/10.1145/582415.582418) — Kalervo Järvelin, Jaana Kekäläinen (2002)
  Journal: ACM Transactions on Information Systems (TOIS); Publisher: ACM; Volume: 20; Pages: 422-446; DOI: [10.1145/582415.582418](https://doi.org/10.1145/582415.582418)
  本文介绍了折扣累积增益（DCG）和归一化折扣累积增益（NDCG），作为评估信息检索系统的主要指标，为其广泛应用奠定了基础。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval.html) — C.D. Manning, P. Raghavan, H. Schütze (2008)
  Publisher: Cambridge University Press
  信息检索领域的经典教材，在搜索和相关性评估的背景下，对包括NDCG在内的排名评估指标进行了全面解释。其中第八章尤为相关。
- [Recommender Systems: An Introduction](https://www.cambridge.org/core/books/recommender-systems-an-introduction/9780521493369) — Dietmar Jannach, Markus Zanker, Alexander Felfernig, Gerhard Friedrich (2010)
  Publisher: Cambridge University Press
  本书对推荐系统领域进行了易于理解的介绍，并专门有一节讨论了评估指标，其中在推荐质量的背景下讨论了NDCG。
- [Recommender Systems Handbook](https://doi.org/10.1007/978-1-0716-2197-4) — Francesco Ricci, Lior Rokach, Bracha Shapira (2022)
  Publisher: Springer US; DOI: [10.1007/978-1-0716-2197-4](https://doi.org/10.1007/978-1-0716-2197-4)
  一本关于推荐系统的综合性手册，收录了顶尖研究人员的贡献。它广泛涵盖了评估方法和指标，包括NDCG，用于评估系统性能。

---

[上一节](06-%E5%B9%B3%E5%9D%87%E7%B2%BE%E5%BA%A6%E5%9D%87%E5%80%BC%20%28MAP%29.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%A1%A1%E9%87%8F%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD.md)

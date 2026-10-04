# 平均绝对误差 (MAE)

来源：[原文](https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-3-metrics-for-regression/mae-metric)

[返回章节目录](README.md) · [返回课程目录](../README.md)

这是“平均绝对误差 (MAE)”部分的详细内容。

### 什么是平均绝对误差 (MAE)？

在评估回归模型时，我们的首要目标是弄清楚模型的预测与实际值之间存在多大偏差。一种直接的衡量方法就是平均绝对误差，简称 MAE。

假设您的模型预测房价。对于一套房屋，模型预测价格为 25 万美元，但实际售价为 26 万美元。误差是 1 万美元（26 万美元 - 25 万美元）。对于另一套房屋，模型预测价格为 31 万美元，但实际售价为 30.5 万美元。这里的误差是 -5 千美元（30.5 万美元 - 31 万美元）。

如果我们直接对这些误差（1 万美元和 -5 千美元）取平均值，正负值可能会相互抵消，从而给出整体表现的误导性信息。为避免这种情况，MAE 使用每个误差的*绝对值*。绝对误差只是误差的大小，不考虑其正负号。因此，我们示例中的绝对误差分别是 $|10,000| = 10,000$ 和 $|-5,000| = 5,000$。

MAE 接着计算您的测试数据集中所有预测的这些绝对误差的*平均值*。它平均地告诉您，无论预测值是过高还是过低，您的预测与真实值之间的偏差有多大。

### 计算 MAE

MAE 的公式是：


$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$


我们来分解一下这个公式：

- $n$ 是测试集中的数据点总数。
- $y_i$ 是第 $i$ 个数据点的实际真实值。
- $\hat{y}_i$（读作“y-hat”）是模型对第 $i$ 个数据点的预测值。
- $|y_i - \hat{y}_i|$ 是该数据点实际值与预测值之间的绝对差（绝对误差）。
- $\sum_{i=1}^{n}$ 表示我们将所有数据点（从第一个 $i=1$ 到最后一个 $i=n$）的这些绝对差加起来。
- $\frac{1}{n}$ 表示我们将总和除以数据点的数量，从而得到平均值。

### 示例计算

我们来看一个小例子。假设我们有一个包含 4 个数据点的测试集，我们的模型对目标变量（如摄氏温度）做出以下预测：

| 数据点 | 实际值 ($y_i$) | 预测值 ($\hat{y}_i$) | 误差 ($y_i - \hat{y}_i$) | 绝对误差 ($|y_i - \hat{y}_i|$) |
| :--------- | :------------------- | :---------------------------- | :------------------------ | :------------------------------------ |
| 1 | 22 | 24 | -2 | 2 |
| 2 | 15 | 14 | 1 | 1 |
| 3 | 30 | 27 | 3 | 3 |
| 4 | 19 | 20 | -1 | 1 |

现在，我们应用 MAE 公式：

1. **计算绝对误差：** 我们已在表格中完成此项：2、1、3、1。
2. **求绝对误差之和：** $2 + 1 + 3 + 1 = 7$。
3. **除以数据点数量 ($n=4$)：** $MAE = \frac{7}{4} = 1.75$。

因此，该模型在这个小型测试集上的 MAE 为 1.75。



[交互图表：每个数据点的绝对误差](https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-3-metrics-for-regression/mae-metric#plot-ukjq5v)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "每个数据点的绝对误差",
    "xaxis": {
      "title": "数据点索引",
      "tickmode": "array",
      "tickvals": [
        1,
        2,
        3,
        4
      ]
    },
    "yaxis": {
      "title": "绝对误差"
    },
    "bargap": 0.2,
    "margin": {
      "l": 50,
      "r": 20,
      "t": 40,
      "b": 40
    }
  },
  "data": [
    {
      "type": "bar",
      "x": [
        1,
        2,
        3,
        4
      ],
      "y": [
        2,
        1,
        3,
        1
      ],
      "marker": {
        "color": "#4dabf7"
      },
      "name": "绝对误差"
    }
  ]
}
```

</details>



> 示例中计算的四个数据点的绝对误差。MAE 代表这些柱子的平均高度。

### 理解 MAE

MAE 值直接衡量了目标变量原始单位中平均预测误差的大小。在我们的例子中，MAE 是 1.75 摄氏度。这意味着，平均而言，模型的温度预测与实际值相差 1.75 摄氏度，无论方向（过高或过低）。

- **越低越好：** 较低的 MAE 通常表示模型拟合度越好，因为预测值平均而言更接近实际值。MAE 为 0 则表示完美预测。
- **背景决定：** MAE 为 1.75 是好是坏？这完全取决于具体情况。如果您预测的是日温度，其值范围从 -10°C 到 40°C，那么平均 1.75°C 的误差可能相当不错。然而，如果您预测的是人体温度，正常值在 37°C 左右，而微小偏差都具有很大的影响，那么 1.75°C 的 MAE 将非常糟糕。请务必将 MAE 与目标变量的范围和量级进行比较。

### MAE 的特点

MAE 具有一些显著的特点：

- **易于理解：** 其主要优点在于易于理解。它直接对应您所关注单位（美元、千克、分数等）中的平均误差大小。
- **对异常值不敏感：** 因为 MAE 使用绝对差值，所以它不像对误差进行平方的指标（例如我们接下来将讨论的均方误差 MSE）那样，对大误差（异常值）给予过高的权重 (weight)。一次非常糟糕的预测对 MAE 的影响远不如对 MSE 的影响那么剧烈。如果您不希望异常值主导评估指标，或者您的数据集中包含您不想过度惩罚的已知异常情况，这会很有用。

然而，它线性处理所有误差（10 美元的误差贡献是 5 美元误差的两倍）这一事实，如果您应用中特别担心大误差，可能就不那么理想。在您需要强烈惩罚大偏差的情况下，MSE 或 RMSE 等其他指标可能更合适。

总的来说，MAE 提供了一种清晰、易于理解的平均预测误差衡量方法，是理解回归模型表现的有用工具，尤其当你偏好对异常预测不那么敏感的指标，或者平均误差大小的直接可读性最受关注时。

## 参考资料

- [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2013)
  Publisher: Springer; Pages: Chapter 2 & Chapter 3
  一本基础教科书，以易懂的方式介绍了统计学习，涵盖了回归模型评估和包括MAE在内的各种指标。
- [sklearn.metrics.mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) — scikit-learn Development Team (2024)
  scikit-learn库中平均绝对误差（MAE）函数的官方文档，提供了技术规范和使用示例。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 3 & Chapter 7; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本关于统计学习方法的综合性书籍，详细介绍了回归模型以及不同损失函数和评估指标的属性。

---

[上一节](02-%E8%AE%A1%E7%AE%97%E9%A2%84%E6%B5%8B%E8%AF%AF%E5%B7%AE.md) · [下一节](04-%E5%9D%87%E6%96%B9%E8%AF%AF%E5%B7%AE%20%28MSE%29.md)

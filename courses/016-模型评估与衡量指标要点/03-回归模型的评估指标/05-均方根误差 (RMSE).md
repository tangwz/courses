# 均方根误差 (RMSE)

来源：[原文](https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-3-metrics-for-regression/rmse-metric)

[返回章节目录](README.md) · [返回课程目录](../README.md)

我们已经了解均方误差 (MSE) 如何衡量实际值 ($y$) 与预测值 ($\hat{y}$) 之间的平均平方差。虽然 MSE 能有效惩罚较大的误差，但其单位是平方的（例如，美元的平方），这有时会使理解起来不太直接。

为了让误差指标回到目标变量的原始单位，我们计算**均方根误差 (RMSE)**。它就是均方误差的平方根。

### 计算方法

RMSE 的公式直接来源于 MSE：


$$
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$


我们来细分一下：

- 如前所述，$y_i$ 是第 $i$ 个观测的实际值，$\hat{y}_i$ 是预测值。
- $(y_i - \hat{y}_i)$ 是预测误差。
- 我们将每个误差平方：$(y_i - \hat{y}_i)^2$。
- 我们将这些平方误差求和：$\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$。
- 我们计算平方误差的平均值（这就是 MSE）：$\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$。
- 最后，我们取 MSE 的平方根得到 RMSE。

### 意义

RMSE 的主要优点在于其易于理解。因为我们取了平方根，RMSE 的单位与原始目标变量 ($y$) 的单位**相同**。

- 如果您预测房屋价格以美元计，RMSE 也以美元为单位。RMSE 为 $15,000 表明模型价格预测与实际价格的典型偏差约为$15,000。
- 如果您预测温度以摄氏度计，RMSE 则以摄氏度为单位。

与 MSE 类似，RMSE 会对大误差给予不成比例的高权重 (weight)，因为误差在求平均值之前被平方了。一个即使只产生少数几个非常大误差的预测模型，其 RMSE 也会比平均误差幅度相似但极端误差较少的模型高出许多。

RMSE 值越低，表明模型对数据的拟合越好。RMSE 为 0 意味着模型对数据集中每个观测都做出了完美预测。

### 计算示例

我们使用一个包含实际温度和预测温度（摄氏度）的小数据集：

| 实际值 ($y_i$) | 预测值 ($\hat{y}_i$) | 误差 ($y_i - \hat{y}_i$) | 平方误差 ($(y_i - \hat{y}_i)^2$) |
| --- | --- | --- | --- |
| 22 | 23 | -1 | 1 |
| 25 | 24 | 1 | 1 |
| 19 | 21 | -2 | 4 |
| 28 | 26 | 2 | 4 |
| **总计** |  |  | **10** |

1. **计算误差：** 找出每个实际值和预测值之间的差值。
2. **计算平方误差：** 将每个误差平方。
3. **计算均方误差 (MSE)：** 对平方误差求平均值。
   
   $$
   MSE = \frac{1 + 1 + 4 + 4}{4} = \frac{10}{4} = 2.5
   $$
   
   MSE 为 2.5 摄氏度平方。
4. **计算均方根误差 (RMSE)：** 取 MSE 的平方根。
   
   $$
   RMSE = \sqrt{MSE} = \sqrt{2.5} \approx 1.58
   $$
   

RMSE 约为 1.58 摄氏度。这表示模型的温度预测通常与实际值相差约 1.58 摄氏度。

### RMSE 与 MAE 和 MSE 的比较

将 RMSE 与我们讨论过的其他误差指标进行比较会很有用：

- **MAE：** 衡量平均绝对误差。与 RMSE 相比，对大误差的敏感度较低。单位与目标变量相同。
- **MSE：** 衡量平均平方误差。由于平方运算，对大误差更敏感。单位是目标变量单位的平方。
- **RMSE：** 衡量平均平方误差的平方根。对大误差敏感（与 MSE 类似）。单位与目标变量相同（与 MAE 类似），使其比 MSE 更易于理解。



[交互图表：误差指标比较（温度示例）](https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-3-metrics-for-regression/rmse-metric#plot-5tzq4p)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "MAE",
        "MSE",
        "RMSE"
      ],
      "y": [
        1.5,
        2.5,
        1.58
      ],
      "type": "bar",
      "name": "指标"
    }
  ],
  "layout": {
    "title": "误差指标比较（温度示例）",
    "xaxis": {
      "title": "指标"
    },
    "yaxis": {
      "title": "误差值（摄氏度或摄氏度平方）"
    },
    "bargap": 0.2
  }
}
```

</details>



> 针对温度预测示例计算的误差指标。MAE 为 1.5°C，MSE 为 2.5°C²，RMSE 约为 1.58°C。请注意，RMSE 提供了与 MAE 相似的误差量级，但包含了平方误差的影响。

选择 MAE 还是 RMSE 通常取决于您是否希望大误差对指标产生按比例更大的影响。如果偶尔出现大误差可以接受，MAE 可能适用。如果大误差特别成问题且应受到严厉惩罚，则通常更倾向于使用 RMSE（或 MSE）。由于 RMSE 使用原始单位，因此在回归任务中报告模型性能时经常选用它。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer
  这本书是统计学习领域的常用资源，解释了像RMSE这样的回归指标及其在模型评估中的作用。（第二版）
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，涵盖了回归指标（如RMSE），并提供了模型性能评估的例子和应用背景。（第三版）
- [Machine Learning (CS229) Lecture Notes](http://cs229.stanford.edu/) — Andrew Ng (2023)
  Publisher: Stanford University
  这些广受认可的讲义提供了机器学习的核心知识，包括对回归问题和相关误差指标的讨论。

---

[上一节](04-%E5%9D%87%E6%96%B9%E8%AF%AF%E5%B7%AE%20%28MSE%29.md) · [下一节](06-%E6%AF%94%E8%BE%83%E5%B9%B3%E5%9D%87%E7%BB%9D%E5%AF%B9%E8%AF%AF%E5%B7%AE%20%28MAE%29%E3%80%81%E5%9D%87%E6%96%B9%E8%AF%AF%E5%B7%AE%20%28MSE%29%20%E5%92%8C%E5%9D%87%E6%96%B9%E6%A0%B9%E8%AF%AF%E5%B7%AE%20%28RMSE%29.md)

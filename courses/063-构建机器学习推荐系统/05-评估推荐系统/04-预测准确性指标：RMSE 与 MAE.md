# 预测准确性指标：RMSE 与 MAE

来源：[原文](https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-5-evaluating-recommendation-systems/prediction-accuracy-metrics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当推荐系统的目标是预测用户可能给某个物品打出的具体分数时，我们需要能够直接衡量这些预测准确度的指标。这类需求在向用户展示预测评分的系统中非常普遍，例如：“根据您的历史记录，您可能会给这部电影打 4.5 分。”在这种情况下，需要使用预测准确性指标，来评估模型预测评分与用户实际评分之间的接近程度。

在这一任务中，最常用且最基础的两个指标是平均绝对误差 (MAE) 和均方根误差 (RMSE)。它们都能量化 (quantization)一组预测中的平均误差，但计算方式略有不同，因此在解释和敏感度上也有所区别。

### 平均绝对误差 (MAE)

平均绝对误差是最直观的误差指标。它衡量一组预测中误差的平均大小，而不考虑误差的方向。它是测试样本中预测值与实际观察值之间绝对差值的平均值，其中每个样本的差值权重 (weight)相同。

MAE 的公式为：


$$
\text{MAE} = \frac{1}{|\hat{R}|} \sum_{(\text{u, i}) \in \hat{R}} |r_{ui} - \hat{r}_{ui}|
$$


其中：

- $\hat{R}$ 是测试集中的用户-物品对集合。
- $|\hat{R}|$ 是测试集中的评分总数。
- $r_{ui}$ 是用户 $u$ 对物品 $i$ 的实际评分。
- $\hat{r}_{ui}$ 是模型为用户 $u$ 和物品 $i$ 预测的评分。

该指标的含义非常直接。MAE 为 0.5 意味着平均而言，模型的预测偏差为 0.5 分。这使得它成为一个非常容易向业务相关方传达的指标。

让我们看看如何在 Python 中进行计算。假设你有一个包含实际评分和预测评分的 pandas DataFrame：

```python
import pandas as pd
import numpy as np

# 样本数据
data = {
    'user_id': [1, 1, 2, 2, 3],
    'item_id': [101, 102, 101, 103, 104],
    'actual_rating': [4, 3, 5, 2, 4],
    'predicted_rating': [3.8, 3.5, 4.5, 2.8, 3.9]
}
df = pd.DataFrame(data)

# 从零开始计算 MAE
df['absolute_error'] = abs(df['actual_rating'] - df['predicted_rating'])
mae = df['absolute_error'].mean()

print(f"计算得出的 MAE: {mae:.4f}")

# 为了方便使用 scikit-learn
from sklearn.metrics import mean_absolute_error

mae_sklearn = mean_absolute_error(df['actual_rating'], df['predicted_rating'])
print(f"Scikit-learn MAE: {mae_sklearn:.4f}")
```

两种方法的结果是一样的，但通常推荐使用 scikit-learn 等成熟的库，因为它们在处理大型数据集时更不容易出错且效率更高。

### 均方根误差 (RMSE)

均方根误差是另一个广泛用于评估评分预测准确性的指标。MAE 是对绝对误差取平均值，而 RMSE 采取了不同的方法：它先对误差进行平方，然后再取平均值，最后对结果开平方。

RMSE 的公式为：


$$
\text{RMSE} = \sqrt{\frac{1}{|\hat{R}|} \sum_{(\text{u, i}) \in \hat{R}} (r_{ui} - \hat{r}_{ui})^2}
$$


计算步骤如下：

1. 计算每个用户-物品对的实际评分与预测评分之差 ($r_{ui} - \hat{r}_{ui}$)。
2. 对该差值进行平方。这一步有两个作用：它使所有误差变为正数，并且赋予较大的误差更高的权重 (weight)。例如，偏差为 2 分的预测对总和的贡献是 4，而 0.5 分的误差贡献仅为 0.25。
3. 计算这些平方误差的平均值（即均方误差，MSE）。
4. 对 MSE 开平方。最后一步将误差转换回与原始评分相同的单位（例如：分），使其更具可解释性。

以下是对应的 Python 实现：

```python
# 继续使用之前的 DataFrame

# 从零开始计算 RMSE
df['squared_error'] = (df['actual_rating'] - df['predicted_rating'])**2
mse = df['squared_error'].mean()
rmse = np.sqrt(mse)

print(f"计算得出的 RMSE: {rmse:.4f}")

# 使用 scikit-learn
from sklearn.metrics import mean_squared_error

# 注意：sklearn 提供的是 mean_squared_error，所以我们需要对其开平方
rmse_sklearn = np.sqrt(mean_squared_error(df['actual_rating'], df['predicted_rating']))
print(f"Scikit-learn RMSE: {rmse_sklearn:.4f}")
```

### RMSE 与 MAE 的比较：大误差的影响

MAE 和 RMSE 之间的主要区别在于它们处理不同幅度误差的方式。由于 RMSE 对误差进行了平方处理，它对大预测误差的惩罚比 MAE 更严厉。对于同一组预测，RMSE 的值永远大于或等于 MAE。两者之间的差距越大，说明样本中个体误差的波动越大。如果差距很大，通常表明模型产生了少数非常大的误差。

让我们通过一个例子来说明。考虑模型预测的两种情形。在情形 A 中，误差较小且平稳。在情形 B 中，大多数误差较小，但存在一个明显的异常值。

- **情形 A（平稳的小误差）：** 实际值 `[4, 5, 3]`，预测值 `[3.5, 4.5, 3.5]`。误差为 `[-0.5, -0.5, 0.5]`。
- **情形 B（一个大误差）：** 实际值 `[4, 5, 3]`，预测值 `[3.5, 4.5, 1.0]`。误差为 `[-0.5, -0.5, -2.0]`。

对于情形 A：

- MAE = $(| -0.5| + |-0.5| + |0.5|) / 3 = 0.5$
- RMSE = $\sqrt{((-0.5)^2 + (-0.5)^2 + (0.5)^2) / 3} = \sqrt{(0.25 + 0.25 + 0.25) / 3} = 0.5$

对于情形 B：

- MAE = $(|-0.5| + |-0.5| + |-2.0|) / 3 = (0.5 + 0.5 + 2.0) / 3 = 1.0$
- RMSE = $\sqrt{((-0.5)^2 + (-0.5)^2 + (-2.0)^2) / 3} = \sqrt{(0.25 + 0.25 + 4.0) / 3} = \sqrt{4.5 / 3} \approx 1.22$

可以注意到，虽然情形 B 中单个大误差使 MAE 翻了一倍（从 0.5 增加到 1.0），但它使 RMSE 增加的幅度更大（从 0.5 增加到 1.22）。下表直观地展示了这种敏感性。



[交互图表：RMSE 对大误差的敏感性](https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-5-evaluating-recommendation-systems/prediction-accuracy-metrics#plot-1mx7j8a)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "RMSE 对大误差的敏感性",
    "xaxis": {
      "title": "指标"
    },
    "yaxis": {
      "title": "误差值"
    },
    "barmode": "group",
    "font": {
      "family": "sans-serif"
    }
  },
  "data": [
    {
      "name": "情形 A (小误差)",
      "x": [
        "MAE",
        "RMSE"
      ],
      "y": [
        0.5,
        0.5
      ],
      "type": "bar",
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "name": "情形 B (一个大误差)",
      "x": [
        "MAE",
        "RMSE"
      ],
      "y": [
        1.0,
        1.22
      ],
      "type": "bar",
      "marker": {
        "color": "#fd7e14"
      }
    }
  ]
}
```

</details>



> 在情形 B 中引入单个较大的预测误差，会导致 RMSE 比起 MAE 出现更剧烈的增长，凸显了其对异常值的敏感性。

### 应该使用哪种指标？

MAE 和 RMSE 之间的选择取决于你的应用对大误差的容忍度。

- **选择 MAE**：如果你希望指标易于理解且对异常值表现得更稳健。如果你的业务认为 2 分的误差严重程度恰好是 1 分误差的两倍，那么 MAE 是合适的选择。它按比例对待所有误差。
- **选择 RMSE**：如果你的系统特别不希望出现大误差。如果单个 2 分的预测偏差比两个各 1 分的偏差带来的负面影响大得多，那么 RMSE 是更好的选择。它对大误差的惩罚会引导模型优化，从而尽量避免此类结果。

在许多实际场景中，RMSE 是评估评分预测模型的默认指标，因为大误差会严重损害用户体验。不过，同时报告这两个指标通常是更好的做法，因为它们之间的差异能反映出模型误差分布的有用信息。

虽然 MAE 和 RMSE 是评估预测准确性的基石，但请记住它们并不是全部。许多推荐系统并不是根据它们预测评分的能力来衡量的，而是根据它们对物品进行排序的能力。为此，我们需要另一类指标，我们将在接下来的内容中进行讲解。

## 参考资料

- [sklearn.metrics Module Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html) — scikit-learn developers (2023)
  scikit-learn中`sklearn.metrics`模块的官方文档，包含`mean_absolute_error`和`mean_squared_error`的定义及实现细节。
- [Recommender Systems: An Introduction](https://link.springer.com/book/10.1007/978-0-387-85820-3) — Francesco Ricci, Lior Rokach, Bracha Shapira, Paul B. Kantor (2011)
  Publisher: Springer; DOI: [10.1007/978-0-387-85820-3](https://doi.org/10.1007/978-0-387-85820-3)
  一本关于推荐系统的综合性教材，讨论了MAE和RMSE在评估评分预测模型中的应用和解释。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本统计学习的经典教材，提供了预测模型中各种误差度量（包括MAE和RMSE）的基本理论和统计基础。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/accuracy.html) — Rob J Hyndman and George Athanasopoulos (2023)
  Publisher: OTexts
  一本权威的在线预测学教材，其中有一章专门细致地比较了MAE和RMSE，解释了它们的统计特性以及对模型评估的影响。

---

[上一节](03-%E4%B8%BA%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E8%AF%84%E4%BC%B0%E6%8B%86%E5%88%86%E6%95%B0%E6%8D%AE.md) · [下一节](05-%E6%8E%92%E5%90%8D%E6%8C%87%E6%A0%87%EF%BC%9APrecision%40K%20%E4%B8%8E%20Recall%40K.md)

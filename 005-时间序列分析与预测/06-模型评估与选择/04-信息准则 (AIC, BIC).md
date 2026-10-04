# 信息准则 (AIC, BIC)

来源：[原文](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-6-model-evaluation-selection/information-criteria-aic-bic)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 MAE、MSE 和 RMSE 等指标能告诉我们模型预测与测试集上实际值的匹配程度，但它们本身不惩罚模型的复杂性。一个非常复杂的模型（例如，具有许多参数 (parameter)的 ARIMA 模型）可能在训练数据上拟合得非常好，甚至可能*过好*，导致在该数据上的误差很低。然而，这种复杂性可能意味着模型捕获了噪声而非潜在的信号，从而导致对新的、未见过的数据泛化能力差。这种现象称为过拟合 (overfitting)。

信息准则，如赤池信息准则 (AIC) 和贝叶斯信息准则 (BIC)，提供了一种通过平衡拟合优度和模型简洁性来选择模型的方法。它们衡量了当给定模型用于表示数据生成过程时所损失的信息量。

### 赤池信息准则 (AIC)

AIC 由赤池弘次开发，它估算预测误差，并以此衡量给定数据集上统计模型的相对质量。AIC 估算模型与真实数据生成过程之间的 Kullback-Leibler 散度。

AIC 的公式通常如下：


$$
AIC = 2k - 2\ln(\hat{L})
$$


其中：

- $k$ 是模型中估计参数 (parameter)的数量（包括残差的方差）。对于 ARIMA(p, d, q) 模型，$k = p + q + 1$（如果包含常数项）或 $k = p + q$（如果不包含常数项）。对于 SARIMA(p,d,q)(P,D,Q)m 模型，$k = p + q + P + Q + 1$（如果包含常数项）。
- $\hat{L}$ 是模型似然函数的最大值。$\ln(\hat{L})$ 是最大似然值的自然对数。

第一项 $2k$ 对参数较多的模型进行惩罚。模型参数越多，这一项的值越大，从而 AIC 值也越高。第二项 $-2\ln(\hat{L})$ 奖励模型的拟合优度。较高的似然值 $\hat{L}$（表示对数据拟合得更好）会导致这一项的值更小，从而降低 AIC。

目标是找到使 AIC 最小的模型。较低的 AIC 值表明模型拟合与复杂性之间有更好的平衡。重要的是要记住 AIC 值是相对的；特定模型的 AIC 值本身没有意义，但当比较拟合到*相同*数据集的不同候选模型时，它会变得有用。

### 贝叶斯信息准则 (BIC)

贝叶斯信息准则 (BIC)，也称为施瓦茨准则 (SIC)，是另一种基于似然函数的模型选择准则，但它对模型复杂性的惩罚比 AIC 更强。

BIC 的公式是：


$$
BIC = \ln(n)k - 2\ln(\hat{L})
$$


其中：

- $n$ 是用于估计的数据点数量（样本大小）。
- $k$ 是模型中估计参数 (parameter)的数量（与 AIC 中相同）。
- $\hat{L}$ 是似然函数的最大值（与 AIC 中相同）。

将 BIC 与 AIC 比较，可以看到参数数量 $k$ 的惩罚项是 $\ln(n)k$ 而不是 $2k$。由于样本大小的自然对数 $\ln(n)$ 对于大多数实际时间序列通常大于 2（$n > e^2 \approx 7.4$），因此 BIC 通常比 AIC 更严厉地惩罚复杂模型。这通常导致 BIC 选择比 AIC 更简单的模型。

与 AIC 类似，BIC 旨在平衡拟合度与复杂性，BIC 值最低的模型更受青睐。它也是一种相对衡量标准，用于比较拟合到相同数据的模型。

### AIC 和 BIC 的实践应用

在 Python 中使用 `statsmodels` 等库拟合 ARIMA 或 SARIMA 模型时，总结输出通常包含 AIC 和 BIC 值。

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# 假设 'data' 是您的平稳时间序列（例如，差分后）
# 例子：拟合一个 ARIMA(1,1,1) 模型
# 注意：实际拟合通常在原始数据上进行，
# 指定 d=1，但为了演示 AIC/BIC，我们假设
# 数据已经平稳，我们正在比较 ARMA(p,q) 阶数。
# 实际使用时，请将 ARIMA(p,d,q) 拟合到原始序列。

# 生成一些用于演示的样本数据
import numpy as np
np.random.seed(42)
n_samples = 100
ar_params = np.array([0.6])
ma_params = np.array([-0.4])
# 为 ARIMA(1,1,1) 结构添加差分
y = sm.tsa.arima_process.arma_generate_sample(ar=ar_params, ma=ma_params, nsample=n_samples)
y = np.cumsum(y) # 积分回原始状态以模拟非平稳性
time_series = pd.Series(y)

# 拟合候选模型
model_111 = ARIMA(time_series, order=(1, 1, 1)).fit()
model_211 = ARIMA(time_series, order=(2, 1, 1)).fit()
model_112 = ARIMA(time_series, order=(1, 1, 2)).fit()

# 获取 AIC 和 BIC 值
print("ARIMA(1,1,1)：")
print(f"  AIC: {model_111.aic:.2f}")
print(f"  BIC: {model_111.bic:.2f}")
print("\nARIMA(2,1,1)：")
print(f"  AIC: {model_211.aic:.2f}")
print(f"  BIC: {model_211.bic:.2f}")
print("\nARIMA(1,1,2)：")
print(f"  AIC: {model_112.aic:.2f}")
print(f"  BIC: {model_112.bic:.2f}")

# --- 示例输出 ---
# ARIMA(1,1,1):
#   AIC: 278.20
#   BIC: 285.99
#
# ARIMA(2,1,1):
#   AIC: 279.98
#   BIC: 290.36
#
# ARIMA(1,1,2):
#   AIC: 280.02
#   BIC: 290.40
```

在此示例中，ARIMA(1,1,1) 模型在三个候选模型中具有最低的 AIC 和 BIC 值。根据这些准则，这表明它为这个特定数据集提供了拟合度与简洁性的最佳平衡。

我们可以将此比较可视化：



[交互图表：使用信息准则进行模型比较](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-6-model-evaluation-selection/information-criteria-aic-bic#plot-5f42r7)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "ARIMA(1,1,1)",
        "ARIMA(2,1,1)",
        "ARIMA(1,1,2)"
      ],
      "y": [
        278.2,
        279.98,
        280.02
      ],
      "name": "AIC",
      "marker": {
        "color": "#4263eb"
      }
    },
    {
      "type": "bar",
      "x": [
        "ARIMA(1,1,1)",
        "ARIMA(2,1,1)",
        "ARIMA(1,1,2)"
      ],
      "y": [
        285.99,
        290.36,
        290.4
      ],
      "name": "BIC",
      "marker": {
        "color": "#12b886"
      }
    }
  ],
  "layout": {
    "title": "使用信息准则进行模型比较",
    "xaxis": {
      "title": "模型"
    },
    "yaxis": {
      "title": "准则值"
    },
    "barmode": "group",
    "template": "plotly_white",
    "width": 600,
    "height": 400
  }
}
```

</details>



> 三个候选 ARIMA 模型的 AIC 和 BIC 值比较。较低的值表示模型拟合度与复杂性之间有更好的权衡。

### AIC 和 BIC 的选择

关于 AIC 或 BIC 哪个普遍“更好”，没有确定答案。

- **AIC** 倾向于选择比 BIC 更复杂的模型。如果目标是预测精度，它可能更受青睐，因为稍微复杂的模型有时可以捕捉数据的更多细节，从而改进预测，尽管存在轻微过拟合 (overfitting)的风险。AIC 是渐进有效的；如果真实模型在候选模型之中，当 $n \to \infty$ 时，AIC 将以概率 1 选中它。然而，它不具有一致性；如果真实模型不是无限复杂的，即使样本量很大，AIC 仍然可能选择过于复杂的模型。
- **BIC** 倾向于选择更简单的模型，因为它有更强的惩罚项，该惩罚项随样本大小 $n$ 的增加而增大。如果目标是简洁性或找到在结构上更接近“真实”潜在过程的模型（假设存在相对简单的真实过程），它可能更受青睐。BIC 具有一致性；如果真实模型在候选模型之中并且具有有限参数 (parameter)，当 $n \to \infty$ 时，BIC 将以概率 1 选中它。

在实践中，同时查看 AIC 和 BIC 通常很有用。如果它们在最佳模型上达成一致，会增加对选择的信心。如果它们意见不一，则突出了一种权衡：AIC 倾向的模型可能提供稍好的拟合，而 BIC 倾向的模型则更简洁。最终的选择可能取决于您的具体目标以及可能进行的进一步分析，例如检查残差（在前面的章节中介绍）和评估在保留集上的预测精度（使用 MAE、RMSE 等指标）。

请记住，AIC 和 BIC 是指导模型选择的工具，尤其在通过 ACF/PACF 分析确定多个合理 ARIMA/SARIMA 阶数时特别有用。它们应与残差诊断和样本外预测评估一起使用，而不是替代它们。

## 参考资料

- [A New Look at the Statistical Model Identification](https://ieeexplore.ieee.org/document/1100705) — Hirotugu Akaike (1974)
  Journal: IEEE Transactions on Automatic Control; Publisher: IEEE; Volume: 19; Pages: 716-723; DOI: [10.1109/TAC.1974.1100705](https://doi.org/10.1109/TAC.1974.1100705)
  介绍了赤池信息准则 (AIC) 作为统计模型识别方法，为平衡模型拟合优度和复杂性奠定了理论基础。
- [Estimating the Dimension of a Model](https://projecteuclid.org/journals/annals-of-statistics/volume-6/issue-2/Estimating-the-Dimension-of-a-Model/10.1214/aos/1176344136.full) — Gideon Schwarz (1978)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 6; Pages: 461-464; DOI: [10.1214/aos/1176344136](https://doi.org/10.1214/aos/1176344136)
  介绍了贝叶斯信息准则 (BIC)，亦称施瓦茨准则，为模型选择提供了不同的视角，对模型复杂性施加了更强的惩罚。
- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: Wiley
  一部关于时间序列分析的经典教科书，涵盖 ARIMA/SARIMA 模型构建、识别、估计和诊断，包括使用 AIC 和 BIC 等信息准则进行模型选择。（第5版）

---

[上一节](03-%E5%B8%B8%E7%94%A8%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87%EF%BC%88MAE%E3%80%81MSE%E3%80%81RMSE%E3%80%81MAPE%EF%BC%89.md) · [下一节](05-%E6%AF%94%E8%BE%83%E4%B8%8D%E5%90%8C%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.md)

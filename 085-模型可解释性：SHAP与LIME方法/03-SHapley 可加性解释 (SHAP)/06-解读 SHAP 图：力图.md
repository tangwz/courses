# 解读 SHAP 图：力图

来源：[原文](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/shap-force-plots)

[返回章节目录](README.md) · [返回课程目录](../README.md)

SHAP 值量化 (quantization)了单个特征值对模型预测的贡献。对于特定预测，这些值通过 KernelSHAP 或 TreeSHAP 等技术得出。理解它们的含义，特别是各个特征值如何使模型的输出偏离基线，非常重要。SHAP 可视化提供了这种理解，而**力图**是呈现这些*局部*解释的主要工具。

可以将力图看作拔河比赛的视觉表现。一方面，您有**基准值**。这表示如果模型没有您正在查看的实例的任何特定特征信息时，它会做出的平均预测。它本质上是模型在背景数据集（通常是训练集）上的平均输出。

另一方面是该特定实例的**最终预测**（输出值）。在这两点之间，特征作为力起作用，使预测变高或变低。

其工作原理如下：

1. **推高预测的特征（正 SHAP 值）**：具有正 SHAP 值的特征有助于相对于基准值提高预测。在标准力图中，这些通常以红色显示。SHAP 值越大，代表该特征的红色部分就越长，表示向上推动作用越强。
2. **拉低预测的特征（负 SHAP 值）**：具有负 SHAP 值的特征有助于相对于基准值降低预测。这些通常以蓝色显示。同样，负 SHAP 值的大小决定了蓝色部分的长度，显示该特征将预测拉低了多少。

核心思想是基准值与所有单个特征 SHAP 值之和等于该预测的最终输出值。用数学表示为：

$\text{模型输出} = \text{基准值} + \sum \text{各特征的 SHAP 值}$

这种累加属性，继承自 Shapley 值，使得力图容易理解。它们准确地说明了模型如何得出特定预测，从基线开始，并加上或减去每个特征的贡献。

### 阅读力图

解读力图涉及查看这些组成部分：

- **识别基准值**：这是起点，通常显示在图上。
- **识别输出值**：这是模型为此实例做出的最终预测。
- **查看力的作用**：
  - 红色部分表示推高预测的特征。注意这些是哪些特征及其相对长度（影响）。
  - 蓝色部分表示拉低预测的特征。注意这些是哪些特征及其相对长度。
- **力的平衡**：该图通过视觉方式平衡这些正负贡献，以弥合基准值与最终输出值之间的差异。

### 例子：解释房价预测

设想您已经训练了一个模型来预测房价，并且您想解释它对特定房屋的预测。假设您的数据集中平均预测价格（基准值）为 $250k。对于一栋特定的房屋，模型预测为$350k。力图可能会显示：

- **基准值**：\$250k
- **预测值**：\$350k
- **特征**：
  - `sqft_living = 2000`：显著推高价格（大的红色部分，例如 +\$80k SHAP 值）。
  - `grade = 8`：推高价格（中等红色部分，例如 +\$40k SHAP 值）。
  - `yr_built = 1950`：略微拉低价格（小的蓝色部分，例如 -\$15k SHAP 值）。
  - `zipcode = 98103`：极小程度拉低价格（很小的蓝色部分，例如 -\$5k SHAP 值）。

力图直观地排列这些信息，显示 `sqft_living` 和 `grade` 的大额正向贡献如何超过 `yr_built` 和 `zipcode` 的小额负向贡献，从而得到高于基准值的最终预测。

虽然原始的 `shap` 库生成交互式 JavaScript 力图，但我们可以使用其他图表类型表示其核心思想。以下瀑布图说明了其累加性质：



[交互图表：SHAP 力图解释（瀑布图显示）](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/shap-force-plots#plot-8qsb97)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "SHAP 力图解释（瀑布图显示）",
    "yaxis": {
      "title": "预测房价（千美元）"
    },
    "xaxis": {
      "type": "category"
    },
    "waterfallgap": 0.3,
    "margin": {
      "l": 50,
      "r": 50,
      "t": 50,
      "b": 50
    }
  },
  "data": [
    {
      "type": "waterfall",
      "name": "预测解释",
      "orientation": "v",
      "measure": [
        "absolute",
        "relative",
        "relative",
        "relative",
        "relative",
        "total"
      ],
      "x": [
        "基准值",
        "sqft_living (+8万)",
        "grade (+4万)",
        "yr_built (-1.5万)",
        "zipcode (-5千)",
        "最终预测"
      ],
      "textposition": "outside",
      "textfont": {
        "size": 10
      },
      "y": [
        250,
        80,
        40,
        -15,
        -5,
        350
      ],
      "connector": {
        "line": {
          "color": "#868e96"
        }
      },
      "increasing": {
        "marker": {
          "color": "#fa5252"
        }
      },
      "decreasing": {
        "marker": {
          "color": "#4dabf7"
        }
      },
      "totals": {
        "marker": {
          "color": "#495057",
          "line": {
            "color": "#495057",
            "width": 1
          }
        }
      }
    }
  ]
}
```

</details>



> 一张瀑布图，说明了 SHAP 力图背后的原理。从基准值（平均预测）开始，正 SHAP 值（红色）会增加预测，而负 SHAP 值（蓝色）会降低预测，从而得出该特定实例的最终预测。

### 使用 Python 生成力图

`shap` 库使生成力图变得简单。假设您已计算出 `explainer` 和 `shap_values`（如前几节所述），并且您有特征数据 `X`，您可以绘制单个实例的解释图（例如索引为 `i` 的实例）：

```python
import shap

# 假设 'explainer' 是一个已拟合的 SHAP 解释器（例如 KernelExplainer, TreeExplainer）
# 假设 'shap_values' 是您数据集的 SHAP 值数组/列表
# 假设 'X' 是您的特征 DataFrame 或 NumPy 数组
# 选择要解释的实例的索引
instance_index = 0

# 加载 JavaScript 可视化库（Jupyter notebook 等环境所需）
shap.initjs()

# 为所选实例生成力图
shap.force_plot(
    explainer.expected_value, # 基准值
    shap_values[instance_index], # 此实例的 SHAP 值
    X.iloc[instance_index] # 此实例的特征值
)
```

此代码通常会在 Jupyter notebook 等环境中渲染交互式图。该图直观地确认了累加关系：`explainer.expected_value + shap_values[instance_index].sum()` 应该非常接近模型对 `X.iloc[instance_index]` 的预测。

### 可视化多个解释

力图主要用于*局部*解释（一次一个预测）。但是，`shap` 库允许将多个力图堆叠在一起，通常是垂直旋转的。这可以通过同时显示许多样本的解释来提供半全局视图。在许多样本中，始终向同一方向推动预测的特征会变得显而易见。

```python
# 为多个实例生成力图（例如前 100 个）
# 注意：对于大型数据集/图表，这可能需要大量计算
shap.force_plot(
    explainer.expected_value,
    shap_values[:100], # 前 100 个实例的 SHAP 值
    X.iloc[:100] # 前 100 个实例的特征值
)
```

这种堆叠视图有助于辨识更广泛的模式，同时仍然基于单个实例的解释。

力图提供了一种强大且有理论依据的方法，可以逐个实例地查看模型预测的内部。它们直接显示每个特征的影响，清楚地说明了为何相对于基线做出了特定预测，这对于调试、验证和建立对模型的信任很重要。

## 参考资料

- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) — Scott M. Lundberg, Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4765-4774
  介绍了 SHAP (SHapley Additive exPlanations) 框架，定义了 SHAP 值以及包括用于局部解释的力图在内的多种可视化方法。
- [SHAP Documentation: Interpreting individual predictions with force plots](https://shap.readthedocs.io/en/latest/api_examples.html#interpreting-individual-predictions-with-force-plots) — Scott Lundberg (2024)
  提供了使用 `shap` Python 库生成和解释 SHAP 力图的官方指南和代码示例。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/shap.html) — Christoph Molnar (2024)
  这本被广泛引用的在线书籍全面概述了模型可解释性方法，包括一个详细的章节解释 SHAP 值及其可视化表示（如力图）。

---

[上一节](05-TreeSHAP%EF%BC%9A%E9%92%88%E5%AF%B9%E6%A0%91%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96.md) · [下一节](07-%E8%A7%A3%E8%AF%BBSHAP%E5%9B%BE%EF%BC%9A%E6%A6%82%E8%A7%88%E5%9B%BE%E4%B8%8E%E4%BE%9D%E8%B5%96%E5%9B%BE.md)

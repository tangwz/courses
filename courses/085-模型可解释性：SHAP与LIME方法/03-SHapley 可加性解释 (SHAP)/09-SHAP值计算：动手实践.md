# SHAP值计算：动手实践

来源：[原文](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/shap-hands-on-practical)

[返回章节目录](README.md) · [返回课程目录](../README.md)

实践例子展示了如何使用`shap` Python库来计算和可视化SHAP值，这有助于理解个体预测和整体模型行为。

我们假设您已经有一个训练好的机器学习 (machine learning)模型，并希望更好地理解其预测。在此例子中，我们设想已经在像Iris数据集这样大家熟悉的数据集上训练了一个梯度提升分类器。

### 设置和模型训练

首先，请确保您已安装所需的库。您主要需要`shap`、`scikit-learn`和`pandas`。

```python
# 如果您尚未安装shap，请安装
# pip install shap

import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris

# 加载Iris数据集
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 为了演示，我们简化为一个二元分类问题
# 类别0 对比 类别1 & 2
y_binary = (y > 0).astype(int)
class_names = ['setosa', 'not setosa'] # 将0映射为'setosa'，1映射为'not setosa'

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 训练一个梯度提升分类器
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"模型已在 {X_train.shape[0]} 个样本上训练。")
print(f"正在 {X_test.shape[0]} 个样本上进行测试。")
```

这就配置好了我们的环境，数据已加载到Pandas DataFrame `X` 中，目标变量为 `y_binary`。我们已经训练了一个 `GradientBoostingClassifier`。

### 使用TreeExplainer计算SHAP值

由于梯度提升是一个基于树的集成模型，计算SHAP值的最有效方式是使用`shap.TreeExplainer`。这个解释器利用树的内部结构进行更快的计算，相比于像`KernelExplainer`这样的模型无关方法更为迅速。

```python
# 1. 初始化解释器
# 我们将训练好的模型传递给解释器。
explainer = shap.TreeExplainer(model)

# 2. 计算SHAP值
# 我们为测试集（或任何我们想解释的数据）计算SHAP值。
# 输出的'shap_values'通常是一个列表（用于多分类）或数组（用于二分类）
# 包含每个特征、每个实例的SHAP值。
# 对于使用scikit-learn模型的二元分类，shap通常返回
# 对应正类别（本例中为类别1）的值。
shap_values = explainer.shap_values(X_test)

# 检查形状：（样本数量，特征数量）
print(f"SHAP值形状: {shap_values.shape}")

# 检查预期值（基准值）
# 这是训练数据（或背景数据）上的平均预测
print(f"解释器预期值（基准值）: {explainer.expected_value}")
```

`shap_values`变量现在包含一个数组，其中每行对应`X_test`中的一个实例，每列对应一个特征。该值本身表示该特征对模型输出（对数几率或概率，取决于模型和解释器设置）在该特定实例上的贡献，与基准值（`explainer.expected_value`）进行对比。

### 可视化局部解释：力图

为了理解单个预测，力图非常有效。它显示了哪些特征使预测结果更高（正SHAP值，通常为红色），哪些特征使预测结果更低（负SHAP值，通常为蓝色）。

让我们解释测试集中第一个实例（`X_test.iloc[0]`）的预测。

```python
# 在环境中初始化JavaScript可视化（绘图所需）
shap.initjs()

# 解释第一个预测
instance_index = 0

# 为第一个实例创建力图
# 我们传入基准值、该实例的SHAP值和特征值
shap.force_plot(explainer.expected_value,
                shap_values[instance_index,:],
                X_test.iloc[instance_index,:],
                matplotlib=False) # 使用Javascript绘图
```

> 力图显示了每个特征对单个实例预测的贡献。使预测结果更高的特征显示为红色，使预测结果更低的特征显示为蓝色。条形的长度表示特征影响的大小。

理解这个图：“基准值”表示所有样本的平均预测分数。列出的特征是那些使该特定实例的预测偏离基准值的特征。如果最终输出值高于基准值，则红色特征（正SHAP值）的集体影响强于蓝色特征（负SHAP值）。您可以清楚地看到哪些特征（如`petal width (cm)`或`sepal length (cm)`）对这个特定预测影响最大以及影响方向。

### 可视化全局解释：汇总图

尽管力图对于个体预测很适用，汇总图则提供了整个数据集（或您计算SHAP值所用的子集）上特征重要性的全局情况。

汇总图有多种样式。默认样式（`plot_type="dot"`）显示了每个特征的SHAP值分布。

```python
# 创建汇总图（点图版）
shap.summary_plot(shap_values, X_test, plot_type="dot")
```



[交互图表：SHAP汇总图 (点图)](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/shap-hands-on-practical#plot-s97tti)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "name": "花瓣长度 (cm)",
      "y": [
        0,
        1,
        2,
        3
      ],
      "x": [
        -1.7283948,
        -1.5396692,
        3.7445736,
        3.6977887
      ],
      "mode": "markers",
      "marker": {
        "color": [
          1.0,
          1.0,
          0.0,
          0.0
        ],
        "colorscale": [
          [
            0.0,
            "#339af0"
          ],
          [
            1.0,
            "#f03e3e"
          ]
        ],
        "colorbar": {
          "title": "特征值"
        },
        "symbol": "circle",
        "size": 8
      },
      "text": [
        "低",
        "低",
        "高",
        "高"
      ],
      "type": "scatter"
    },
    {
      "name": "花瓣宽度 (cm)",
      "y": [
        0,
        1,
        2,
        3
      ],
      "x": [
        -1.4847989,
        -1.384871,
        2.7968922,
        2.8268008
      ],
      "mode": "markers",
      "marker": {
        "color": [
          1.0,
          1.0,
          0.0,
          0.0
        ],
        "colorscale": [
          [
            0.0,
            "#339af0"
          ],
          [
            1.0,
            "#f03e3e"
          ]
        ],
        "colorbar": {
          "title": "特征值"
        },
        "symbol": "circle",
        "size": 8
      },
      "text": [
        "低",
        "低",
        "高",
        "高"
      ],
      "type": "scatter"
    },
    {
      "name": "萼片宽度 (cm)",
      "y": [
        0,
        1,
        2,
        3
      ],
      "x": [
        -0.038678713,
        0.001217056,
        -0.02692078,
        -0.015495319
      ],
      "mode": "markers",
      "marker": {
        "color": [
          0.0,
          0.0,
          0.0,
          1.0
        ],
        "colorscale": [
          [
            0.0,
            "#339af0"
          ],
          [
            1.0,
            "#f03e3e"
          ]
        ],
        "colorbar": {
          "title": "特征值"
        },
        "symbol": "circle",
        "size": 8
      },
      "text": [
        "高",
        "高",
        "高",
        "低"
      ],
      "type": "scatter"
    },
    {
      "name": "萼片长度 (cm)",
      "y": [
        0,
        1,
        2,
        3
      ],
      "x": [
        -0.028417198,
        0.01815932,
        0.01598907,
        0.0021416685
      ],
      "mode": "markers",
      "marker": {
        "color": [
          0.0,
          0.0,
          1.0,
          1.0
        ],
        "colorscale": [
          [
            0.0,
            "#339af0"
          ],
          [
            1.0,
            "#f03e3e"
          ]
        ],
        "colorbar": {
          "title": "特征值"
        },
        "symbol": "circle",
        "size": 8
      },
      "text": [
        "高",
        "高",
        "低",
        "低"
      ],
      "type": "scatter"
    }
  ],
  "layout": {
    "title": {
      "text": "SHAP汇总图 (点图)"
    },
    "xaxis": {
      "title": "SHAP值 (对模型输出的影响)"
    },
    "yaxis": {
      "title": null,
      "tickvals": [
        0,
        1,
        2,
        3
      ],
      "ticktext": [
        "花瓣长度 (cm)",
        "花瓣宽度 (cm)",
        "萼片宽度 (cm)",
        "萼片长度 (cm)"
      ],
      "autorange": "reversed"
    },
    "showlegend": false,
    "margin": {
      "l": 150
    },
    "hovermode": "closest",
    "coloraxis": {
      "colorbar": {
        "title": {
          "text": "特征值"
        }
      },
      "colorscale": [
        [
          0.0,
          "#339af0"
        ],
        [
          1.0,
          "#f03e3e"
        ]
      ]
    }
  }
}
```

</details>



> 测试集中每个特征的SHAP值分布。每个点代表一个特征在单个预测中的SHAP值。颜色表示特征值（红色=高，蓝色=低）。

理解：

- **特征重要性：** 特征根据所有样本上SHAP绝对值之和进行排序（最重要的在顶部）。在此，`petal length (cm)`和`petal width (cm)`最具影响。
- **影响分布：** 水平分布显示了每个特征影响的范围。宽分布表明特征的影响力根据实例不同而显著变化。
- **关联性：** 颜色表示原始特征值（高或低）。对于`petal length (cm)`，高值（红色点）通常具有高的正SHAP值，推动预测结果倾向类别1（“非狗尾草”）。低值（蓝色点）具有负SHAP值，推动预测结果倾向类别0（“狗尾草”）。`sepal width (cm)`整体影响较小（点集中在零附近），且特征值与影响方向之间的关联性不那么明显。

另一种方式是，条形图显示了每个特征的平均SHAP绝对值，更清楚地展示了整体重要性的大小。

```python
# 创建汇总图（条形图版）
shap.summary_plot(shap_values, X_test, plot_type="bar")
```



[交互图表：SHAP全局特征重要性（平均SHAP绝对值）](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/shap-hands-on-practical#plot-113jl3p)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        1.9711091,
        1.8314105,
        0.024458377,
        0.018768571
      ],
      "y": [
        "花瓣长度 (cm)",
        "花瓣宽度 (cm)",
        "萼片宽度 (cm)",
        "萼片长度 (cm)"
      ],
      "orientation": "h",
      "marker": {
        "color": "#1c7ed6"
      }
    }
  ],
  "layout": {
    "title": {
      "text": "SHAP全局特征重要性（平均SHAP绝对值）"
    },
    "xaxis": {
      "title": "平均(|SHAP值|) (对模型输出影响的平均大小)"
    },
    "yaxis": {
      "autorange": "reversed"
    },
    "margin": {
      "l": 150
    }
  }
}
```

</details>



> 测试集中每个特征的平均SHAP绝对值。较高的条形表示对预测的平均影响越大。

这个条形图证实了，在这个数据集中，花瓣长度和宽度对模型预测的平均影响最大。

### 考察特征依赖性：依赖图

依赖图显示了单个特征的SHAP值如何随该特征值的变化而变化。它们还可以通过根据另一个特征的值来着色点，从而展示关联作用。

让我们研究模型输出对`petal width (cm)`的依赖性，并看看它是否与`petal length (cm)`有关联。

```python
# 为'petal width (cm)'创建依赖图
# 根据'petal length (cm)'着色点以检查关联
shap.dependence_plot("petal width (cm)", shap_values, X_test, interaction_index="petal length (cm)")
```



[交互图表：SHAP依赖图：花瓣宽度 (cm)](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/shap-hands-on-practical#plot-nmb068)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        0.2,
        0.2,
        0.2,
        0.4,
        0.2,
        0.1,
        0.2,
        0.2,
        0.1,
        0.2,
        0.4,
        0.2,
        0.1,
        0.2,
        0.6,
        0.4,
        0.3,
        0.2,
        0.2,
        0.2
      ],
      "y": [
        -1.384871,
        -1.4847989,
        -1.4847989,
        -1.4847989,
        -1.4847989,
        -1.4847989,
        -1.384871,
        -1.4847989,
        -1.4847989,
        -1.4847989,
        -1.4847989,
        -1.4847989,
        -1.4847989,
        -1.384871,
        -1.4847989,
        -1.384871,
        -1.384871,
        -1.4847989,
        -1.4847989,
        -1.4847989
      ],
      "mode": "markers",
      "marker": {
        "color": [
          1.4,
          1.4,
          1.3,
          1.5,
          1.4,
          1.4,
          1.5,
          1.4,
          1.5,
          1.3,
          1.5,
          1.5,
          1.0,
          1.3,
          1.4,
          1.4,
          1.4,
          1.7,
          1.5,
          1.7
        ],
        "colorbar": {
          "title": "花瓣长度 (cm)"
        },
        "colorscale": [
          [
            0.0,
            "#339af0"
          ],
          [
            1.0,
            "#f03e3e"
          ]
        ]
      },
      "type": "scatter"
    }
  ],
  "layout": {
    "xaxis": {
      "title": "花瓣宽度 (cm)"
    },
    "yaxis": {
      "title": "petal width (cm)的SHAP值"
    },
    "title": {
      "text": "SHAP依赖图：花瓣宽度 (cm)"
    },
    "hovermode": "closest"
  }
}
```

</details>



> 'petal width (cm)'的值与其对应SHAP值之间的关系。颜色表示'petal length (cm)'的值。

理解：

- **主要影响：** 该图清楚地显示了正向关系。随着`petal width (cm)`的增加，其SHAP值趋于增加，这意味着较高的花瓣宽度强烈地将预测推向“非狗尾草”。在花瓣宽度约为0.7-0.8厘米处有一个显著的跳跃。
- **关联作用：** 根据`petal length (cm)`着色有助于观察关联作用。请注意，高`petal width (cm)`（右侧）的点始终具有高`petal length (cm)`（红色）。相反，低`petal width (cm)`（左侧）的点大多具有低`petal length (cm)`（蓝色）。这种强关联在Iris数据集中是预期的。如果存在显著关联，您可能会在给定花瓣宽度下看到颜色的垂直分离（例如，在花瓣宽度=1.5时，红色点的SHAP值系统性地高于或低于蓝色点）。在这种情况下，关联作用看起来很小。

通过生成和分析这些图，您不再只是了解模型的预测结果，而是理解它为什么做出该预测，确定局部和全局最具影响的特征，并考察潜在的特征关联。SHAP的这种实践应用对调试、验证以及在您的机器学习 (machine learning)模型中建立信任具有重要价值。

## 参考资料

- [A Unified Approach to Interpreting Model Predictions](https://neurips.cc/media/NIPS2017/file/8a20a8621978632d76c43147fa2c6772-Paper.pdf) — Scott M. Lundberg, Gabriel G. Erion, Hugh Chen, Jonathan Gillam, Philip K. Denny (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.5555/3295222.3295230](https://doi.org/10.5555/3295222.3295230)
  建立了SHAP框架及其基于Shapley值的理论基础，并介绍了KernelSHAP和TreeSHAP等解释器。
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/) — Scott M. Lundberg and SHAP Contributors (2024)
  `shap` Python库的官方资源，包含安装、API、解释器和可视化工具的使用指南。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/shap.html) — Christoph Molnar (2024)
  一份综合指南，详细讨论了SHAP、其特性、不同类型的SHAP值以及与其他可解释性方法的比较。
- [Consistent Feature Attribution for Tree Ensembles](https://arxiv.org/abs/1908.08474) — Scott M. Lundberg, Gabriel G. Erion, Su-In Lee (2020)
  Journal: arXiv preprint arXiv:1908.08474; DOI: [10.48550/arXiv.1908.08474](https://doi.org/10.48550/arXiv.1908.08474)
  进一步研究了TreeSHAP，重点关注其在树模型上的计算效率和一致性，与本节的实际示例直接相关。

---

[上一节](08-SHAP%20%E5%9C%A8%20Python%20%E4%B8%AD%E7%9A%84%E5%AE%9E%E7%8E%B0.md) · [下一节](../04-%E6%AF%94%E8%BE%83%E4%B8%8E%E5%BA%94%E7%94%A8%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%80%A7%E6%8A%80%E6%9C%AF/01-LIME%20%E4%B8%8E%20SHAP%EF%BC%9A%E5%8C%BA%E5%88%AB.md)

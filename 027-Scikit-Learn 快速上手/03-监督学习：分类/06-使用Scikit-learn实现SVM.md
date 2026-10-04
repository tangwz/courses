# 使用Scikit-learn实现SVM

来源：[原文](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/implementing-svm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

支持向量 (vector)机（SVM）可以使用Scikit-learn应用于分类任务。用于分类的主要实现是 `SVC` 类（支持向量分类），它位于 `sklearn.svm` 模块中。

就像其他Scikit-learn估计器一样，`SVC` 遵循熟悉的 `fit`/`predict` 模式。我们首先实例化模型，可能会配置其超参数 (parameter) (hyperparameter)，然后在我们的数据上训练它，最后使用它对新的、未见过的数据点进行预测。

### 基本用法

让我们从一个使用默认参数 (parameter)的简单例子开始。我们将使用 `make_blobs` 生成的一些样本数据，它能创建适用于分类任务的良好聚类。请记住，SVM对特征尺度敏感，因此应用缩放（如 `StandardScaler`）通常是必要的预处理步骤。我们将在第4章中详细介绍预处理，但这里将应用基本缩放以进行演示。

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 生成样本数据
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42, cluster_std=1.5)

# 2. 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 缩放特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # 在测试数据上使用 transform，而不是 fit_transform

# 4. 实例化SVC模型
# 默认值: kernel='rbf', C=1.0, gamma='scale'
svm_classifier = SVC(random_state=42)

# 5. 训练模型
svm_classifier.fit(X_train_scaled, y_train)

# 6. 进行预测
y_pred = svm_classifier.predict(X_test_scaled)

# 7. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确度: {accuracy:.4f}")
# 预期输出: Model Accuracy: 1.0000 (或类似，数据集易于分离)
```

在这个例子中，我们使用 `SVC` 的默认设置创建了一个实例。最主要的默认设置是 `kernel='rbf'`，它使用径向基函数核，这是处理非线性可分数据的常见选择。然后，我们使用缩放后的训练数据训练 (`fit`) 模型，并在缩放后的测试数据上进行了预测 (`predict`)。

### 选择合适的核函数

`kernel` 超参数 (parameter) (hyperparameter)决定了 `SVC` 如何转换数据以找到最优分离超平面。你根据对数据结构的理解来选择核函数。

- **`'linear'`**: 如果你认为数据大多是线性可分的，就使用这个。它拟合一个简单的超平面。
- **`'rbf'` (Radial Basis Function)**: 这是默认值，也是许多问题的良好起点。它可以将数据映射到无限维空间，并创建复杂、非线性的决策边界。它由 `gamma` 超参数控制。
- **`'poly'`**: 使用多项式函数来映射数据。多项式的次数通过 `degree` 超参数设置（默认是3）。
- **`'sigmoid'`**: 使用类似于神经网络 (neural network)中的函数。通常用于特定应用。

在创建 `SVC` 实例时指定核函数：

```python
# 线性核
svm_linear = SVC(kernel='linear', random_state=42)
# svm_linear.fit(X_train_scaled, y_train) ...

# 多项式核（3次）
svm_poly = SVC(kernel='poly', degree=3, random_state=42)
# svm_poly.fit(X_train_scaled, y_train) ...

# RBF核（默认，显式显示）
svm_rbf = SVC(kernel='rbf', random_state=42)
# svm_rbf.fit(X_train_scaled, y_train) ...
```

让我们在样本数据上可视化由线性核和RBF核生成的决策边界。



[交互图表：使用线性核函数的SVM](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/implementing-svm#plot-10hsk8r)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "使用线性核函数的SVM"
    },
    "xaxis": {
      "title": {
        "text": "特征 1 (已缩放)"
      },
      "zeroline": false,
      "showgrid": false
    },
    "yaxis": {
      "title": {
        "text": "特征 2 (已缩放)"
      },
      "zeroline": false,
      "showgrid": false
    },
    "width": 500,
    "height": 400,
    "margin": {
      "l": 50,
      "r": 20,
      "t": 40,
      "b": 50
    }
  },
  "data": [
    {
      "type": "contour",
      "x": [
        -3.0,
        -2.5,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0
      ],
      "y": [
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5
      ],
      "z": [
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1
        ]
      ],
      "colorscale": [
        [
          0,
          "#ffc9c9"
        ],
        [
          1,
          "#a5d8ff"
        ]
      ],
      "opacity": 0.4,
      "showscale": false,
      "name": "决策边界"
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        -1.0,
        -0.8,
        -1.2,
        1.0,
        1.2,
        0.8
      ],
      "y": [
        0.5,
        0.7,
        0.3,
        -0.5,
        -0.7,
        -0.3
      ],
      "name": "类别 0",
      "marker": {
        "color": "#fa5252",
        "size": 8
      }
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        1.0,
        0.8,
        1.2,
        -1.0,
        -1.2,
        -0.8
      ],
      "y": [
        0.5,
        0.3,
        0.7,
        -0.5,
        -0.3,
        -0.7
      ],
      "name": "类别 1",
      "marker": {
        "color": "#4c6ef5",
        "size": 8
      }
    }
  ]
}
```

</details>



### 超参数 (parameter) (hyperparameter)：`C` 和 `gamma`

除了核函数的选择，另外两个超参数显著影响SVM的行为：

- **`C` (正则化 (regularization)参数)**: 这个参数控制着**平滑决策边界和正确分类训练点之间的权衡**。**较小的 `C`** 值会创建**更宽的间隔**（更简单的模型），容许更多的错误分类。**较大的 `C`** 值旨在训练数据上实现**零错误分类**，可能导致间隔更窄，模型更复杂，易于过拟合 (overfitting)。可以将其视为对错误分类的惩罚。
- **`gamma` ('rbf'、'poly'和'sigmoid'核的系数)**: 这个参数定义了**单个训练示例的影响范围**。**较小的 `gamma`** 意味着**更大的影响半径**，导致更平滑、更泛化的决策边界。**较大的 `gamma`** 意味着**较小的影响半径**，导致高度弯曲的决策边界，可能非常紧密地拟合训练数据（可能过拟合）。当 `gamma` 设置为 `'scale'`（默认值）时，它使用 $1 / (\text{特征数量} * \text{X.方差()})$，这通常是一个良好的起点。

这些超参数对调整你的SVM模型很重要，以便在你的特定数据集上获得最佳性能。

让我们看看不同的 `C` 值如何影响决策边界。我们将使用相同的合成数据和RBF核。

```python
# 重新生成和缩放数据以获得一致的绘图
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42, cluster_std=1.5)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练不同C值的SVM
svm_low_C = SVC(kernel='rbf', C=0.1, random_state=42)
svm_low_C.fit(X_scaled, y)

svm_high_C = SVC(kernel='rbf', C=1000, random_state=42)
svm_high_C.fit(X_scaled, y)

# 生成等高线图Z值的辅助函数
def generate_z_for_plot(classifier, X_data):
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return xx, yy, Z

xx_low_C, yy_low_C, Z_low_C = generate_z_for_plot(svm_low_C, X_scaled)
xx_high_C, yy_high_C, Z_high_C = generate_z_for_plot(svm_high_C, X_scaled)
```



[交互图表：使用RBF核函数的SVM (C=0.1)](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/implementing-svm#plot-1i68996)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "使用RBF核函数的SVM (C=0.1)"
    },
    "xaxis": {
      "title": {
        "text": "特征 1 (已缩放)"
      },
      "zeroline": false,
      "showgrid": false
    },
    "yaxis": {
      "title": {
        "text": "特征 2 (已缩放)"
      },
      "zeroline": false,
      "showgrid": false
    },
    "width": 500,
    "height": 400,
    "margin": {
      "l": 50,
      "r": 20,
      "t": 40,
      "b": 50
    }
  },
  "data": [
    {
      "type": "contour",
      "x": [
        -3.0,
        -2.5,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0
      ],
      "y": [
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5
      ],
      "z": [
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -0.5,
          -0.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -0.5,
          -0.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -0.5,
          -0.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -0.5,
          -0.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -0.5,
          -0.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -0.5,
          -0.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -0.5,
          -0.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -0.5,
          -0.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -0.5,
          -0.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -0.5,
          -0.5,
          1.5,
          1.5,
          1.5,
          1.5
        ]
      ],
      "colorscale": [
        [
          0,
          "#ffc9c9"
        ],
        [
          1,
          "#a5d8ff"
        ]
      ],
      "opacity": 0.4,
      "showscale": false,
      "name": "决策边界"
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        -1.0,
        -0.8,
        -1.2,
        1.0,
        1.2,
        0.8
      ],
      "y": [
        0.5,
        0.7,
        0.3,
        -0.5,
        -0.7,
        -0.3
      ],
      "name": "类别 0",
      "marker": {
        "color": "#fa5252",
        "size": 8
      }
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        1.0,
        0.8,
        1.2,
        -1.0,
        -1.2,
        -0.8
      ],
      "y": [
        0.5,
        0.3,
        0.7,
        -0.5,
        -0.3,
        -0.7
      ],
      "name": "类别 1",
      "marker": {
        "color": "#4c6ef5",
        "size": 8
      }
    }
  ]
}
```

</details>



> **使用RBF核函数且C值较低（C=0.1）的SVM决策边界。** 请注意更宽的间隔，它允许一些错误分类以实现更好的泛化能力。



[交互图表：使用RBF核函数的SVM (C=1000)](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/implementing-svm#plot-3c1jtp)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "使用RBF核函数的SVM (C=1000)"
    },
    "xaxis": {
      "title": {
        "text": "特征 1 (已缩放)"
      },
      "zeroline": false,
      "showgrid": false
    },
    "yaxis": {
      "title": {
        "text": "特征 2 (已缩放)"
      },
      "zeroline": false,
      "showgrid": false
    },
    "width": 500,
    "height": 400,
    "margin": {
      "l": 50,
      "r": 20,
      "t": 40,
      "b": 50
    }
  },
  "data": [
    {
      "type": "contour",
      "x": [
        -3.0,
        -2.5,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0
      ],
      "y": [
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5
      ],
      "z": [
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5
        ],
        [
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          -1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5,
          1.5
        ]
      ],
      "colorscale": [
        [
          0,
          "#ffc9c9"
        ],
        [
          1,
          "#a5d8ff"
        ]
      ],
      "opacity": 0.4,
      "showscale": false,
      "name": "决策边界"
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        -1.0,
        -0.8,
        -1.2,
        1.0,
        1.2,
        0.8
      ],
      "y": [
        0.5,
        0.7,
        0.3,
        -0.5,
        -0.7,
        -0.3
      ],
      "name": "类别 0",
      "marker": {
        "color": "#fa5252",
        "size": 8
      }
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        1.0,
        0.8,
        1.2,
        -1.0,
        -1.2,
        -0.8
      ],
      "y": [
        0.5,
        0.3,
        0.7,
        -0.5,
        -0.3,
        -0.7
      ],
      "name": "类别 1",
      "marker": {
        "color": "#4c6ef5",
        "size": 8
      }
    }
  ]
}
```

</details>



> **使用RBF核函数且C值较高（C=1000）的SVM决策边界。** 请观察更窄的间隔，它优先考虑正确分类所有训练点。

### 了解支持向量 (vector)

SVM的一个独特之处是它们依赖于**支持向量**。这些是训练集中最接近决策边界（超平面）的数据点。它们是“支持”超平面并影响其位置和方向的示例。远离间隔的点不影响超平面的定义。这使得SVM在计算上更高效，因为它们在预测时不需要考虑所有训练数据，只需考虑支持向量。

训练 `SVC` 模型后，你可以访问支持向量：

```python
# 假设 svm_classifier 已被训练
print(f"每个类别的支持向量数量: {svm_classifier.n_support_}")
print(f"支持向量的索引: {svm_classifier.support_}")
print(f"支持向量（已缩放）: {svm_classifier.support_vectors_}")
```

### 多类别分类

尽管SVM的核心思想是二元分类，但 `SVC` 也可以处理多类别问题。Scikit-learn的 `SVC` 默认实现**一对多 (OvR) 策略**。在这种方法中，为每个类别训练一个单独的二元SVM，以对抗所有其他类别。在预测时，选择其相应二元SVM置信度分数最高的类别。

或者，你可以指定 `decision_function_shape='ovo'` 来使用**一对一 (OvO) 策略**。在这里，为每对类别训练一个二元SVM。如果你有 $N$ 个类别，这意味着要训练 $N(N-1)/2$ 个SVM。最终预测由这些成对分类器之间的多数投票决定。OvO在训练时通常计算成本更高，但有时能带来更好的准确度。

这是一个使用 `make_classification` 处理多类别数据的例子：

```python
from sklearn.datasets import make_classification

# 生成多类别数据
X_multi, y_multi = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                       n_informative=2, n_clusters_per_class=1,
                                       n_classes=3, random_state=42)

# 缩放特征
scaler_multi = StandardScaler()
X_multi_scaled = scaler_multi.fit_transform(X_multi)

# 使用OvR策略（默认）训练SVC
svm_multi_ovr = SVC(kernel='rbf', random_state=42)
svm_multi_ovr.fit(X_multi_scaled, y_multi)

print("\n多类别SVM（OvR策略）已训练。")
```

可视化多类别决策边界比二元决策边界更复杂，但仍能对模型如何在特征空间中分离不同类别提供有价值的理解。

### 总结

Scikit-learn的 `SVC` 为在分类任务中实现支持向量 (vector)机提供了一种强大且灵活的方式。通过理解并适当调整其 `kernel`、`C` 和 `gamma` 超参数 (parameter) (hyperparameter)，你可以有效地构建出对未见过数据具有良好泛化能力的模型，即使对于非线性可分问题也是如此。请记住，适当的特征缩放通常是SVM获得最佳性能的重要前提。

## 参考资料

- [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) — Scikit-learn Developers (2023)
  Scikit-learn中支持向量分类类的官方文档，详细说明了其参数和方法。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 12: Support Vector Machines and Extensions
  一本涵盖统计学习方法的综合教科书，其中包括对支持向量机的详细论述。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2023)
  Publisher: O'Reilly Media; Pages: Chapter 5: Support Vector Machines
  一本广泛使用的Scikit-learn机器学习算法实现指南，其中包含专门介绍SVM的章节。

---

[上一节](05-%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%20%28SVM%29%20%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md) · [下一节](07-%E5%88%86%E7%B1%BB%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md)

# 使用 Python 实现 LIME

来源：[原文](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-2-lime-local-interpretability/lime-python-implementation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用 Python 实现 LIME 涉及名为 `lime` 的库。这个库提供工具，用于解释使用 scikit-learn、TensorFlow、Keras、PyTorch 等常见机器学习 (machine learning)框架训练的分类器和回归器的单个预测。

### 安装 LIME 库

首先，请确保你已经安装了 `lime` 库。你可以使用 pip 进行安装：

```bash
pip install lime
```

你还需要 `numpy` 和 `scikit-learn` 等常用数据科学库，用于数据处理和模型构建。

### 解释表格数据预测

最常见的使用场景是解释在结构化表格数据上训练的模型预测。`lime` 库为此目的提供了 `LimeTabularExplainer` 类。

我们来看一个使用熟悉的鸢尾花数据集和 scikit-learn RandomForestClassifier 的例子。

#### 1. 准备数据和模型

假设你已将数据加载到 NumPy 数组或 Pandas DataFrame 中，并已训练了一个分类器。

```python
import numpy as np
import sklearn
import sklearn.ensemble
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

# 加载和准备数据
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练一个模型（例如，随机森林）
model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"训练准确度: {model.score(X_train, y_train):.2f}")
print(f"测试准确度: {model.score(X_test, y_test):.2f}")

# 从测试集中选择一个要解释的实例
instance_idx = 0
instance_to_explain = X_test[instance_idx]
actual_class = class_names[y_test[instance_idx]]
predicted_class = class_names[model.predict(instance_to_explain.reshape(1, -1))[0]]

print(f"\n要解释的实例（索引: {instance_idx}）: {instance_to_explain}")
print(f"实际类别: {actual_class}")
print(f"模型预测类别: {predicted_class}")
```

#### 2. 创建 LIME 解释器

下一步是创建 `LimeTabularExplainer` 的一个实例。这个对象需要你的训练数据信息，以便了解特征值的分布，从而生成扰动。

```python
# 创建 LIME 解释器
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,         # LIME 用于生成扰动的数据
    feature_names=feature_names,   # 特征名称列表
    class_names=class_names,       # 类别名称列表
    mode='classification'          # 指定 'classification' 或 'regression'
    # 可选：discretize_continuous=True, verbose=True, etc.
)
```

`LimeTabularExplainer` 的重要参数 (parameter)：

- `training_data`：一个 NumPy 数组，代表训练数据集。LIME 用它来计算特征统计数据（均值、标准差），用于扰动和离散化。提供实际训练数据很常见，但使用代表性样本也可以。
- `feature_names`：一个字符串列表，对应你数据的列名。
- `class_names`：一个字符串列表，代表目标类别名称。
- `mode`：根据你的模型类型，设置为 `'classification'` 或 `'regression'`。

#### 3. 定义预测函数

LIME 需要访问你的模型的预测函数。重要的是，这个函数必须接收一个扰动样本的 NumPy 数组（其中每一行是一个样本），并以 NumPy 数组的形式返回模型的预测概率（对于分类）或预测值（对于回归）。

对于 scikit-learn 分类器，`predict_proba` 方法通常提供此功能。

```python
# 定义 LIME 将使用的预测函数
# 它接收扰动数据（NumPy 数组）并返回概率（NumPy 数组）
predict_fn = lambda x: model.predict_proba(x)
```

确保输出形状正确：分类为 `(num_samples, num_classes)`，回归为 `(num_samples,)`。

#### 4. 生成解释

现在，在你的解释器对象上调用 `explain_instance` 方法。

```python
# 解释所选实例
explanation = explainer.explain_instance(
    data_row=instance_to_explain,      # 你想要解释的实例
    predict_fn=predict_fn,             # 上面定义的预测函数
    num_features=len(feature_names),   # 解释中特征的最大数量
    num_samples=5000                   # 用于扰动的样本数量
    # 可选：top_labels=1（仅解释预测得分最高的类别）
)
```

`explain_instance` 的重要参数：

- `data_row`：你想要解释的特定实例（作为一维 NumPy 数组）。
- `predict_fn`：上一步创建的函数封装。
- `num_features`：解释中包含特征的最大数量。LIME 根据重要性对特征进行排序并返回最重要的那些。
- `num_samples`：LIME 在 `data_row` 周围生成用于训练局部替代模型的扰动样本数量。更高的值可以带来更稳定的解释，但会增加计算时间。

#### 5. 解释解释输出

`explain_instance` 返回的 `explanation` 对象包含局部解释。你可以通过多种方式访问它：

- **作为列表：** `explanation.as_list()` 返回一个元组列表，其中每个元组包含 `(feature_description, weight)`。权重 (weight)表示特征对被解释的*特定类别*（默认为预测类别）的预测贡献。正权重支持预测，负权重反对预测。

  ```python
  # 获取预测类别的解释列表
  explanation_list = explanation.as_list()
  print(f"\n对预测 '{predicted_class}' 的解释:")
  for feature, weight in explanation_list:
      print(f"- {feature}: {weight:.4f}")
  ```
- **可视化：** LIME 提供方便的绘图功能。

  - `explanation.show_in_notebook()`：直接在 Jupyter 环境中渲染 HTML 可视化。
  - `explanation.as_pyplot_figure()`：返回一个 matplotlib 图形对象，用于自定义或保存。

我们使用 Plotly 创建一个特征贡献的简单条形图可视化，模仿 LIME 内置函数可能生成的图表。



[交互图表：鸢尾花实例的 LIME 解释（预测类别）](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-2-lime-local-interpretability/lime-python-implementation#plot-15vtg55)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "鸢尾花实例的 LIME 解释（预测类别）",
    "xaxis": {
      "title": "贡献权重"
    },
    "yaxis": {
      "title": "特征规则",
      "automargin": true,
      "categoryorder": "total ascending"
    },
    "margin": {
      "l": 150,
      "r": 20,
      "t": 60,
      "b": 50
    }
  },
  "data": [
    {
      "type": "bar",
      "y": [
        "花瓣宽度 (cm) <= 1.70",
        "花瓣长度 (cm) <= 4.90",
        "萼片长度 (cm) <= 5.80",
        "萼片宽度 (cm) > 2.80"
      ],
      "x": [
        0.18,
        0.12,
        -0.05,
        0.03
      ],
      "orientation": "h",
      "marker": {
        "color": [
          "#37b24d",
          "#37b24d",
          "#f03e3e",
          "#37b24d"
        ]
      }
    }
  ]
}
```

</details>



> 单个鸢尾花预测的 LIME 特征贡献可视化示例。正权重（绿色）的特征增加了预测类别的概率，而负权重（红色）的特征则降低了概率。这些规则（例如，“花瓣宽度 (cm) <= 1.70”）源自 LIME 的内部离散化或基于实例周围的扰动样本。

该可视化清晰地显示了哪些特征值（如果连续特征被离散化，通常表示为 `feature <= value` 或 `value < feature <= value2` 等规则）将预测推向了预测类别（正权重），哪些则阻碍了它（负权重），这针对的是这个*特定实例*。

### 解释回归模型

解释回归模型的过程非常相似。主要区别在于：

1. 创建 `LimeTabularExplainer` 时，设置 `mode='regression'`。
2. `predict_fn` 应该返回实际预测值（一维 NumPy 数组），而不是概率。对于 scikit-learn 回归器，这通常意味着使用 `model.predict` 方法。
3. 权重 (weight)的解释略有变化：它们代表与特定实例的该特征规则相关联的预测输出值的近似变化。

```python
# 回归示例
# 假设 'reg_model' 是一个训练好的 scikit-learn 回归器
# explainer_reg = lime.lime_tabular.LimeTabularExplainer(..., mode='regression', class_names=['TargetValue'])
# predict_fn_reg = lambda x: reg_model.predict(x)
# explanation_reg = explainer_reg.explain_instance(instance_to_explain, predict_fn_reg, num_features=5)
# explanation_reg.show_in_notebook() # 或使用其他方法
```

### 注意事项

- **预测函数封装：** 确保你的 `predict_fn` 接受 NumPy 数组并以正确形状返回概率/值，通常是最困难的部分。仔细检查它的行为。
- **分类特征：** 在 `LimeTabularExplainer` 构造函数中使用 `categorical_features` 和 `categorical_names` 参数 (parameter)指定分类特征，以便在扰动期间进行正确处理。
- **稳定性：** LIME 解释由于扰动中涉及随机抽样，有时在不同运行之间会略有差异。增加 `num_samples` 可以提高稳定性，但会花费更长时间。
- **计算成本：** 生成解释，特别是当 `num_samples` 较大时，可能计算密集。

本节提供了一个使用 LIME Python 库实现 LIME 来处理表格数据的实用指南。通过创建解释器、定义预测函数和调用 `explain_instance`，你可以为你的黑盒 (black box)模型预测生成局部、可解释的解释，从而获得关于*为什么*做出特定决策的宝贵洞察。下一章将介绍 SHAP，这是另一种具有不同理论根据的强大技术。

## 参考资料

- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778) — Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16); Publisher: ACM; Pages: 1135–1144; DOI: [10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
  首次提出LIME算法的原始研究论文，该算法用于模型无关的局部解释。
- [LIME: Local Interpretable Model-agnostic Explanations for Python - Documentation](https://lime.readthedocs.io/en/latest/) — Marco Tulio Ribeiro and the LIME Contributors (2024)
  `lime` Python库的官方文档，详细说明了不同数据类型的安装和使用示例。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/lime.html) — Christoph Molnar (2024)
  一本全面的在线书籍，其中专门一章详细解释了LIME的方法论和应用。

---

[上一节](05-%E8%A7%A3%E8%AF%BB%20LIME%20%E8%A7%A3%E9%87%8A.md) · [下一节](07-LIME%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7%E4%B8%8E%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9.md)

---
course: "basics-ml-deployment"
chapter: "preparing-model-deployment"
lesson: "using-joblib-persistence"
sourceId: 3963
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-2-preparing-model-deployment/using-joblib-persistence"
title: "使用 Joblib 保存模型"
description: "了解 Joblib 库，它常用于保存 scikit-learn 模型和大型 NumPy 数组。"
order: 4
plots: []
sourceHash: "cf49199afa145fe7881917a0879788fc92e20ee6fe44051afe7563d7625d1987"
sourceCorrections: []
---

Python 内置的 `pickle` 模块提供了一种通用方法来序列化 Python 对象，而科学计算的 Python 社区通常依赖于另一个名为 `Joblib` 的库。Joblib 是 SciPy 生态系统的一部分，包含用于高效序列化的工具，特别针对包含大型 NumPy 数组的对象进行了优化。

### 为何使用 Joblib？

`Joblib` 的序列化函数（`joblib.dump` 和 `joblib.load`）可以直接替代 `pickle.dump` 和 `pickle.load`，但在处理机器学习 (machine learning)模型，特别是来自 scikit-learn 等库的模型时，具有一个重要优势。这些模型常存储大量的数值数据数组（例如模型权重 (weight)或参数 (parameter)）。Joblib 专门设计用于比标准 `pickle` 模块更有效地处理这些 NumPy 数组，从而带来：

1. **更小的文件大小：** Joblib 在保存包含大型 NumPy 数组的对象时，通常可以生成更小的文件，尤其是在启用压缩时。
2. **更快的加载/保存速度：** 对数值数据的优化可以带来更快的序列化和反序列化时间。

由于这些优点，`joblib` 已成为保存和加载 scikit-learn 模型及管道的推荐方式。

### 安装 Joblib

如果你安装了 scikit-learn，Joblib 很可能已作为依赖项安装。但是，如果你需要单独安装它，可以使用 pip：

```bash
pip install joblib
```

### 使用 `joblib.dump` 保存模型

使用 Joblib 保存模型与使用 `pickle` 非常相似。你需要使用 `joblib.dump` 函数，传入要保存的对象和文件名。

假设你有一个名为 `model` 的已训练 scikit-learn 模型对象。以下是保存它的方法：

```python
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 示例：训练一个简单模型
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
model = LogisticRegression()
model.fit(X, y)

# --- 保存模型 ---
# 常见的文件扩展名是 '.joblib' 或 '.pkl'，如果压缩则为 '.gz'
filename = 'my_trained_model.joblib'
joblib.dump(model, filename)

print(f"模型已保存到 {filename}")
```

在这段代码中：

- 我们导入了 `joblib` 库。
- 我们训练了一个占位符 `LogisticRegression` 模型（请将其替换为你的实际训练模型）。
- `joblib.dump(model, filename)` 将我们训练好的 `model` 对象保存到 `filename` 指定的文件中。

Joblib 会自动检测对象是否包含大型 NumPy 数组并进行优化。你也可以使用 `joblib.dump` 中的 `compress` 参数 (parameter)显式控制压缩级别（例如，`joblib.dump(model, 'model_compressed.joblib.gz', compress=3)`），这可以在保存/加载时间略长一些的代价下，额外缩小文件大小。

### 使用 `joblib.load` 加载模型

要将模型加载回 Python 环境以进行预测，你需要使用 `joblib.load` 函数，并提供保存文件的路径。

```python
import joblib

# --- 加载模型 ---
filename = 'my_trained_model.joblib'
loaded_model = joblib.load(filename)

print(f"模型已从 {filename} 加载")

# 现在你可以使用加载的模型，例如进行预测
# 假设 'new_data' 是模型兼容的输入
# predictions = loaded_model.predict(new_data)
# print(predictions)
```

`loaded_model` 对象现在是你保存的原始 `model` 对象的完整重构，可以使用它进行推断。

### Joblib 与 Pickle 总结

- **使用 `pickle`：** 用于通用 Python 对象序列化。它是内置的，适用于大多数标准 Python 数据类型和对象。
- **使用 `joblib`：** 主要用于处理包含大型 NumPy 数组的对象，这在科学计算中很常见，尤其是在 scikit-learn 库中。对于这些特定情况，它能提供性能更佳和可能更小的文件大小。

可以将 Joblib 的持久化函数看作是 Pickle 的一个特别版本，它针对机器学习 (machine learning)工作流中常见的数据结构进行了优化。

> **安全提示：** 与 pickle 文件类似，Joblib 文件也可能包含恶意代码。请仅从你完全信任的来源加载 Joblib 文件。

使用 Joblib 可以确保你的 scikit-learn 模型被高效保存，使其更易于管理和传输。然而，仅仅保存模型对象并非全部。你还需要考虑正确使用它所需的运行环境和预处理步骤，我们将在接下来讨论这些。

## 参考资料

- [Joblib: persistence documentation](https://joblib.readthedocs.io/en/latest/persistence.html) — Joblib Developers (2024)
  Joblib持久化模块的官方文档，提供有关保存和加载Python对象（特别是包含大型NumPy数组的对象）的详细信息。
- [20.1. pickle - Python object serialization](https://docs.python.org/3/library/pickle.html) — Python Software Foundation (2024)
  Python `pickle`模块的官方文档，提供对通用Python对象序列化和反序列化的全面理解。

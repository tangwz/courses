---
course: "applied-data-science"
chapter: "model-deployment-fundamentals"
lesson: "saving-loading-models"
sourceId: 1905
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-5-model-deployment-fundamentals/saving-loading-models"
title: "保存和加载训练好的模型"
description: "将训练好的模型持久化以供后续使用或部署的方法。"
order: 1
plots: []
sourceHash: "663a34ebf36797079d150279acd5b0a7b7b1b2c7af32332caf352d1dcde1b982"
sourceCorrections: []
---

机器学习 (machine learning)模型训练完成后，结果通常是程序内存中的一个对象，包含学习到的参数 (parameter)和结构。为了稍后使用此模型，可能是在不同的进程中、不同的机器上，或者应用程序重启后，你需要一种方法将其状态保存到文件，然后再加载回内存。这个过程通常被称为序列化（保存）和反序列化（加载）。如果没有它，每次想使用模型时都需要重新训练，这对于实际运用来说既低效又不切实际。

Python生态系统提供了持久化机器学习模型的标准方法，特别是那些使用scikit-learn等库构建的模型。

### 为何保存模型？

保存训练好的模型的主要原因包括：

1. **持久性：** 模型代表着大量的计算投入。保存它们可以保留这份投入。
2. **可复用性：** 保存的模型可以多次加载并用于预测，无需重新训练。
3. **部署：** 保存的模型是部署到生产系统（如Web服务或批量预测任务）中，用于在新数据上进行预测的工件。
4. **共享：** 你可以与同事或其他应用程序共享训练好的模型文件。

### 使用 `pickle` 进行序列化

Python内置的 `pickle` 模块提供了一种标准方法来序列化和反序列化Python对象。由于scikit-learn模型是Python对象，你可以使用 `pickle` 保存它们。

**保存模型：**

要保存模型，你需要以二进制写入模式（`'wb'`）打开文件，并使用 `pickle.dump()`。

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# 假设 'X' 是你的特征矩阵（例如，一个 Pandas DataFrame）
# 'y' 是你的目标变量（例如，一个 Pandas Series）
# 示例占位数据：
X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6], 'feature2': [10, 12, 11, 14, 15, 13]})
y = pd.Series([0, 0, 0, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练一个简单模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 定义文件名
model_filename = 'logistic_regression_model.pkl'

# 将模型保存到文件
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"模型已保存到 {model_filename}")
```

**加载模型：**

要将模型加载回内存，你需要以二进制读取模式（`'rb'`）打开文件，并使用 `pickle.load()`。

```python
import pickle
from sklearn.linear_model import LogisticRegression # 需要此导入以便 pickle 重构对象

# 定义保存模型的文件夹
model_filename = 'logistic_regression_model.pkl'

# 从文件加载模型
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

print("模型加载成功。")

# 现在你可以使用 loaded_model 进行预测
# 示例：对测试集进行预测（确保 X_test 可用）
# predictions = loaded_model.predict(X_test)
# print(predictions)
```

**`pickle` 的限制：**

- **安全性：** 加载 pickle 文件可以执行任意代码。**切勿**从不受信任的来源加载 pickle 文件。
- **版本兼容性：** pickle 文件对保存时使用的 Python 版本和库（如 scikit-learn、NumPy、Pandas）的版本很敏感。使用不同版本加载 pickle 文件可能会失败或产生不正确的结果。
- **可读性：** pickle 文件是二进制的，不可读。

### 使用 `joblib` 处理大型数据

尽管 `pickle` 可以使用，但 `joblib` 库（`pip install joblib`）提供了 `pickle.dump` 和 `pickle.load` 的替代方案，对于包含大型 NumPy 数组的对象（这在 scikit-learn 模型中很常见）通常效率更高。scikit-learn 本身也常建议使用 `joblib` 来保存和加载模型。

**使用 `joblib` 保存模型：**

接口与 `pickle` 非常相似。

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 假设 'X' 和 'y' 的定义与之前相同
X = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200)})
y = pd.Series([0]*50 + [1]*50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练一个可能更大的模型
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 定义文件名
model_filename = 'random_forest_model.joblib'

# 使用 joblib 保存模型
joblib.dump(model, model_filename)

print(f"模型已保存到 {model_filename}")
```

**使用 `joblib` 加载模型：**

```python
import joblib
from sklearn.ensemble import RandomForestClassifier # 重构所需导入

# 定义文件名
model_filename = 'random_forest_model.joblib'

# 使用 joblib 加载模型
loaded_model = joblib.load(model_filename)

print("使用 joblib 模型加载成功。")

# 使用加载的模型
# predictions = loaded_model.predict(X_test)
# print(predictions)
```

**`joblib` 相较于 `pickle` 在机器学习 (machine learning)模型上的优势：**

- **效率：** 对于包含大型 NumPy 数组的对象，其速度可以快很多，并且生成的文件更小。
- **透明性：** 它本质上是一个针对科学计算对象优化的 `pickle` 专用封装。

它与 `pickle` 具有相同的安全顾虑和版本兼容性敏感度。在保存和加载之间，请始终确保你的环境保持一致。

> 使用 pickle 或 joblib 等序列化库保存和加载机器学习模型的工作流程。

### 框架专用格式

请注意，许多机器学习 (machine learning)框架，特别是深度学习 (deep learning)框架，提供了它们自己的专门函数和格式来保存和加载模型。这些格式通常针对特定框架的架构进行优化，不仅可以保存模型权重 (weight)，还可以保存模型结构和优化器状态。

- **TensorFlow/Keras：** 通常使用 `model.save('my_model.h5')` (HDF5 格式) 或 `model.save('my_model_directory')` (SavedModel 格式)。通过 `tf.keras.models.load_model()` 进行加载。
- **PyTorch：** 使用 `torch.save(model.state_dict(), 'model_state.pth')` 保存学习到的参数 (parameter)（推荐），或者 `torch.save(model, 'model.pth')` 保存整个模型对象（灵活性较低）。加载时需要先重建模型结构，然后使用 `model.load_state_dict(torch.load('model_state.pth'))`。
- **XGBoost/LightGBM：** 有自己的 `model.save_model()` 和相应的加载函数，通常保存为专用二进制或文本格式。

使用这些框架时，请查阅其文档，了解模型持久化的最佳实践。对于使用这些库构建的模型，通常优选使用原生格式。

### 重要事项

- **版本一致性：** 这是常见的错误源。加载模型时使用的 Python 和核心库（scikit-learn、NumPy、Pandas 等）的版本应与保存时使用的版本匹配。差异可能导致加载错误或预测中出现微小问题。使用虚拟环境（如 `venv` 或 `conda`）并明确列出依赖项（例如，在 `requirements.txt` 文件中）对于管理这一点非常重要。
- **安全风险：** 如前所述，从不受信任的来源加载 pickle 文件（`.pkl`、`.joblib` 以及可能的某些框架专用格式）存在安全风险。这些文件可以被制作成在加载时执行恶意代码。只加载你或受信任方创建的文件。
- **保存预处理步骤：** 训练好的模型通常只是预测流程的一部分。如果你对训练数据应用了预处理步骤，如缩放（例如 `StandardScaler`）、编码（例如 `OneHotEncoder`）或填充，那么在进行预测之前，你*必须*对任何新数据应用*完全相同*的转换（使用*相同的*已拟合缩放器/编码器对象）。因此，你需要将这些已拟合的预处理对象与模型一同保存。一种常见做法是将预处理步骤和模型封装在 scikit-learn 的 `Pipeline` 对象中，然后保存整个管线。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# 假设 X_train, y_train 已定义

# 创建一个管线
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# 拟合整个管线
pipe.fit(X_train, y_train)

# 保存整个管线对象
pipeline_filename = 'full_pipeline.joblib'
joblib.dump(pipe, pipeline_filename)
print(f"管线已保存到 {pipeline_filename}")

# 稍后，加载管线
loaded_pipe = joblib.load(pipeline_filename)

# 现在你可以使用 loaded_pipe.predict(new_data)
# 该管线会自动处理缩放和预测
# new_predictions = loaded_pipe.predict(X_test)
# print(new_predictions)
```

正确保存和加载模型是机器学习 (machine learning)投入实际应用的一个基本步骤。选择正确的方法（`pickle`、`joblib` 或框架专用格式）并仔细管理依赖项和预处理步骤，可以确保你的模型能够可靠地部署并用于预测。

## 参考资料

- [\`pickle\` - Python object serialization](https://docs.python.org/3/library/pickle.html) — Python Software Foundation (2024)
  Python标准库中关于`pickle`的文档，详细说明了对象序列化、反序列化和安全问题。
- [Save and load Keras models](https://www.tensorflow.org/guide/keras/save_and_serialize) — Neel Kovelamudi, Francois Chollet (2023)
  关于使用HDF5和SavedModel等格式保存Keras模型的官方指南。
- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) — Matthew Inkawhich (2018)
  Publisher: PyTorch Foundation
  PyTorch关于模型保存和加载的教程，涵盖`state_dict`和完整模型保存策略。

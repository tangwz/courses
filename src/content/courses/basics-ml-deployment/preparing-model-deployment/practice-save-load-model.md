---
course: "basics-ml-deployment"
chapter: "preparing-model-deployment"
lesson: "practice-save-load-model"
sourceId: 3973
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-2-preparing-model-deployment/practice-save-load-model"
title: "动手实践：保存和加载简单模型"
description: "逐步指南，介绍如何使用pickle或joblib训练、保存和重新加载一个简单的scikit-learn模型。"
order: 7
plots: []
sourceHash: "fb74b659b163fc3109b4242daebf74a543dac88a6321526d1b860b70c6f4745d"
sourceCorrections: []
---

保存训练好的模型是部署的必要步骤，使用Python的`pickle`和`joblib`库进行序列化可以实现这一过程。一个实际的例子将演示模型持久化的主要流程：训练一个非常简单的机器学习 (machine learning)模型，将其保存到文件中，然后重新加载它来做预测。

### 环境设置

首先，确保您已安装所需的库。我们主要使用`scikit-learn`来构建模型和生成数据，以及`joblib`，它通常被推荐用于scikit-learn对象。尽管`pickle`是内置的，但`joblib`在处理包含大型NumPy数组的对象时通常能提供更高的效率。

如果尚未安装，您可以使用pip添加它们：

```bash
pip install scikit-learn joblib
```

### 训练简单模型

让我们首先创建并训练一个基础的分类模型。我们将使用`scikit-learn`生成一些合成数据并训练一个`LogisticRegression`模型。

```python
# 导入所需库
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np # 稍后将需要numpy用于样本数据

# 生成合成分类数据
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=5, random_state=42)

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练逻辑回归模型
model = LogisticRegression(solver='liblinear', random_state=42)
print("正在训练模型...")
model.fit(X_train, y_train)
print("模型训练完成。")

# 可选：评估已训练的模型（保存前）
y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"训练数据上的准确率: {train_accuracy:.4f}")

y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"测试数据上的准确率: {test_accuracy:.4f}")
```

此时，`model`对象仅存在于您的计算机内存中。如果脚本运行结束，训练好的参数 (parameter)将丢失。

### 使用Pickle保存模型

`pickle`是Python用于对象序列化的标准库。它几乎可以保存任何Python对象。

为了保存模型，我们以二进制写入模式（`'wb'`）打开一个文件，并使用`pickle.dump()`。

```python
import pickle

# 定义保存模型的文件名
pickle_filename = 'logistic_regression_model.pkl'

print(f"正在使用pickle将模型保存到 {pickle_filename}...")
# 以二进制写入模式 ('wb') 打开文件
with open(pickle_filename, 'wb') as file:
    # 使用pickle.dump序列化模型对象并写入文件
    pickle.dump(model, file)
print("模型已成功使用pickle保存。")
```

您现在应该在您的目录中看到一个名为`logistic_regression_model.pkl`的文件。该文件包含您已训练的`LogisticRegression`模型的序列化表示。

### 使用Pickle加载模型

现在，假设我们在另一个脚本中或已重新启动了环境。我们可以使用`pickle.load()`将保存的模型重新加载到内存中。我们需要以二进制读取模式（`'rb'`）打开文件。

```python
import pickle
import numpy as np # 确保numpy已导入，如果样本数据需要的话
from sklearn.metrics import accuracy_score # 用于验证预测结果

# 定义模型保存的文件名
pickle_filename = 'logistic_regression_model.pkl'

print(f"正在使用pickle从 {pickle_filename} 加载模型...")
# 以二进制读取模式 ('rb') 打开文件
with open(pickle_filename, 'rb') as file:
    # 使用pickle.load从文件反序列化对象
    loaded_model_pickle = pickle.load(file)
print("模型已成功使用pickle加载。")

# 让我们验证加载的模型是否有效
# 创建一些样本数据（例如，第一个测试样本）
sample_data = X_test[0].reshape(1, -1) # 重塑以进行单次预测
expected_prediction = y_test[0]

# 使用加载的模型进行预测
prediction = loaded_model_pickle.predict(sample_data)
print(f"使用加载的pickle模型对样本数据进行预测: {prediction[0]}")
print(f"预期预测结果: {expected_prediction}")

# 可选：再次验证测试集上的准确率
y_pred_loaded = loaded_model_pickle.predict(X_test)
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)
print(f"使用加载的pickle模型在测试数据上的准确率: {loaded_accuracy:.4f}")
```

如您所见，加载的模型与原始模型表现完全一致，产生相同的预测结果和准确率。

### 使用Joblib保存模型

`joblib`对于保存scikit-learn模型特别有用，因为它在处理包含大型NumPy数组的对象时通常效率更高，而这种对象在机器学习 (machine learning)中很常见。其语法与`pickle`非常相似。

```python
import joblib

# 定义保存模型的文件名
joblib_filename = 'logistic_regression_model.joblib'

print(f"正在使用joblib将模型保存到 {joblib_filename}...")
# 使用joblib.dump序列化并保存模型
# joblib高效处理文件的打开/关闭
joblib.dump(model, joblib_filename)
print("模型已成功使用joblib保存。")
```

这会创建一个名为`logistic_regression_model.joblib`的文件。

### 使用Joblib加载模型

使用`joblib.load()`加载用`joblib`保存的模型同样简单。

```python
import joblib
import numpy as np # 确保numpy已导入，如果样本数据需要的话
from sklearn.metrics import accuracy_score # 用于验证预测结果

# 定义模型保存的文件名
joblib_filename = 'logistic_regression_model.joblib'

print(f"正在使用joblib从 {joblib_filename} 加载模型...")
# 使用joblib.load反序列化模型
loaded_model_joblib = joblib.load(joblib_filename)
print("模型已成功使用joblib加载。")

# 验证加载的模型
sample_data = X_test[0].reshape(1, -1)
expected_prediction = y_test[0]

prediction = loaded_model_joblib.predict(sample_data)
print(f"使用加载的joblib模型对样本数据进行预测: {prediction[0]}")
print(f"预期预测结果: {expected_prediction}")

# 可选：验证测试集上的准确率
y_pred_loaded_joblib = loaded_model_joblib.predict(X_test)
loaded_accuracy_joblib = accuracy_score(y_test, y_pred_loaded_joblib)
print(f"使用加载的joblib模型在测试数据上的准确率: {loaded_accuracy_joblib:.4f}")
```

同样，使用`joblib`加载的模型与原始模型表现完全一致。

### 保存预处理步骤

通常，您的模型依赖于在训练期间应用的特定数据预处理步骤，例如缩放或编码。在进行预测之前，这些步骤*也必须*应用于任何新数据。因此，您需要将拟合好的预处理器对象与模型一起保存。

让我们扩展这个例子，使用scikit-learn中的`StandardScaler`。

```python
# 导入所需库
from sklearn.preprocessing import StandardScaler
import joblib

# 假设X_train是您的原始训练数据
print("正在拟合StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # 拟合并转换训练数据
print("Scaler已拟合。")

# 现在，在缩放后的数据上训练模型
model_scaled = LogisticRegression(solver='liblinear', random_state=42)
print("正在缩放后的数据上训练模型...")
model_scaled.fit(X_train_scaled, y_train)
print("模型训练完成。")

# --- 同时保存scaler和模型 ---

scaler_filename = 'standard_scaler.joblib'
model_scaled_filename = 'scaled_logistic_regression_model.joblib'

print(f"正在将scaler保存到 {scaler_filename}...")
joblib.dump(scaler, scaler_filename)
print("Scaler已保存。")

print(f"正在将缩放数据上训练的模型保存到 {model_scaled_filename}...")
joblib.dump(model_scaled, model_scaled_filename)
print("缩放模型已保存。")

# --- 同时加载以进行预测 ---

print("\n正在加载scaler和缩放模型以进行预测...")
loaded_scaler = joblib.load(scaler_filename)
loaded_scaled_model = joblib.load(model_scaled_filename)
print("Scaler和缩放模型已加载。")

# --- 联合使用它们 ---
# 获取原始测试数据（例如，X_test）
# 重要：在新数据上使用transform()，而不是fit_transform()
X_test_scaled = loaded_scaler.transform(X_test)

# 使用加载的模型对缩放后的测试数据进行预测
y_pred_loaded_scaled = loaded_scaled_model.predict(X_test_scaled)
loaded_scaled_accuracy = accuracy_score(y_test, y_pred_loaded_scaled)

print(f"\n使用加载的scaler和模型在测试数据上的准确率: {loaded_scaled_accuracy:.4f}")
```

在这个例子中，我们使用`joblib`将`scaler`和`model_scaled`保存为单独的文件。稍后进行预测时，我们首先加载这两个对象，然后使用加载的`scaler`对新数据进行`transform`处理，再将其提供给加载的`model_scaled`。未能应用*完全相同*的缩放将导致不正确的预测。

您也可以将相关对象（如模型及其scaler）一起保存到一个字典或列表中，并序列化这一个对象，尽管管理单独的文件通常更清晰。

### 总结

在本次实践练习中，您学习了如何：

1. 训练一个简单的`scikit-learn`模型。
2. 使用`pickle`和`joblib`将训练好的模型保存到文件。
3. 将保存的模型从文件重新加载到内存。
4. 通过进行预测来验证加载的模型是否正常工作。
5. 认识到与模型一同保存预处理步骤（如scaler）的重要性，并在预测期间一致地应用它们。

保存和加载模型是准备机器学习 (machine learning)工作以进行部署或共享的基本步骤。通过序列化您的模型及其必要组件，您可以确保它们可以在原始训练环境之外可靠地使用。

## 参考资料

- [Model persistence](https://scikit-learn.org/stable/modules/model_persistence.html) — scikit-learn developers (2023)
  官方文档，解释了如何在scikit-learn中保存和加载机器学习模型，并特别推荐使用`joblib`。
- [joblib User Guide - Persistence](https://joblib.readthedocs.io/en/latest/persistence.html) — joblib developers (2023)
  `joblib`的官方用户指南，详细说明了其高效序列化和反序列化的功能，对于包含大型NumPy数组的对象尤其有用。
- [pickle - Python object serialization](https://docs.python.org/3/library/pickle.html) — Python Software Foundation (2023)
  Python内置`pickle`模块的官方文档，介绍了其用于序列化和反序列化Python对象结构的方法。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于构建和部署机器学习系统的综合书籍，提供了超越基本序列化的模型持久化、版本控制和操作考虑的更广阔背景。

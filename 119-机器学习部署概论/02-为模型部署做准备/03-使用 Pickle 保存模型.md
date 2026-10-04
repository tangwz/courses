# 使用 Pickle 保存模型

来源：[原文](https://apxml.com/zh/courses/basics-ml-deployment/chapter-2-preparing-model-deployment/using-pickle-persistence)

[返回章节目录](README.md) · [返回课程目录](../README.md)

机器学习 (machine learning)模型一旦训练完成，它便存在于计算机内存中，随时可以进行预测。然而，如果没有持久化机制，当Python脚本关闭或计算机关机时，该模型及其所有学到的参数 (parameter)都会消失。为了以后使用模型或与他人共享，需要一种方法将其保存到文件并在需要时重新加载。这个过程称为序列化，Python 内置的 `pickle` 模块是执行此任务的 主要 工具之一。

你可以把 pickle 处理想象成“冻结”一个 Python 对象，保存它的状态，以便之后可以完美地重建。`pickle` 模块接收你内存中的 Python 对象（比如你训练好的机器学习模型、一个列表、一个字典等），并将其转换为字节序列。这个字节流可以直接写入文件。当你再次需要该对象时，你从文件中读取字节流，并使用 `pickle` 将其“解冻”或反序列化回内存中一个功能完备的 Python 对象。

### Pickle 的工作原理：存储与加载

你将从 `pickle` 模块使用的两个主要函数是 `dump()` 和 `load()`。

1. **`pickle.dump(obj, file)`**：此函数接收你的 Python 对象（`obj`），并将其经过 pickle 处理的表示写入已打开的文件对象（`file`）。务必以二进制写入模式（`'wb'`）打开文件，因为 pickle 生成的是字节流，而非人类可读的文本。
2. **`pickle.load(file)`**：此函数从已打开的文件对象（`file`）中读取经过 pickle 处理的对象表示，并返回重建的 Python 对象。相应地，你需要以二进制读取模式（`'rb'`）打开文件。

我们来看一个简单示例。假设我们有一个字典，代表一些模型参数 (parameter)（在实际场景中，这将是你的实际训练好的模型对象）：

```python
import pickle

# 假设此字典代表一些学习到的参数
model_parameters = {
    'feature_scaling': 'standard',
    'coefficients': [0.5, -1.2, 0.8],
    'intercept': 2.1
}

# 定义我们将保存对象的文件的名称
filename = 'model_params.pkl'

# --- 保存对象 (序列化) ---
# 以二进制写入模式 ('wb') 打开文件
try:
    with open(filename, 'wb') as file:
        # 使用 pickle.dump 序列化对象并写入文件
        pickle.dump(model_parameters, file)
    print(f"对象已成功保存到 {filename}")
except Exception as e:
    print(f"保存对象出错: {e}")

# --- 加载对象 (反序列化) ---
# 假设我们关闭了脚本，现在正在重新加载它
loaded_parameters = None # 初始化变量

# 以二进制读取模式 ('rb') 打开文件
try:
    with open(filename, 'rb') as file:
        # 使用 pickle.load 从文件中反序列化对象
        loaded_parameters = pickle.load(file)
    print(f"对象已成功从 {filename} 加载")
    print("加载的参数:", loaded_parameters)

    # 验证是否相同
    print("加载的对象与原始对象相同吗?", loaded_parameters == model_parameters)

except FileNotFoundError:
    print(f"错误: 文件 '{filename}' 未找到。")
except Exception as e:
    print(f"加载对象出错: {e}")
```

当你运行此代码时，它首先创建 `model_parameters` 字典。然后，它以二进制写入模式（`'wb'`）打开 `model_params.pkl`，并且 `pickle.dump()` 将字典转换为字节并保存到该文件中。保存后，代码通过以二进制读取模式（`'rb'`）打开相同文件并使用 `pickle.load()` 从保存的字节中重建字典来模拟加载。最后，它打印加载的字典并确认它与原始字典匹配。

### 机器学习 (machine learning)模型的 Pickle 处理

这个完全相同的过程也适用于来自 scikit-learn 等库的训练好的机器学习模型。一个训练好的 scikit-learn 模型（例如 `LinearRegression`、`RandomForestClassifier`）只是一个包含所有学习信息（如系数、特征重要性、树结构等）的 Python 对象。你可以直接将这个训练好的模型对象传递给 `pickle.dump()` 来保存它。

```python
# 假设 'model' 是你训练好的 scikit-learn 模型对象
# model = train_my_model(...) # 你的训练代码在这里

# 将训练好的模型保存到文件
model_filename = 'trained_model.pkl'
try:
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print("模型保存成功。")
except NameError:
    print("注意: 变量 'model' 未定义。这是占位符代码。")
except Exception as e:
    print(f"保存模型出错: {e}")

# 之后，在另一个脚本或函数中加载模型
# loaded_model = None
# try:
#     with open(model_filename, 'rb') as file:
#         loaded_model = pickle.load(file)
#     print("模型加载成功。")
#     # 现在你可以使用 loaded_model.predict(...)
# except FileNotFoundError:
#     print(f"错误: 模型文件 '{model_filename}' 未找到。")
# except Exception as e:
#     print(f"加载模型出错: {e}")
```

### 一点提醒：安全性

务必注意 `pickle` 文件对于恶意构造的数据是不安全的。`pickle` 模块可以在反序列化（`pickle.load()`）过程中执行任意代码。因此，**绝不要从不受信任或未经认证的来源加载 pickle 文件**。仅在你自行创建或来自你隐式信任的来源的文件上使用 `pickle.load()`。

`pickle` 提供了一种简单直接的方法来持久化许多 Python 对象，包括训练好的机器学习 (machine learning)模型。它是 Python 标准库的一部分，因此你无需安装额外的东西即可使用它。然而，对于某些类型的对象，特别是 scikit-learn 模型中常见的 NumPy 大数组，另一个名为 `joblib` 的库可能会提供优势，我们接下来会讨论。

## 参考资料

- [\`pickle\` - Python object serialization](https://docs.python.org/3/library/pickle.html) — Python core developers (2024)
  提供了Python `pickle` 模块的全面文档，详细介绍了其功能、用法以及序列化和反序列化的重要安全注意事项。
- [Persistence model](https://scikit-learn.org/stable/modules/model_persistence.html) — Scikit-learn developers (2024)
  概述了保存和加载scikit-learn模型的方法，包括使用 `pickle` 进行模型持久化的实际示例。
- [Designing Machine Learning Systems: New Ways of Thinking in Product Development](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于构建和部署机器学习系统的重要书籍，为模型持久化在更广泛的机器学习部署流程中的作用提供了背景信息。

---

[上一节](02-%E6%A8%A1%E5%9E%8B%E5%BA%8F%E5%88%97%E5%8C%96%E7%AE%80%E4%BB%8B.md) · [下一节](04-%E4%BD%BF%E7%94%A8%20Joblib%20%E4%BF%9D%E5%AD%98%E6%A8%A1%E5%9E%8B.md)

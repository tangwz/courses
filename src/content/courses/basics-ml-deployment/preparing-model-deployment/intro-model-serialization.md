---
course: "basics-ml-deployment"
chapter: "preparing-model-deployment"
lesson: "intro-model-serialization"
sourceId: 3958
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-2-preparing-model-deployment/intro-model-serialization"
title: "模型序列化简介"
description: "了解序列化及其在保存 Python 对象（包括机器学习模型）中的作用。"
order: 2
plots: []
sourceHash: "f271e41a860e556c11d9e6d5a14ae9c9af3467481cc9f90e3aa1eecfcd77b895"
sourceCorrections: []
---

你已成功训练了一个机器学习 (machine learning)模型。它驻留在计算机内存中，保存着从数据中学到的所有模式。但当你的脚本运行完毕或重启计算机时会怎样呢？这个模型，就像程序中的任何其他变量一样，会消失。为了后续将这个训练好的模型用于预测，或与他人分享，你需要一种方法来保存其状态。这种将内存中的对象转换为可存储或传输格式的过程，称为**序列化**。

可以将其视为对模型对象进行一次快照，准确地捕捉其训练后的状态。序列化将这个活跃的Python对象，包括其学到的参数 (parameter)和结构，转换为字节流。然后，这个字节流可以轻松写入磁盘文件。之后，你可以读取这个文件并执行逆向过程，即**反序列化**，将完全相同的模型对象重新构建回内存。

### 机器学习 (machine learning)模型为何需要序列化？

- **持久化**：它使得你能够将训练好的模型保存到磁盘，这样每次需要时无需重新训练。训练过程可能计算成本高昂且耗时，因此保存结果很重要。
- **可移植性**：序列化的模型文件可以传输到不同的机器或环境中。这对于部署是基本要求，因为在一个系统上训练的模型通常需要在另一个系统上运行（例如生产服务器）。
- **可复用性**：你可以随时加载已保存的模型，对新数据进行预测，而无需重新运行训练代码。

### 基本流程

一般流程如下所示：

1. **训练**：在 Python 中开发并训练你的机器学习 (machine learning)模型。
2. **序列化**：使用序列化库将训练好的模型对象转换为字节流。
3. **存储**：将此字节流写入文件（例如，`my_model.pkl` 或 `my_model.joblib`）。
4. **（后续/其他地方）加载**：从文件中读取字节流。
5. **反序列化**：使用相同的库将字节流反转回功能性 Python 模型对象。
6. **预测**：使用加载的模型对新数据进行预测。

下图展示此过程：

> 该流程展示了从内存中训练好的模型对象，经由序列化转换为存储的字节流（文件），再通过反序列化回到可能不同环境中的可用模型对象。

在 Python 中，有几个库可以执行序列化。对于机器学习任务，最常用的是 Python 内置的 `pickle` 模块和 `joblib` 库，后者特别针对机器学习模型中常见的大型数值数组进行了优化。以下章节将展示如何使用这些工具来有效保存和加载你的模型。

## 参考资料

- [\`pickle\` - Python object serialization](https://docs.python.org/3/library/pickle.html) — Python Software Foundation (2024)
  Python内置序列化模块的官方文档，说明其功能和对象持久化存储的用法。
- [\`joblib\` - Running Python functions as pipeline jobs](https://joblib.readthedocs.io/en/latest/) — `joblib` developers (2024)
  `joblib`库的官方文档，强调其对大型数值数组的效率以及在机器学习模型持久化中的应用。
- [Persistence](https://scikit-learn.org/stable/modules/model_persistence.html) — Scikit-learn developers (2024)
  一个广泛使用的Scikit-learn机器学习库提供的实用指南，关于如何保存和加载模型，通常推荐使用`joblib`或`pickle`。
- [Machine Learning Design Patterns](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/) — Valliappa Lakshmanan, Sara Robinson, and Michael Munn (2020)
  Publisher: O'Reilly Media
  本书介绍了构建可靠和可扩展的机器学习系统的设计模式，包括模型部署和序列化的方法。

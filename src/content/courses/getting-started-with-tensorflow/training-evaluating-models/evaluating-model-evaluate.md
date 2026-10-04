---
course: "getting-started-with-tensorflow"
chapter: "training-evaluating-models"
lesson: "evaluating-model-evaluate"
sourceId: 1740
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models/evaluating-model-evaluate"
title: "使用 model.evaluate() 评估模型表现"
description: "使用 evaluate 方法评估您训练好的模型在未见过的数据上的表现。"
order: 5
plots: ["plots/1740-0.json"]
sourceHash: "aada1474ea655b6972b2f1e80322f5ecc361abff60432dcd5f29eeaa7db3a383"
sourceCorrections: []
---

在使用 `model.fit()` 训练模型后，接下来可以评估模型在从未见过的数据上的表现。尽管在 `fit()` 期间使用的验证数据能给出训练过程中泛化能力的一个提示，但在一个专门的 *测试集* 上进行最终评估，能提供一个对模型在实际情况中表现的偏差更小的衡量。`model.evaluate()` 方法就在这里发挥作用了。

`model.evaluate()` 方法使用提供的测试数据集，计算损失以及您在编译模型时指定的任何其他度量。它对整个数据集进行批量单次处理，计算度量，并返回最终值。与 `model.fit()` 不同，它不进行任何权重 (weight)更新，仅仅测量表现。

### 使用 `model.evaluate()`

要使用 `model.evaluate()`，您需要两项内容：

1. 您训练好的 Keras 模型对象。
2. 测试数据集，包含输入特征和对应的真实标签（或目标值）。这些数据必须独立于训练集和验证集。

该方法可以接受多种格式的测试数据，类似于 `model.fit()`：

- **NumPy 数组**：将测试特征和标签作为独立的 NumPy 数组传入：`model.evaluate(x_test, y_test)`。
- **`tf.data.Dataset`**：传入一个 `tf.data.Dataset` 对象，该对象生成 `(特征, 标签)` 的元组：`model.evaluate(test_dataset)`。对于大型数据集，通常更推荐使用 `tf.data.Dataset`，因为它与下一章讨论的输入管道结合良好。

我们来看一个常见的用法，假设您有测试数据 `x_test` 和 `y_test`：

```python
# 假设 'model' 是您已编译和训练的 Keras 模型
# 假设 'x_test' 和 'y_test' 是您的测试特征和标签

print("正在评估模型在测试数据上的表现...")
results = model.evaluate(x_test, y_test, batch_size=128, verbose=1)

# 'results' 变量包含损失和度量值
# 顺序与 model.metrics_names 一致
print("测试损失:", results[0])
print("测试准确率:", results[1]) # 假设准确率是指定的第一个度量

# 或者，如果您使用命名度量进行编译：
# results_dict = model.evaluate(x_test, y_test, return_dict=True)
# print(results_dict)
```

### 理解输出

`model.evaluate()` 方法默认返回一个标量损失值，以及模型编译时包含的任何额外度量值。

- 如果您在 `model.compile()` 期间传入了一个度量列表，`evaluate()` 会返回一个列表，其中第一个元素是测试损失，后续元素是按提供顺序计算的每个度量的值。您可以检查 `model.metrics_names` 以查看与返回的值对应的名称。
- 如果您将 `return_dict=True` 传入 `evaluate()`，它会返回一个字典，将度量名称（包括 'loss'）映射到其计算出的标量值。这通常更易读。

`verbose` 参数 (parameter)控制评估期间的输出：

- `verbose=0`：静默模式，无输出。
- `verbose=1`：进度条（默认）。
- `verbose=2`：每轮一个输出行（在这里不那么适用，因为是单次评估，但与 `fit` 保持一致）。

### 解释评估结果

评估的主要目的是估计模型的泛化能力。将从 `model.evaluate()` 获得的测试损失和度量与在 `model.fit()` 最终阶段观察到的验证损失和度量进行比较。

- **好的泛化能力**：如果测试度量值接近验证度量值，这表示您的模型对未见过的数据有良好的泛化能力。
- **潜在的过拟合 (overfitting)**：如果测试度量值明显差于验证度量值（例如，损失高很多，准确率低），您的模型可能对训练数据过拟合了，并且验证集可能未能完全代表数据，或者超参数 (parameter) (hyperparameter)可能在不经意间根据验证表现进行了调整。此时可能需要正则化 (regularization)、Dropout 或收集更多样化的训练/验证数据等方法。



![模型表现比较](plots/1740-0.json)



> 示例比较：显示训练损失/准确率、验证损失/准确率（可能来自训练结束时），以及通过 `model.evaluate()` 获得的最终测试损失/准确率。验证和测试度量值接近，表明泛化能力尚可，尽管表现低于训练数据本身，这是正常的。

请记住，测试集只应用于最终评估，*并且*应在所有训练和超参数调整（基于验证集）完成后进行。使用测试集的结果来指导后续模型开发会使它作为无偏表现估计的目的失效。严格评估您的模型能增加对其在部署时预期表现的信心。

## 参考资料

- [tf.keras.Model.evaluate](https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate) — TensorFlow Authors (2024)
  Keras `evaluate()` 方法的官方文档，详细介绍了其参数、用法和返回值。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  一本实用指南，解释了模型评估、训练/验证/测试集的作用以及如何使用 Keras 和 TensorFlow 解释结果。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 97-106
  提供了泛化、过拟合以及专用测试集在机器学习中重要性的理论基础。参见第5章。

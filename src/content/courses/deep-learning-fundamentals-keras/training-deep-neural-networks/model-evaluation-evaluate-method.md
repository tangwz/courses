---
course: "deep-learning-fundamentals-keras"
chapter: "training-deep-neural-networks"
lesson: "model-evaluation-evaluate-method"
sourceId: 4925
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-3-training-deep-neural-networks/model-evaluation-evaluate-method"
title: "模型评估：evaluate() 方法"
description: "了解如何使用 evaluate() 方法评估已训练模型在未见过数据上的表现。"
order: 8
plots: ["plots/4925-0.json"]
sourceHash: "27ebed6b497ed95e998f8b66ccd9890fea783755fe6fce553933b35259785839"
sourceCorrections: []
---

深度神经网络 (neural network)的训练通常涉及定义模型架构并使用 `fit()` 等方法。在此学习过程中，监控验证数据上的表现能提供有用的信息。然而，验证集在训练期间常被隐式或显式地用于做出决策，例如调整超参数 (parameter) (hyperparameter)或应用提前停止。因此，其性能指标可能无法完全公正地评估模型在新颖、未见过的数据上的表现。

为了对模型的泛化能力进行最终的、客观的评估，您需要在单独的数据集上进行评估。该数据集在训练或验证调优期间被保留，且从未使用过。这就是测试集的作用，Keras 提供了 `evaluate()` 方法来实现这一特定用途。

### 使用 `evaluate()` 评估泛化能力

`evaluate()` 方法在作为输入提供的数据集上计算损失以及您在 `compile()` 步骤中指定的任何其他指标。它对数据进行一次遍历，生成预测并将其与真实标签进行比较，以计算性能指标。与 `fit()` 不同，它不执行任何权重 (weight)更新。其唯一作用是进行评估。

可以这样理解：

- **训练数据：** 学生学习的材料。
- **验证数据：** 学习期间进行的练习测验，用于检查理解情况并调整学习习惯。
- **测试数据：** 学习完成后进行的期末考试，用于衡量对以前未见过问题的整体掌握程度。

在测试集上使用 `evaluate()` 为您的模型提供了这个“期末考试”分数。

### 如何使用 `evaluate()`

使用该方法非常简单。假设您有一个已训练好的 `model`，并且您的测试数据已拆分为特征 (`x_test`) 和标签 (`y_test`)，您可以像这样调用 `evaluate()` ：

```python
# 假设 'model' 是您已训练好的 Keras 模型
# 假设 'x_test' 和 'y_test' 是您的测试特征和标签

# 在测试数据上评估模型
results = model.evaluate(x_test, y_test, batch_size=128) # 您可以选择指定批次大小

# 打印结果
print("测试损失:", results[0])
print("测试准确率:", results[1]) # 假设准确率是编译的第二个指标

# 如果您有更多指标，可以通过索引访问：
# print("测试指标 2:", results[2])
# ...以此类推。
```

`evaluate()` 方法需要测试特征和相应的真实标签。它返回一个列表（如果只编译了损失，则返回一个标量），其中包含损失值，然后是 `model.compile()` 中指定的任何指标的值。返回列表中指标的顺序与编译时 `metrics` 参数 (parameter)中提供的顺序一致。

例如，如果您像这样编译模型：

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision()])
```

那么 `model.evaluate(x_test, y_test)` 返回的列表将包含：

1. `results[0]`: 在测试集上计算的分类交叉熵损失。
2. `results[1]`: 在测试集上计算的准确率。
3. `results[2]`: 在测试集上计算的精确度。

### 解释评估结果

`evaluate()` 返回的值表示模型在从未遇到过的数据上的表现。这通常被认为是衡量模型在实际应用场景中可能表现如何的最可靠方式。

比较测试指标（来自 `evaluate()`）与训练期间观察到的最终验证指标（来自 `fit()` 的输出）很有帮助。

- **相似表现：** 如果测试表现与验证表现接近，这表明您的验证集具有代表性，并且模型泛化能力良好。
- **测试表现较差：** 与验证集相比，测试集上的表现明显下降可能表明您在模型开发或超参数 (parameter) (hyperparameter)调优过程中隐式地对验证集过拟合 (overfitting)了。这也可能意味着测试集与训练/验证集相比具有略微不同的特点。
- **测试表现较好：** 这种情况虽然不那么常见，但测试集上略好的表现可能由于统计学上的偶然性而发生，尤其是在测试集较小的情况下。大幅度的提升可能需要对数据划分进行审查。

以下图表显示了一个典型场景，其中测试表现略低于验证表现，而验证表现又低于训练表现。



![模型在不同数据集上的典型表现](plots/4925-0.json)



> 训练集、验证集和测试集上获得的最终准确率分数的比较。从验证准确率到测试准确率略有下降是常见且预期的。

总而言之，`model.evaluate()` 是 Keras 中用于获取已训练模型在未见过数据上最终性能评估的标准函数。它提供了了解模型泛化能力所需的指标，在部署模型或决定下一步改进措施之前，它是一个很好的衡量标准。

## 参考资料

- [Evaluate Keras Model](https://keras.io/api/models/model_training_api/#evaluate-method) — Keras Team (2023)
  Publisher: Keras Team
  Keras `evaluate()` 方法的官方文档，详细说明了其用法和参数。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; DOI: [10.1007/978-1-4842-7935-1](https://doi.org/10.1007/978-1-4842-7935-1)
  使用 Keras 进行模型评估的实践指南，包括对测试集和泛化能力的讨论。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 105-115, 411-417
  提供了模型评估、泛化以及使用独立的训练、验证和测试数据集的重要性的基础理论。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Training Neural Networks, Part 2: Tips and Tricks](http://cs231n.github.io/neural-networks-2/) — Stanford University CS231n Course Staff (2023)
  涵盖了过拟合、泛化以及在神经网络训练和评估中正确使用验证集和测试集等重要主题。

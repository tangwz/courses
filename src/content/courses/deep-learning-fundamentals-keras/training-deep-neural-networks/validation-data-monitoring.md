---
course: "deep-learning-fundamentals-keras"
chapter: "training-deep-neural-networks"
lesson: "validation-data-monitoring"
sourceId: 4923
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-3-training-deep-neural-networks/validation-data-monitoring"
title: "验证数据与性能监控"
description: "如何使用验证数据在训练期间监控模型性能并检测过拟合。"
order: 7
plots: ["plots/4923-0.json", "plots/4923-1.json"]
sourceHash: "90ce442fa11927efb637b8dfcb4b2e5084e1aac3f6fdc86b7815e458e33aa386"
sourceCorrections: []
---

当使用 `fit()` 方法训练神经网络 (neural network)时，优化器会努力使在*训练数据*上计算出的损失函数 (loss function)最小化。在这些数据上报告的指标，例如训练准确率，能说明模型对所见特定样本的学习情况。然而，我们的最终目标不仅仅是在模型已见过的数据上表现良好；而是要构建一个能很好地*泛化*到全新、未见数据的模型。如何在训练过程中评估这种泛化能力呢？这就是验证数据的作用所在。

仅仅监控训练损失和指标可能会产生误导。模型可能会变得极其擅长预测训练样本，实际上是“记忆”了它们，包括它们的噪声和特性。这种现象，称为*过拟合 (overfitting)*，会导致在训练集上表现出色，但在遇到新数据时表现不佳。模型未能捕获泛化所需的潜在规律。

### 划分验证数据

为了更实际地评估模型在未见数据上的表现，我们需要定期在一个模型未参与训练的单独数据集上对其进行评估。这就是*验证集*。在Keras中，在训练期间引入验证有两种主要方式：

1. **使用 `validation_split`：** 您可以专门保留一部分训练数据用于验证。Keras 会在训练开始*前*自动划出指定百分比的数据，并在每个周期结束时专门将其用于验证。如果您没有预定义的验证集，这会很方便。

   ```python
   # 假设 x_train 和 y_train 包含您的全部训练数据
   history = model.fit(x_train, y_train,
                       epochs=30,
                       batch_size=64,
                       validation_split=0.2) # 使用最后 20% 的数据进行验证
   ```
2. **使用 `validation_data`：** 您可以提供一个元组 `(x_val, y_val)`，其中包含一个完全独立的验证数据集的特征和标签。这通常是更受欢迎的方式，因为它让您能更好地控制，以确保验证集具有代表性且与训练过程独立。

   ```python
   # 假设 x_train_p, y_train_p 是训练部分
   # x_val, y_val 是单独的验证集
   history = model.fit(x_train_p, y_train_p,
                       epochs=30,
                       batch_size=64,
                       validation_data=(x_val, y_val)) # 使用提供的验证集
   ```

重要的是，验证数据应来自与训练数据相同的底层分布，但其中包含模型在训练更新过程中*未曾*见过的样本。模型的权重 (weight)*绝不会*根据验证集的表现进行更新；它纯粹用于监控。

### 监控性能指标

当您使用 `fit()` 方法并结合 `validation_split` 或 `validation_data` 时，Keras 会在每个周期结束时在验证集上评估模型。结果会被添加到 `fit()` 返回的 `History` 对象中。这些指标通常以 `val_` 为前缀。例如，如果您使用 `loss='binary_crossentropy'` 和 `metrics=['accuracy']` 编译模型，`history.history` 字典将包含如下键：

- `loss`：每个周期的训练损失。
- `accuracy`：每个周期的训练准确率。
- `val_loss`：每个周期的验证损失。
- `val_accuracy`：每个周期的验证准确率。

绘制这些指标随周期变化的图表是理解训练动态的常规做法。

### 解释训练与验证曲线

分析训练与验证损失及指标的图表，对于诊断过拟合 (overfitting)或欠拟合 (underfitting)等潜在问题非常基础。

- **理想情况：** 训练和验证损失都稳步下降并收敛到相似的低值。同样，训练和验证准确率上升并停留在相似的高值。这表明模型正在学习可泛化的规律。
- **过拟合：** 训练损失持续下降（或训练准确率提高），但在若干周期后，验证损失开始上升（或验证准确率停滞或下降）。这种背离表明模型开始记忆训练数据，并失去泛化能力。训练和验证曲线之间的差距显著拉大。



![训练与验证损失对比（过拟合示例）](plots/4923-0.json)



> 训练损失稳步下降，而验证损失在第20个周期后开始上升，表明出现过拟合。

- **欠拟合：** 训练和验证损失都保持在高位，或者下降非常缓慢并在高值处停滞。同样，准确率保持在低位。这表明模型过于简单，或者训练时间不足以捕获数据中的潜在规律。



![训练与验证损失对比（欠拟合示例）](plots/4923-1.json)



> 训练和验证损失都较高且下降缓慢，表明出现欠拟合。

"因此，监控验证性能非常重要。它提供了模型在未见数据上表现的替代指标，并有助于确定训练期间模型达到最佳泛化能力的时刻。及早识别过拟合，可以使您在模型在新数据上的性能下降之前停止训练。像*早期停止*和*正则化 (regularization)*这样的方法（我们将在第6章讨论），直接利用验证监控来提升模型的泛化能力。目前，请记住在 `fit()` 期间观察 `val_` 指标是您评估训练是否朝正确方向前进的主要工具。"

## 参考资料

- [Train and evaluate models](https://keras.io/api/models/model_training_apis/) — Keras team (2024)
  Publisher: Keras
  Keras模型训练的官方指南，解释了`fit()`方法的验证和性能监控参数。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  深度学习基础的权威资源，包含泛化、过拟合和验证集的作用。
- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.archambault.ca/books/hands-on-machine-learning-with-scikit-learn-keras-tensorflow-concepts-tools-and-techniques-to-build-intelligent-systems-3rd-ed-aur%C3%A9lien-g%C3%A9ron/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 10 (Training Deep Neural Networks)
  实用指南，涵盖Keras实现细节、模型评估以及训练和验证曲线的解读。

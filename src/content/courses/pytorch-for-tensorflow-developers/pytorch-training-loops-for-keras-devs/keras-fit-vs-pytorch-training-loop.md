---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-training-loops-for-keras-devs"
lesson: "keras-fit-vs-pytorch-training-loop"
sourceId: 5758
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs/keras-fit-vs-pytorch-training-loop"
title: "训练模式：TensorFlow 的 fit 方法与 PyTorch 训练循环"
description: "了解 TensorFlow Keras 的高级 `fit()` 方法与 PyTorch 中手动训练循环之间的差异。"
order: 1
plots: []
sourceHash: "57e7ac57f645409bb81065dc5c4f5b870bfe787c8563851b4a69cd9c2ba9e0ef"
sourceCorrections: []
---

如果您的背景是 TensorFlow Keras，您对高效的训练流程会很熟悉。您定义模型，然后使用 `model.compile()` 来指定优化器、损失函数 (loss function)和任何评估指标。之后，只需一次 `model.fit()` 调用，提供您的训练数据、训练轮数（epochs）和批量大小（batch size），即可启动整个训练流程。Keras 会高效地管理数据迭代、梯度计算和模型权重 (weight)更新，所有这些都隐藏在此便捷的 API 之后。

PyTorch 处理模型训练的方式不同。与 `fit()` 这种高级的、包罗万象的函数不同，PyTorch 要求您自己构建训练循环。这最初可能看起来工作量更大，但它提供了一个重要的好处：对训练过程的每一步都拥有完全的透明度和控制权。这种理念与 PyTorch 整体的“定义即运行”特性相符，即操作在声明时即执行，从而提供灵活性并简化调试。

### Keras 的 `fit()` 方法：回顾

在 Keras 中，训练过程在很大程度上被抽象化了。典型的工作流程包括：

1. **编译（`model.compile()`）：** 您指定优化器（例如，`'adam'`、`tf.keras.optimizers.SGD`）、损失函数 (loss function)（例如，`'categorical_crossentropy'`、`tf.keras.losses.MeanSquaredError`）以及可选的评估指标（例如，`['accuracy']`）。
2. **训练（`model.fit()`）：** 您传入训练数据（特征和标签）、训练轮数、批量大小，以及可选的验证数据和回调函数。Keras 随后处理：
   - 迭代指定的训练轮数。
   - 将数据分成批次。
   - 对于每个批次：
     - 执行前向传播以获取预测结果。
     - 计算损失。
     - 执行反向传播 (backpropagation)以计算梯度。
     - 使用所选优化器更新模型权重 (weight)。
   - 评估评估指标。
   - 在不同时间点（例如，训练轮结束时、批次结束时）执行回调函数。

这种方式因其简洁性以及快速启动并运行标准模型的能力而显得十分强大。

```python
# TensorFlow Keras 示例
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# history = model.fit(train_images, train_labels,
#                     epochs=10,
#                     validation_data=(test_images, test_labels))
```

这个简洁的 Keras 代码片段在内部封装了一系列复杂的运算。

### PyTorch 训练循环：显式构建

在 PyTorch 中，您是训练循环的设计者。您编写标准 Python 代码来迭代训练轮数和数据批次，并显式调用训练每一步所需的函数。一个典型的 PyTorch 训练循环包括以下核心组成部分：

1. **外层循环（训练轮数）：** 迭代所需的训练轮数。
2. **内层循环（批次）：** 迭代您的 `DataLoader`，它提供数据批次。
3. **梯度清零：** 在计算当前批次的梯度之前，您必须显式清除之前批次累积的所有梯度。这通过 `optimizer.zero_grad()` 完成。
4. **前向传播：** 将输入数据通过模型以获取预测结果：`outputs = model(inputs)`。
5. **计算损失：** 使用所选损失函数 (loss function)计算模型预测与真实目标之间的损失：`loss = loss_fn(outputs, targets)`。
6. **反向传播 (backpropagation)：** 计算损失相对于所有 `requires_grad=True` 的模型参数 (parameter)的梯度。这由 `loss.backward()` 启动。
7. **更新权重 (weight)（优化器步进）：** 使用优化器的 `step()` 方法，根据计算出的梯度更新模型参数：`optimizer.step()`。
8. **日志记录与评估指标（可选但推荐）：** 跟踪损失值、计算准确率或其他相关评估指标，并记录它们以监控训练进展。

以下是 PyTorch 训练循环的骨架表示：

```python
# PyTorch 训练循环片段
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# loss_fn = torch.nn.CrossEntropyLoss()

# for epoch in range(num_epochs):
#     model.train() # 将模型设置为训练模式
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         # 将数据移动到相应的设备（例如，GPU）
#         inputs, labels = inputs.to(device), labels.to(device)

#         # 1. 梯度清零
#         optimizer.zero_grad()

#         # 2. 前向传播
#         outputs = model(inputs)

#         # 3. 计算损失
#         loss = loss_fn(outputs, labels)

#         # 4. 反向传播
#         loss.backward()

#         # 5. 更新权重
#         optimizer.step()

#         running_loss += loss.item()
#     # 打印训练轮统计数据，进行验证等。
```

### 模式对比

根本区别在于抽象程度与显式控制之间的对比。

- **抽象（Keras `fit()`）：**

  - **优点：** 级别高，样板代码少，适合标准任务的快速实现。隐藏了训练循环的复杂性。
  - **缺点：** 定制非标准训练行为或调试训练过程中深层问题可能更具挑战性，因为循环逻辑是内部的。
- **显式控制（PyTorch 循环）：**

  - **优点：** 对每一步都有完全控制。易于插入自定义逻辑、检查中间张量、修改梯度或实现复杂的训练流程（例如，GAN、强化学习 (reinforcement learning)算法）。对训练机制提供更高的透明度。调试通常更直接，因为您可以在 Python 代码的任何位置使用 `print()` 或调试器。
  - **缺点：** 需要编写更多代码。正确实现循环的责任在于开发者。

> 这个图表显示了 Keras 的 `model.fit()` 封装的特性与 PyTorch 训练循环的显式、分步构建之间的对比。

### 对开发的影响

**灵活性和定制性：** 如果您需要实现新颖的训练算法，在优化器步进之前以特定方式修改梯度，或者集成 Keras 回调函数不易支持的复杂日志记录，那么 PyTorch 方法本身就更具灵活性。您只是在编写 Python 代码，因此任何可以用 Python 表示的逻辑都可以成为您训练循环的一部分。

**调试：** 调试 PyTorch 训练循环感觉更直接。由于您编写了该循环，因此可以在任何时候插入 `print()` 语句或使用 Python 的 `pdb` 调试器来查看张量形状、值和梯度。例如，检查激活或梯度中的 `NaN` 值是简单的。

**了解机制：** 亲自编写训练循环会促使您更透彻地了解模型训练期间发生的情况。这对于故障排除以及培养对不同组件（优化器、损失函数 (loss function)、学习率调度器）如何共同发挥作用的直觉都有好处。

虽然 PyTorch 的方法意味着训练过程的初始设置会多一点，但它提供的控制和透明度备受重视，尤其是在研究环境或解决不完全符合预定义训练抽象的复杂问题时。随着您在本章中的学习，您将了解到如何实现该循环的每个组成部分，从选择损失函数和优化器到计算评估指标和管理训练流程。

## 参考资料

- [Training and evaluation with the built-in methods](https://www.tensorflow.org/guide/keras/train_and_evaluate) — TensorFlow Authors (2023)
  Publisher: TensorFlow
  提供 Keras 高级训练 API 的官方指南，包括 `model.compile()` 和 `model.fit()`，解释其简化的方法。
- [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — PyTorch Documentation Authors (2017)
  Publisher: PyTorch Foundation
  说明如何从头构建一个完整的 PyTorch 训练循环，展示对每个训练步骤的明确控制。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras 创建者撰写的书籍，对 Keras 及其训练抽象的设计理念进行了全面说明。

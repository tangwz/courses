---
course: "introduction-to-neural-networks"
chapter: "building-first-neural-network"
lesson: "monitoring-training"
sourceId: 1991
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-5-building-first-neural-network/monitoring-training"
title: "监控训练进程"
description: "在训练过程中记录损失和准确率指标。"
order: 5
plots: ["plots/1991-0.json"]
sourceHash: "8014b93e25aea7092a801915c629d1ef66fcff9b52ea6fc01783983bb34f1e98"
sourceCorrections: []
---

一旦您设置好了网络架构，初始化了参数 (parameter)，并启动了训练循环来执行前向传播、计算损失、执行反向传播 (backpropagation)以及通过梯度下降 (gradient descent)更新权重 (weight)，您如何知道它是否真的在*学习*有用的东西呢？仅仅运行循环是不够的。您需要观察这个过程，以理解其表现和有效性。监控训练进程对于构建成功的模型来说是根本的。

如果没有监控，您实际上是在盲目训练。您将不知道模型是否在改进，学习速度如何，或者何时停止训练过程。随时间观察重要指标可以提供关于学习动态的信息，并有助于发现潜在问题。

### 要记录的指标

在训练期间，有两种主要的指标是不可或缺的：

1. **损失函数 (loss function)值：** 这是优化算法（如梯度下降 (gradient descent)）试图最小化的直接度量。追踪损失会告诉您，根据所选目标（例如，回归任务中的均方误差，分类任务中的交叉熵损失），模型的预测是否正在接近实际目标值。

   - **训练损失：** 在当前迭代中用于梯度计算和权重 (weight)更新的数据批次上计算。训练损失的下降通常表明模型正在学习适应训练数据。
   - **验证损失：** 定期（例如，在每个时期结束时）在模型*未曾*训练过的独立验证数据集上计算。此指标评估模型对未见过的数据的泛化能力。比较训练损失和验证损失对于识别过拟合 (overfitting)有益。
2. **性能指标：** 尽管损失引导着优化，但它可能不是衡量模型执行特定任务*表现如何*的最直观度量。因此，我们还记录与问题相关的指标：

   - **准确率：** 对于分类任务，这通常是正确分类的样本百分比。与损失类似，我们记录训练准确率和验证准确率。
   - **其他指标：** 根据任务的不同，其他指标可能更适合（例如，不平衡分类的精确率、召回率、F1分数；回归的平均绝对误差(MAE)或R方）。

### 观察随时间变化的指标

真正的意义在于观察这些指标在整个训练过程中如何变化，通常是跨越多个时期。

- **日志记录：** 最简单的监控形式是定期（例如，在每个时期之后或甚至在一定数量的批次之后）将指标值打印到控制台或日志文件中。

```python
# 训练循环中的位置
for epoch in range(num_epochs):
    # --- 训练阶段 ---
    model.train() # 将模型设置为训练模式
    running_train_loss = 0.0
    running_train_accuracy = 0.0
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate training metrics
        running_train_loss += loss.item() * inputs.size(0)
        # (根据输出和标签计算准确率)
        # running_train_accuracy += calculate_accuracy(outputs, labels) * inputs.size(0)

当神经网络进行训练时，其架构被配置，参数被初始化，并且一个循环执行前向传播，计算损失，执行反向传播，并通过梯度下降更新权重。在此过程中，如何判断网络是否真正学到了有用的东西？仅仅让训练循环运行是不够的。观察这个过程对于理解其行为和有效性是必要的。监控训练进度是构建成功模型的基础。

    # --- 验证阶段 ---
    model.eval() # 将模型设置为评估模式
    running_val_loss = 0.0
    running_val_accuracy = 0.0
    with torch.no_grad(): # 禁用梯度计算
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)
            # (根据输出和标签计算准确率)
            # running_val_accuracy += calculate_accuracy(outputs, labels) * inputs.size(0)

    epoch_val_loss = running_val_loss / len(validation_loader.dataset)
    # epoch_val_accuracy = running_val_accuracy / len(validation_loader.dataset)

    # 记录当前时期的指标
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {epoch_train_loss:.4f} | "
          # f"Train Acc: {epoch_train_accuracy:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}")
          # f"Val Acc: {epoch_val_accuracy:.4f}")

    # 存储指标以便后续绘图
    train_loss_history.append(epoch_train_loss)
    val_loss_history.append(epoch_val_loss)
    # train_acc_history.append(epoch_train_accuracy)
    # val_acc_history.append(epoch_val_accuracy)
```

- **可视化：** 将指标随时期绘制成图，可以更清晰地呈现训练趋势。可视化使发现规律、收敛行为以及过拟合 (overfitting)或欠拟合 (underfitting)等潜在问题变得更容易。

### 训练曲线的解读

让我们查看在绘制训练和验证指标时可能看到的常见模式：

- **健康训练：** 训练损失和验证损失都稳步下降，然后趋于平稳。验证损失可能略高于训练损失，但它们通常遵循相似的趋势。相应地，训练准确率和验证准确率（或其他性能指标）上升并趋于平稳。这表明模型正在学习并泛化良好。
- **过拟合 (overfitting)：** 训练损失持续下降，而验证损失在某一点之后开始上升（或比训练损失明显更早地趋平）。类似地，训练准确率可能持续上升，而验证准确率停滞甚至下降。这表明模型对训练数据学习得太好，包括其噪声，并且正在失去对新数据的泛化能力。这是一个常见问题，第6章讨论的技术（如正则化 (regularization)和早期停止）用于解决它。
- **欠拟合 (underfitting)：** 训练损失和验证损失都保持较高水平或下降非常缓慢，并在高值处趋于平稳。两组的性能指标（如准确率）都保持较低。这表明模型对于任务可能过于简单，训练时长不足，或者优化过程中存在问题（例如，学习率过小）。
- **学习率问题：** 损失曲线的形状有时可以暗示学习率$\alpha$存在问题。

  - *过高：* 损失可能会剧烈波动甚至发散（不受控制地增加）。
  - *过低：* 损失下降非常缓慢，可能需要过长的时间才能收敛，或者陷入次优局部最小值。

这里有一个展示训练损失和验证损失的示例图，可能在训练后期显示出过拟合的迹象：



![各时期训练和验证损失](plots/1991-0.json)



> 训练损失（蓝色）持续下降，而验证损失（橙色）最初下降，但在时期10之后开始上升，表明出现过拟合。

监控这些指标不仅仅是观察；它关乎做出明智的决定。您是否需要提前停止训练？是否应该调整学习率？是否需要不同的模型架构？模型是否需要正则化？分析训练进程提供了必要的反馈，以回答这些问题并指导构建有效的神经网络 (neural network)。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖深度学习的理论基础，包括训练过程、优化、损失函数和泛化。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=f_B3EAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  实用指南，展示如何使用常用库实现和监控神经网络训练，包括讨论跟踪指标和诊断训练问题（第4版）。
- [CS231n: Convolutional Neural Networks for Visual Recognition. Lecture Notes.](http://cs231n.github.io/) — Fei-Fei Li, Justin Johnson, and Serena Yeung (2024)
  官方课程讲义，清晰地解释了神经网络训练，包括损失函数、优化以及监控和解释训练曲线的实用建议。

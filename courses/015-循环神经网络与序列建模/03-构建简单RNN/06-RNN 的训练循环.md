# RNN 的训练循环

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-3-building-simple-rnns/training-loop-rnns)

[返回章节目录](README.md) · [返回课程目录](../README.md)

您已经使用所选的深度学习 (deep learning)框架构建了简单循环神经网络 (neural network) (RNN)的组成部分。您了解如何定义层并构建模型以处理序列数据。但仅有模型结构本身并不能学习。下一步是训练它，这涉及向其展示数据，衡量其预测的错误程度，并随着时间推移调整其内部参数 (parameter)（权重 (weight)和偏置 (bias)）以提升这些预测。这一迭代过程在训练循环中进行管理。

让我们分解专为 RNN 模型设计的典型训练循环的结构和组成部分。尽管具体语法在 TensorFlow 和 PyTorch 之间会有细微差别，但基本思想和工作流程保持一致。

### 核心训练周期

从根本上说，训练神经网络 (neural network)（包括 RNN）是一个优化问题。我们希望找到能使特定损失函数 (loss function)最小化的模型参数 (parameter)，该函数量化 (quantization)了模型预测与实际目标值之间的误差。训练循环通过重复执行以下步骤来促成这一过程：

1. **数据获取：** 从数据集中获取一批输入序列及其对应的目标序列。
2. **前向传播：** 将输入序列通过 RNN 模型，生成输出预测。
3. **损失计算：** 使用所选的损失函数（例如，回归任务的均方误差，分类任务的交叉熵）将模型的预测与真实目标序列进行比较，计算损失。
4. **反向传播 (backpropagation)（梯度计算）：** 计算损失函数相对于模型中每个可训练参数的梯度。对于 RNN，此计算通过网络的层以及通过循环连接*随时间反向*传播梯度，使用第 2 章中讨论的“随时间反向传播 (BPTT)”算法。
5. **参数更新：** 使用优化器（例如 Adam、SGD、RMSprop）调整模型参数。优化器使用计算出的梯度向（理想情况下）最小化损失的方向迈进。
6. **重复：** 对多个批次重复步骤 1-5，直到整个数据集处理完毕（完成一个*训练周期*）。然后，对多个训练周期重复整个过程。

### 循环的可视化

我们可以将此流程可视化为一个周期：

> 典型的训练循环会迭代多个训练周期和批次，对每个批次执行前向传播、损失计算、反向传播 (backpropagation)（BPTT）和参数 (parameter)更新。

### 代码中的组成部分

让我们看看伪代码结构。假设您已经定义了 `model`、`loss_function`、`optimizer`，并且有一个 `data_loader` 可以生成批次的 `(input_sequences, target_sequences)`。

```python
# --- 超参数 ---
num_epochs = 10
learning_rate = 0.001
# ... 其他超参数

# --- 模型、损失、优化器 ---
# model = build_your_rnn_model() # 在前几节中定义
# loss_function = choose_appropriate_loss() # 例如，均方误差、交叉熵
# optimizer = choose_optimizer(model.parameters(), lr=learning_rate) # 例如，Adam

# --- 训练循环 ---
for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0.0
    num_batches = 0

    # 遍历数据批次
    for input_sequences, target_sequences in data_loader:
        # 1. 清除上一步的梯度（重要！）
        optimizer.zero_grad() # 语法在不同框架之间略有差异

        # 2. 前向传播：获取模型预测
        # 如果适用，确保数据位于正确设备（CPU/GPU）上
        predictions = model(input_sequences)

        # 3. 损失计算：比较预测与目标
        # 如有必要，重塑预测/目标以匹配损失函数要求
        loss = loss_function(predictions, target_sequences)

        # 4. 反向传播：计算梯度
        loss.backward() # 这会在 RNN 中触发 BPTT

        # 可选：梯度裁剪（有助于防止梯度爆炸，见第 4 章）
        # framework.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. 优化器步骤：更新模型权重
        optimizer.step()

        # --- 追踪（可选但推荐）---
        epoch_loss += loss.item() # .item() 从损失张量中获取标量值
        num_batches += 1

    # 训练周期结束
    average_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} finished. Average Loss: {average_epoch_loss:.4f}")

print("训练完成。")
```

### RNN 的重要考量

- **梯度清零：** 在每次反向传播 (backpropagation)之前清除梯度（`optimizer.zero_grad()` 或类似操作）非常必要。否则，来自先前批次的梯度将累积，导致更新不正确。
- **输入/输出形状：** 确保您的 `input_sequences`、`target_sequences` 和 `predictions` 具有模型和损失函数 (loss function)所期望的形状。这通常需要仔细处理批次、时间步和特征维度。
- **状态管理：** 在 `SimpleRNN`、`LSTM` 或 `GRU` 层等简单的框架实现中，隐藏状态通常在*每个批次*内部进行管理。对于每个新批次，状态会自动重置。对于更高级的应用或手动实现，您可能需要显式管理隐藏状态，在批次之间传递或有策略地重置它。
- **梯度裁剪：** 正如伪代码中提到的，RNN 在 BPTT 期间有时会遇到梯度爆炸问题（梯度变得过大），特别是对于长序列。梯度裁剪是一种常用技术，通过在梯度范数超过特定阈值时将其缩小来缓解此问题。我们将在第 4 章中更多地讨论这一点。
- **设备放置：** 对于大型模型或数据集，您通常会在 GPU 上训练。确保您的模型和数据张量已移至适当的设备（例如，PyTorch 中的 `.to(device)` 或 TensorFlow 中的 `tf.device` 上下文 (context)管理器）。

这种结构化的循环提供了机制，根据模型观察到的数据迭代地优化您的 RNN 模型。下一节“动手实践：简单序列预测”将利用这些思想并使用特定的深度学习 (deep learning)框架来在具体任务上训练 RNN。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面教科书，内容涵盖反向传播、优化算法、循环神经网络以及梯度裁剪等训练考量。
- [CS231n Convolutional Neural Networks for Visual Recognition: Optimization (Part 2)](http://cs231n.github.io/optimization-2/) — Andrej Karpathy, Justin Johnson, and Fei-Fei Li (2023)
  Publisher: Stanford University
  提供对反向传播算法、梯度计算和训练神经网络所需的优化策略的清晰解释。

---

[上一节](05-%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95RNN%E6%A8%A1%E5%9E%8B.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%AE%80%E5%8D%95%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B.md)

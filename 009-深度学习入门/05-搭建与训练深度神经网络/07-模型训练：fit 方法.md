# 模型训练：fit 方法

来源：[原文](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-5-building-training-dnns/training-the-model-fit)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练深度学习 (deep learning)模型是它从数据中学习的主要步骤。此过程始于模型定义好其层和激活函数 (activation function)，并用指定的损失函数 (loss function)（用于衡量误差）和优化器（如 Adam 或 SGD，用于指导权重 (weight)更新）编译完成。训练涉及向模型提供准备好的数据，让优化器迭代地调整权重，以最小化选定的损失函数。

训练神经网络 (neural network)本质上是一个迭代过程。我们并非只向模型展示数据一次。相反，我们反复地让它接触数据，使其逐步掌握基本规律。每次迭代包含以下几个步骤：

1. **前向传播 (Forward Pass)：** 一批训练数据（输入）逐层通过网络，生成预测（输出）。
2. **损失计算 (Loss Calculation)：** 模型的预测值与编译时定义的损失函数所依据的实际目标值（标签）进行比较。这会得出一个单一的数值，表示模型在该批次上的误差程度。
3. **反向传播 (backpropagation) (Backward Pass)：** 计算出的损失用于计算损失相对于网络中每个权重和偏差的梯度。这表明每个参数 (parameter)对误差的贡献程度以及调整方向。
4. **权重更新 (Weight Update)：** 优化器使用计算出的梯度及其内部逻辑（例如，包含学习率、动量）来更新模型的权重和偏差，旨在减少下一次迭代中的损失。

这个循环对许多数据批次重复进行。

深度学习框架提供了方便的方式来管理这个训练循环。在 Keras 中，这通常通过一个名为 `fit` 的方法完成。在 PyTorch 中，你通常需要明确地编写循环，从而提供更精细的控制。无论具体的实现如何，核心思想保持不变，你需要指定几个重要参数：

### 训练数据

这是模型将学习的输入数据（`X_train`）和相应的目标标签（`y_train`）。我们假设这些数据已按前述内容进行了预处理（例如，缩放、重塑）。

### 周期数（Epochs）

一个**周期**代表对*整个*训练数据集的一次完整遍历。如果你的数据集有 10,000 个样本，并且你训练 10 个周期，模型在训练期间将看到每个样本 10 次（尽管可能在不同的批次和顺序中）。

选择周期数很重要：

- **周期数过少：** 模型可能没有充分接触数据以有效学习，导致*欠拟合 (underfitting)*。
- **周期数过多：** 模型可能会过度学习训练数据中的噪声和特定细节，导致对新的、未见过的数据泛化能力下降。这称为*过拟合 (overfitting)*。我们还会浪费计算资源。像早期停止（稍后会介绍）这样的技术有助于减轻这种情况。

### 批次大小（Batch Size）

为了避免在每次权重 (weight)更新时一次性处理整个数据集（这对于大型数据集而言计算上不可行），我们通常将训练数据分成更小的块，称为**小批次**。**批次大小**定义了在模型权重更新之前，每个前向/反向传播 (backpropagation)中处理的训练样本数量。

- **批量梯度下降 (gradient descent)：** 批次大小 = 整个训练集的大小。计算精确梯度，但速度慢且占用大量内存。
- **随机梯度下降（SGD）：** 批次大小 = 1。每个样本处理后更新权重。迭代速度快，但噪声梯度可能导致收敛不稳定。
- **小批量梯度下降：** 批次大小设置在 1 和数据集大小之间（通常是 2 的幂，如 32、64、128、256）。这是最常见的方法，在梯度精度、计算效率和内存使用之间取得了平衡。

批次大小的选择影响：

- **内存需求：** 更大的批次需要更多内存（GPU RAM）。
- **训练速度：** 更大的批次有时因硬件并行化而处理更快，但每次更新的计算时间更长。更小的批次意味着每个周期内更新更频繁。
- **梯度噪声：** 更小的批次会给梯度估计引入更多噪声，这有时有助于跳出局部最小值，但也可能阻碍收敛。更大的批次提供更稳定的梯度估计。

> 一个周期涉及处理整个数据集，通常会分割成小批次。对于每个批次，模型会执行前向传播，计算损失，执行反向传播，并通过优化器更新其权重。这个循环在周期内的所有批次中重复进行，然后重复多个周期。

### 示例：PyTorch 训练循环

让我们看看这在 PyTorch 中是如何呈现的。假设你拥有 `model`、`criterion`（损失函数 (loss function)，例如 `nn.CrossEntropyLoss`）、`optimizer`（例如 `optim.Adam`）以及一个 `train_loader`（一个提供数据批次的 `DataLoader`）。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 假设这些在其他地方定义：
# model: 你的神经网络模型 (nn.Module 的子类)
# criterion: 你的损失函数 (例如, nn.CrossEntropyLoss())
# optimizer: 你的优化器 (例如, optim.Adam(model.parameters(), lr=0.001))
# X_train_tensor, y_train_tensor: 作为 PyTorch 张量的训练数据
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 如果可能，在 GPU 上运行

# --- 超参数 ---
BATCH_SIZE = 64
NUM_EPOCHS = 10

# --- 准备 DataLoader ---
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# shuffle=True 对于训练很重要，可确保每个周期批次不同
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True) 

# 将模型移动到正确的设备（CPU 或 GPU）
# model.to(device) 

# --- 训练循环 ---
model.train() # 将模型设置为训练模式（对 Dropout、BatchNorm 等层很重要）

print("开始训练...")
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    num_batches = len(train_loader)

    for i, batch in enumerate(train_loader):
        # 1. 从批次获取数据并移动到设备
        # inputs, labels = batch[0].to(device), batch[1].to(device) 
        inputs, labels = batch # 为简单起见，假设数据已在正确设备或 CPU 上

        # 2. 将参数梯度清零（在反向传播前必不可少）
        optimizer.zero_grad()

        # 3. 前向传播：计算预测
        outputs = model(inputs)

        # 4. 计算损失
        loss = criterion(outputs, labels)

        # 5. 反向传播：计算梯度
        loss.backward()

        # 6. 优化：根据梯度更新权重
        optimizer.step()

        # 累加损失用于报告
        running_loss += loss.item() # .item() 从损失张量中获取标量值

        # 可选：定期打印进度
        if (i + 1) % 100 == 0 or (i + 1) == num_batches: # 每 100 个小批次或在周期结束时打印
             print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{i + 1}/{num_batches}], Loss: {loss.item():.4f}')
             # 注意：对于运行平均损失：{running_loss / (i + 1):.4f}

    epoch_loss = running_loss / num_batches
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] completed. Average Loss: {epoch_loss:.4f}')

print('训练完成')

# --- Keras 等效代码 ---
# 为作比较，Keras 的等效代码封装了此循环：
# history = model.fit(X_train_tensor.numpy(), y_train_tensor.numpy(),
#                     epochs=NUM_EPOCHS,
#                     batch_size=BATCH_SIZE,
#                     shuffle=True,
#                     verbose=2) # verbose 控制打印信息的详细程度
# print('训练完成')
```

这个 PyTorch 循环明确执行了前面列出的步骤：为每个批次清零梯度、前向传播、损失计算、反向传播 (backpropagation)和优化器步进。外层循环遍历指定数量的周期。设置 `model.train()` 很重要，因为某些层在训练和评估期间表现不同。Keras 将此循环抽象为 `model.fit()` 调用，根据你提供的 `epochs` 和 `batch_size` 参数 (parameter)，在内部管理批次迭代、混洗和更新。

执行这个训练循环（无论是显式地还是通过 `fit` 这样的方法）就是学习发生的地方。模型的参数根据训练数据进行调整，目标是使损失函数最小化。然而，仅仅运行循环是不够的。我们需要观察训练的进展*情况*，以便做出明智的决策。下一节将介绍如何在训练期间监控损失和准确率等指标，以了解模型行为并诊断潜在问题。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  本书是理解深度学习算法的基石，包括神经网络训练机制、反向传播、梯度下降以及周期和批量大小等概念。
- [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — PyTorch Documentation (2024)
  Journal: PyTorch Tutorials
  一个展示PyTorch中显式训练循环的官方教程，阐述了前向传播、损失计算、反向传播和权重更新的步骤。
- [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) — David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams (1986)
  Journal: Nature; Volume: 323; Pages: 533-536; DOI: [10.1038/323533a0](https://doi.org/10.1038/323533a0)
  普及反向传播算法的开创性论文，对于理解训练过程中神经网络权重如何调整至关重要。

---

[上一节](06-%E9%85%8D%E7%BD%AE%E6%A8%A1%E5%9E%8B%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E4%B8%8E%E4%BC%98%E5%8C%96%E5%99%A8%E9%80%89%E6%8B%A9.md) · [下一节](08-%E7%9B%91%E6%8E%A7%E8%AE%AD%E7%BB%83%E8%BF%9B%E5%B1%95%EF%BC%88%E6%8D%9F%E5%A4%B1%E4%B8%8E%E6%8C%87%E6%A0%87%EF%BC%89.md)

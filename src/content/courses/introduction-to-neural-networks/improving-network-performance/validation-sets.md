---
course: "introduction-to-neural-networks"
chapter: "improving-network-performance"
lesson: "validation-sets"
sourceId: 2027
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-6-improving-network-performance/validation-sets"
title: "验证集的作用"
description: "How to use validation data to monitor generalization and tune hyperparameters."
order: 2
plots: ["plots/2027-0.json"]
sourceHash: "f46d56cd7f15ecdf0a8e4e325c9713c0f989e55012a5c8d31ad3f14fd25294f1"
sourceCorrections: []
---

正如我们所讨论的，仅仅衡量神经网络 (neural network)在训练数据上的表现并不能说明全部情况。一个模型可能在其训练样本上达到非常低的误差，但当面对新的、未见过的数据时却表现糟糕。这种差异突出了*学习*与*记忆*之间的区别。我们需要一种方法来评估模型在开发过程中*实际*表现如何，而无需接触最终测试数据。这正是**验证集**的作用。

### 训练期间评估泛化能力

核心思路很简单：在训练开始前，您会留出部分数据集，这部分数据在训练过程中模型*绝不会*接触到（这意味着，模型的梯度将不会基于这些数据计算）。这部分预留的数据就是您的验证集。其余部分则用于模型训练（即训练集）。

通常，您可以这样划分数据：

- **训练集：** 用于计算损失，并通过反向传播 (backpropagation)和梯度下降 (gradient descent)更新网络的权重 (weight)和偏差。模型直接从这些数据中学习。（例如，数据量的60-80%）
- **验证集：** 在训练期间（例如，每个周期后）定期使用，以评估模型在未训练数据上的表现（损失、准确率等）。此评估有助于监测泛化能力，并就训练过程本身做出决策。（例如，数据量的10-20%）
- **测试集：** 在所有训练和模型选择完成后*只使用一次*，以获得模型在真正未见过数据上表现的最终、无偏估计。（例如，数据量的10-20%）

区分验证集和测试集很重要。可以将验证集看作您的彩排或模拟考试。您用它来微调 (fine-tuning)方法（调整超参数 (parameter) (hyperparameter)，决定何时停止训练）。测试集是期末考试，只进行一次，以了解您真实的学习效果。在开发过程中反复使用测试集，就好像提前研究期末考试题目一样；您的最终分数将无法准确反映您的知识水平。

### 验证集如何指导训练

验证集在开发过程中主要有两个用途：

1. **监测过拟合 (overfitting)：** 通过跟踪每个周期后训练集和验证集上的性能指标（如损失或准确率），您可以观察模型泛化能力如何。
   - 最初，训练损失和验证损失通常都会下降。
   - 如果模型开始过拟合，训练损失将继续下降（因为模型记住了训练数据），但验证损失将趋于平稳或开始*上升*。这种背离是一个清晰的迹象，表明模型正在丧失其泛化能力。



![训练损失 vs. 验证损失](plots/2027-0.json)



> 训练损失持续下降，而验证损失在大约第30个周期开始上升，这表明过拟合开始发生。

2. **指导超参数 (parameter) (hyperparameter)调整和模型选择：** 如何选择最佳学习率？您的网络应有多少层或神经元？哪种正则化 (regularization)技术效果最好？验证集提供了回答这些问题所需的客观衡量标准。您可以使用不同的超参数设置或架构训练模型，并比较它们在验证集上的表现。通常，在对测试集进行最终评估之前，会选择在验证集上表现最佳的配置作为首选模型。

### 实际操作

在训练循环中加入验证检查相对简单。在每个周期后（或有时更频繁），您会使用验证数据进行一次前向传播，计算损失和任何其他相关指标（如准确率），但非常重要的是，您**不会**基于这些验证数据执行反向传播 (backpropagation)或更新权重 (weight)。

以下是一个类似 Python 的伪代码片段，说明了此过程：

```python
# 假设数据集已划分为 train_data, validation_data
# 假设 model, optimizer, loss_function 已定义

for epoch in range(num_epochs):
    # --- 训练阶段 ---
    model.train() # 将模型设置为训练模式（与 Dropout 等层相关）
    total_train_loss = 0
    for batch in train_data:
        inputs, targets = batch
        
        optimizer.zero_grad() # 重置梯度
        
        outputs = model(inputs) # 前向传播
        loss = loss_function(outputs, targets) # 计算损失
        
        loss.backward() # 反向传播
        optimizer.step() # 更新权重
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_data)
    print(f"Epoch {epoch+1}: Training Loss = {avg_train_loss}")

    # --- 验证阶段 ---
    model.eval() # 将模型设置为评估模式
    total_val_loss = 0
    with torch.no_grad(): # 禁用验证的梯度计算
        for batch in validation_data:
            inputs, targets = batch
            outputs = model(inputs) # 仅前向传播
            loss = loss_function(outputs, targets)
            total_val_loss += loss.item()
            # 如果需要，计算准确率等其他指标
            
    avg_val_loss = total_val_loss / len(validation_data)
    print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss}")
    
    # --- 检查点 / 早停逻辑（使用 avg_val_loss）---
    # （更多内容请参阅早停部分）
    # 如果验证损失有所改善，则保存模型等。

# --- 最终评估（在训练循环之后）---
# 根据验证表现加载最佳模型
# 在 test_data 上进行评估（该数据之前从未被使用）
```

因此，验证集是构建高效机器学习 (machine learning)模型的迭代过程中不可或缺的一部分。它提供了指导训练、防止过拟合 (overfitting)以及选择最有可能在新、未见过数据上表现良好的模型配置所需的反馈。没有它，您将如同盲人摸象，不确定模型在训练数据上的改进是否真的能转化为有意义的泛化能力。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  涵盖了泛化、过拟合以及使用验证集进行模型选择和监控的基本原理。
- [Training Neural Networks Part 1: Setting up the Data and the Loss](https://cs231n.github.io/neural-networks-1/) — Stanford University CS231n Course Staff (2023)
  为读者提供了关于数据划分（训练集、验证集、测试集）、监控过拟合和神经网络中早停的易懂介绍。
- [PyTorch Documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch Contributors (2023)
  Publisher: PyTorch Foundation
  PyTorch官方文档，描述了`nn.Module`的`train()`和`eval()`方法以及`torch.no_grad()`上下文管理器，这些对于实现训练和验证循环非常重要。

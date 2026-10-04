---
course: "deep-learning-regularization-optimization"
chapter: "combining-techniques-practical"
lesson: "early-stopping-regularization"
sourceId: 5012
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-8-combining-techniques-practical/early-stopping-regularization"
title: "提前停止作为正则化"
description: "说明提前停止作为一种简单而有效的正则化技术。"
order: 4
plots: ["plots/5012-0.json"]
sourceHash: "1593b8729421a52c1712f122006f98b32740c50bf5888576ec3e6f746d086d6b"
sourceCorrections: []
---

如你所见，随着时间推移监测训练和验证指标，可提供关于模型学习和泛化能力的有用信息。训练损失通常随着模型更好地拟合训练数据而减少，但未见过的数据表现如何呢？验证集在此发挥作用。如果我们根据训练周期绘制训练损失和验证损失（或像准确率这样的其他相关指标），当过拟合 (overfitting)发生时，我们常看到一种明显模式：训练损失持续下降，但验证损失开始上升。这种差异表明模型开始记忆训练数据，包括其噪声，而不是掌握普遍规律。

提前停止直接利用了这一观察。它是一种简单而有效的正则化 (regularization)形式，当模型在验证集上的表现停止提升或开始变差时，立即停止训练过程。你无需训练固定的、可能过多的周期数，而是监测选定的验证指标，并在该指标显示泛化表现已达顶点时停止训练。

### 提前停止的工作方式

其基本理念简单明了：

1. **监测：** 每个训练周期后（或有时更频繁），使用选定指标（例如，验证损失、验证准确率、F1 分数）在专门的验证集上评估模型表现。
2. **记录：** 持续记录迄今为止达到的最佳验证指标以及与该表现相关的模型权重 (weight)。
3. **决定：** 如果验证指标在预设的连续周期数（称为“耐心期”）内没有提升，则停止训练过程。
4. **恢复：** 最终使用的模型通常是产生最佳验证表现的周期中保存的模型，而不是停止前的最后一个训练步骤的模型。

考虑典型的学习曲线：



![带有提前停止点的学习曲线](plots/5012-0.json)



> 训练损失持续下降，而验证损失最初下降，但在第 12 个周期后开始上升。提前停止将在此点附近中止训练，并恢复第 12 个周期的模型状态。

### 为什么提前停止是一种正则化 (regularization)形式？

当模型变得过于复杂并学习到训练数据中无法泛化的具体内容时，就会发生过拟合 (overfitting)。梯度下降 (gradient descent)等训练算法会迭代调整模型参数 (parameter)（权重 (weight)）以最小化训练损失。如果允许运行时间过长，这些参数可能会达到高度适应训练集的值。

在深度学习 (deep learning)模型训练中，为避免模型过拟合，提前停止通过限制优化过程来发挥作用。通过根据验证表现停止训练，实际上是在优化进程的较早阶段选择了一个模型。这些较早的模型通常具有较小的权重幅值，并且与训练了更多周期的模型相比，复杂程度较低。从这个意义上说，提前停止隐式地限制了模型的复杂性，类似于 L1 或 L2 正则化显式地惩罚大权重。它有助于在欠拟合 (underfitting)（停止过早）和过拟合（停止过晚）之间找到一个平衡点。

### 实际实施细节

实施提前停止需要做出一些决定：

- **监测指标：** 验证损失是常见选择，因为它与优化目标直接相关。然而，对于分类任务，根据目标，准确率、F1 分数或 AUC 等指标可能更合适。
- **耐心期：** 验证指标可能存在噪声；表现可能会略微波动或暂时变差，然后才再次提升。`patience` 参数 (parameter)定义了在实际停止前需要等待多少个周期才能看到提升。通常，5-10 个周期的耐心期比较常见，这允许模型有时间从验证表现的暂时下降中恢复。
- **保存最佳模型：** 每当获得新的最佳验证分数时，保存模型的状态（权重 (weight)和可能的优化器状态）很重要。当停止发生时，你将恢复到这个保存的最佳状态。

下面是一个类似 PyTorch 的伪代码片段：

```python
# 初始化跟踪变量
best_validation_loss = float('inf')
epochs_without_improvement = 0
patience = 10 # 示例耐心值

# 假设 model, train_loader, valid_loader, optimizer, criterion 已定义

for epoch in range(num_epochs):
    # --- 训练循环 ---
    model.train()
    for batch in train_loader:
        # ... 训练步骤: 梯度清零, 前向传播, 计算损失, 反向传播, 优化器步进 ...
        pass # 训练批次循环的占位符

    # --- 验证循环 ---
    model.eval()
    current_validation_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            # ... 验证步骤: 前向传播, 损失计算 ...
            current_validation_loss += calculated_loss.item()

    average_validation_loss = current_validation_loss / len(valid_loader)
    print(f"周期 {epoch+1}: 验证损失 = {average_validation_loss:.4f}")

    # --- 提前停止逻辑 ---
    if average_validation_loss < best_validation_loss:
        best_validation_loss = average_validation_loss
        epochs_without_improvement = 0
        # 保存最佳模型状态
        torch.save(model.state_dict(), 'best_model.pth')
        print("验证损失提升，保存模型。")
    else:
        epochs_without_improvement += 1
        print(f"验证损失在 {epochs_without_improvement} 个周期内没有提升。")

    if epochs_without_improvement >= patience:
        print(f"在 {epoch+1} 个周期后提前停止。")
        break # 退出训练循环

# 训练结束后或停止后加载最佳模型权重
model.load_state_dict(torch.load('best_model.pth'))
print("已根据验证表现加载最佳模型权重。")
```

### 优点与考量

**优点：**

- **简单：** 相对容易理解和实现。
- **有效：** 在实际中通常能很好地防止过拟合 (overfitting)。
- **高效：** 可通过避免不必要的训练周期来节省大量计算时间。
- **自身无需超参数 (parameter) (hyperparameter)调整：** 尽管耐心期是一个参数，但提前停止不会引入像 L1/L2 中的 $\lambda$ 那样需要仔细调整的复杂正则化 (regularization)超参数。

**考量：**

- **需要验证集：** 需要一个从训练中分离出来的数据集，这会减少可用于训练的数据量。
- **不优化其他超参数：** 它停止训练，但不能直接帮助调整学习率、正则化强度或模型架构。
- **敏感性：** 表现很大程度上取决于验证集的代表性和潜在噪声。`patience` 参数有助于减轻这一点，但需要一定的判断。

提前停止是深度学习 (deep learning)实践者工具箱中的一种有价值的工具。它常与其他正则化技术（如 L2 或 Dropout）一起使用，通过直接监测模型在训练期间的泛化能力，为防止过拟合提供额外保障。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  对深度学习概念，包括作为正则化技术的提前停止，提供了全面的学术和实践解释。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 738
  一本备受推崇的教科书，涵盖机器学习的理论原则，包括正则化方法和模型选择策略的讨论。
- [tf.keras.callbacks.EarlyStopping class](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) — TensorFlow Authors (2023)
  Publisher: TensorFlow
  TensorFlow 中 Keras API 实现提前停止的官方文档，详细说明了耐心值和监控指标等可配置参数。

---
course: "deep-learning-regularization-optimization"
chapter: "dropout-regularization"
lesson: "practice-adding-dropout"
sourceId: 4944
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-3-dropout-regularization/practice-adding-dropout"
title: "动手实践：添加 Dropout 层"
description: "将 Dropout 加入神经网络模型的实践环节。"
order: 8
plots: ["plots/4944-0.json"]
sourceHash: "b058e398bda66c5d481265475c96a0958521f27a7bdcb71fc8da3cfac5bf9e41"
sourceCorrections: []
---

通过向神经网络 (neural network)添加 Dropout 层，可以应用这一正则化 (regularization)技术。使用 PyTorch 等框架将 Dropout 集成到模型中非常简单。接下来将演示如何添加 `nn.Dropout` 层，并讨论其对训练过程的影响。

假设我们有一个用于分类任务的简单多层感知器 (MLP)。一个没有 Dropout 的架构如下所示：

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleMLP, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size1)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu_2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.output_layer(x)
        return x

# 示例实例化
# model_no_dropout = SimpleMLP(input_size=784, hidden_size1=256, hidden_size2=128, output_size=10)
# print(model_no_dropout)
```

这是一个标准的前馈网络。如果此模型在数据集上容易出现过拟合 (overfitting)，我们可以引入 Dropout 层。

### 添加 Dropout 层

PyTorch 中的 `nn.Dropout` 模块实现了 Dropout 技术。它接受 Dropout 概率 `p` 作为参数 (parameter)，即训练期间任何给定神经元输出被设置为零的概率。通常的做法是将 Dropout 层放在隐藏层的激活函数 (activation function)之后。

以下是修改 `SimpleMLP` 以包含 Dropout 的方式：

```python
import torch
import torch.nn as nn

class MLPWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_prob=0.5):
        super(MLPWithDropout, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size1)
        self.relu_1 = nn.ReLU()
        # 在第一个隐藏层激活后应用 Dropout
        self.dropout_1 = nn.Dropout(p=dropout_prob)
        self.layer_2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu_2 = nn.ReLU()
        # 在第二个隐藏层激活后应用 Dropout
        self.dropout_2 = nn.Dropout(p=dropout_prob)
        self.output_layer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x) # 应用 dropout
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.dropout_2(x) # 应用 dropout
        x = self.output_layer(x)
        return x

# 使用默认 Dropout 概率 0.5 的示例实例化
model_with_dropout = MLPWithDropout(input_size=784, hidden_size1=256, hidden_size2=128, output_size=10)
print(model_with_dropout)

# 或者指定不同的概率
# model_with_dropout_p25 = MLPWithDropout(input_size=784, hidden_size1=256, hidden_size2=128, output_size=10, dropout_prob=0.25)
# print(model_with_dropout_p25)
```

在修改后的版本中：

1. 我们向构造函数添加了 `dropout_prob` 参数，默认值为 0.5，这是一个常用的初始值。
2. 我们使用指定的概率实例化了 `nn.Dropout` 层（`self.dropout_1`、`self.dropout_2`）。
3. 在 `forward` 方法中，我们在隐藏层的 ReLU 激活之后立即应用这些 Dropout 层。注意，Dropout 通常不应用于输出层。

### 训练模式与评估模式

使用 Dropout（以及批归一化 (normalization)等其他层）时，必须区分训练和评估阶段。

- **训练期间：** 激活 Dropout，随机将神经元输出置零以防止共适应。
- **评估/推断期间：** 使用完整的网络能力。Dropout 应该关闭，通常 Dropout 前一层的输出会按 $(1-p)$ 的因子进行缩放，以补偿训练期间活跃神经元较少的情况（在调用 `model.eval()` 时，PyTorch 的 `nn.Dropout` 会自动处理此缩放，这被称为“反向 Dropout”技术）。

PyTorch 模型通过特定的模式来处理这些行为。必须显式切换它们：

- `model.train()`：将模型设为训练模式。Dropout 层处于开启状态。
- `model.eval()`：将模型设为评估模式。Dropout 层处于关闭状态，激活值会得到正确缩放。

以下是典型训练循环中的代码结构：

```python
# 假设 model, train_loader, val_loader, optimizer, criterion 已定义

num_epochs = 10

for epoch in range(num_epochs):
    # --- 训练阶段 ---
    model_with_dropout.train() # 将模型设置为训练模式
    train_loss = 0.0
    for data, target in train_loader: # 遍历训练批次
        optimizer.zero_grad()
        output = model_with_dropout(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")

    # --- 验证阶段 ---
    model_with_dropout.eval() # 将模型设置为评估模式
    val_loss = 0.0
    with torch.no_grad(): # 验证时不计算梯度
        for data, target in val_loader: # 遍历验证批次
            output = model_with_dropout(data)
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

# --- 在测试集上进行最终评估 ---
# model_with_dropout.eval() # 确保模型处于评估模式
# with torch.no_grad():
#    # 执行测试...
```

在验证或测试时忘记切换到 `model.eval()` 是常见错误。这会导致预测结果具有随机性（由于 Dropout 仍处于开启状态），并因输出未经过正确缩放而导致性能衡量不准。

### 可视化 Dropout 的效果

Dropout 通常有助于缩小训练性能与验证性能之间的差距，从而减少过拟合 (overfitting)。虽然使用 Dropout 可能会使训练损失略微升高或收敛变慢（因为网络在每次迭代中都在发生变化），但与出现过拟合且未开启 Dropout 的模型相比，验证损失通常更低且更稳定。



![Effect of Dropout on Learning Curves](plots/4944-0.json)



> 有无 Dropout 模型的训练和验证损失曲线对比。可以看到，未启用 Dropout 的模型验证损失开始上升（过拟合），而启用 Dropout 的模型验证损失保持在较低且更稳定的水平，尽管其训练损失略高。

### 进一步实验

通过以上内容，你已经了解了基本的实现方法。接下来可以尝试：

- **Dropout 率 (`p`)：** 尝试不同的数值（如 0.2、0.3、0.5）。更高的数值提供更强的正则化 (regularization)，但如果设置过高，可能会降低收敛速度或导致欠拟合 (underfitting)。
- **放置位置：** 虽然在全连接层的激活函数 (activation function)后放置 Dropout 很常见，但也可以尝试在激活前放置，或仅在特定层使用。
- **组合使用：** 将 Dropout 与 L2 权重 (weight)衰减等其他正则化技术结合，观察它们的共同作用。

添加 Dropout 是应对过拟合 (overfitting)的有效手段。请务必正确使用 `model.train()` 和 `model.eval()`，以确保它在不同阶段按预期工作。

## 参考资料

- [Dropout: A Simple Way to Prevent Overfitting in Neural Networks](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research (JMLR); Volume: 15; Pages: 1929-1958
  这篇开创性论文介绍了Dropout技术，展示了其在防止神经网络过拟合方面的有效性。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本权威教科书，在深度学习的广泛背景下，为Dropout正则化提供了全面的理论基础和详细解释。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Lecture Notes on Regularization](http://cs231n.github.io/neural-networks-2/#reg) — Stanford University (2023)
  提供对Dropout清晰直观的解释，包括实践实现细节、反向Dropout的概念及其在防止过拟合方面的作用。

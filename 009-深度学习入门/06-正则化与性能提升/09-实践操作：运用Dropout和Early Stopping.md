# 实践操作：运用Dropout和Early Stopping

来源：[原文](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-6-regularization-performance-improvement/practice-dropout-early-stopping)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将Dropout和Early Stopping正则化 (regularization)技术集成到典型的PyTorch训练流程中，是对抗过拟合 (overfitting)并提升模型泛化能力的实用方法。这涉及在馈网络训练过程中应用这些方法。

设想我们已为一个分类任务训练了一个神经网络 (neural network)，也许与第5章中的MNIST数字分类器相似。在训练了多个周期后，我们可能会看到训练损失持续降低，但验证损失却开始上升。这种分歧是过拟合的一个典型表现：模型对训练数据（包括其中的噪声）学习得过于细致，并失去了对未见数据的泛化能力。Dropout和Early Stopping是应对此问题的有效手段。

### 场景设置：一个基本网络与训练循环

首先，我们使用PyTorch的`nn.Module`定义一个简单的全连接神经网络 (neural network)。该网络将作为我们添加正则化 (regularization)之前的基准。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 假设数据加载器train_loader和val_loader已就绪
# 假设input_size、hidden_size、output_size已定义

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, output_size)
        # Softmax通常包含在损失函数中（nn.CrossEntropyLoss）

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x

# 初始化基准模型
model_baseline = SimpleNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss() # 示例损失函数
optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=0.001) # 示例优化器
```

我们的标准训练循环会遍历周期，执行前向和反向传播 (backpropagation)，并计算损失。我们还需要在每个周期内进行验证步骤，以监督未见数据上的表现。

```python
# 训练循环的基本结构（为简洁起见省略了细节）
# def train_epoch(model, loader, criterion, optimizer):
#     model.train() # 将模型设置为训练模式
#     # ... 训练步骤 ...
#     return average_training_loss

# def validate_epoch(model, loader, criterion):
#     model.eval() # 将模型设置为评估模式
#     # ... 验证步骤 ...
#     return average_validation_loss, average_validation_accuracy

# num_epochs = 20
# for epoch in range(num_epochs):
#     train_loss = train_epoch(model_baseline, train_loader, criterion, optimizer_baseline)
#     val_loss, val_acc = validate_epoch(model_baseline, val_loader, criterion)
#     print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
```

运行此基准模型可能会出现前面描述的过拟合 (overfitting)模式。

### 实现Dropout

为了加入Dropout，我们在模型定义中引入`nn.Dropout`层。常用做法是将其置于激活函数 (activation function)之后，尤其是在全连接层中。`p`参数 (parameter)指定元素被置零的概率。`p`的典型值在0.2到0.5之间。

让我们修改`SimpleNet`以包含Dropout：

```python
class NetWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(NetWithDropout, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob) # 添加了Dropout层
        self.layer_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x) # 在激活后应用dropout
        x = self.layer_2(x)
        return x

# 初始化带Dropout的模型
model_dropout = NetWithDropout(input_size, hidden_size, output_size, dropout_prob=0.5)
optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=0.001)
```

在训练阶段前使用`model.train()`，在验证或测试阶段前使用`model.eval()`是很要紧的。`nn.Dropout`在这些模式下的行为不同：

- `model.train()`：Dropout处于活跃状态，随机将神经元置零。
- `model.eval()`：Dropout处于非活跃状态。Dropout层之前的输出会按等于dropout概率`p`的因子缩小，以考虑评估时更多神经元活跃的事实。这种缩放确保了训练和评估之间，激活的预期总和大致保持一致。当调用`model.eval()`时，PyTorch会自动处理此缩放。

然后，你将使用相同的训练循环结构来训练`model_dropout`。你应该会看到与基准模型相比，训练损失和验证损失之间的差距更小，这表明过拟合 (overfitting)程度降低了。

### 实现Early Stopping

Early Stopping监督验证表现（例如，验证损失），当此指标在连续指定数量的周期内停止提升时，便会停止训练，这个数量被称为“耐心”（patience）。这阻止模型继续训练进入过拟合 (overfitting)状态。

我们可以在训练循环中手动实现Early Stopping。我们需要变量来追踪迄今为止达到的最佳验证损失，以及自上次提升以来的周期数。我们还需要一个保存最佳模型状态的机制。

```python
import copy

# 包含Early Stopping的训练循环
num_epochs = 50 # 允许更多周期，因为early stopping可能会更早停止训练
patience = 5 # 停止前等待提升的周期数
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

# 初始化模型（可以是带Dropout或基准模型）
model_early_stop = NetWithDropout(input_size, hidden_size, output_size, dropout_prob=0.5) # 带Dropout的示例
optimizer_early_stop = optim.Adam(model_early_stop.parameters(), lr=0.001)

print("Starting training with Early Stopping...")
for epoch in range(num_epochs):
    # 训练一个周期
    model_early_stop.train() # 设置为训练模式
    # 假设train_loader、criterion、optimizer_early_stop已定义
    # ... (你一个周期的训练循环逻辑) ...
    avg_train_loss = calculate_average_train_loss() # 占位符

    # 验证模型
    model_early_stop.eval() # 设置为评估模式
    current_val_loss = 0.0
    current_val_acc = 0.0
    with torch.no_grad(): # 验证时禁用梯度计算
        # 假设val_loader已定义
        # ... (你一个周期的验证循环逻辑) ...
        avg_val_loss = calculate_average_val_loss() # 占位符
        avg_val_acc = calculate_average_val_accuracy() # 占位符

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

    # Early Stopping检查
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # 保存最佳模型状态
        best_model_state = copy.deepcopy(model_early_stop.state_dict())
        print(f"Validation loss improved to {best_val_loss:.4f}。保存模型状态。")
    else:
        epochs_no_improve += 1
        print(f"Validation loss did not improve for {epochs_no_improve} epoch(s)。")

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs。")
        # 停止前加载最佳模型状态
        if best_model_state:
            model_early_stop.load_state_dict(best_model_state)
            print("已加载训练期间找到的最佳模型状态。")
        break # 退出训练循环

# 循环结束后，model_early_stop包含验证损失最佳周期时的权重
print("训练完成。")
```

在这个修改后的循环中，我们在每个周期后检查验证损失。如果它有所提升，我们就重置计数器并保存模型的state dictionary。如果它没有提升，我们就增加计数器。一旦计数器达到`patience`限制，训练就会停止，我们会加载我们之前保存的最佳模型状态。

### 可视化影响

比较基准模型、带有Dropout的模型，以及带有Dropout并结合Early Stopping的模型的验证损失曲线，通常能清楚地显示其益处。



[交互图表：Dropout和Early Stopping对验证损失的影响](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-6-regularization-performance-improvement/practice-dropout-early-stopping#plot-1bjsmvv)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "Dropout和Early Stopping对验证损失的影响",
    "xaxis": {
      "title": "周期"
    },
    "yaxis": {
      "title": "验证损失"
    },
    "legend": {
      "title": "模型"
    }
  },
  "data": [
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20
      ],
      "y": [
        0.6,
        0.5,
        0.42,
        0.35,
        0.3,
        0.26,
        0.23,
        0.21,
        0.22,
        0.24,
        0.27,
        0.3,
        0.33,
        0.36,
        0.39,
        0.42,
        0.45,
        0.48,
        0.51,
        0.54
      ],
      "mode": "lines",
      "name": "基准（无正则化）",
      "line": {
        "color": "#ff6b6b"
      }
    },
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20
      ],
      "y": [
        0.62,
        0.53,
        0.45,
        0.39,
        0.34,
        0.3,
        0.27,
        0.25,
        0.23,
        0.21,
        0.2,
        0.19,
        0.195,
        0.2,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.26
      ],
      "mode": "lines",
      "name": "带Dropout",
      "line": {
        "color": "#228be6"
      }
    },
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15
      ],
      "y": [
        0.62,
        0.53,
        0.45,
        0.39,
        0.34,
        0.3,
        0.27,
        0.25,
        0.23,
        0.21,
        0.2,
        0.19,
        0.195,
        0.2,
        0.21
      ],
      "mode": "lines+markers",
      "name": "Dropout + Early Stopping",
      "line": {
        "color": "#12b886"
      }
    }
  ]
}
```

</details>



> 各周期验证损失曲线的比较。基准模型在第8个周期后损失增加，显示出明显的过拟合 (overfitting)。Dropout有助于保持较低的验证损失。Early Stopping在第15个周期左右停止训练，此时（带dropout的）验证损失不再有明显提升，这阻止了进一步不必要的训练，并可能选取一个更接近最佳验证表现点（第12个周期）的模型。

该图表显示了基准模型的验证损失如何开始增加，表明过拟合。带有Dropout的模型显示出泛化能力的提升，更长时间地保持较低的验证损失。在Dropout之上应用Early Stopping，阻止了在验证表现趋于平稳或恶化后继续不必要的训练，节省了计算资源，并通过在验证损失最低点附近停止，可能获得一个在未见数据上表现更好的模型。

通过运用Dropout和Early Stopping，你将获得实用的工具来构建更可靠的深度学习 (deep learning)模型，使其能从训练数据更好地泛化到新的、未见示例。请记住监督训练和验证指标，以理解这些方法如何影响你模型的学习过程。

## 参考资料

- [Dropout: A Simple Way to Prevent Overfitting Neural Networks](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Publisher: JMLR Editorial Board; Volume: 15; Pages: 1929-1958; DOI: [10.5555/2620311.2620813](https://doi.org/10.5555/2620311.2620813)
  介绍Dropout作为一种正则化技术以减少神经网络过拟合的开创性论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 10.4.3 (Dropout), 7.8 (Early Stopping)
  一本关于深度学习的综合教材，详细解释了Dropout和Early Stopping等正则化技术。
- [torch.nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) — PyTorch Documentation (2024)
  PyTorch官方Dropout模块文档，详细说明其参数、用法以及在训练和评估模式下的行为。
- [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — PyTorch Documentation (2024)
  PyTorch官方教程，演示了神经网络的完整训练流程，其中包括实现Early Stopping所需的关键验证指标监测。

---

[上一节](08-%E8%B6%85%E5%8F%82%E6%95%B0%E6%90%9C%E7%B4%A2%E7%AD%96%E7%95%A5%EF%BC%88%E7%BD%91%E6%A0%BC%E6%90%9C%E7%B4%A2%E3%80%81%E9%9A%8F%E6%9C%BA%E6%90%9C%E7%B4%A2%EF%BC%89.md) · [下一节](../07-%E4%B8%93%E7%94%A8%E6%9E%B6%E6%9E%84%E4%BB%8B%E7%BB%8D/01-%E5%89%8D%E9%A6%88%E7%BD%91%E7%BB%9C%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)

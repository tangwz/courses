---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-training-loops-for-keras-devs"
lesson: "model-evaluation-pytorch"
sourceId: 5763
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs/model-evaluation-pytorch"
title: "PyTorch 中的模型评估循环"
description: "在 PyTorch 中构建模型评估循环，其目的类似于 TensorFlow 的 `model.evaluate()`。"
order: 6
plots: []
sourceHash: "7f4891b3f5527039197a6d0f9d8a4c85226876da85865d9f899c1e8ad60922c2"
sourceCorrections: []
---

训练好 PyTorch 模型后，或者在训练期间为了监控进度，您需要使用模型从未见过的数据集（例如验证集或测试集）来评估其表现。如果您是从 TensorFlow Keras 转过来的，您会熟悉便捷的 `model.evaluate()` 方法，该方法在您编译模型后只需一次调用即可处理评估过程。

在 PyTorch 中，与训练类似，评估模型需要编写一个明确的循环。这使您能够完全控制该过程，进行自定义指标计算，并详细了解模型的行为。评估循环的结构与训练循环非常相似，但有几个重要区别：您不会计算梯度或更新模型权重 (weight)。

## PyTorch 评估循环的步骤

构建评估循环通常包含以下步骤：

1. **将模型设置为评估模式**：这对于在训练和推理 (inference)期间行为不同的层（如 Dropout 和 Batch Normalization）非常重要。
2. **禁用梯度计算**：由于您不是在训练，因此无需计算或存储梯度。这可以节省内存和计算资源。
3. **遍历数据**：循环遍历评估数据集，通常由 `DataLoader` 提供。
4. **执行前向传播**：从模型获取预测。
5. **计算损失和指标**：计算损失以及任何其他性能指标（例如准确率、精确率、召回率）。
6. **聚合指标**：收集并平均所有批次的指标，以获得整个数据集的性能总览。

让我们更详细地了解这些重要组成部分。

### 将模型设置为评估模式：`model.eval()`

在开始评估之前，您必须通过调用 `model.eval()` 将模型切换到评估模式。这很重要，因为某些层，尤其是 `torch.nn.Dropout` 和 `torch.nn.BatchNorm1d` / `torch.nn.BatchNorm2d` / `torch.nn.BatchNorm3d`，在训练和评估期间具有不同的行为。

- **Dropout 层**：在训练期间，dropout 会随机将层的部分输出归零，以防止过拟合 (overfitting)。在评估期间，您希望使用整个网络，因此 dropout 层会被关闭（即，它们成为直通层）。
- **Batch Normalization 层**：在训练期间，批归一化 (normalization)层使用当前输入批次的均值和标准差来归一化其输出，并且它们还维护训练数据总体均值和标准差的运行估计值。在评估期间，它们使用这些运行估计值来归一化输入，从而确保行为一致。

调用 `model.eval()` 会递归地为模型中的所有模块设置模式。反之，当您切换回训练模式时，您将调用 `model.train()` 以将这些层恢复到其训练行为。

```python
# 假设 'model' 是您的 PyTorch nn.Module 实例
model.eval()
print("模型处于评估模式。")
# ... 执行评估 ...

# 如果您之后需要切换回训练模式
# model.train()
# print("模型已回到训练模式。")
```

忘记调用 `model.eval()` 可能会导致评估结果不一致且具有误导性，因为 dropout 仍将处于活跃状态，并且批归一化层将使用批次统计数据而非学习到的总体统计数据。

### 禁用梯度计算：`torch.no_grad()`

在评估期间，您只关心模型的输出，而不关心更新其权重 (weight)。因此，计算梯度是不必要且计算成本高昂的。PyTorch 提供了一个上下文 (context)管理器 `torch.no_grad()`，它在其作用域内禁用梯度计算。

使用 `torch.no_grad()` 提供两个主要优势：

1. **减少内存消耗**：此块内的操作不会跟踪历史，因此 PyTorch 不需要存储用于反向传播 (backpropagation)的中间值。
2. **提高速度**：计算可以更快，因为它们不涉及构建梯度计算图的开销。

以下是它的用法：

```python
import torch

# 假设模型、数据和目标已定义并位于正确的设备上
# 在此循环的此部分之前应已调用 model.eval()

with torch.no_grad():
    # 前向传播
    predictions = model(data_batch)
    # 损失计算（评估期间可选，但通常有用）
    # loss = criterion(predictions, target_batch)
    # 其他指标计算
    # accuracy = calculate_accuracy(predictions, target_batch)
```

在 `with torch.no_grad():` 块内执行的任何张量操作都将具有 `requires_grad=False`，即使其输入在块外部具有 `requires_grad=True`。

### 评估循环的结构

现在，让我们将这些元素组合成一个典型的评估循环。此函数将您的模型、评估集的 `DataLoader` 以及损失函数 (loss function)（判据）作为输入。

```python
import torch
import torch.nn as nn

# 示例：定义一个简单模型、判据和一个用于说明的虚拟数据加载器
# 在实际场景中，这些将是您实际训练过的模型和数据
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2) # 10 个输入特征，2 个输出类别
    def forward(self, x):
        return self.fc(x)

# 用于说明的虚拟数据
dummy_eval_data = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(5)] # 5 个批次，每个批次 32 个样本
eval_loader = torch.utils.data.DataLoader(dummy_eval_data, batch_size=None) # batch_size=None 是因为数据已经分批

model = SimpleModel() # 假设此模型已训练
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # 将模型设置为评估模式

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0) # 累积损失，按批次大小加权

            # 计算准确率
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    print(f'评估结果：平均损失：{avg_loss:.4f}，准确率：{accuracy:.4f} ({correct_predictions}/{total_samples})')
    return avg_loss, accuracy

# 执行评估
eval_loss, eval_accuracy = evaluate_model(model, eval_loader, criterion, device)
```

在此 `evaluate_model` 函数中：

1. 首先调用 `model.eval()`。
2. `torch.no_grad()` 包裹了整个数据循环，确保不计算梯度。
3. 在循环内部，数据批次被移动到正确的 `device`。
4. 执行前向传播 `outputs = model(inputs)`。
5. 使用 `criterion` 计算 `loss`。我们在求和之前将 `loss.item()` 乘以 `inputs.size(0)`（批次大小），因为损失函数通常返回批次的平均损失。为了获得数据集的总损失，我们需要将这些求和，然后除以总样本数。
6. 对于准确率，`torch.max(outputs, 1)` 为每个样本找到得分最高的类别。`correct_predictions` 和 `total_samples` 的数量被累积。
7. 遍历所有批次后，计算并报告 `avg_loss` 和总体 `accuracy`。

这个结构具有高度适应性。您可以轻松地从 `torchmetrics` 等库中添加其他指标，或在循环内实现自定义指标。

### 与 `tf.keras.evaluate()` 的比较

如果您习惯使用 TensorFlow，Keras 的 `model.evaluate(eval_dataset)` 方法会隐式地执行所有这些步骤。它接收您的评估数据集，遍历它，计算配置的损失和指标，然后返回结果。

PyTorch 的方法虽然需要更明确的代码，但提供几个优势：

- **透明度**：您可以清楚地看到评估的每一步发生了什么。
- **灵活性**：您可以轻松插入自定义逻辑、调试中间值或计算非标准指标，而无需遵循特定的 API 或回调系统。
- **理解**：自己编写循环可以加深您对评估过程的理解。

虽然 Keras 提供了便利，但 PyTorch 的方式提供更细粒度的控制，这在研究或处理复杂评估场景时特别有用。

评估模型后，您将使用这些指标（如平均损失和准确率）来比较不同模型、执行超参数 (parameter) (hyperparameter)调整，或决定您的模型是否可以部署。如果您在训练期间进行评估（例如，在每个 epoch 后对验证集进行评估），这些指标还可以为提前停止决策提供依据，帮助您防止过拟合 (overfitting)并节省训练时间。

## 参考资料

- [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  一个PyTorch官方教程，演示了完整的训练和评估循环，展示了`model.eval()`和`torch.no_grad()`在标准工作流中的实际应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供深度学习模型评估的理论基础，包括验证集、Dropout和Batch Normalization行为的原理。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本实用的指南，详细介绍了PyTorch训练和评估循环的实现，包含具体的代码示例。
- [\`tf.keras.Model.evaluate\` API](https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate) — TensorFlow Developers (2024)
  Keras `evaluate` 方法的官方API文档，提供了与PyTorch显式评估循环进行比较的背景。

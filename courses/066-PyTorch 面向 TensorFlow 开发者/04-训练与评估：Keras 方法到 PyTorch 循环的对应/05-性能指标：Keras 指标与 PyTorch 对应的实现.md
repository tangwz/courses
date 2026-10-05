# 性能指标：Keras 指标与 PyTorch 对应的实现

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs/metrics-pytorch-tf)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练神经网络 (neural network)时，优化器的主要目标是最小化损失函数 (loss function)。然而，损失值本身，特别是对于交叉熵等复杂函数，不总能直观地衡量模型在其预期任务上的表现。此时，性能指标就显得很重要。准确率、精确率、召回率或 F1 分数等指标能让你更清楚地了解模型的能力。

如果你习惯使用 TensorFlow Keras，你会很熟悉在 `model.compile()` 方法中方便地指定指标：

```python
# Keras 示例
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision()])
```

Keras 会在训练 (`model.fit()`) 和评估 (`model.evaluate()`) 期间自动计算并报告这些指标。这些 Keras 指标，例如 `tf.keras.metrics.Accuracy()`，是带状态的对象。它们会自动在各个批次中累积值（例如，正确预测总数和样本总数），并计算最终指标值。

PyTorch 遵循其更明确的设计理念，要求你更多地参与到指标计算过程中。PyTorch 没有与 Keras 自动指标追踪系统直接对应、内置且与训练循环结构集成的功能。相反，你通常需要自行完成这些计算。

### 手动计算 PyTorch 指标

对于简单的指标，在训练或评估循环中直接进行张量操作通常就足够了。例如，要在分类任务中计算一个批次的准确率：

```python
# PyTorch 手动计算批次准确率
# outputs: 模型输出的原始 logits
# labels: 真实标签
_, predicted_classes = torch.max(outputs.data, 1)
correct_predictions = (predicted_classes == labels).sum().item()
batch_accuracy = correct_predictions / labels.size(0)
```

为了得到一个周期的准确率，你需要将该周期内所有批次的 `correct_predictions` 和样本总数 (`labels.size(0)`) 相加，然后进行除法运算：

```python
# 累积以获得周期级别准确率
# 在周期循环前初始化
epoch_total_correct = 0
epoch_total_samples = 0

# 在批次循环内部（获取 correct_predictions 和 labels.size(0) 后）
# epoch_total_correct += correct_predictions
# epoch_total_samples += labels.size(0)

# 在周期循环之后
# epoch_accuracy = epoch_total_correct / epoch_total_samples
```

这种手动方法让你有充分的控制权，但可能会变得重复且容易出错，特别是对于像 AUC 或 F1 分数这样需要在批次间细致管理状态的复杂指标。

### TorchMetrics：一个专门的 PyTorch 指标库

为了弥补这一不足并为指标提供更便捷、类似 Keras 的体验，PyTorch 生态系统提供了 **TorchMetrics**。这个由 PyTorch Lightning 团队开发的库，可以在任何 PyTorch 项目中独立使用。它提供了多种常用的机器学习 (machine learning)指标，它们具有以下特点：

- **有状态的**: 它们自动累积批次间的统计数据。
- **可组合的**: 易于集成到你的 PyTorch 代码中。
- **对分布式训练友好**: 能够在分布式环境中正确同步指标。
- **设备无关**: 可以在 CPU 或 GPU 上运行。

你首先需要安装它：
`pip install torchmetrics`

下面是在多类别分类问题中使用 `TorchMetrics` 计算准确率的示例：

```python
import torch
import torchmetrics

# 定义类别数量和设备
NUM_CLASSES = 10 # CIFAR-10 示例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化指标，确保它在正确的设备上
accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)

# --- 在你的训练或评估循环内部 ---
# for images, labels in data_loader:
#     images, labels = images.to(device), labels.to(device)
#     outputs = model(images) # 获取模型预测 (logits)
#
#     # 使用当前批次的预测和目标更新指标状态
#     # TorchMetrics 通常期望分类任务的原始 logits
#     # 或根据指标及其参数的概率。
#     # 对于 'multiclass' 任务的 Accuracy，它在内部应用 argmax。
#     accuracy_metric.update(outputs, labels)
# --- 批次循环结束 ---

# 处理完一个周期中的所有批次后：
# 计算所有累积批次的指标
epoch_accuracy = accuracy_metric.compute()
print(f"周期准确率: {epoch_accuracy.item():.4f}")

# 重置指标的内部状态，以便进行下一个周期或评估阶段
accuracy_metric.reset()
```

使用 `TorchMetrics` 主要涉及以下四个步骤：

1. **初始化**: 创建指标实例（例如，`torchmetrics.Accuracy`、`torchmetrics.Precision`、`torchmetrics.F1Score`）。通常需要指定 `task` 参数 (parameter)（例如，“二分类”、“多类别”或“多标签”）以及适用的 `num_classes`。将指标对象发送到与你的数据和模型相同的设备上。
2. **更新**: 在数据循环的每次迭代中，调用指标的 `update(predictions, targets)` 方法。这会累积当前批次的必要统计数据。
3. **计算**: 在遍历所有批次后（通常在一个周期结束时），调用 `compute()` 方法以获取最终指标值。
4. **重置**: 调用 `reset()` 来清除指标的内部状态，使其为下一个周期或新的评估运行做好准备。

`TorchMetrics` 提供了一套全面的分类、回归、图像分析等指标，显著简化了 PyTorch 中的指标追踪。

### 使用 Scikit-learn 进行指标计算

另一种方法，特别是当 `TorchMetrics` 中没有特定指标或需要快速、临时评估时，是使用 `sklearn.metrics`。由于 scikit-learn 函数在 NumPy 数组上操作，你需要：

1. 使用 `.detach()` 从计算图中分离张量。
2. 使用 `.cpu()` 将它们移到 CPU。
3. 使用 `.numpy()` 将它们转换为 NumPy 数组。

```python
from sklearn.metrics import precision_score

# 假设 all_preds_list 和 all_labels_list 包含
# 一个周期的所有预测和标签，它们都收集为 CPU 张量
# 例如，all_preds_cpu = torch.cat(all_preds_list).cpu().numpy()
#       all_labels_cpu = torch.cat(all_labels_list).cpu().numpy()

# precision = precision_score(all_labels_cpu, all_preds_cpu, average='macro')
# print(f"Scikit-learn 精确率 (macro): {precision:.4f}")
```

虽然这种方法可行，但与 `TorchMetrics` 相比，它在 GPU 加速的训练循环中进行逐批更新的效率通常较低。它通常更适用于周期结束或最终模型评估。

### 将指标集成到 PyTorch 评估循环中

让我们将其整合起来，看看 `TorchMetrics` 如何融入标准的 PyTorch 评估循环：

```python
# 假设：model, val_dataloader, criterion (损失函数), device 已定义
# 并且已初始化一个 TorchMetrics 指标：
# val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
# val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(device)


model.eval()  # 将模型设置为评估模式
running_val_loss = 0.0

# 在评估开始时重置指标
val_accuracy.reset()
# val_f1.reset() # 如果使用多个指标

with torch.no_grad():  # 禁用评估的梯度计算
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_val_loss += loss.item() * inputs.size(0)

        # 更新指标
        val_accuracy.update(outputs, labels)
        # val_f1.update(outputs, labels)

# 遍历所有验证数据后计算最终指标
epoch_val_loss = running_val_loss / len(val_dataloader.dataset)
final_val_accuracy = val_accuracy.compute()
# final_val_f1 = val_f1.compute()

print(f"验证损失: {epoch_val_loss:.4f}")
print(f"验证准确率: {final_val_accuracy.item():.4f}")
# print(f"验证 F1 分数: {final_val_f1.item():.4f}")
```

在这个示例中，`model.eval()` 很重要，因为它会将 Dropout 和 Batch Normalization 等层设置为评估模式。`torch.no_grad()` 会禁用梯度计算，这会加速推理 (inference)过程并减少内存使用。指标是按批次更新的，并在最后统一计算一次。

### Keras 与 PyTorch 指标：比较

下表总结了指标处理的主要不同之处：



[交互图表：交互图表](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs/metrics-pytorch-tf#plot-5yrd4b)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "table",
      "header": {
        "values": [
          [
            "<b>特性</b>"
          ],
          [
            "<b>Keras (tf.keras.metrics)</b>"
          ],
          [
            "<b>PyTorch / TorchMetrics</b>"
          ]
        ],
        "align": [
          "left",
          "left"
        ],
        "font": {
          "family": "Arial",
          "size": 11,
          "color": "white"
        },
        "fill": {
          "color": "#495057"
        },
        "line": {
          "width": 1,
          "color": "#dee2e6"
        }
      },
      "cells": {
        "values": [
          [
            "集成",
            "用法",
            "状态管理",
            "灵活性",
            "生态系统"
          ],
          [
            "内置，在 `model.compile()` 中指定",
            "主要通过 `model.fit()`/`model.evaluate()` 自动处理",
            "有状态的指标对象自动处理累积",
            "提供标准集合，通过继承 `tf.keras.metrics.Metric` 可创建自定义指标",
            "Keras 核心 API 的一部分，集成度高"
          ],
          [
            "手动计算或通过外部库（例如，`TorchMetrics`）",
            "在训练/评估循环中显式调用 `update()`、`compute()`、`reset()`",
            "基本计算需手动累积，或使用 `TorchMetrics` 等库中的有状态对象",
            "灵活性高，可实现任何自定义逻辑。`TorchMetrics` 提供多种标准指标。",
            "`TorchMetrics` 是广泛使用的社区标准。`sklearn.metrics` 可作为替代方案使用。"
          ]
        ],
        "align": [
          "left",
          "left"
        ],
        "font": {
          "family": "Arial",
          "size": 10,
          "color": "#495057"
        },
        "fill": {
          "color": [
            "#f8f9fa",
            "#f8f9fa"
          ]
        },
        "line": {
          "width": 1,
          "color": "#dee2e6"
        },
        "height": 30
      }
    }
  ],
  "layout": {
    "autosize": true,
    "margin": {
      "l": 20,
      "r": 20,
      "t": 5,
      "b": 5
    }
  }
}
```

</details>



> Keras 和 PyTorch 中指标处理的比较。PyTorch 提供精细的控制，而 `TorchMetrics` 则提供方便的、类似 Keras 的有状态指标对象。

### 使用指标的实用考量

- **选择合适的指标**: 指标的选择高度依赖于你面临的具体问题。例如，在不平衡数据集中，准确率可能会产生误导。在这种情况下，对于分类任务，精确率、召回率、F1 分数或 ROC 曲线下面积（AUC）可能会提供更多信息。`TorchMetrics` 提供了许多此类指标。
- **不同任务的指标**:
  - **分类**: 准确率、精确率、召回率、F1 分数、AUC、混淆矩阵。
  - **回归**: 均方误差 (MSE)、均方根误差 (RMSE)、平均绝对误差 (MAE)、R 平方 ($R^2$)。
  - `TorchMetrics` 通常要求指定 `task`（例如，“二分类”、“多类别”或“回归”）以及其他参数 (parameter)，如 `num_classes` 或 `average`（用于 F1/精确率/召回率）。
- **记录指标**: 在每个周期后记录（训练和验证）指标是常见做法。这有助于你监控训练进度，发现过拟合 (overfitting)等问题（即训练指标改善但验证指标停滞或变差），以及比较不同的实验或超参数 (hyperparameter)设置。你可以将日志输出到控制台、文件，或专门的实验追踪工具，如 TensorBoard（使用 `torch.utils.tensorboard.SummaryWriter`）或 Weights & Biases。

通过掌握如何在 PyTorch 中手动或借助 `TorchMetrics` 等辅助库计算和追踪性能指标，你能更全面地了解模型的学习过程及其真实表现，而不仅仅是训练损失。这种显式控制，虽然比 Keras 的 `compile`/`fit` 模式需要更多的设置，但与 PyTorch 赋予开发者对其模型训练流程完全透明和掌控的理念是一致的。

## 参考资料

- [TorchMetrics Documentation](https://torchmetrics.readthedocs.io/en/stable/) — PyTorch Lightning Team (2024)
  提供了在 PyTorch 中使用 TorchMetrics 计算各种性能指标的全面指导，包括安装和 API 使用。
- [Keras metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics) — TensorFlow Authors (2024)
  官方文档，详细介绍了 Keras 框架中状态化度量对象的 API 和使用方法，有助于理解 Keras 的实现方式。
- [Scikit-learn User Guide: Metrics and Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) — Scikit-learn developers (2024)
  解释了 scikit-learn 中可用的各种模型评估指标，包括它们的定义和实际使用。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，涵盖机器学习概念，并详细解释了精确度、召回率和 F1 分数等分类指标。

---

[上一节](04-PyTorch%20%E4%B8%AD%E7%9A%84%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97%E4%B8%8E%E6%9D%83%E9%87%8D%E6%9B%B4%E6%96%B0.md) · [下一节](06-PyTorch%20%E4%B8%AD%E7%9A%84%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E5%BE%AA%E7%8E%AF.md)

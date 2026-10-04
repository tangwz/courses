---
course: "getting-started-with-pytorch"
chapter: "monitoring-debugging-models"
lesson: "visualizing-training-tensorboard"
sourceId: 2229
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-8-monitoring-debugging-models/visualizing-training-tensorboard"
title: "使用 TensorBoard 可视化训练进度"
description: "通过 `torch.utils.tensorboard` 使用 TensorBoard 记录标量（损失、准确率）、图像和模型图。"
order: 5
plots: ["plots/2229-0.json"]
sourceHash: "01dca90e4ef42d0cc6017bb4b63db8b8c35cb68bc8bd2af0b6f9e738124b980b"
sourceCorrections: []
---

虽然打印语句和手动记录可以提供损失或准确率等指标的快照，但它们往往无法清晰地展示整个训练过程中的趋势和变化。损失是在持续下降，还是在剧烈波动？验证准确率是否停滞不前？使用可视化工具回答这些问题会容易得多。TensorBoard 是一个功能强大的可视化工具包，最初为 TensorFlow 开发，它通过 `torch.utils.tensorboard` 模块与 PyTorch 顺畅结合。它能让您在基于网络的仪表板中跟踪和可视化模型训练的各个方面。

### 设置 SummaryWriter

PyTorch 中用于将数据记录到 TensorBoard 的主要接口是 `SummaryWriter` 类。您通常会在训练脚本的开头实例化它。

```python
from torch.utils.tensorboard import SummaryWriter
import torch # 假设 torch 已被导入

# 创建一个 SummaryWriter 实例
# 这将创建一个类似 'runs/experiment_name' 的目录
# 如果未提供参数，则默认为 'runs/CURRENT_DATETIME_HOSTNAME'
log_dir = 'runs/my_first_experiment'
writer = SummaryWriter(log_dir)

print(f"TensorBoard 日志目录: {log_dir}")
# 之后可以通过以下命令查看: tensorboard --logdir runs
```

`SummaryWriter` 会将事件文件写入指定的 `log_dir`。TensorBoard 读取这些文件来生成可视化图表。最佳做法是为不同实验（例如，改变超参数 (parameter) (hyperparameter)）使用不同的目录，以便您可以轻松比较各个运行。

### 记录标量值

TensorBoard 最常用的场景是记录随时间变化的标量值，例如损失和准确率。这通过 `add_scalar` 方法完成。

`writer.add_scalar(tag, scalar_value, global_step=None)`

- `tag` (字符串): 标量的名称，例如 'Training Loss' 或 'Validation Accuracy'。在标签中使用斜杠（例如，'Loss/train', 'Loss/validation'）有助于在 TensorBoard 用户界面中组织图表。
- `scalar_value` (浮点数或整数): 您想记录的值。请注意，这应该是一个 CPU 标量值。如果您的损失在 GPU 上，您需要使用 `.item()` 将其移至 CPU。
- `global_step` (整数): 与此数据点关联的步数，通常表示 epoch 数或批次迭代计数。这决定了图中 x 轴的值。

让我们看看如何将其整合到一个典型的训练和验证循环结构中：

```python
# --- 假设这些已定义: ---
# model: 您的 torch.nn.Module
# train_loader, valid_loader: 您的 DataLoaders
# criterion: 您的损失函数 (例如, nn.CrossEntropyLoss())
# optimizer: 您的优化器 (例如, optim.Adam(model.parameters()))
# num_epochs: 训练的 epoch 数量
# device: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ---------------------------------

model.to(device)

for epoch in range(num_epochs):
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    total_train_samples = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0) # 累加按批次大小加权的损失
        total_train_samples += inputs.size(0)

        # 每 N 次迭代（例如 100 次）记录批次损失
        log_interval = 100
        if i % log_interval == log_interval - 1:
            current_step = epoch * len(train_loader) + i
            avg_batch_loss = running_loss / (log_interval * train_loader.batch_size) # 近似区间平均值
            writer.add_scalar('Loss/train_batch', avg_batch_loss, current_step)
            # 注意: 这只是一个示例记录方案

    epoch_loss = running_loss / total_train_samples # 该 epoch 的平均损失
    writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, 训练损失: {epoch_loss:.4f}')

    # --- 验证阶段 ---
    model.eval() # 设置模型为评估模式
    validation_loss = 0.0
    correct = 0
    total_val_samples = 0
    with torch.no_grad(): # 禁用梯度计算
        for data in valid_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_val_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = validation_loss / total_val_samples
    accuracy = 100.0 * correct / total_val_samples

    writer.add_scalar('Loss/validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, 验证损失: {avg_val_loss:.4f}, 准确率: {accuracy:.2f}%')

# 训练完成后关闭写入器
writer.close()
print("训练完成。TensorBoard 日志已保存。")
```

在这个例子中:

1. 我们在训练循环之前实例化了 `SummaryWriter`。
2. 在训练循环中，我们定期记录每个批次的平均训练损失以及每个 epoch 的平均损失。
3. 在验证循环中，我们计算并记录每个 epoch 的平均验证损失和准确率。
4. 我们将 `epoch` 数字用作 epoch 级别指标的 `global_step`。对于批次级别指标，我们根据 epoch 和批次索引计算组合步数。
5. 重要的是，我们在训练结束后调用 `writer.close()` 以确保所有缓冲数据都写入磁盘。

### 启动和使用 TensorBoard

您的脚本运行并生成日志文件后，您可以从终端启动 TensorBoard 界面。导航到包含 `runs`（或自定义日志）目录的父目录并运行：

```bash
tensorboard --logdir runs
```

如果您使用了特定的目录，例如 `logs/my_experiment_1`，您将使用：

```bash
tensorboard --logdir logs/my_experiment_1
```

TensorBoard 通常会启动一个网络服务器，通常在 `http://localhost:6006`。在您的网络浏览器中打开此地址。您应该会看到一个仪表板，您可以在其中查看记录的标量、比较不同运行（如果您的 `logdir` 中有多个子目录），并观察 epoch 或步数的变化趋势。



![TensorBoard 指标示例](plots/2229-0.json)



> 一个类似于 TensorBoard 可能显示的图表示例，展示了训练损失、验证损失和验证准确率在不同 epoch 上的变化。

### 关于标量

记录标量是基础操作，但 `SummaryWriter` 也提供了可视化其他类型数据的方法，这对于更具体的调试情况很有用：

- `add_histogram(tag, values, global_step)`: 跟踪张量值随时间的变化分布。这对于监控不同层中权重 (weight)或梯度的分布非常有用，可以帮助诊断梯度消失或梯度爆炸等问题。
- `add_graph(model, input_to_model)`: 可视化模型的架构。您传入 `nn.Module` 和一个示例输入张量。TensorBoard 会显示操作图，这有助于验证连接和形状。请注意，动态控制流（例如依赖于张量值的 `if` 语句）可能无法完全显示。
- `add_image(tag, img_tensor, global_step)`: 记录图像。在计算机视觉任务中很有用，可以在训练期间查看示例输入、输出或生成的图像。`img_tensor` 的格式需要仔细处理（例如 `CHW` 或 `NCHW`）。
- `add_embedding(mat, metadata, label_img, global_step, tag)`: 使用 PCA 或 t-SNE 等技术将高维嵌入 (embedding)（如词嵌入或图像特征）可视化到较低维空间。

对于中级课程，掌握 `add_scalar` 是最重要的一步。随着遇到更复杂的调试难题，试用 `add_histogram` 用于权重/梯度以及 `add_graph` 用于模型结构是很好的后续步骤。

使用 TensorBoard 将调试从解读一串数字转变为分析视觉趋势。它能帮助了解收敛速度、潜在的过拟合 (overfitting)（比较训练损失与验证损失），以及学习过程的稳定性，使其成为实际深度学习 (deep learning)开发中必不可少的工具。

## 参考资料

- [Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) — PyTorch Team (2024)
  Publisher: PyTorch Foundation
  PyTorch工作流中集成TensorBoard的实用指南，涵盖SummaryWriter和各种日志记录方法。
- [TensorBoard](https://www.tensorflow.org/tensorboard) — Google (2024)
  TensorBoard功能和能力的官方概述，包括标量日志记录和可视化，这对于在PyTorch中使用它至关重要。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  提供了PyTorch的覆盖，包括模型训练、监控和调试技术，其中自然会包括TensorBoard等工具。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，讨论了监控训练和验证指标对于理解模型性能和诊断问题的重要性。

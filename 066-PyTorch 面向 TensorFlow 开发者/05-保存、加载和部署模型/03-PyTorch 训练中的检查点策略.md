# PyTorch 训练中的检查点策略

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-5-pytorch-model-persistence-deployment/checkpointing-pytorch-training)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练机器学习 (machine learning)模型是一个耗时的过程，特别是使用大型数据集或复杂架构时。中断是常见的，无论是由于系统崩溃、资源限制，还是仅仅因为需要暂停后稍后继续。检查点是指定期保存模型和训练过程状态的做法。这确保了你可以从上次保存的点恢复训练，避免进度丢失和计算资源浪费。它对获取模型的中间版本也很有用，这些版本在验证数据上的表现可能比最终模型更好，特别是在发生过拟合 (overfitting)时。

### 检查点应包含哪些内容

训练机器学习 (machine learning)模型是一个耗时的过程。为了防止因中断而丢失进度或能够随时恢复训练，检查点机制必不可少。一个全面的检查点应该允许尽可能准确地恢复训练过程。仅仅保存模型的权重 (weight)（`state_dict`）通常不足以顺利恢复。为了确保训练的平稳恢复，通常需要保存更多内容。

- **模型 `state_dict`**：这个字典包含模型的所有可学习参数 (parameter)（权重和偏置 (bias)）。通过 `model.state_dict()` 获取。
- **优化器 `state_dict`**：Adam 或 SGD 等优化器也有内部状态（例如，动量缓冲区、参数的学习率）。通过 `optimizer.state_dict()` 保存它，优化器就能从上次停止的地方精确地继续。
- **当前 Epoch 编号**：了解 epoch 编号对于恢复训练循环、管理学习率调度和日志记录很重要。
- **最后记录的损失**：保存检查点 epoch 的训练和验证损失对于跟踪进度很有用。
- **学习率调度器状态**：如果您使用学习率调度器（例如 `torch.optim.lr_scheduler`），其状态也应使用 `scheduler.state_dict()` 保存，以确保学习率在恢复后正确地按照其调度进行。
- **其他自定义变量**：任何其他重建训练状态所需的脚本特定变量，例如随机数生成器状态（如果需要非常具体的复现性，尽管这可能很复杂），或自定义指标。

以下是说明检查点文件典型组成部分的图表：

> 检查点文件通常包含模型的参数、优化器状态、当前 epoch、损失值，以及可选的学习率调度器状态。

### 实现检查点策略

PyTorch 让你完全控制何时以及如何保存检查点。这与 TensorFlow 的 Keras API 不同，Keras API 中的检查点通常由 `tf.keras.callbacks.ModelCheckpoint` 等回调函数处理。尽管 Keras 回调函数提供便利，PyTorch 的手动方法提供了更大的灵活性。

以下是一些常见策略：

#### 1. 定期保存（每 N 个 Epoch）

这是一种直接的策略，即每固定数量的 epoch 保存一个检查点。这可以确保你有合理的频繁备份。

```python
# 在你的训练循环中
# 假设 model、optimizer、epoch、current_loss 已定义

SAVE_EVERY_N_EPOCHS = 10
# ... 在一个 epoch 完成后 ...
if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
    checkpoint_path = f'./checkpoints/model_epoch_{epoch+1}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss,
        # 'scheduler_state_dict': scheduler.state_dict(), # 如果你使用调度器
    }, checkpoint_path)
    print(f"检查点在 epoch {epoch+1} 保存到 {checkpoint_path}")
```

主要的权衡是磁盘空间与备份的粒度。保存过于频繁会占用大量存储，而保存过于不频繁则有丢失更多进度的风险。

#### 2. 保存表现最佳的模型

通常，主要目标是保存验证数据集上表现最佳的模型。这有助于避免保存过拟合 (overfitting)的模型。

```python
# 在训练循环外部初始化
best_val_loss = float('inf')
# ...
# 在你的训练循环中，验证阶段之后
# 假设 val_loss 已计算
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model_path = './checkpoints/best_model.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), # 对最佳模型来说是可选的，但对于微调很有用
        'val_loss': best_val_loss,
    }, best_model_path)
    print(f"新最佳模型在 epoch {epoch+1} 保存，验证损失为: {best_val_loss:.4f}")
```

如果“最佳模型”的主要目的是推理 (inference)，你可以选择只保存模型的 `state_dict`，但如果你稍后决定从这个最佳状态进行微调 (fine-tuning)，包含优化器和 epoch 也会很有用。

#### 3. 保存最新检查点

除了保存最佳模型或定期检查点外，通常保存模型的最新状态也很有用。这通常在每个保存间隔被覆盖，并用于在训练中断时立即恢复。

```python
# 在你的训练循环中，可能每个 epoch 或每隔几个 epoch
latest_checkpoint_path = './checkpoints/latest_checkpoint.pth'
torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': current_loss, # 或 current_val_loss
    # 'scheduler_state_dict': scheduler.state_dict(),
}, latest_checkpoint_path)
print(f"最新检查点在 epoch {epoch+1} 保存")
```

你可以结合这些策略。例如，每个 epoch 保存 `latest_checkpoint.pth`，每 N 个 epoch 保存 `model_epoch_{N}.pth`，以及每当验证性能提升时保存 `best_model.pth`。

### 组织你的检查点

良好的组织很重要，特别是对于长时间实验。

- **命名约定**：使用有意义的名称。包含 epoch 编号（例如 `checkpoint_epoch_050.pth`）或验证指标（例如 `model_val_acc_0.92.pth`）会非常有帮助。
- **目录结构**：为每个实验的检查点创建一个专用目录（例如 `checkpoints/my_experiment/`）。
- **管理磁盘空间**：对于非常长的训练运行，你可能会实施一项策略，只保留最后 K 个检查点或前 N 个最佳模型来管理磁盘使用。

### 从检查点加载以恢复训练

要恢复训练，你需要将保存的状态重新加载到模型、优化器和其他相关变量中。

```python
# 在开始训练循环之前，或在脚本的开头
# 首先定义模型、优化器，以及可选的调度器
# model = YourModelClass(*args)
# optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

start_epoch = 0
# 你想加载的检查点路径
checkpoint_to_load_path = './checkpoints/latest_checkpoint.pth' # 或特定 epoch 的检查点

if os.path.exists(checkpoint_to_load_path):
    checkpoint = torch.load(checkpoint_to_load_path) # 如果需要，添加 map_location

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    last_loss = checkpoint.get('loss', float('inf')) # 对于可选键使用 .get

    # if 'scheduler_state_dict' in checkpoint and scheduler is not None:
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"从 epoch {start_epoch} 恢复训练，损失为 {last_loss:.4f}")
else:
    print("未找到检查点，从头开始训练。")

# 你的训练循环将从 start_epoch 开始
# for epoch in range(start_epoch, NUM_EPOCHS):
#     # ... 训练逻辑 ...
```

加载检查点时，特别是当你可能在不同环境（例如，GPU 训练的模型到 CPU 进行推理 (inference)，或反之）之间移动时，请在 `torch.load()` 中使用 `map_location` 参数 (parameter)：

- `torch.load(PATH, map_location=torch.device('cpu'))` 将在 GPU 上训练的模型加载到 CPU。
- `torch.load(PATH, map_location='cuda:0')` 加载到特定的 GPU。

加载 `state_dict` 后，如果你要恢复训练，请记住调用 `model.train()`；如果你加载模型是为了推理，则调用 `model.eval()`，以设置 Dropout 和 Batch Normalization 等层的合适模式。

### 最佳实践与注意事项

- **测试你的检查点**：务必测试你的保存和加载机制，确保其正常工作。一个常见错误是忘记保存或加载优化器状态，这可能导致学习进度重置。
- **原子性**：保存检查点时，特别是如果进行覆盖（如 `latest_checkpoint.pth`），更安全的方法是先保存到临时文件，然后原子地将其重命名为最终路径。这可以防止脚本在保存操作期间崩溃导致文件损坏。

  ```python
  temp_path = checkpoint_path + ".tmp"
  torch.save(state, temp_path)
  os.replace(temp_path, checkpoint_path) # os.replace 是原子操作
  ```
- **存储**：将检查点保存到可靠的存储介质。对于重要实验，考虑将其备份到不同位置。
- **灵活性**：PyTorch 的手动方法意味着你可以广泛地自定义检查点，例如记录额外的元数据，或与 TensorBoard 或 Weights & Biases 等实验跟踪工具配合使用以记录检查点事件。

通过实施检查点策略，你可以使你的 PyTorch 训练流程更具适应性和易于管理，确保宝贵的计算时间不会丢失，并且你始终可以获取模型最有前景的版本。这是模型开发的一个基本组成部分，允许进行更多实验和更安全地执行长时间运行的训练任务。

## 参考资料

- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) — Matthew Inkawhich (2024)
  PyTorch 官方文档，详细介绍了如何保存和加载模型 `state_dict`、优化器 `state_dict` 以及其他训练组件，对于恢复训练和推理至关重要。
- [os - Miscellaneous operating system interfaces](https://docs.python.org/3/library/os.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  Python `os` 模块的官方文档，特别介绍了 `os.replace` 等函数，用于原子文件操作，是稳健保存检查点的好方法。
- [Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — Sasank Chilamkurthy (2017)
  Publisher: PyTorch Foundation
  一个全面的 PyTorch 教程，演示了完整训练循环的实现，包括根据验证准确性保存表现最佳模型的实用示例。

---

[上一节](02-%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E5%AE%8C%E6%95%B4%E6%A8%A1%E5%9E%8B%E4%B8%8E%E4%BB%85%E4%BF%9D%E5%AD%98%E5%8F%82%E6%95%B0.md) · [下一节](04-%E5%9C%A8PyTorch%E4%B8%AD%E6%9F%A5%E7%9C%8B%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E5%92%8C%E6%9D%83%E9%87%8D.md)

# 训练控制：Keras 回调与 PyTorch 自定义逻辑

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs/keras-callbacks-vs-pytorch-hooks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

TensorFlow Keras 提供了一个便捷的 `model.fit()` 方法，并通过回调系统来管理训练过程的各个方面，而 PyTorch 则鼓励一种更亲力亲为的方式。PyTorch 中的训练循环是显式编写的。这种显式性也延伸到如何实现训练控制机制，例如早停、模型检查点和动态学习率调整。相较于预定义的回调对象，这些逻辑直接集成到训练脚本中。这提供了细粒度的控制，并清楚地了解每一步发生的情况。

### Keras 回调：快速回顾以作背景介绍

如果你曾大量使用 Keras，你可能对它的回调系统很熟悉。回调是传递给 `fit()` 方法的对象，它们可以在训练的不同阶段执行操作（例如，在每个训练周期开始或结束时，在每个批次之前或之后）。常见的例子有：

- `ModelCheckpoint`: 在某个频率下保存模型或权重 (weight)。
- `EarlyStopping`: 当监控的指标不再提升时停止训练。
- `ReduceLROnPlateau`: 当监控的指标不再提升时降低学习率。
- `TensorBoard`: 记录事件以便使用 TensorBoard 可视化。

这些回调抽象了底层逻辑，使添加常见功能变得容易。在 PyTorch 中，你会通过自己编写逻辑来达到类似的效果。

### 在 PyTorch 循环中实现训练控制

让我们看看如何在标准的 PyTorch 训练循环中实现这些常见的训练控制模式。

#### 1. 早停

早停有助于防止过拟合 (overfitting)，它会在模型在验证集上的性能连续指定数量的训练周期（通常称为“耐心”）不再提升时停止训练。

为了实现早停，你需要：

- 监控一个验证指标（例如，验证损失）。
- 记录迄今为止最好的指标值。
- 维护一个没有提升的训练周期计数器。
- 如果此计数器超过你的耐心阈值，则停止训练。

以下是你可能将其集成到训练循环中的方式：

```python
# 假设这些已在其他地方定义：
# model, train_loader, val_loader, optimizer, criterion
# num_epochs, patience

best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    # --- 训练阶段 ---
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    # --- 验证阶段 ---
    model.eval()
    current_val_loss = 0.0
    with torch.no_grad():
        for val_data, val_labels in val_loader:
            val_outputs = model(val_data)
            loss = criterion(val_outputs, val_labels)
            current_val_loss += loss.item()

    current_val_loss /= len(val_loader)
    print(f"训练周期 {epoch+1}/{num_epochs}, 验证损失: {current_val_loss:.4f}")

    # 早停逻辑
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        epochs_no_improve = 0
        # （可选）在此处保存最佳模型（请参阅下一节）
        torch.save(model.state_dict(), 'best_model_checkpoint.pth')
        print(f"验证损失有所提升。已保存最佳模型。")
    else:
        epochs_no_improve += 1
        print(f"验证损失在 {epochs_no_improve} 个训练周期内没有提升。")

    if epochs_no_improve >= patience:
        print(f"在 {epoch+1} 个训练周期后触发早停。")
        break 
```

在此代码片段中，`patience` 是你定义的一个整数（例如，5 或 10）。如果验证损失在 `patience` 个训练周期内没有提升，训练循环将终止。

#### 2. 模型检查点

模型检查点涉及到在训练期间保存模型的状态（通常是其学习到的参数 (parameter)）。这有几个作用：

- 如果训练中断，你可以稍后恢复训练。
- 你可以取回在验证集上表现最好的模型，即使训练继续进行，性能后来因过拟合而下降。

如上面的早停示例所示，一种常见的策略是每当验证指标提升时就保存模型：

```python
# 在验证阶段，计算 current_val_loss 之后
if current_val_loss < best_val_loss:
    best_val_loss = current_val_loss
    epochs_no_improve = 0
    torch.save(model.state_dict(), 'best_model_checkpoint.pth') # 保存模型
    print(f"训练周期 {epoch+1}: 验证损失提升到 {best_val_loss:.4f}，正在保存模型...")
else:
    # ... （epochs_no_improve 逻辑）
```

你也可以按固定的训练周期间隔保存检查点，无论验证性能如何，如果你想保留模型历史记录或用于非常长的训练运行。

#### 3. 学习率调整

在训练期间调整学习率是一种常用技术，以提升收敛速度和最终模型性能。PyTorch 提供了 `torch.optim.lr_scheduler` 模块，它提供了多种随时间改变学习率的方式。这类似于 Keras 的 `LearningRateScheduler` 或 `ReduceLROnPlateau` 回调。

要使用学习率调整器：

1. 像往常一样定义你的优化器。
2. 实例化一个调整器，并将优化器传递给它。
3. 在训练循环中的适当位置调用 `scheduler.step()`（通常在每个训练周期之后，或者有时在每个批次之后，具体取决于调整器）。对于像 `ReduceLROnPlateau` 这样的调整器，你需要将要监控的指标（例如，验证损失）传递给 `step()` 方法。

让我们来看一个使用 `ReduceLROnPlateau` 的例子，它在监控的指标停止提升时降低学习率，类似于其 Keras 中的对应功能：

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 假设优化器已定义
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 初始化调整器
# 'min' 模式表示当监控的量停止下降时，学习率将被降低
# factor 是学习率将被降低的因子。new_lr = lr * factor
# patience 是学习率将被降低之前，指标没有提升的训练周期数。
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# ... （在你的训练循环中，验证阶段之后） ...
# 在计算完该训练周期的 current_val_loss 之后：
scheduler.step(current_val_loss)
```

当调用 `scheduler.step(current_val_loss)` 时，调整器会根据其 `patience` 设置检查 `current_val_loss` 是否有所提升。如果没有，它会按指定的 `factor` 降低 `optimizer` 的学习率。`verbose=True` 参数会在学习率调整时打印一条消息。

其他常见的调整器包括：

- `StepLR`: 每 `step_size` 个训练周期将学习率乘以因子 `gamma`。
- `ExponentialLR`: 每个训练周期将学习率乘以因子 `gamma`。
- `CosineAnnealingLR`: 使用余弦退火策略调整学习率。

核心是在正确的频率调用 `scheduler.step()`（通常是每个训练周期一次，对于按批次调整的调整器，是在 `optimizer.step()` 之后立即调用，对于像 `ReduceLROnPlateau` 这样的按训练周期调整的调整器，是在验证之后调用）。

#### 4. 自定义指标记 (token)录

尽管 Keras 有一个 `TensorBoard` 回调可以自动记录许多指标，但在 PyTorch 中，你通常会使用 `torch.utils.tensorboard.SummaryWriter` 来显式记录值。你可以在循环中的任何位置记录训练损失、验证损失、准确率、学习率或任何其他自定义指标。

```python
from torch.utils.tensorboard import SummaryWriter

# 初始化写入器
writer = SummaryWriter('runs/my_experiment_name')

# ... （在你的训练循环中） ...

# 记录训练损失（按批次或按训练周期平均）
# 假设 train_loss 是按批次计算的
# writer.add_scalar('Loss/train_batch', train_loss.item(), epoch * len(train_loader) + batch_idx)

# 记录验证损失（按训练周期）
# 假设 current_val_loss 是按训练周期计算的
writer.add_scalar('Loss/validation_epoch', current_val_loss, epoch)

# 记录学习率（按训练周期）
writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

# ... （训练后） ...
writer.close()
```

然后你运行 TensorBoard 并指向 `runs` 目录以可视化这些指标。这让你能精确控制记录什么以及何时记录。

### 控制流差异可视化

训练控制的处理方式有显著不同。在 Keras 中，回调有点像 `fit` 方法执行周期中的插件。在 PyTorch 中，这些控制机制是你自定义循环的组成部分。

> 训练循环结构对比。Keras 回调在 `model.fit()` 的特定点执行。PyTorch 将控制逻辑直接集成到自定义编写的循环中。

### 方法对比：灵活性对比便利性

- **灵活性和透明度**：PyTorch 将控制逻辑直接嵌入 (embedding)到训练循环中的方式提供了最大的灵活性。你可以实现高度定制的行为，在任何时候检查任何变量，并准确了解你的代码在做什么。Keras 回调系统虽然方便，但有时感觉像一个“黑箱”，并且，超出标准选项的自定义行为可能需要更深入地了解其内部工作方式。
- **冗长性与样板代码**：PyTorch 灵活性的代价是，对于早停或模型检查点等常见任务，你会编写更多代码，相比于简单地添加 Keras 回调。Keras 在减少这些标准场景下的样板代码方面表现出色。
- **控制点**：在 Keras 中，回调有绑定到特定事件的预定义方法（例如，`on_epoch_end`, `on_batch_begin`）。在 PyTorch 中，你精确决定你的逻辑在构建的循环结构中何时执行。这可以是批次之后、验证之前、优化器步进之后。控制权完全在你手中。
- **状态管理**：使用 PyTorch，你直接管理控制逻辑所需的状态（例如 `best_val_loss` 或 `epochs_no_improve`），将其作为训练脚本中的变量，或者，如果你以类的方式组织训练循环，则在类中管理。Keras 回调通常封装自己的状态。

### 创建可复用的“类似回调”的结构

尽管 PyTorch 没有像 Keras 那样正式的回调系统，你仍然可以创建可复用的组件。如果你发现自己在不同项目中反复编写相同的早停逻辑，你可以将其封装到 Python 函数甚至简单的类中。

例如，一个早停函数可能如下所示：

```python
def check_early_stopping(current_val_loss, best_val_loss, epochs_no_improve, patience):
    stop_training = False
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        epochs_no_improve = 0
        # 可能会返回一个标志以保存模型
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        stop_training = True
        print(f"早停已触发。耐心值: {patience}。")

    return best_val_loss, epochs_no_improve, stop_training

# 在你的循环中：
# best_val_loss, epochs_no_improve, should_stop = check_early_stopping(...)
# if should_stop:
#     break
```

这不会复制 Keras 回调的完整事件驱动特性，但它促进了在你的显式 PyTorch 循环中常见模式的代码复用。更高级的模式，例如那些涉及钩子直接修改梯度或激活的模式，将在第 6 章讨论 `torch.nn.Module.register_forward_hook` 和类似功能时提到。对于大多数训练控制，直接循环集成是 PyTorch 的标准做法。

总之，在 PyTorch 中管理训练控制涉及到在你的训练和验证循环中编写显式 Python 逻辑。尽管这最初看起来比使用 Keras 回调更复杂，但它提供了高度的透明度和定制性，使你能够根据自己的需求精确地调整训练过程。随着你对 PyTorch 越来越熟悉，你可能会欣赏这种对模型训练机制的直接控制。

## 参考资料

- [Keras Callbacks](https://keras.io/api/callbacks/) — Keras team (2024)
  Keras回调的官方文档，解释了其目的、可用类型和实现细节。
- [PyTorch Documentation: Training your models](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — PyTorch team (2024)
  Publisher: PyTorch Foundation
  一个PyTorch教程，展示了标准训练循环，涵盖了前向、反向和优化等必要步骤。
- [PyTorch Documentation: torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — PyTorch team (2025)
  Publisher: PyTorch Foundation
  PyTorch学习率调度器的官方文档，详细介绍了各种策略及其API用法。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本深度学习的基础教材，为早期停止和正则化等训练策略提供了理论基础。
- [PyTorch Documentation: torch.utils.tensorboard.SummaryWriter](https://pytorch.org/docs/stable/tensorboard.html) — PyTorch team (2022)
  官方文档解释了如何将TensorBoard与PyTorch集成，用于记录指标和可视化训练进度。

---

[上一节](06-PyTorch%20%E4%B8%AD%E7%9A%84%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E5%BE%AA%E7%8E%AF.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AA%E5%AE%8C%E6%95%B4%E7%9A%84%E8%AE%AD%E7%BB%83%E5%92%8C%E8%AF%84%E4%BC%B0%E5%BE%AA%E7%8E%AF.md)

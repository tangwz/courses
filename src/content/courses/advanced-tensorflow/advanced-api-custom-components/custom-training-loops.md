---
course: "advanced-tensorflow"
chapter: "advanced-api-custom-components"
lesson: "custom-training-loops"
sourceId: 2881
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-4-advanced-api-custom-components/custom-training-loops"
title: "编写自定义训练循环"
description: "通过使用 tf.GradientTape 编写您自己的训练循环，获得对训练过程的完全控制。"
order: 5
plots: []
sourceHash: "c901388ce02a2990a815bb86148e8fc2b0daea7b913178cfa8179d1cbe144742"
sourceCorrections: []
---

虽然 `model.fit()` 为大多数标准训练场景提供了便捷的抽象，但常常会出现需要对训练过程进行更精细控制的情况。实现自定义训练循环可以赋予您这种控制能力，允许进行非标准梯度更新、复杂的指标计算，或与Keras回调不易处理的外部系统集成。这里将详细说明如何使用TensorFlow的核心组件构建这些循环。

### 为何编写自定义训练循环？

当您需要时，可以选择自定义训练循环：

- **实现非标准训练算法：** 例如，训练生成对抗网络 (GAN)（GANs）通常涉及使用不同损失函数 (loss function)，交替更新模型的不同部分（生成器和判别器）。
- **应用自定义梯度处理：** 您可能需要裁剪梯度，根据梯度值应用自定义正则化 (regularization)，或在特定条件下仅更新模型的特定层。
- **获得指标更新和日志记录的细粒度控制：** 尽管Keras回调提供了定制功能，但自定义循环在何时以及如何计算、汇总和记录指标方面提供了完全的自主权。
- **集成外部逻辑：** 您可能需要与其他系统交互，或在训练迭代中执行复杂的状体管理。
- **理解底层机制：** 从头开始构建训练循环有助于巩固您对TensorFlow和Keras如何执行优化的理解。

### 核心组件

自定义训练循环主要协调这些组件之间的交互：

1. **模型：** 您的 `tf.keras.Model` 实例（通过Sequential、函数式API或子类化创建）。
2. **数据集：** 一个 `tf.data.Dataset`，提供输入特征和目标标签的批次数据。
3. **损失函数 (loss function)：** 一个可调用函数（如 `tf.keras.losses.SparseCategoricalCrossentropy`），根据模型预测和真实标签计算损失值。
4. **优化器：** `tf.keras.optimizers.Optimizer` 的一个实例（例如 `tf.keras.optimizers.Adam`），负责将梯度应用于模型的可训练变量。
5. **`tf.GradientTape`：** 自动微分的引擎。它会记录在其上下文 (context)内执行的操作，允许您计算目标（通常是损失）相对于源变量（通常是模型的可训练变量）的梯度。
6. **指标：** `tf.keras.metrics.Metric` 实例（例如 `tf.keras.metrics.Accuracy`），用于在训练和评估期间跟踪性能。

### 自定义训练循环的结构

其基本结构涉及嵌套循环：一个用于迭代周期（epochs）的外部循环，以及一个用于每个迭代周期内批次（batches）的内部循环。

以下是内部（批次）循环中典型步骤的分解：

1. **获取数据：** 从数据集迭代器中获取下一批数据（`x_batch`，`y_batch`）。
2. **开始记录：** 打开 `tf.GradientTape` 上下文 (context)。TensorFlow将监控在此块内访问的涉及可训练 `tf.Variable` 对象的运算。

   ```python
   with tf.GradientTape() as tape:
       # 此处记录的操作
   ```
3. **前向传播：** 用输入批次调用模型以获取预测。如果您的模型包含 `Dropout` 或 `BatchNormalization` 等在训练和推断期间行为不同的层，请确保传递 `training=True`。

   ```python
   y_pred = model(x_batch, training=True)
   ```
4. **计算损失：** 使用您选择的损失函数 (loss function)，计算模型预测和真实标签之间的损失。

   ```python
   loss_value = loss_fn(y_batch, y_pred)
   # 如果损失函数涉及层添加的正则化项，
   # 您可能需要添加 model.losses
   loss_value += sum(model.losses)
   ```
5. **计算梯度：** 使用tape计算损失相对于模型的可训练变量的梯度。

   ```python
   grads = tape.gradient(loss_value, model.trainable_variables)
   ```
6. **应用梯度：** 让优化器使用计算出的梯度更新模型的权重 (weight)。

   ```python
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   ```
7. **更新指标：** 根据批次结果更新您所选指标的状态。

   ```python
   train_accuracy_metric.update_state(y_batch, y_pred)
   train_loss_metric.update_state(loss_value)
   ```

外部（迭代周期）循环处理诸如遍历数据集以完成一次完整传递、在每个迭代周期开始时重置指标、记录结果以及可能运行一个验证循环等任务。

> 一个自定义训练循环的典型流程图，显示了迭代周期和批次迭代以及梯度计算和应用。

### 示例：一个基本的自定义训练循环

让我们用一个简单的例子来说明。假设您有一个编译好的Keras模型（`model`）、一个优化器（`optimizer`）、一个损失函数 (loss function)（`loss_fn`）、训练数据（`train_dataset`）以及指标（`train_loss_metric`，`train_accuracy_metric`）。

```python
import tensorflow as tf

# 假设 model, optimizer, loss_fn, train_dataset 已定义
# 假设 train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
# 假设 train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

epochs = 5

# 定义用于提升性能的训练步骤函数
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)
        # 如果存在正则化损失，请添加
        loss += sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 更新指标
    train_loss_metric.update_state(loss)
    train_accuracy_metric.update_state(y_batch, predictions)

# 训练循环
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch+1}")

    # 在每个迭代周期开始时重置指标
    train_loss_metric.reset_state()
    train_accuracy_metric.reset_state()

    # 遍历数据集的批次
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        train_step(x_batch_train, y_batch_train)

        # 每N个批次记录一次（可选）
        if step % 100 == 0:
             print(f"Step {step}: Loss: {train_loss_metric.result():.4f}, Accuracy: {train_accuracy_metric.result():.4f}")

    # 在每个迭代周期结束时显示指标
    train_loss = train_loss_metric.result()
    train_acc = train_accuracy_metric.result()
    print(f"Epoch {epoch+1}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")

    # 可选：在此处运行一个验证循环，结构类似
    # 但无需 GradientTape 和梯度应用。请记住调用
    # model(x_val_batch, training=False)。
```

### 善用 `tf.function` 提升性能

请注意示例中应用于 `train_step` 函数的 `@tf.function` 装饰器。这对于性能很重要。TensorFlow会分析装饰函数内的Python代码，并生成一个优化的计算图。`train_step` 的后续调用会直接执行此图，从而在大多数操作中绕过较慢的Python解释器。

使用 `@tf.function` 时，请注意以下几点：

- **追踪：** TensorFlow会追踪函数（在Python中运行它），以便在首次使用一组唯一的输入签名（形状和数据类型）调用时构建图。在函数内部创建Python变量或执行带有副作用的操作可能导致意想不到的行为或频繁的重新追踪，从而降低性能提升。对于需要在图内跨调用保持和修改的状态，请使用 `tf.Variable`。
- **控制流：** 如果条件或循环边界依赖于张量，请使用TensorFlow的控制流操作（`tf.cond`，`tf.while_loop`），而不是Python的 `if`/`for`/`while`，以确保它们是图的一部分。

### 处理训练与推断行为

许多层，尤其是 `tf.keras.layers.BatchNormalization` 和 `tf.keras.layers.Dropout`，在训练和推断期间行为不同。批归一化 (normalization)在训练期间更新其移动均值和方差，但在推断期间使用它们进行归一化。Dropout在训练期间随机将激活设为零，但在推断期间不活跃。

在调用模型时正确传入 `training` 参数 (parameter)非常重要：

- `model(inputs, training=True)`：在训练步骤期间的 `GradientTape` 上下文 (context)内。
- `model(inputs, training=False)`：在执行验证或训练后进行预测时。

忘记这一点可能导致结果不正确或模型无法正常训练。

### 自定义循环 对比 `model.fit()`

在 `model.fit()` 和自定义循环之间进行选择涉及便利性与控制权之间的权衡：

- **在以下情况下使用 `model.fit()`：**
  - 您的训练过程是标准的（前向传播、损失、梯度、优化器步骤）。
  - 您希望快速实现，并受益于Keras回调、简易的分布式策略集成和进度记录等内置功能。
  - 您的定制需求可以通过自定义层、损失、指标或标准回调来满足。
- **在以下情况下使用自定义训练循环：**
  - 您需要完全控制梯度计算和应用过程（例如，梯度裁剪、GANs）。
  - 您需要实现复杂、非标准的更新规则或指标计算。
  - 您正在进行需要修改核心训练算法的研究。
  - 您希望更好地理解训练机制。

掌握自定义训练循环有助于您实现几乎任何训练算法，突破标准工作流，从而精确地定制TensorFlow以满足您的高级建模要求。

## 参考资料

- [Automatic differentiation with tf.GradientTape](https://www.tensorflow.org/guide/autodiff) — TensorFlow Authors (2023)
  此官方文档解释了`tf.GradientTape`用于自动微分的原理和用法，这是在任何自定义训练循环中计算梯度的基础机制。
- [Better performance with tf.function](https://www.tensorflow.org/guide/function) — TensorFlow Authors (2023)
  Publisher: TensorFlow
  本指南提供了关于使用`@tf.function`优化TensorFlow代码以提升性能的见解，这是实现自定义训练步骤时的一个重要考量。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本涵盖TensorFlow和Keras的综合性书籍，提供了实际示例和自定义组件的详细解释，包括与构建自定义训练循环相关的方面（第4版）。

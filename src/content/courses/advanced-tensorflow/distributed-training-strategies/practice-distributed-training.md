---
course: "advanced-tensorflow"
chapter: "distributed-training-strategies"
lesson: "practice-distributed-training"
sourceId: 2866
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-3-distributed-training-strategies/practice-distributed-training"
title: "实践：实现分布式训练"
description: "设置并运行一个使用`tf.distribute`策略的分布式训练任务。"
order: 9
plots: []
sourceHash: "353890ed6ce9c99ec8964e52622efbcbf8afd144bc6fdcdbadbc4225c4875e9c"
sourceCorrections: []
---

逐步演示使用TensorFlow的`tf.distribute.Strategy`来设置和运行分布式训练任务。这个动手练习将帮助你巩固对如何在多个设备上扩展训练的理解。为保证简单性和广泛适用性，主要关注`tf.distribute.MirroredStrategy`，它处理单机内多GPU上的同步训练。还将简单介绍多工作器训练所需的设置。

### 前提条件

确保你已安装TensorFlow。如果你的系统上有多个GPU，TensorFlow应能自动识别它们。你可以这样验证：

```python
import tensorflow as tf
import os
import json

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  print(f"Detected {len(gpus)} GPU(s):")
  for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")
else:
  print("No GPU detected. MirroredStrategy requires at least one GPU,")
  print("but works best with multiple GPUs. CPU execution will be slower.")
  print("Consider using tf.distribute.OneDeviceStrategy('/cpu:0') for CPU practice.")

# 定义用于打乱数据集的缓冲区大小
BUFFER_SIZE = 10000
# 定义全局批处理大小。这将在副本之间拆分。
GLOBAL_BATCH_SIZE = 64 * len(gpus) if gpus else 64 # 示例：每个副本64
# 定义训练的轮次数量
EPOCHS = 5
```

如果你没有多个GPU，`MirroredStrategy`仍将在单个GPU甚至CPU上运行（尽管没有从分布式获得的性能提升）。其原理保持不变。

### 1. 准备数据集

分布式训练需要细致处理数据集。每个副本（处理单元，通常是GPU）需要处理输入数据的不同分片。`tf.data` API与`tf.distribute.Strategy`结合。当你在策略范围内将`tf.data.Dataset`传递给`model.fit`时，TensorFlow会自动处理数据在副本之间的分片。

让我们使用`tf.data`创建一个简单的合成数据集：

```python
# 创建一个简单的合成数据集
def create_synthetic_dataset(num_samples=10000, num_features=10):
  # 生成随机特征和二元标签
  X = tf.random.normal(shape=(num_samples, num_features))
  coeffs = tf.random.normal(shape=(num_features, 1))
  logits = tf.matmul(X, coeffs) + tf.random.normal(shape=(num_samples, 1), stddev=0.1)
  y = tf.cast(logits > 0, tf.int64)
  return tf.data.Dataset.from_tensor_slices((X, y))

# 创建数据集
dataset = create_synthetic_dataset()

# 打乱并批处理数据集
# 注意：这里的批处理大小是全局批处理大小
dataset = dataset.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Dataset element spec: {dataset.element_spec}")
print(f"Global batch size: {GLOBAL_BATCH_SIZE}")
if gpus:
    print(f"Per-replica batch size: {GLOBAL_BATCH_SIZE // len(gpus)}")
```

`GLOBAL_BATCH_SIZE`是所有副本在一个步骤中处理的总批处理大小。`MirroredStrategy`将自动用此全局批处理大小除以副本数量，以确定每个副本的批处理大小。`prefetch(tf.data.AUTOTUNE)`用于提升性能，允许数据预处理与模型执行同时进行。

### 2. 定义模型

本例中我们将使用一个标准的Keras Sequential模型。重要的是我们*在哪里*定义模型。

```python
def build_model(num_features=10):
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(16, activation='relu', input_shape=(num_features,)),
      tf.keras.layers.Dense(8, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  return model
```

### 3. 选择并初始化策略

现在，我们选择我们的分布式策略。对于单机多GPU的情况，`MirroredStrategy`是常用选择。

```python
# 如果GPU可用，使用MirroredStrategy。否则，对CPU使用OneDeviceStrategy。
if gpus:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas.")
else:
    # 用于演示目的，当CPU或单个GPU场景时的备用方案
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0') # 或者如果存在一个GPU，使用'/gpu:0'
    print("Using OneDeviceStrategy (fallback).")

# 计算实际的副本数量
num_replicas = strategy.num_replicas_in_sync
```

### 4. 在策略范围内创建模型和优化器

这是重要步骤。为确保模型变量和优化器状态在相应设备上创建并正确镜像，它们的创建*必须*发生在`strategy.scope()`内部。

```python
# 在策略范围内创建模型和优化器
with strategy.scope():
  # 模型构建
  model = build_model()

  # 优化器定义
  optimizer = tf.keras.optimizers.Adam()

  # 损失函数定义
  loss_object = tf.keras.losses.BinaryCrossentropy(
      from_logits=False,
      reduction=tf.keras.losses.Reduction.NONE # 对于分布式训练很重要！
  )

  # 定义指标
  train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

  # 编译模型（可选但推荐用于Keras的fit/evaluate）
  # 注意：这里我们使用一个虚拟损失，因为我们将在下面手动计算它
  # 或者如果使用model.fit，直接提供loss_object
  # model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
```

请注意损失函数 (loss function)的`reduction=tf.keras.losses.Reduction.NONE`参数 (parameter)。当使用`tf.distribute.Strategy`训练时，损失应在副本批处理内的*每个样本*上计算。然后，策略会正确处理这些损失在所有副本上的聚合（通常是将它们求和并除以*全局*批处理大小）。如果你使用了默认的`SUM_OVER_BATCH_SIZE`归约，策略的后续聚合将导致不正确的缩放。

### 5. 实现训练步骤

我们可以使用与`tf.distribute.Strategy`结合的标准`model.fit` API。或者，让我们演示在分布式环境中，使用`strategy.run`和`strategy.reduce`的自定义训练步骤是怎样的。

```python
# 计算每个副本损失的函数
def compute_loss(labels, predictions):
  per_example_loss = loss_object(labels, predictions)
  # 我们在副本内部计算局部批处理大小的平均损失。
  # 策略将在之后处理副本间的平均。
  # 或者，你可以在这里求和并在归约后除以全局批处理大小。
  return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

# 定义训练步骤函数
@tf.function
def distributed_train_step(inputs):
  features, labels = inputs

  def step_fn(features, labels):
    with tf.GradientTape() as tape:
      predictions = model(features, training=True)
      loss = compute_loss(labels, predictions)

    # 计算梯度
    gradients = tape.gradient(loss, model.trainable_variables)
    # 应用梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 更新指标
    train_accuracy.update_state(labels, predictions)
    return loss

  # 在每个副本上运行步骤函数
  per_replica_losses = strategy.run(step_fn, args=(features, labels))

  # 聚合副本间的（例如损失）结果
  # 使用带有SUM的归约并除以num_replicas，或者直接使用MEAN。
  mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
  return mean_loss # 返回全局批处理的平均损失
```

**`distributed_train_step`中的要点：**

- `@tf.function`装饰器将训练步骤编译成TensorFlow图以提升性能。
- `strategy.run(step_fn, ...)`在每个副本上执行`step_fn`，提供相应的数据切片。`step_fn`包含一个副本的核心前向传播、损失计算和梯度计算/应用。
- `compute_loss`计算副本处理的*局部*批处理的平均损失，但使用`tf.nn.compute_average_loss`针对*全局*批处理大小进行了适当缩放。
- `strategy.reduce(...)`聚合从每个副本返回的值（如损失）。在这里，我们对每个副本的损失求和并返回总和，这代表了全局批处理的平均损失，因为`compute_loss`已包含全局批处理大小的缩放。

### 6. 运行训练循环

现在，我们遍历轮次和步骤，调用我们的分布式训练步骤。

```python
print("开始分布式训练...")

for epoch in range(EPOCHS):
  total_loss = 0.0
  num_batches = 0
  # 在每个轮次开始时重置指标
  train_accuracy.reset_state()

  # 遍历分布式数据集
  for batch_inputs in dataset:
    # 运行分布式训练步骤
    batch_loss = distributed_train_step(batch_inputs)
    total_loss += batch_loss
    num_batches += 1
    # 可选：在轮次内打印进度
    # if num_batches % 50 == 0:
    #   print(f"  Epoch {epoch+1}, Batch {num_batches}, Loss: {batch_loss.numpy():.4f}, Accuracy: {train_accuracy.result().numpy():.4f}")

  # 计算该轮次的平均损失
  epoch_loss = total_loss / num_batches
  epoch_accuracy = train_accuracy.result()

  print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

print("分布式训练完成。")
```

### 使用`model.compile`和`model.fit`

或者，如果你更喜欢更高层的Keras API，你可以使用`model.compile`和`model.fit`。确保模型、优化器和指标在`strategy.scope()`内创建。当提供`tf.data.Dataset`时，`model.fit`将自动处理分布式。

```python
# 如果尚未完成，在范围内重新创建模型和优化器
with strategy.scope():
  model_fit = build_model()
  optimizer_fit = tf.keras.optimizers.Adam()
  # 除非有特殊需求，否则对编译/拟合使用标准归约
  loss_fit = tf.keras.losses.BinaryCrossentropy(from_logits=False)
  model_fit.compile(optimizer=optimizer_fit, loss=loss_fit, metrics=['accuracy'])

print("\n开始使用model.fit进行分布式训练...")

# model.fit自动处理数据集分布式
history = model_fit.fit(dataset, epochs=EPOCHS, verbose=1)

print("使用model.fit的分布式训练完成。")
print("历史记录:", history.history)
```

使用`model.fit`通常更简单，但理解使用`strategy.run`的自定义循环能让你更深入地理解分布式机制如何运作。

### 多工作器训练（`MultiWorkerMirroredStrategy`）的考量

尽管`MirroredStrategy`处理*一台*机器上的多个设备，但`MultiWorkerMirroredStrategy`扩展到*多台*机器（工作器）。其设置包括：

1. **`TF_CONFIG` 环境变量：** 每个工作器都需要设置一个`TF_CONFIG`环境变量（通常作为JSON字符串）。此变量告知工作器整个集群的设置情况（所有工作器的地址、当前工作器的类型和索引）。

   ```json
   // 工作器0（机器A上）的TF_CONFIG示例
   {
       "cluster": {
           "worker": ["machine-a.example.com:20000", "machine-b.example.com:20000"]
       },
       "task": {"type": "worker", "index": 0}
   }

   // 工作器1（机器B上）的TF_CONFIG示例
   {
       "cluster": {
           "worker": ["machine-a.example.com:20000", "machine-b.example.com:20000"]
       },
       "task": {"type": "worker", "index": 1}
   }
   ```
2. **策略实例化：** 你实例化`tf.distribute.MultiWorkerMirroredStrategy()`而不是`MirroredStrategy`。
3. **代码执行：** 你在所有工作机器上运行*相同*的Python脚本。TensorFlow使用`TF_CONFIG`进行协调。
4. **数据集分片：** 确保你的`tf.data`管道在工作器间正确分片数据。通常，这涉及在数据集选项中设置`tf.data.experimental.AutoShardPolicy.DATA`或`tf.data.experimental.AutoShardPolicy.FILE`，或者根据`TF_CONFIG`手动分片。
5. **检查点：** 检查点（例如，使用`tf.train.CheckpointManager`）对于多工作器设置的容错能力很重要。检查点通常应保存到所有工作器可访问的共享文件系统。

以下是说明`MultiWorkerMirroredStrategy`的图表：

> `MultiWorkerMirroredStrategy`设置涉及多台机器，每台机器可能有多块GPU。通信和梯度同步根据在每个工作器上设置的`TF_CONFIG`变量进行管理。数据集在工作器间分片。

设置`MultiWorkerMirroredStrategy`除了Python代码本身之外，还需要基础设施管理（网络、环境变量），这就是我们将动手部分集中在`MirroredStrategy`上的原因。然而，核心TensorFlow代码结构（使用`strategy.scope()`，调整数据管道）保持相似。

这个实践练习演示了使用`tf.distribute.Strategy`实现分布式训练的基本步骤。通过在策略范围内定义你的模型、优化器和训练步骤，并确保你的数据管道得到正确处理，你可以使用多个设备来大幅加速你的训练任务。

## 参考资料

- [Distributed training with TensorFlow](https://www.tensorflow.org/guide/distributed_training) — TensorFlow Authors (2023)
  涵盖了多种 TensorFlow 分布式训练策略、配置及其有效使用方法的官方指南。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125964/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供了实用的示例和详细说明，用于实现机器学习概念，包括 TensorFlow 中的分布式训练策略（第3版）。
- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Authors (2023)
  官方指南，介绍如何使用 `tf.data` API 优化数据管道，以实现高效的输入处理，适用于分布式设置。

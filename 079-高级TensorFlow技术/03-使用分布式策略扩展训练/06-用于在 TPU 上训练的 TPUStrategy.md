# 用于在 TPU 上训练的 TPUStrategy

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-3-distributed-training-strategies/tpu-strategy)

[返回章节目录](README.md) · [返回课程目录](../README.md)

对于大规模机器学习 (machine learning)任务，在硬件加速器之间分配工作负载是常见的做法。虽然 GPU 通常与各种分布式训练策略配合使用，但 Google 的张量处理单元（TPU）提供了专用的硬件加速，明确为这些高要求计算而设计。TPU 在密集矩阵乘法方面表现出特别的效率，并具有高带宽内存（HBM），这使它们非常适合训练深度神经网络 (neural network)，特别是大型语言模型和视觉转换器。

要在 TensorFlow 中使用 TPU 的功能，您需要使用 `tf.distribute.TPUStrategy`。此策略抽象化了在 TPU 设备上多个核心之间，甚至在构成“TPU Pod”的多个 TPU 设备之间进行计算通信和协调的复杂性。

### 了解 TPUs 和 `TPUStrategy`

通常来说，一个 TPU 设备包含多个 TPU 核心（Google Cloud 中现代 TPU 通常有 8 个）。`TPUStrategy` 实现同步数据并行，类似于 `MirroredStrategy`，但针对 TPU 架构进行了优化。当您使用 `TPUStrategy` 时：

1. **模型复制：** 您的模型会在每个可用的 TPU 核心上复制一份。
2. **数据分片：** 每批输入数据都会在副本之间进行分割（分片）。每个 TPU 核心处理其独有的数据分片。
3. **梯度聚合：** 在正向和反向传播 (backpropagation)后，每个核心上计算的梯度会通过 TPU 核心之间专用高速互连机制高效聚合。
4. **权重 (weight)更新：** 聚合后的梯度用于更新模型权重，并且这些更新会在下一个训练步骤之前同步到所有副本。

这种同步方法确保了训练期间所有核心之间模型的一致性。

> 使用 `TPUStrategy` 的数据和梯度流向。主机 CPU 负责协调，将数据分片分发到 TPU 核心并聚合梯度。

### 连接到 TPU

在使用 `TPUStrategy` 之前，您的 TensorFlow 程序需要定位并连接到可用的 TPU 资源。这通常通过 `tf.distribute.cluster_resolver.TPUClusterResolver` 来完成。这个实用工具会自动检测 Google Colab、Kaggle Notebooks 或 Google Cloud AI Platform Notebooks 等环境中的 TPU 配置。

```python
import tensorflow as tf
import os

try:
    # 尝试检测并初始化 TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f'在 TPU {tpu.master()} 上运行')
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("TPU 策略已初始化。")
    print(f"加速器数量: {strategy.num_replicas_in_sync}")

except ValueError:
    # 如果未检测到 TPU，则回退到默认策略（CPU 或单 GPU）
    print("未找到 TPU。使用默认策略。")
    strategy = tf.distribute.get_strategy()
```

这段代码首先尝试解析 TPU 集群。如果成功，它会连接到集群，初始化 TPU 系统，并创建一个 `TPUStrategy` 实例。如果未找到 TPU（例如，在本地运行但无权访问 TPU），它会优雅地回退到默认策略。`strategy.num_replicas_in_sync` 属性会显示有多少个 TPU 核心可用于同步训练。

### 在策略范围内训练

与其他分发策略类似，您的训练设置的核心组件，特别是模型创建和优化器实例化，必须在 `strategy.scope()` 中进行：

```python
# 定义一个构建模型的函数（示例）
def build_model():
    # 使用标准 Keras API
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# 定义一个创建数据集的函数（示例）
def create_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = tf.reshape(tf.cast(x_train, tf.float32) / 255.0, (-1, 784))
    y_train = tf.one_hot(y_train, 10)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # 重要：在分发 *之前* 进行混洗、重复和批处理
    dataset = dataset.shuffle(60000).repeat().batch(batch_size)
    # 预取以提升性能
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# 确定全局批次大小
# TPU 在大批次大小下表现最佳，通常是每个核心 128 的倍数。
PER_REPLICA_BATCH_SIZE = 128
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync
print(f"全局批次大小: {GLOBAL_BATCH_SIZE}")

# 创建数据集
train_dataset = create_dataset(GLOBAL_BATCH_SIZE)

# --- 策略范围内的操作 ---
with strategy.scope():
    # 模型构建
    model = build_model()

    # 优化器实例化
    optimizer = tf.keras.optimizers.Adam()

    # 损失函数和指标
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    # 模型编译（可选但 Keras 中常见）
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[train_accuracy])
# --- 策略范围结束 ---

# 标准 Keras model.fit 可与策略配合使用
EPOCHS = 5
STEPS_PER_EPOCH = 60000 // GLOBAL_BATCH_SIZE # 示例计算

print("开始训练...")
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH)
print("训练完成。")
```

请注意，用于构建、编译和拟合模型的核心 Keras 代码基本保持不变。当调用 `model.fit` 时，`TPUStrategy` 会处理底层的分发逻辑。

### TPU 训练的注意事项

尽管 `TPUStrategy` 简化了分发，但要达到最佳性能通常需要关注 TPU 特定的细节：

1. **高效的 `tf.data` 数据流：** TPU 速度极快。您的输入数据流（`tf.data.Dataset`）必须高度优化，以持续为 TPU 核心提供数据。使用 `dataset.cache()`、`dataset.prefetch(tf.data.AUTOTUNE)`、并行映射操作（`num_parallel_calls=tf.data.AUTOTUNE`），并确保在分发 *之前* 正确进行批处理。输入瓶颈是 TPU 上常见的性能问题。
2. **批次大小：** TPU 在大批次大小下能达到最佳性能。`GLOBAL_BATCH_SIZE` 通常应是 `128 * strategy.num_replicas_in_sync` 的倍数。通常需要通过实验来确定您的特定模型和 TPU 配置的最佳大小。
3. **张量形状：** TPU 倾向于使用静态张量形状。尽管对动态形状的支持正在改进，但在图构建过程中使用固定输入大小并避免产生未知维度的操作，可以提升性能和兼容性。
4. **支持的操作：** 并非所有 TensorFlow 操作都有优化的 TPU 实现。虽然覆盖范围很广，但复杂的自定义操作或不常见的操作可能需要修改或回退到 CPU 执行，这可能减慢训练速度。查阅 TensorFlow 官方文档以获取 TPU 支持的操作列表。
5. **BFloat16 格式：** TPU 对 `bfloat16` 数字格式具有原生硬件支持。这种格式提供与 `float32` 相似的动态范围，但内存占用减半，通常可以加速计算并减少内存使用，而无需进行损失缩放（`float16` 混合精度中通常需要）。您通常可以通过 Keras 策略轻松启用 `bfloat16` 计算：`tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')`。

### 调试和性能分析

调试 TPU 上的分布式训练可能比在单个设备上更复杂。

- **TensorBoard 性能分析器：** TensorBoard 性能分析器是一个重要工具。它提供关于 TPU 利用率、TPU 上不同操作的执行时间、潜在的输入数据流瓶颈以及通信开销的详细信息。分析性能分析器的输出对于优化 TPU 训练性能非常重要。
- **日志记录：** 使用标准的日志记录实践，但请记住日志可能来自不同的 TPU 工作器或主机 CPU。整合日志会有帮助。
- **从简单开始：** 先使用较小的模型或更简单的设置来验证 TPU 连接和基本训练循环功能，然后再扩展到复杂模型。

`TPUStrategy` 为使用 Google 专用 TPU 硬件提供了强大的抽象。通过了解其工作原理并关注输入数据流效率、批次大小和支持的操作，您可以显著加速大型复杂 TensorFlow 模型的训练。

## 参考资料

- [tf.distribute.TPUStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/TPUStrategy) — TensorFlow Authors (2024)
  Publisher: TensorFlow
  `TPUStrategy`类的官方API文档，解释了其在TPU上进行分布式训练的用法和参数。
- [Use TPUs](https://www.tensorflow.org/guide/tpu) — TensorFlow Authors (2023)
  Publisher: TensorFlow
  Google TPU与TensorFlow结合使用的综合指南，涵盖设置、训练实践和性能优化。
- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://ieeexplore.ieee.org/document/8000735) — Norman P. Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, Rick Boyle, Pierre-luc Cantin, Clifford Chao, Chris Clark, Jeremy Coriell, Mike Daley, Matt Dau, Jeffrey Dean, Ben Gelb, Tara Vazir Ghaemmaghami, Rajendra Gottipati, William Gulland, Robert Shute, Eric Han (2017)
  Journal: Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA); Publisher: IEEE; Pages: 1-12; DOI: [10.1109/ISCA.2017.20](https://doi.org/10.1109/ISCA.2017.20)
  介绍Google第一代张量处理单元（TPU）架构和性能特点的奠基性研究论文。
- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Authors (2024)
  Publisher: TensorFlow
  关于使用`tf.data` API优化输入管道以避免数据瓶颈的官方指南，这对于高效的TPU训练至关重要。
- [Faster training with mixed precision](https://www.tensorflow.org/guide/mixed_precision) — TensorFlow Authors (2024)
  Publisher: TensorFlow
  解释如何实现混合精度训练的官方指南，包括使用`bfloat16`以提高TPU上的训练速度和内存效率。

---

[上一节](05-ParameterServerStrategy%20%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3.md) · [下一节](07-%E6%9C%89%E6%95%88%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C.md)

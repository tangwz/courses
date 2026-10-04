---
course: "advanced-tensorflow"
chapter: "distributed-training-strategies"
lesson: "mirrored-strategy"
sourceId: 2847
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-3-distributed-training-strategies/mirrored-strategy"
title: "用于单机多 GPU 训练的 MirroredStrategy"
description: "使用 MirroredStrategy 在单机多 GPU 上实现同步数据并行。"
order: 3
plots: []
sourceHash: "7dcf2bb17d888facf4ba53bd42df6546d57bc686037cfd12542ecdfb6c8c88cd"
sourceCorrections: []
---

当您的模型或数据集所需的资源超出单机上单个 GPU 的有效供给能力，但又未达到需要多台机器的规模时，`tf.distribute.MirroredStrategy` 提供了一种直接的方法来使用您的单主机上所有可用的 GPU。这是一种同步数据并行策略，它将您的整个模型复制到每个可用 GPU 上，在每个副本上处理输入数据的不同部分，并确保所有模型副本保持同步。

### MirroredStrategy 的运作方式

`MirroredStrategy` 的核心思路很简单：复制模型、分割任务并同步更新。以下是该流程的分解：

1. **模型复制：** 当您在 `MirroredStrategy` 作用域内定义模型时，TensorFlow 会在每个指定的 GPU（默认是所有可见 GPU）上创建模型变量的完整副本。
2. **数据分片：** 每一批输入的训练数据都会在可用 GPU 副本之间平均分配。例如，如果您有 4 个 GPU 和一个全局批大小为 1024，每个 GPU 副本将处理 256 个样本的独特切片（$1024 / 4 = 256$）。这就是“数据并行”的体现。
3. **独立的正向和反向传播 (backpropagation)：** 每个副本都在其分配的数据切片上独立执行正向传播（计算预测和损失）和反向传播（计算梯度）。
4. **梯度同步（All-Reduce）：** 这是同步步骤。在优化器更新模型变量之前，每个副本上计算的梯度需要合并。`MirroredStrategy` 通常使用高效的 “all-reduce” 算法（通常使用 NVIDIA 的 NCCL 库实现 GPU 间通信）来汇总所有副本的梯度。
5. **同步变量更新：** 一旦聚合梯度在所有副本上可用，每个副本都会使用优化器对其本地模型变量副本应用完全相同的更新。由于每个副本都以相同的初始权重 (weight)开始并在每个步骤中接收相同的梯度更新，模型变量在整个训练过程中保持所有 GPU 上的同步（“镜像”）。

> 数据在 GPU 间分割。每个 GPU 独立计算梯度。梯度通过 all-reduce 操作聚合，然后同步更新应用于所有模型副本。

这种同步特性确保了一致性，但在 all-reduce 步骤中引入了通信开销。效率提升取决于并行处理节省的计算时间是否超过了通信成本。

### 实现 MirroredStrategy

将 `MirroredStrategy` 集成到您的 TensorFlow 代码中，尤其是使用 Keras 时，只需极少的改动。主要机制是 `strategy.scope()` 上下文 (context)管理器。

1. **实例化策略：** 创建 `MirroredStrategy` 的实例。您可以选择指定要使用的设备，但默认情况下，它使用 TensorFlow 可见的所有 GPU。

   ```python
   import tensorflow as tf

   # 检查可用 GPU
   gpus = tf.config.list_physical_devices('GPU')
   print(f"可用 GPU 数量: {len(gpus)}")

   # 实例化策略（默认使用所有可用 GPU）
   strategy = tf.distribute.MirroredStrategy()
   print(f"策略中的设备数量: {strategy.num_replicas_in_sync}")
   ```
2. **使用策略作用域：** 最重要的改动是将模型构建、优化器实例化和 `model.compile()` 调用放在 `strategy.scope()` *内部*。这会告诉 TensorFlow 根据策略的规则管理变量和操作（即，在副本间镜像变量）。

   ```python
   # 全局定义批大小
   # 每步处理的有效批大小是 global_batch_size
   # 每个副本处理 global_batch_size // num_replicas_in_sync
   BUFFER_SIZE = 10000
   GLOBAL_BATCH_SIZE = 64 * strategy.num_replicas_in_sync # 示例：按 GPU 数量缩放批大小
   PER_REPLICA_BATCH_SIZE = GLOBAL_BATCH_SIZE // strategy.num_replicas_in_sync

   # 创建数据集（使用 MNIST 的示例）
   (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
   x_train = x_train[..., tf.newaxis] / 255.0

   train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
   train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)

   # 重要：在策略作用域内构建模型、优化器并编译
   with strategy.scope():
       # 模型定义（简单的 CNN 示例）
       model = tf.keras.Sequential([
           tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
           tf.keras.layers.MaxPooling2D(),
           tf.keras.layers.Flatten(),
           tf.keras.layers.Dense(128, activation='relu'),
           tf.keras.layers.Dense(10, activation='softmax')
       ])

       # 优化器定义
       optimizer = tf.keras.optimizers.Adam()

       # 编译模型
       model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     optimizer=optimizer,
                     metrics=['accuracy'])

   # 训练模型 - Keras 自动处理分布式
   print("开始使用 MirroredStrategy 进行训练...")
   model.fit(train_dataset, epochs=5)
   print("训练完成。")
   ```

请注意，训练循环（`model.fit`）本身不需要在作用域内。如果模型是在策略作用域内编译的，Keras 的 `fit` 方法会策略感知并自动处理分布式细节。

### 数据处理

当将 `tf.data.Dataset` 与 `MirroredStrategy` 配合使用时，TensorFlow 会自动处理批次在副本间的分割。您可以像往常一样定义数据集管道，指定 *全局* 批大小（所有 GPU 每步处理的样本总数）。TensorFlow 会将此全局批次除以副本数量（`strategy.num_replicas_in_sync`），以确定每个副本的批大小。

通常建议将您的全局批大小与所使用的 GPU 数量成比例缩放，以保持每个 GPU 的工作负载一致，并可能提高训练速度和稳定性，尽管最佳批大小通常需要实验。

### 实际考量

- **通信开销：** all-reduce 步骤需要 GPU 之间进行通信。在具有快速互连（如 NVLink）的系统中，此开销相较于计算时间通常很小。在连接较慢（例如，标准 PCIe）的系统中，通信可能成为瓶颈，限制您获得的加速。性能分析（第 2 章介绍）对于识别此类瓶颈非常重要。
- **变量放置：** `MirroredStrategy` 将变量直接放置在每个 GPU 副本上。请确保您的模型适合单个 GPU 的内存。
- **同步性：** 虽然简化了一致性，但同步更新意味着整个步骤的持续时间由最慢的副本决定。
- **应用场景：** 当模型适合单个 GPU 的内存时，`MirroredStrategy` 非常适合加速在配备多个 GPU 的单机上进行训练。相较于多 worker 策略，它为计算密集型工作负载提供了显著的加速，且实现复杂度相对较低。

通过了解 `MirroredStrategy` 如何复制模型、分发数据和同步梯度，您可以有效地使用单机上的多个 GPU 来更快地训练您的 TensorFlow 模型。记住要在 `strategy.scope()` 内定义模型和优化器，并适当调整全局批大小。

## 参考资料

- [Train a Keras model with tf.distribute.MirroredStrategy](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy) — TensorFlow Authors (2024)
  官方TensorFlow指南，提供了使用MirroredStrategy与Keras的实际实现细节和代码示例。
- [Distributed training with TensorFlow](https://www.tensorflow.org/guide/distributed_training) — TensorFlow Authors (2024)
  TensorFlow分布式训练策略的全面概述，包括MirroredStrategy的背景和替代方案。
- [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  TensorFlow的MirroredStrategy常用于高效GPU间梯度同步（all-reduce）的基础库文档。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 19
  这本广泛使用的书籍包含关于TensorFlow分布式训练的章节，为MirroredStrategy等策略提供了更广阔的背景和示例。

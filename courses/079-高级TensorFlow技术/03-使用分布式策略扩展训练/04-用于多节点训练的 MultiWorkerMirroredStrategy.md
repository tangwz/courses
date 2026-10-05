# 用于多节点训练的 MultiWorkerMirroredStrategy

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-3-distributed-training-strategies/multi-worker-mirrored-strategy)

[返回章节目录](README.md) · [返回课程目录](../README.md)

TensorFlow 的 `MirroredStrategy` 能够有效利用单机上的多个 GPU。然而，某些机器学习 (machine learning)任务需要更高的计算能力，或者数据集太大，无法在单个节点的内存系统中轻松容纳。为了将训练扩展到多台机器，TensorFlow 提供了 `tf.distribute.MultiWorkerMirroredStrategy`。

此策略在多台机器上实施同步数据并行，这些机器通常被称为“工作器”。它与 `MirroredStrategy` 类似：每个工作器获得模型的完整副本，处理输入数据的独立部分，在本地计算梯度，然后参与一个集合操作，以同步所有工作器上的这些梯度，之后再更新模型变量。主要区别在于，通信和同步现在通过连接不同机器的网络进行。

### 多工作器镜像策略如何运作

1. **初始化：** 当策略初始化时，它需要集群设置信息——哪些机器参与以及每台机器扮演的角色。这通常使用 `TF_CONFIG` 环境变量进行配置。
2. **模型复制：** 就像 `MirroredStrategy` 一样，当在策略范围内定义时，模型的变量会在所有参与工作器上的所有 GPU 之间创建和镜像。
3. **数据分片：** 输入数据集 (`tf.data.Dataset`) 会自动分片，通常根据工作器数量和每个工作器的 GPU 数量。每个 GPU 处理全局批次的不同部分。`tf.data.experimental.AutoShardPolicy` 常用以正确处理此分发。
4. **前向传播：** 每个工作器上的每个 GPU 在其分配的数据切片上执行前向传播。
5. **梯度计算：** 每个 GPU 根据其本地数据切片计算损失相对于模型变量的梯度。
6. **全约化同步：** 这是核心通信步骤。所有工作器上所有 GPU 计算的梯度会使用高效的全约化算法（例如 NVIDIA GPU 的 NCCL）进行聚合。此操作汇总所有副本的梯度，并使结果对所有副本可用。每个副本随后拥有相同的平均梯度。工作器之间的网络带宽和延迟在此成为重要因素。
7. **变量更新：** 每个副本使用同步后的梯度以相同方式更新其模型的本地变量副本。因为在全约化步骤后梯度在各处都相同，变量在所有工作器和 GPU 之间保持同步。

> 包含两个工作器（每个工作器有两个 GPU）的 `MultiWorkerMirroredStrategy` 概述。数据分片，梯度在本地计算，通过网络进行全约化同步，并用于以相同方式更新模型副本。

### 设置集群：`TF_CONFIG` 变量

为了让工作器相互发现和协调，TensorFlow 依赖于通过 `TF_CONFIG` 环境变量指定的集群配置。参与训练任务的*每台*工作器机器上都必须设置此变量。它是一个包含两个主要部分的 JSON 字符串：

1. **`cluster`**：定义所有参与工作器的网络地址（主机名/IP 和 端口），并为其分配角色（通常只是 `worker`）。
2. **`task`**：指定*当前*工作器进程在集群定义中的角色 (`type`) 和索引 (`index`)。

这是一个包含两个工作器设置的 `TF_CONFIG` 示例：

**在工作器 0 上：**

```json
export TF_CONFIG='{
    "cluster": {
        "worker": ["worker0.example.com:2222", "worker1.example.com:2222"]
    },
    "task": {"type": "worker", "index": 0}
}'
```

**在工作器 1 上：**

```json
export TF_CONFIG='{
    "cluster": {
        "worker": ["worker0.example.com:2222", "worker1.example.com:2222"]
    },
    "task": {"type": "worker", "index": 1}
}'
```

- `"cluster"` 字典在 `"worker"` 键下列出了所有工作器。主机名 (`worker0.example.com`、`worker1.example.com`) 和端口 (`2222`) 必须在机器之间可访问。
- `"task"` 字典告知每个进程其在此集群中的特定身份。工作器 0 的 `index` 为 `0`，工作器 1 的 `index` 为 `1`。

正确设置 `TF_CONFIG` 对于多工作器训练非常重要。编排系统，如 Kubernetes（常与 Kubeflow 搭配使用），通常会自动将适当的 `TF_CONFIG` 注入每个工作器容器。如果手动运行，必须确保在Python脚本启动前设置此变量。

### 在代码中实现多工作器训练

将 `MultiWorkerMirroredStrategy` 集成到 Keras 代码中与使用 `MirroredStrategy` 非常相似。主要步骤是：

1. **确保 `TF_CONFIG` 已设置：** 这在 Python 代码之外发生，在脚本运行的环境中。
2. **实例化策略：** 创建 `tf.distribute.MultiWorkerMirroredStrategy()` 的实例。TensorFlow 会自动解析 `TF_CONFIG` 环境变量。
3. **在策略范围内定义模型和优化器：** 将模型创建、编译和优化器实例化包装在 `with strategy.scope():` 中。这确保变量以分布式方式创建。
4. **准备数据集：** 使用 `tf.data.Dataset` 进行输入管道。策略通常与自动分片策略配合得最好。确保数据集加载逻辑高效，因为它在分布式环境中可能成为瓶颈。
5. **调用 `model.fit`：** 使用标准的 Keras `model.fit` API。策略在后台处理梯度聚合和变量更新。

```python
import tensorflow as tf
import os
import json

# 假设 TF_CONFIG 已在环境中设置
# 示例：对于工作器 0
# os.environ['TF_CONFIG'] = json.dumps({
#     'cluster': {
#         'worker': ['host1:port', 'host2:port']
#     },
#     'task': {'type': 'worker', 'index': 0}
# })

# 1. 实例化策略
# 可以指定通信选项，例如 GPU 的 NCCL
# strategy = tf.distribute.MultiWorkerMirroredStrategy(
#     communication_options=tf.distribute.experimental.CommunicationOptions(
#         implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
#     )
# )
strategy = tf.distribute.MultiWorkerMirroredStrategy()

print(f"设备数量: {strategy.num_replicas_in_sync}")

# 准备分布式数据集
BUFFER_SIZE = 10000
GLOBAL_BATCH_SIZE = 64 * strategy.num_replicas_in_sync # Scale batch size
# 示例：创建一个模拟数据集
features = tf.random.uniform((1000, 10))
labels = tf.random.uniform((1000, 1))
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)

# 定义数据集分发选项
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)

# 2. 在策略范围内定义模型和优化器
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print("模型和优化器已在策略范围内创建。")

# 3. 使用 model.fit 训练模型
# 策略自动处理分发
print("开始训练...")
history = model.fit(dataset, epochs=5, verbose=2) # Verbose=2 通常更适合多工作器
print("训练完成。")

# 保存模型通常只需要在一个工作器（主工作器）上保存
# 或使用特定的保存选项。详细信息请参阅 TensorFlow 文档。
# 示例：仅在工作器 0 上保存
# task_type = os.environ.get('TF_CONFIG')
# if task_type:
#    tf_config = json.loads(task_type)
#    if tf_config['task']['type'] == 'worker' and tf_config['task']['index'] == 0:
#        model.save('my_multi_worker_model.keras')
# else: # 单工作器情况
#    model.save('my_single_worker_model.keras')
```

### 多工作器训练的注意事项

- **网络带宽/延迟：** 全约化性能严重依赖连接工作器的网络。慢速或高延迟网络可能成为一个重要的瓶颈，可能抵消增加更多工作器带来的好处。使用高性能互连设备（如 InfiniBand）和优化的通信库（如 NVIDIA GPU 的 NCCL）很重要。
- **全局批次大小：** 由于每个工作器处理批次的一部分，*全局*批次大小（所有副本批次大小之和）与副本数量 (`num_replicas_in_sync`) 成比例增加。可能需要调整学习率或其他超参数 (parameter) (hyperparameter)以适应这个更大的有效批次大小。常见做法是线性缩放学习率，尽管这并非普遍适用。
- **数据分片：** 确保 `tf.data` 管道正确地在工作器之间分片数据。通常建议使用 `AutoShardPolicy.DATA` 或 `AutoShardPolicy.FILE`（如果从多个文件读取）。不正确的分片可能导致工作器处理重叠数据或某些工作器处于空闲状态。
- **容错性：** 同步训练意味着如果一个工作器失败，整个任务通常会停止。生产环境通常需要容错机制，例如使用 `tf.keras.callbacks.BackupAndRestore` 或实现带有检查点策略的自定义训练循环，以处理工作器重启。
- **保存检查点/模型：** 在多工作器设置中，通常只有一个工作器（通常指定为“主”工作器，通常是工作器索引0）应负责保存检查点或最终模型等任务，以避免冲突和重复写入。代码通常包含基于 `TF_CONFIG` 中工作器任务索引的检查。

`MultiWorkerMirroredStrategy` 是一个强大的工具，用于在多台机器上扩展同步训练。它的设置需要仔细配置 `TF_CONFIG` 环境变量并考虑网络性能，但它允许使用熟悉的 Keras API 为要求高的训练任务使用更多的计算资源。

## 参考资料

- [tf.distribute.MultiWorkerMirroredStrategy Class](https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy) — TensorFlow Team (2024)
  官方文档，提供了 TensorFlow 多工作器分布式训练策略的详细 API 规范和使用示例。
- [Distributed training with TensorFlow](https://www.tensorflow.org/guide/distributed_training) — TensorFlow Team (2024)
  官方指南，提供了 TensorFlow 中分布式训练策略的广泛概述，包括多工作器设置的概念性解释和实际示例。
- [Distributed Deep Learning: A Guide to Scalable Training](https://www.oreilly.com/library/view/distributed-deep-learning/9781492067451/) — Peter Mark, Dinesh Suresh (2020)
  Publisher: O'Reilly Media
  一本全面的指南，涵盖了分布式深度学习的原理和实践，包括数据并行、同步训练和架构考虑。
- [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl) — NVIDIA (2024)
  Publisher: NVIDIA
  官方资源，详细介绍了 NVIDIA 用于 GPU 间和节点间通信的库，这对于分布式训练中高效的 All-Reduce 操作至关重要。

---

[上一节](03-%E7%94%A8%E4%BA%8E%E5%8D%95%E6%9C%BA%E5%A4%9A%20GPU%20%E8%AE%AD%E7%BB%83%E7%9A%84%20MirroredStrategy.md) · [下一节](05-ParameterServerStrategy%20%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3.md)

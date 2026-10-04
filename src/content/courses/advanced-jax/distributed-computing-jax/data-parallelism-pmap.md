---
course: "advanced-jax"
chapter: "distributed-computing-jax"
lesson: "data-parallelism-pmap"
sourceId: 2846
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-3-distributed-computing-jax/data-parallelism-pmap"
title: "使用 pmap 实现数据并行"
description: "使用 `pmap` 在设备间分发数据和复制模型，以进行并行训练。"
order: 4
plots: []
sourceHash: "63b2c02138812255c6599cd937c89afe8d78ebffcee6d41d9a001a4bc33bfbd2"
sourceCorrections: []
---

数据并行是一种常见且有效的策略，能加快机器学习 (machine learning)任务，尤其是模型训练的速度。其基本思路简单明了：您将模型复制到多个计算设备（如 GPU 或 TPU）上，并为每个副本提供输入数据批次的不同切片或分片。每个设备使用相同的模型参数 (parameter)独立处理其数据分片。`pmap` 是 JAX 实现这种 SPMD（单程序多数据）模式的主要方式。

请记住，`pmap` 会将一个函数映射到数组上，这些数组的主轴与所用的设备数量一致。当用于数据并行时，这意味着：

1. **模型复制：** 您 `pmap` 的函数（通常是模型的前向传播或整个训练步骤）会在所有参与的设备上被隐式复制。如果表示模型参数的函数参数 *没有* 映射的主轴，JAX 会自动对其进行广播，从而让每个设备获得相同的副本。
2. **数据分片：** 具有与设备数量匹配的主轴的输入数组会沿该轴自动拆分。第一个切片送往第一个设备，第二个切片送往第二个设备，依此类推。这就是数据批次的分发方式。

我们通过一个例子来说明这一点。假设我们有一个简单的预测函数，并希望使用数据并行在可用设备上并行运行它。

首先，我们需要做一些准备工作：识别设备并定义一个函数。

```python
import jax
import jax.numpy as jnp

# 获取可用设备数量
num_devices = jax.local_device_count()
print(f"Number of devices: {num_devices}")

# 示例函数（例如，一个简化的模型层）
def predict(params, inputs):
  # 一个简单的线性变换
  return jnp.dot(inputs, params['w']) + params['b']

# 生成模拟参数（权重和偏置）
# 这些参数将在设备间复制
key = jax.random.PRNGKey(0)
input_dim = 10
output_dim = 5
params = {
    'w': jax.random.normal(key, (input_dim, output_dim)),
    'b': jax.random.normal(key, (output_dim,))
}

# 生成全局数据批次
global_batch_size = 32 * num_devices # 示例总批次大小
dummy_data = jax.random.normal(key, (global_batch_size, input_dim))

print(f"Parameter shapes: w={params['w'].shape}, b={params['b'].shape}")
print(f"Global data batch shape: {dummy_data.shape}")
```

现在，数据并行的重要步骤是准备输入数据。`pmap` 要求输入数据数组具有与设备数量（`num_devices`）相等的主维度。我们需要相应地重塑 `dummy_data`。

```python
# 为 pmap 重塑数据：[设备数量, 每个设备的批次大小, 特征数]
batch_per_device = global_batch_size // num_devices
sharded_data = dummy_data.reshape((num_devices, batch_per_device, input_dim))

print(f"Sharded data shape: {sharded_data.shape}")
# 预期形状：(设备数量, 每个设备的批次大小, 输入维度)
```

数据正确分片后，我们现在可以应用 `pmap`。请注意，`params` 是直接传递的。由于它没有与 `num_devices` 匹配的主维度，JAX 会知道应该将其广播（复制）到每个设备。然而，`sharded_data` 具有正确的主维度，因此它将被拆分。

```python
# 将 pmap 应用于预测函数
# params 被广播，sharded_data 沿第一个轴拆分
parallel_predict = jax.pmap(predict, in_axes=(None, 0))

# 运行并行计算
# 我们不需要显式复制 params，pmap 会处理广播
sharded_predictions = parallel_predict(params, sharded_data)

# 确保计算完成，然后检查形状
sharded_predictions.block_until_ready()

print(f"Output predictions shape: {sharded_predictions.shape}")
# 预期形状：(设备数量, 每个设备的批次大小, 输出维度)
```

`in_axes` 参数规定了 `pmap` 应该如何处理每个输入参数：

- `None`：广播此参数。相同的值被发送到所有设备。这对于模型参数来说很常见。
- `0`：映射此参数的第一个轴（轴 0）。这意味着数组沿轴 0 拆分，每个切片被发送到不同的设备。这在数据并行中是输入数据的常规做法。

输出 `sharded_predictions` 也具有与设备数量一致的主维度。每个切片 `sharded_predictions[i]` 包含在设备 `i` 上使用其输入数据的一部分（`sharded_data[i]`）和复制的 `params` 计算得到的结果。

> `pmap` 用于数据并行时的数据流程。全局批次在设备间被拆分（分片）。模型参数通常会被复制（广播）到每个设备。每个设备独立计算其结果。输出沿设备轴堆叠。

这个例子说明了使用 `pmap` 对分片数据并行应用函数的核心机制。在典型的训练场景中，`predict` 函数将是一个更大的 `train_step` 函数的一部分，该函数还会计算损失和梯度。尽管每个设备上的前向传播和损失计算是独立的，但每个设备上计算出的梯度通常需要在更新模型参数之前在所有设备之间合并（例如，平均）。这个重要的聚合步骤需要集合通信原语，我们将在下一节中进行了解。

## 参考资料

- [SPMD on Multiple Devices with \`pmap\`](https://jax.readthedocs.io/en/latest/jax-101/06-using_pmap.html) — Vladimir Mikulik, Roman Ring (2024)
  解释了 `pmap` 如何用于多设备上的单程序多数据（SPMD）编程，包括其 `in_axes` 参数以及数据和参数的分布处理。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面解释了深度学习模型的数据并行和其他分布式训练策略，提供了基础背景知识。
- [Distributed Machine Learning Patterns](https://blackwells.co.uk/bookshop/product/Distributed-Machine-Learning-Patterns-by-Yuan-Tang/9781617299025) — Yuan Tang (2024)
  Publisher: Manning Publications
  涵盖了分布式机器学习的各种模式，包括数据并行、模型并行和通信策略，为扩展机器学习工作负载提供了实践见解。

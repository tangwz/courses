# 将 pmap 与训练框架结合使用

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-6-large-scale-model-training-jax/combining-pmap-training-frameworks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 JAX 提供了功能强大的 `pmap` 变换，用于在多个设备上分发计算，但构建复杂的模型及其训练循环需要良好的组织性。Flax 或 Haiku 等高级神经网络 (neural network)库提供了抽象功能，用于定义模型、管理参数 (parameter)和处理状态，这大大简化了开发工作。将 `pmap` 与这些框架集成，可以使我们同时获得结构化模型构建和高效数据并行化的优点。

核心思想仍然是 `pmap` 实现的单程序多数据 (SPMD) 方法。我们编写函数（通常是训练步骤），就像它在单个设备上运行一样，但 `pmap` 会将其转换为在多个设备上并发运行，每个设备都在输入数据的不同切片上操作。这些框架有助于管理模型参数和优化器状态，这些在分布式设置中需要得到正确处理。

### 复制模型状态

当使用 `pmap` 进行数据并行时，模型本身通常会在所有参与设备上进行复制。每个设备都持有模型参数 (parameter)的完整副本。同样，优化器的状态（例如 Adam 中的动量缓冲区）也需要复制，以便每个设备可以根据其本地数据分片计算潜在的参数更新。

Flax 等框架通常在结构化容器（例如 Python 字典或专门的 dataclass，通常称为“训练状态”）中管理参数和状态。在 `pmap` 处理的训练循环开始之前，我们在主机 (CPU) 上初始化模型参数和优化器状态，然后明确地将此状态复制到所有可用设备上。JAX 提供了有助于实现这种复制的实用工具。

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax # Common optimizer library used with Flax

# 假设 'model' 是一个 Flax nn.Module 实例
# 假设 'optimizer' 是一个 Optax 优化器实例

# 初始化示例（在主机 CPU 上）
key = jax.random.PRNGKey(0)
dummy_input = jnp.ones([1, 28, 28, 1]) # 输入形状示例
params = model.init(key, dummy_input)['params']
tx = optax.adam(learning_rate=1e-3)
optimizer_state = tx.init(params)

# 创建一个 TrainState 对象（Flax 的常见模式）
# 它将参数、优化器状态和 apply_fn 捆绑在一起
class TrainState(train_state.TrainState):
  pass # 如果需要，可以添加自定义字段

state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# 在设备间复制状态
num_devices = jax.local_device_count()
replicated_state = jax.device_put_replicate(state, jax.local_devices())

print(f"状态已在 {num_devices} 个设备上复制。")
# 检查示例：单个设备上参数的形状
print(jax.tree_util.tree_map(lambda x: x.shape, replicated_state.params)['Dense_0']['kernel'])
# 检查示例：跨设备参数的形状（注意前导维度）
print(jax.tree_util.tree_map(lambda x: x.shape, state.params)['Dense_0']['kernel'])
```

请注意 `jax.device_put_replicate` 如何创建一个状态版本，其中每个叶节点（参数数组、优化器状态数组）都有一个等于设备数量的前导维度。

### 数据分片

对于数据并行，全局训练数据批次需要均匀分割到设备上。如果您的设备数量为 `N`，全局批次大小为 `B`，则每个设备将处理一个大小为 `B // N` 的本地批次。这种分片需要在数据传递给 `pmap` 处理的函数之前进行。

```python
global_batch_size = 64
local_batch_size = global_batch_size // num_devices

# 假设 'global_images' 和 'global_labels' 是 NumPy 数组
# 形状为 [global_batch_size, ...]

def shard_batch(batch):
  """重塑数据并在设备间分片。"""
  return jax.tree_util.tree_map(
      lambda x: x.reshape((num_devices, local_batch_size) + x.shape[1:]),
      batch
  )

# 数据示例（替换为实际数据加载）
global_images = jnp.ones([global_batch_size, 28, 28, 1])
global_labels = jnp.ones([global_batch_size], dtype=jnp.int32)
batch = {'image': global_images, 'label': global_labels}

sharded_batch = shard_batch(batch)

# 验证形状
print("全局图像形状:", global_images.shape)
print("分片图像形状:", sharded_batch['image'].shape)
# 输出应显示：分片图像形状: (num_devices, local_batch_size, 28, 28, 1)
```

`shard_batch` 实用工具使用 `jax.tree_util.tree_map` 来处理任意批次结构（如字典），并重塑每个数据数组，使其具有与设备数量匹配的前导维度。

### 定义和映射训练步骤

分布式训练循环的核心是执行单个优化步骤的函数。此函数通常计算损失、梯度，并确定单个设备本地批次的参数 (parameter)更新。使用框架时，这通常涉及调用模型的 `apply` 方法和使用标准的 JAX 自动微分 (`jax.value_and_grad`)。

一个重要方面是处理梯度聚合。由于每个设备仅根据其本地数据分片计算梯度，因此在更新复制的模型参数之前，这些梯度需要在所有设备上平均。这确保了参数更新反映了整个全局批次的损失梯度。`jax.lax.pmean` 集合操作在 `pmap` 处理的函数内部用于此目的。

```python
def compute_loss(params, batch, apply_fn):
  """计算批次的交叉熵损失。"""
  logits = apply_fn({'params': params}, batch['image'])
  one_hot_labels = jax.nn.one_hot(batch['label'], num_classes=10)
  loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.mean(loss)

def train_step(state, batch):
  """在单个设备的数据分片上执行一次训练步骤。"""
  # 计算本地批次的损失和梯度
  grad_fn = jax.value_and_grad(compute_loss)
  loss, grads = grad_fn(state.params, batch, state.apply_fn)

  # **** 重要：在设备间平均梯度 ****
  # 'batch_axis' 是我们为 pmap 维度指定的名称
  averaged_grads = jax.lax.pmean(grads, axis_name='batch_axis')

  # 使用平均后的梯度更新状态
  new_state = state.apply_gradients(grads=averaged_grads)

  # 也可以在此处计算和聚合指标（例如，准确率）
  # metrics = {'loss': loss, 'accuracy': compute_accuracy(logits, batch['label'])}
  # averaged_metrics = jax.lax.pmean(metrics, axis_name='batch_axis')

  return new_state, loss # 返回更新后的状态和本地损失

# 现在，对 train_step 函数进行 pmap 处理
# 指定 pmean 中使用的 axis_name
p_train_step = jax.pmap(train_step, axis_name='batch_axis')

# --- 在训练循环中 ---
# 假设 'sharded_batch' 已按前面所示准备好
# replicated_state 包含在设备间复制的状态

# 执行并行训练步骤
replicated_state, local_losses = p_train_step(replicated_state, sharded_batch)

# local_losses 的形状将是 (num_devices,)
# 用于日志记录的设备间平均损失（可选，在主机上完成）
avg_loss = jnp.mean(local_losses)
print(f"设备间平均损失: {avg_loss:.4f}")

# replicated_state 现在包含更新后的参数和
# 优化器状态，在所有设备上保持一致。
```

在此示例中：

1. `compute_loss` 定义了如何使用存储在 `state` 中的模型的 `apply_fn` 来计算给定参数集和批次的损失。
2. `train_step` 使用 `jax.value_and_grad` 计算损失和梯度。
3. `jax.lax.pmean(grads, axis_name='batch_axis')` 对由名称 `'batch_axis'` 标识的 `pmap` 操作中所有参与设备的梯度树 (`grads`) 进行平均。
4. `state.apply_gradients` 是 `flax.training.train_state.TrainState` 提供的方法，它使用优化器 (`state.tx`) 通过提供的梯度更新参数 (`state.params`)。
5. `jax.pmap(train_step, axis_name='batch_axis')` 创建 `train_step` 的并行版本。`axis_name` 参数很重要；它将 `pmap` 操作与函数内部使用的集合操作（如 `pmean`）连接起来。
6. 当使用 `replicated_state` 和 `sharded_batch` 调用 `p_train_step` 时，JAX 在每个设备上执行 `train_step`，使用其对应的状态和数据切片。`pmean` 操作同步设备以平均梯度。
7. 该函数返回更新后的 `replicated_state`（在所有设备上保持一致）和在每个设备的本地分片上计算的损失。

### 处理 RNG 密钥

像 dropout 这样的随机操作在分布式设置中需要仔细处理随机数生成器 (RNG) 密钥。简单地复制相同的密钥会导致所有设备上生成相同的 dropout 掩码，从而抵消随机性带来的益处。

常见策略是在主机上拆分主 PRNG 密钥，并为每个设备提供*不同*的子密钥。Flax 等框架通常提供机制来自动处理此问题，在初始化或应用模型时，通常要求您传递特定的 RNG 流（例如，一个用于“params”初始化，一个用于“dropout”）。使用 `pmap` 时，您需要确保这些按设备划分的密钥正确地传递到 `pmap` 处理的函数中。通常，这涉及在 `pmap` 处理的函数*外部*拆分密钥，并将生成的分片密钥作为输入参数 (parameter)的一部分包含在内。

```python
# --- 训练循环外部 ---
main_key = jax.random.PRNGKey(42)

# --- 训练循环内部 ---
# 为当前步骤拆分
step_key, main_key = jax.random.split(main_key)

# 为模型内的 dropout 等操作在设备间拆分
dropout_keys = jax.random.split(step_key, num_devices)

# 修改 train_step 以接受和使用 dropout 密钥
def train_step_with_rng(state, batch, dropout_rng):
   # ... 在 compute_loss 或 apply_fn 内部 ...
   # logits = apply_fn({'params': params}, batch['image'],
   #                   rngs={'dropout': dropout_rng})
   # ... 函数的其余部分 ...
   # 记住用于梯度等的 pmean
   pass # 修改后函数体的占位符

p_train_step_with_rng = jax.pmap(train_step_with_rng, axis_name='batch_axis')

# 用分片的 dropout 密钥调用
# replicated_state, local_losses = p_train_step_with_rng(replicated_state, sharded_batch, dropout_keys)
```

将 `pmap` 与 Flax 或 Haiku 等框架结合使用，提供了一种可扩展且有组织的方式来实现数据并行训练。该框架处理模型定义和状态管理，而 `pmap` 编排在多个设备上的分发和执行，包括使用集合操作进行必要的梯度同步。这种模式对于在现代加速器硬件上高效训练大型模型十分重要。

## 参考资料

- [Distributed JAX](https://jax.readthedocs.io/en/latest/advanced_tutorials.html) — The JAX authors (2024)
  Publisher: JAX documentation
  JAX 分布式计算的官方指南，涵盖 `pmap`、`pmean` 等集体操作以及单程序多数据 (SPMD) 范式。
- [Distributed Training with Flax and JAX](https://flax.readthedocs.io/en/latest/guides/nnx_overview/scaling_up_to_multiple_devices.html) — Flax team (2024)
  Publisher: Flax Documentation
  Flax 官方指南，展示如何将 Flax 模型和 `TrainState` 与 JAX 的 `pmap` 集成，以进行数据并行训练，包括状态复制、数据分片和梯度聚合。
- [JAX: Composable transformations of Python+NumPy programs](https://www.jmlr.org/papers/v24/22-0534/22-0534.pdf) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Skye Wanderman-Milne, and Adam Paszke (2023)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 24; Pages: 1-31; DOI: [10.5555/3579549.3579555](https://doi.org/10.5555/3579549.3579555)
  介绍 JAX 的基础论文，阐述了其设计原则、自动微分以及通过 XLA 进行编译的机制，这些构成了其并行化能力的基础。

---

[上一节](03-%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0%E5%92%8C%E7%8A%B6%E6%80%81%E7%9A%84%E5%A4%84%E7%90%86.md) · [下一节](05-%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF.md)

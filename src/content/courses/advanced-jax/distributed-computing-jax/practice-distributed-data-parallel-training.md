---
course: "advanced-jax"
chapter: "distributed-computing-jax"
lesson: "practice-distributed-data-parallel-training"
sourceId: 2862
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-3-distributed-computing-jax/practice-distributed-data-parallel-training"
title: "实践：分布式数据并行训练"
description: "使用 `pmap` 和集合操作实现一个简单的数据并行训练循环。"
order: 9
plots: []
sourceHash: "166a463afc1471a3de33b49430e7ce6e4b015c054fc96ba1892f4f052931e4dc"
sourceCorrections: []
---

构建一个基本的数据并行训练循环，需要模拟多设备（如GPU或TPU核心）环境，并使用 `pmap` 在这些设备上分派一个简单的机器学习 (machine learning)任务。目标是训练一个模型，其中每个设备处理一部分数据批次，局部计算梯度，随后共同完成平均梯度的计算，以进行同步参数 (parameter)更新。

此练习展现了单程序多数据 (SPMD) 执行与 `pmap` 的核心工作流程：为单个设备编写代码，然后 `pmap` 负责复制执行并协调必要的通信。

### 设置与依赖

首先，我们导入JAX、NumPy，并检查可用设备的数量。在此示例中，即使您在CPU上运行，我们也会模拟多个设备，但此代码适用于多GPU或TPU配置。

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax import pmap, grad, value_and_grad, jit
from jax.lax import pmean

# 检查可用设备
num_devices = jax.local_device_count()
print(f"Number of available devices: {num_devices}")

# 如果在CPU上运行，JAX可以模拟多个设备进行pmap测试
# 取消注释下一行，例如在CPU上模拟4个设备
# jax.config.update('jax_platforms', 'cpu') # 如有需要，强制使用CPU
# jax.config.update("jax_cpu_device_count", 4)
# num_devices = jax.local_device_count() # 如果进行模拟，更新设备数量
# print(f"Number of simulated devices: {num_devices}")

# 确保我们至少有2个设备以获得有意义的示例
if num_devices < 2:
    print("Warning: This example is best run with multiple devices (real or simulated).")
    # 您仍然可以运行它，但pmap将无法提供并行优势。

# 生成一个PRNG密钥
key = jax.random.PRNGKey(0)
```

### 定义一个简单模型和损失函数 (loss function)

为了简单起见，我们将使用一个基本的线性回归模型。任务是找到权重 (weight) `w` 和偏置 (bias) `b`，使得 $y \approx Xw + b$。

```python
def linear_model(params, x):
  """一个简单的线性模型预测函数。"""
  w, b = params
  return jnp.dot(x, w) + b

def mean_squared_error(params, x_batched, y_batched):
  """计算均方误差损失。"""
  predictions = linear_model(params, x_batched)
  error = predictions - y_batched
  loss = jnp.mean(error**2)
  return loss
```

### 准备样本数据并分片

我们需要能跨设备拆分的数据。使用 `pmap` 进行数据并行处理的常规做法是，确保输入数组的*第一个*维度与设备数量匹配。沿此维度 (`data[i]`) 的每个切片都会发送到第 `i` 个设备。

```python
# 生成合成数据
feature_dim = 5
num_samples = 100 * num_devices # 确保总样本数可被设备数量整除

key, w_key, b_key, x_key, noise_key = jax.random.split(key, 5)

# 真实参数（我们希望模型学习这些）
true_w = jax.random.normal(w_key, (feature_dim,))
true_b = jax.random.normal(b_key, ())

# 生成特征X和目标y
X = jax.random.normal(x_key, (num_samples, feature_dim))
noise = jax.random.normal(noise_key, (num_samples,)) * 0.1
y = jnp.dot(X, true_w) + true_b + noise

# 为pmap重塑数据：为设备添加一个前导维度
# 每个设备将获得 batch_size_per_device = num_samples // num_devices 个样本
batch_size_per_device = num_samples // num_devices
sharded_X = X.reshape((num_devices, batch_size_per_device, feature_dim))
sharded_y = y.reshape((num_devices, batch_size_per_device))

print(f"Total samples: {num_samples}")
print(f"Samples per device: {batch_size_per_device}")
print(f"Shape of sharded X: {sharded_X.shape}") # 应该是 (num_devices, batch_size_per_device, feature_dim)
print(f"Shape of sharded y: {sharded_y.shape}") # 应该是 (num_devices, batch_size_per_device)
```

### 定义分布式训练步骤

这是本示例的主要部分。我们将创建一个函数，它被设计用于在*单个*设备上运行，计算其本地数据分片的损失和梯度。随后，我们将使用 `pmap` 在所有设备上并行运行此函数。重要的是，在 `pmap` 化函数内部，我们将在执行参数 (parameter)更新之前，使用 `jax.lax.pmean` 平均每个设备独立计算的梯度。

```python
# 定义在单个设备上计算损失和梯度的函数
compute_loss_and_grads = value_and_grad(mean_squared_error)

# 定义将被pmap化的训练步骤函数
# 它接收当前参数和单个设备的数据分片
def distributed_train_step(params, x_shard, y_shard, learning_rate):
  """在设备分片数据上执行一个训练步骤。"""
  # 1. 在每个设备上局部计算损失和梯度
  loss, grads = compute_loss_and_grads(params, x_shard, y_shard)

  # 2. 使用pmean在所有设备上平均梯度
  # 'batch' 是我们将在pmap中定义的轴名称
  # pmean 计算映射在 'batch' 轴上的设备的平均值
  avg_grads = pmean(grads, axis_name='batch')

  # 3. 更新参数（所有设备上保持一致）
  # 简单的梯度下降更新
  new_params = jax.tree_map(lambda p, g: p - learning_rate * g, params, avg_grads)

  # 我们也对损失进行平均，以便报告（可选，但有用）
  avg_loss = pmean(loss, axis_name='batch')

  return new_params, avg_loss

# 使用pmap创建训练步骤的并行版本
# - `axis_name='batch'` 为被映射的维度提供一个名称。
#   此名称由函数内的pmean等集合操作使用。
# - `in_axes=(0, 0, 0, None)` 指定输入如何映射：
#   - params: 复制（None表示在所有设备上使用相同的参数）
#   - x_shard: 沿轴0分片（使用第一个维度作为设备维度）
#   - y_shard: 沿轴0分片
#   - learning_rate: 复制（None表示在所有设备上使用相同的值）
# - `out_axes=0` 指定输出（new_params, avg_loss）应被
#   沿轴0堆叠。然而，由于参数更新在所有设备上都是一致的，
#   `new_params` 沿轴0的所有元素都将相同。
#   在pmean之后，avg_loss在所有设备上也将相同。
# 我们在pmap内部使用jit=True以提高性能（通常是默认值，但此处明确指定）。
p_train_step = pmap(
    distributed_train_step,
    axis_name='batch',
    in_axes=(0, 0, 0, None),
    out_axes=0, # 输出参数和损失将被分片，但在设备间一致
    static_broadcasted_argnums=(3,) # learning_rate 在每次调用中不变
)
```

> 注意：`static_broadcasted_argnums` 参数告知 `pmap`（和 `jit`），该索引处的参数（学习率）在具有相同值的多次调用中保持不变。如果只有其他参数改变，这有助于避免重新编译。`in_axes` 规范非常重要：`0` 表示沿其第一个轴拆分相应的参数，将结果切片分发到设备。`None` 表示复制该参数；每个设备获得相同的副本。

### 初始化参数 (parameter)并复制

在开始训练循环之前，我们需要初始化模型参数（`w` 和 `b`），然后将它们复制到所有设备上。`pmap` 期望的输入要么是分片的，要么是复制的。由于参数在开始时应在所有设备上一致（并保持同步），我们选择复制它们。

```python
# 随机初始化参数
key, w_init_key, b_init_key = jax.random.split(key, 3)
initial_w = jax.random.normal(w_init_key, (feature_dim,))
initial_b = jax.random.normal(b_init_key, ())
params = (initial_w, initial_b)

# 将初始参数复制到所有设备
# 方法1：使用jax.tree_map和jnp.array堆叠
# replicated_params = jax.tree_map(lambda x: jnp.array([x] * num_devices), params)

# 方法2：使用pmap广播行为的常见模式
# 定义一个只返回其输入的辅助函数
def broadcast(x):
  return x
# 将此函数pmap化，in_axes=None意味着输入x被广播
p_broadcast = pmap(broadcast, in_axes=None, out_axes=0)
replicated_params = p_broadcast(params)

print("Shape of initial w:", initial_w.shape)
print("Shape of replicated w:", replicated_params[0].shape) # 应该是 (num_devices, feature_dim)
print("Shape of initial b:", initial_b.shape)
print("Shape of replicated b:", replicated_params[1].shape) # 应该是 (num_devices,)
```

### 训练循环

现在我们可以运行训练循环了。在每一步中，我们用当前复制的参数 (parameter)和分片数据调用 `p_train_step` 函数。该函数负责并行执行、通过 `pmean` 进行梯度平均以及同步更新。

```python
learning_rate = 0.05
num_epochs = 50 # 此处每个epoch使用一次分片的完整数据集

print("\n开始分布式训练...")
current_params = replicated_params

for epoch in range(num_epochs):
  # 执行并行训练步骤
  # 传入复制的参数和本“epoch”的分片数据
  current_params, loss = p_train_step(current_params, sharded_X, sharded_y, learning_rate)

  # p_train_step返回的损失是复制的（在所有设备上相同，
  # 因为我们使用了pmean）。我们只需要一个设备的值。
  # jax.device_get 将数据从设备传输到主机（如果频繁执行可能会很慢）。
  # 访问第一个元素[0]也有效，并且在循环中通常更受青睐。
  epoch_loss = loss[0] # 从第一个设备获取损失

  if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# 训练结束后，current_params中的参数仍然是复制的。
# 从一个设备获取最终参数。
final_params = jax.tree_map(lambda x: x[0], current_params)

print("\n训练完成。")
print("Learned w:", final_params[0])
print("True w:   ", true_w)
print("Learned b:", final_params[1])
print("True b:   ", true_b)
```

### 验证

在每次调用 `p_train_step` 后，`current_params` 变量会持有更新后的参数 (parameter)，这些参数在设备间是复制的。由于梯度平均（`pmean`）确保所有设备计算出相同的平均梯度，并且更新规则是确定的，因此所有设备上的参数在整个训练过程中应保持一致。例如，您可以通过检查设备0和设备1之间权重 (weight) `w` 的 `jnp.allclose(current_params[0][0], current_params[0][1])` 来验证这一点。

### 总结

在本实践部分，我们成功地使用 `pmap` 实现了一个数据并行训练循环：

1. **数据分片：** 我们重塑了输入数据 `X` 和 `y`，使它们的第一个维度与设备数量匹配。
2. **SPMD 函数：** 我们定义了 `distributed_train_step`，它封装了单个设备的逻辑：在其数据分片上计算损失和梯度。
3. **集合通信：** 在此函数内部，我们使用了带有 `axis_name` ('batch') 的 `jax.lax.pmean` 来平均所有参与 `pmap` 的设备所计算的梯度。
4. **同步更新：** 使用平均梯度，每个设备执行了完全相同的参数 (parameter)更新，确保参数保持同步。
5. **`pmap` 转换：** 我们将 `pmap` 应用于 `distributed_train_step`，指定 `axis_name` 并使用 `in_axes` 来控制参数的分布方式（`0` 用于分片数据，`None` 用于复制的参数和学习率）。
6. **复制：** 在启动循环之前，我们明确复制了初始参数。

此模式是扩展许多JAX计算（特别是深度学习 (deep learning)训练）以支持多加速器的基本方法。您可以通过替换线性模型和损失函数 (loss function)为更复杂的神经网络 (neural network)，并加入Adam等优化器来调整此模板，而数据并行的核心 `pmap` 结构在很大程度上保持不变。

## 参考资料

- [Multi-device programming with \`pmap\`](https://jax.readthedocs.io/en/latest/notebooks/gpu_autodiff_pmap.html) — JAX documentation contributors (2024)
  Journal: JAX Documentation; Publisher: Google
  JAX官方关于`pmap`并行执行和`pmean`等集合操作的指南，对于理解本实践示例至关重要。
- [Accurate, Large Minibatch SGD: A White Paper](https://arxiv.org/abs/1706.02677) — Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He (2017)
  Journal: arXiv preprint arXiv:1706.02677; DOI: [10.48550/arXiv.1706.02677](https://doi.org/10.48550/arXiv.1706.02677)
  一篇基础性论文，描述了大规模同步随机梯度下降的原理和考虑因素，该方法通过梯度平均的数据并行训练得以实现。
- [Cloud TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture) — Google Cloud (2024)
  Journal: Google Cloud Documentation
  解释了Cloud TPU如何为SPMD（单程序多数据）执行而设计，为JAX中使用的`pmap`编程模型提供了背景。

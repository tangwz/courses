---
course: "advanced-jax"
chapter: "distributed-computing-jax"
lesson: "handling-axis-names-pmap"
sourceId: 2852
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-3-distributed-computing-jax/handling-axis-names-pmap"
title: "pmap 中轴名称的处理"
description: "使用轴名称，使集合操作的指定更明确、更可靠。"
order: 6
plots: []
sourceHash: "073bfb4bfda16afd1d7e72aa7650f938787e0a8983bd6c1cd1c95da4b04dee32"
sourceCorrections: []
---

当编写打算用 `pmap` 进行并行执行的函数时，特别是那些涉及 `lax.psum` 或 `lax.pmean` 等集合通信操作的函数，明确指定通信应在*哪一组设备之间*进行是很要紧的。JAX 通常在基本示例中能正确推断集合操作是作用于 `pmap` 映射的单个轴。然而，这种隐式行为在更复杂的场合，例如嵌套的 `pmap` 调用或旨在在不同并行环境重复使用的函数中，可能会变得不明确或导致问题。

为了使意图更清楚，JAX 允许你为 `pmap` 映射的轴命名。你可以通过 `axis_name` 参数 (parameter)来提供这个名称。

```python
import jax
import jax.numpy as jnp
from jax import lax, pmap

# 定义要映射的函数
def scaled_sum(x):
  # 执行一些计算
  scaled_x = x * 2.0
  # 对参与 pmap 的设备结果求和
  # 为集合操作明确指定轴的名称
  total_sum = lax.psum(scaled_x, axis_name='devices')
  return total_sum

# 设备数量
n_devices = jax.local_device_count()
print(f"使用 {n_devices} 个设备。")

# 创建一些输入数据，并将其分片到设备上
data = jnp.arange(n_devices, dtype=jnp.float32)

# 应用 pmap，提供 axis_name 'devices'
# 此名称将字符串 'devices' 绑定到映射轴（输入数据的第 0 轴）
parallel_computation = pmap(scaled_sum, axis_name='devices')

# 运行计算
result = parallel_computation(data)

print("输入数据:", data)
# 所有设备上的结果应相同: sum(输入 * 2)
# 示例：如果数据是 [0., 1.]，结果是 [2., 2.]，因为 sum(0*2 + 1*2) = 2
# 示例：如果数据是 [0., 1., 2., 3.]，结果是 [12., 12., 12., 12.]，因为 sum(0*2 + 1*2 + 2*2 + 3*2) = 12
print("结果（所有设备上相同）:", result)

# 手动验证总和
expected_sum = jnp.sum(data * 2.0)
print("预期总和:", expected_sum)

# 注意：输出 'result' 将在所有设备上复制。
# 访问 result[0] 可以得到计算的总和。
assert jnp.allclose(result[0], expected_sum)
```

在这个例子中，`pmap(scaled_sum, axis_name='devices')` 将 `scaled_sum` 函数应用于输入 `data` 数组的每个元素，这些元素分布在可用的设备上。`data` 的第一个维度（大小为 `n_devices`）是映射轴。我们将名称 `'devices'` 指定给这个轴。

在 `scaled_sum` 函数内部，`lax.psum(scaled_x, axis_name='devices')` 调用明确告知集合操作沿着名为 `'devices'` 的轴对 `scaled_x` 的值进行求和。如果没有 `axis_name`，`lax.psum(scaled_x)` 在这个简单情况中可能仍然通过隐式地对映射轴求和而正常工作，但使用名称消除了任何不明确之处。

### 使用轴名称的优势

1. **清晰度：** 代码变得能够自我说明。看到 `lax.psum(..., axis_name='batch')` 清楚地表明求和操作是在被并行处理的批次维度上执行的。
2. **稳定性：** 它避免了在更复杂计算中出现问题。如果你嵌套 `pmap` 调用，每个层级都可以有独立的 `axis_name`。集合操作可以通过引用对应的名称来针对特定的并行层级。例如，你可能有 `pmap(..., axis_name='model_replicas')` 嵌套在 `pmap(..., axis_name='data_shards')` 内部。使用 `axis_name='data_shards'` 的集合操作将在持有不同数据分片但相同模型副本的设备间进行，而 `axis_name='model_replicas'` 将在持有不同模型副本但相同数据分片的设备间进行（这是某些模型并行方法中常见的模式）。
3. **可组合性：** 使用命名轴设计的函数更容易被重复使用。一个在名为 `'spatial'` 的轴上执行归约操作的函数，可以在任何映射了该名称轴的 `pmap` 调用中使用，而不论是否存在其他命名轴。

### 例子：数据并行中的梯度平均

`pmap` 中集合操作的一个常见应用是在数据并行训练期间平均梯度。每个设备计算其本地数据分片的梯度，这些梯度需要在更新模型参数 (parameter)之前在所有设备之间进行平均。

```python
import jax
import jax.numpy as jnp
from jax import lax, pmap, grad

# 模拟损失函数（例如，均方误差）
def loss_fn(params, local_batch):
  # 替换为实际的模型计算和损失
  predictions = params['w'] * local_batch['x'] + params['b']
  error = predictions - local_batch['y']
  return jnp.mean(error**2)

# 在单个设备上计算梯度的函数
def compute_gradients(params, local_batch):
  return grad(loss_fn)(params, local_batch)

# 包含梯度平均的更新步骤
def parallel_update_step(params, sharded_batch):
  # 在每个设备上局部计算梯度
  local_grads = compute_gradients(params, sharded_batch)

  # 使用命名轴 'data_parallel_axis' 在设备间平均梯度
  # 直接使用 pmean 进行平均。使用 psum 后再除以数量也有效。
  avg_grads = lax.pmean(local_grads, axis_name='data_parallel_axis')

  # 简单的梯度下降更新（可替换为 Adam 等）
  learning_rate = 0.01
  # 注意：在实际情况中，对任意 pytree 结构使用 jax.tree_map
  new_params = {
      'w': params['w'] - learning_rate * avg_grads['w'],
      'b': params['b'] - learning_rate * avg_grads['b']
  }
  return new_params

# 获取设备数量
n_devices = jax.local_device_count()

# 初始化模拟参数（最初在所有设备上相同）
params = {'w': jnp.ones(()), 'b': jnp.zeros(())}

# 创建分片到设备上的模拟数据
# 形状：(设备数量, 每个设备的批次, ...)
xs = jnp.arange(n_devices * 4, dtype=jnp.float32).reshape((n_devices, 4, 1))
ys = (xs * 2.0) + 0.5 # 真实 w=2.0, b=0.5
sharded_batch = {'x': xs, 'y': ys}

# 使用带有轴名称的 pmap 定义并行更新函数
p_update_step = pmap(parallel_update_step, axis_name='data_parallel_axis')

# 执行一个更新步骤
new_params = p_update_step(params, sharded_batch)

# 参数基于所有设备的平均梯度进行更新
# new_params 将在设备间复制。访问单个设备上的参数：
print("原始参数:", params)
print("更新后的参数（设备 0 上）:", jax.tree_map(lambda x: x[0], new_params))
```

在此训练步骤中，`pmap` 将 `parallel_update_step` 函数映射到 `sharded_batch` 的第一个轴上，我们将其命名为 `'data_parallel_axis'`。函数内部的 `lax.pmean(local_grads, axis_name='data_parallel_axis')` 确保梯度平均操作明确地在参与此数据并行计算的设备之间进行。

尽管 JAX 在简单情况下可能正确推断出轴，但使用 `axis_name` 明确命名它是一种非常值得推荐的做法。它使你的分布式计算显著更清楚，更不容易出现细微的问题，并且更易于维护和组合，尤其当你的并行策略变得更复杂时。

## 参考资料

- [jax.pmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) — JAX developers (2024)
  JAX pmap 转换的官方文档，详细说明其用法、参数以及 axis_name 如何促进显式设备映射和集体操作。
- [jax.lax collective primitives (psum, pmean)](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.html) — The JAX Authors (2024)
  Publisher: The JAX Authors
  JAX 集体通信原语（如 lax.psum 和 lax.pmean）的官方文档，解释了它们如何与 axis_name 结合使用以协调跨设备的计算。
- [Distributed arrays and automatic sharding](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_sharding.html) — JAX developers (2024)
  一份 JAX 指南，解释了分布式数组和分片的进阶概念，其中 axis_name 对于定义超出基本 pmap 的并行计算以及管理数据分布至关重要。

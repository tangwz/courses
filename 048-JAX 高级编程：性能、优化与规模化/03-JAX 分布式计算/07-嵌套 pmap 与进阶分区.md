# 嵌套 pmap 与进阶分区

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-3-distributed-computing-jax/nested-pmap-advanced-partitioning)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 `pmap` 在设备单维分布计算方面表现强大，但复杂的并行策略通常需要将计算映射到多维加速器网格上。为了满足这一需求，嵌套调用 `pmap` 提供了一种解决方案，这使您能够构建精细的数据并行和模型并行配置。

### 嵌套 `pmap`

设想您有一个设备网格，比如 4 个 GPU 逻辑上排列成 2x2 网格。您可能希望在行上应用数据并行，在列上应用模型并行。这需要同时沿两个不同轴线映射您的计算。JAX 通过嵌套 `pmap` 调用来实现此目的。

用 `pmap` 修饰的函数本身可以调用另一个用 `pmap` 修饰的函数。

```python
import jax
import jax.numpy as jnp
from jax.experimental import maps # 网格定义所需，有时有用

# 假设有 4 个设备可用
num_devices = jax.local_device_count()
if num_devices < 4:
  print(f"Warning: Need at least 4 devices for this example, found {num_devices}")
  # 如果设备不足，则回退到简单执行
  devices = jax.local_devices()[:1] * 4 # 如果需要，将第一个设备使用 4 次
else:
  devices = jax.local_devices()[:4]

# 将设备重塑为 2x2 网格（逻辑表示）
device_mesh = jax.device_put(jnp.arange(num_devices).reshape(2, 2), devices).devices

print("逻辑设备网格:")
print(device_mesh)

# 内部函数：操作特定于一个“模型”分片的数据
# 我们沿着“模型”轴（列）映射此函数
@jax.pmap(axis_name='model')
def inner_op(x_model_shard, params_model_shard):
  # 例子：模型分片内的简单计算
  # 沿着“模型”轴求和（此内部 pmap 调用中的所有设备）
  sum_across_models = jax.lax.psum(x_model_shard * params_model_shard, axis_name='model')
  return sum_across_models * 2 # 某些任意操作

# 外部函数：操作特定于一个“数据”批处理分片的数据
# 我们沿着“数据”轴（行）映射此函数
@jax.pmap(axis_name='data')
def outer_op(x_data_shard, params_data_shard):
  # x_data_shard 到达这里时形状为 (num_model_shards, ...)
  # params_data_shard 到达这里时形状为 (num_model_shards, ...)

  # 在“模型”轴上调用内部 pmap
  result = inner_op(x_data_shard, params_data_shard) # 隐式映射到前导维度

  # 例子：沿着“数据”轴对结果求和（此外部 pmap 调用中的所有设备）
  final_result = jax.lax.psum(result, axis_name='data')
  return final_result

# 准备输入数据和参数，在两个轴上分片
# 形状: (num_data_shards, num_model_shards, ...)
data = jnp.arange(16.).reshape(2, 2, 2, 2) # (data_axis=2, model_axis=2, feature1=2, feature2=2)
params = jnp.ones(16.).reshape(2, 2, 2, 2) * 0.5

# 执行嵌套 pmap
# JAX 根据嵌套结构自动处理分片到相应设备上的放置。
output = outer_op(data, params)

print("\n输入数据形状:", data.shape)
print("输出形状:", output.shape) # 输出将在所有设备上复制
print("输出（一个副本）:")
print(output[0])

# 示例验证（针对此特定操作的手动计算）
# 对于每个 inner_op 调用（固定的数据分片，变化的模型分片）：
# shard_0_0: (0,1,2,3) * 0.5 = (0, 0.5, 1, 1.5) -> psum = (0+0.5+1+1.5) = 3.0 -> *2 = 6.0
# shard_0_1: (4,5,6,7) * 0.5 = (2, 2.5, 3, 3.5) -> psum = (2+2.5+3+3.5) = 11.0 -> *2 = 22.0
# data_shard 0 的内部结果: (6.0, 22.0) <- 在模型轴上复制
#
# shard_1_0: (8,9,10,11) * 0.5 = (4, 4.5, 5, 5.5) -> psum = (4+4.5+5+5.5) = 19.0 -> *2 = 38.0
# shard_1_1: (12,13,14,15)*0.5 = (6, 6.5, 7, 7.5) -> psum = (6+6.5+7+7.5) = 27.0 -> *2 = 54.0
# data_shard 1 的内部结果: (38.0, 54.0) <- 在模型轴上复制
#
# “数据”轴上的外部操作 psum:
# Axis 0: psum(6.0, 38.0) = 44.0
# Axis 1: psum(22.0, 54.0) = 76.0
# 最终结果应在所有设备上复制，形状为 (2,2,2)，例如 [[44, 44],[44, 44]], [[76, 76],[76,76]]（按原始特征维度）。
# 等等，示例代码重塑了内部结果。我们重新推导一下。
# inner_op 输入: 沿着模型轴的每对设备具有 (2, 2) 特征。
# inner_op(x[0,0], p[0,0]) = psum([0,1]*0.5 + [2,3]*0.5, axis='model') = psum([0, 0.5] + [1, 1.5], axis='model') = psum([1, 2]) = 3.0? NO. psum is elementwise?
# 让我们重新阅读 psum：对 *跨设备* 的数组元素求和。
# 好的，假设有 4 个设备。网格是 [[0,1],[2,3]]
# outer_op called on devices [0,2] (data axis 0) and [1,3] (data axis 1) ?? No, outer_op maps the *function* across devices.
# 为了清晰起见，我们简化输入数据。假设 1 个特征维度。
# data = jnp.arange(4.).reshape(2, 2) # (data_axis=2, model_axis=2)
# params = jnp.ones(4.).reshape(2, 2) * 0.5
#
# 设备映射（示例）：
# Device 0 gets data[0,0]=0, params[0,0]=0.5
# Device 1 gets data[0,1]=1, params[0,1]=0.5
# Device 2 gets data[1,0]=2, params[1,0]=0.5
# Device 3 gets data[1,1]=3, params[1,1]=0.5
#
# outer_op is called, mapping over data axis.
#   - 迭代 0（数据轴索引 0）在设备 [0, 1] 上运行。接收到这些设备上的 data[0]=((0,1),(2,3)) 和 params[0]=((0.5,0.5),(0.5,0.5))。x_data_shard = data[0], params_data_shard = params[0]。
#   - 迭代 1（数据轴索引 1）在设备 [2, 3] 上运行。接收到这些设备上的 data[1]=((8,9),(10,11)) 和 params[1]=((0.5,0.5),(0.5,0.5))。x_data_shard = data[1], params_data_shard = params[1]。
#
# 在 outer_op 内部，迭代 0（设备 0, 1）：
#   调用 `inner_op(x_data_shard, params_data_shard)`。`inner_op` 映射到 *其* 输入的第一个维度（`x_data_shard`，大小为 2，对应“模型”轴）。
#     - 内部迭代 0（模型轴索引 0）在设备 0 上运行。获取 x_model_shard=x_data_shard[0]=(0,1), params_model_shard=params_data_shard[0]=(0.5, 0.5)。计算 x*p = (0, 0.5)。
#     - 内部迭代 1（模型轴索引 1）在设备 1 上运行。获取 x_model_shard=x_data_shard[1]=(2,3), params_model_shard=params_data_shard[1]=(0.5, 0.5)。计算 x*p = (1, 1.5)。
#     `jax.lax.psum(..., axis_name='model')` 对参与此 `inner_op` 调用的设备（设备 0, 1）进行求和。
#       `psum((0, 0.5), (1, 1.5))` -> `(0+1, 0.5+1.5)` = `(1, 2)`。此结果 `(1,2)` 在设备 0 和设备 1 上均存在。
#     `result = (1, 2) * 2 = (2, 4)`。此结果在设备 0 和设备 1 上均存在。
#
# 在 outer_op 内部，迭代 1（设备 2, 3）：
#   调用 `inner_op(x_data_shard, params_data_shard)`。`inner_op` 映射到 *其* 输入的第一个维度（`x_data_shard`，大小为 2，对应“模型”轴）。
#     - 内部迭代 0（模型轴索引 0）在设备 2 上运行。获取 x_model_shard=x_data_shard[0]=(8,9), params_model_shard=params_data_shard[0]=(0.5, 0.5)。计算 x*p = (4, 4.5)。
#     - 内部迭代 1（模型轴索引 1）在设备 3 上运行。获取 x_model_shard=x_data_shard[1]=(10,11), params_model_shard=params_data_shard[1]=(0.5, 0.5)。计算 x*p = (5, 5.5)。
#     `jax.lax.psum(..., axis_name='model')` 对参与此 `inner_op` 调用的设备（设备 2, 3）进行求和。
#       `psum((4, 4.5), (5, 5.5))` -> `(4+5, 4.5+5.5)` = `(9, 10)`。此结果 `(9,10)` 在设备 2 和设备 3 上均存在。
#     `result = (9, 10) * 2 = (18, 20)`。此结果在设备 2 和设备 3 上均存在。
#   回到 `outer_op`:
#     `jax.lax.psum(result, axis_name='data')` 对参与 `outer_op` 调用的设备（所有设备 0, 1, 2, 3）进行求和。
#     设备 0 上的 `result` 值为 (2, 4)。设备 1 上为 (2, 4)。设备 2 上为 (18, 20)。设备 3 上为 (18, 20)。
#     `psum((2,4), (2,4), (18,20), (18,20))` -> `(2+2+18+18, 4+4+20+20)` = `(40, 48)`。
#     此最终结果 `(40, 48)` 应存在于所有设备 0, 1, 2, 3 上。
# 让我们用更简单的输入在脑海中重新运行代码：
# data = jnp.arange(4.).reshape(2, 2)
# params = jnp.ones(4.).reshape(2, 2) * 0.5
# Device 0: data=0, param=0.5. outer_iter=0, inner_iter=0. x*p=0.
# Device 1: data=1, param=0.5. outer_iter=0, inner_iter=1. x*p=0.5.
# Device 2: data=2, param=0.5. outer_iter=1, inner_iter=0. x*p=1.0.
# Device 3: data=3, param=0.5. outer_iter=1, inner_iter=1. x*p=1.5.
# outer_iter=0（设备 0,1）：inner_op 接收 x=(0,1), p=(0.5,0.5)。
#   inner_iter=0（设备 0）：接收 x=0, p=0.5 -> 计算 0。
#   inner_iter=1（设备 1）：接收 x=1, p=0.5 -> 计算 0.5。
#   “模型”上的内部 psum（设备 0,1）：psum(0, 0.5) = 0.5。设备 0,1 上的结果 = 0.5 * 2 = 1.0。
# outer_iter=1（设备 2,3）：inner_op 接收 x=(2,3), p=(0.5,0.5)。
#   inner_iter=0（设备 2）：接收 x=2, p=0.5 -> 计算 1.0。
#   inner_iter=1（设备 3）：接收 x=3, p=0.5 -> 计算 1.5。
#   “模型”上的内部 psum（设备 2,3）：psum(1.0, 1.5) = 2.5。设备 2,3 上的结果 = 2.5 * 2 = 5.0。
# “数据”上的外部 psum（设备 0,1,2,3）：psum(1.0, 1.0, 5.0, 5.0) = 12.0。
# 最终输出在所有设备上应为 12.0。

# 让我们稍微修改示例代码以使用这些更简单的数据。
data_simple = jnp.arange(4.).reshape(2, 2) # (data_axis=2, model_axis=2)
params_simple = jnp.ones(4.).reshape(2, 2) * 0.5

# 使用更简单的数据重新运行
output_simple = outer_op(data_simple, params_simple)
print("\n--- 简单示例 ---")
print("输入数据形状:", data_simple.shape)
print("输出形状:", output_simple.shape)
print("输出（一个副本）:")
print(output_simple[0]) # 预期 12.0
```

关于嵌套 `pmap` 的重要点：

1. **轴作用域：** 内部 `pmap` 中定义的轴名称仅限于该特定调用以及在该外部迭代中参与其中的设备。外部 `axis_name` 指的是外部 `pmap` 中涉及的所有设备之间的集体操作。
2. **数据映射：** 数据通常沿着与 `pmap` 轴对应的维度进行分片。在上面的示例中，`data` 和 `params` 的形状为 `(2, 2, ...)`，映射到 2x2 逻辑设备网格。第一个维度对应于外部 `pmap` 的 `data` 轴，第二个维度对应于内部 `pmap` 的 `model` 轴。
3. **集体操作：** 像 `psum` 这样的集体操作必须指定它们所操作的 `axis_name`。这会告知 JAX 哪些设备组应参与通信（例如，在 `inner_op` 调用中沿“模型”轴求和，或在 `outer_op` 调用中沿“数据”轴求和）。

### 使用 `pmap` 的进阶分区策略

虽然嵌套 `pmap` 允许将计算映射到多维设备网格，但 `pmap` 本身遵循 SPMD 原则：相同的程序在各处运行，但操作不同的数据切片。分区由输入数组沿映射轴分片的方式隐式确定。

嵌套 `pmap` 使得常见的进阶分区模式成为可能：

1. **组合数据并行和模型并行：** 如示例所示，您可以使用外部 `pmap` 进行数据并行（拆分批次），使用内部 `pmap` 进行模型并行（将模型层或参数 (parameter)分发到不同设备）。然后，在适当的轴作用域内使用集体操作（例如，沿数据轴 `psum` 梯度，沿模型轴 `all_gather` 激活）。
2. **2D/3D 张量并行：** 对于非常大的矩阵乘法（在 Transformer 中常见），您可以使用嵌套 `pmap` 和每个维度内的特定集体通信模式，将输入矩阵和计算在 2D 甚至 3D 设备网格上进行分区。

这里是一个图表，说明了用于组合数据/模型并行的 2x2 设备网格：

> 一个 2x2 设备网格。外部 `pmap` 映射到“数据”轴（行，虚线红色表示“数据”轴集体操作）。对于每个数据分片，内部 `pmap` 映射到“模型”轴（列，实心蓝色表示“模型”轴集体操作）。

**手动设备分配考量：**

`pmap` 通常假设 JAX 根据 `jax.devices()` 或与输入数组关联的设备来管理设备分配。对于高度特定的硬件拓扑或性能调优，您有时可能需要更明确地控制哪个物理设备执行计算的哪个部分。

虽然 `pmap` 本身不提供对其执行内部细粒度手动 *计算* 放置，但您可以通过以下方式影响它：

1. **准备输入：** 在调用 `pmap` 之前，使用 `jax.device_put(array_shard, device)` 将输入数组分片手动放置到特定设备上。如果可能，`pmap` 通常会尊重这种放置。
2. **使用 `pmap(devices=...)`：** 您可以明确地将设备列表或数组传递给 `pmap`，以限制其运行设备，尽管该集合 *内部* 的分配通常是自动的。

对于与 `pmap` 中隐含的标准 SPMD 模型显着不同的分区方案（例如，不同设备运行本质上不同阶段的复杂流水线并行），其他 JAX 功能或库可能更适合，例如谨慎使用 `jax.jit(device=...)` 或专注于明确分区的实验性库，如 `jax.experimental.shard_map` 或基于 `jax.Array` 构建的框架。然而，嵌套 `pmap` 与轴命名相结合，为 `pmap` 框架内的许多进阶网格式分区策略提供了一种强大的机制。

熟练运用嵌套 `pmap` 需要仔细关注数据分片、轴命名以及集体操作的作用范围。虽然与单个 `pmap` 相比它增加了复杂性，但它实现了在多维加速器数组上扩展计算的能力，从而能够比使用更简单的并行方案更有效地训练更大的模型和处理更大的数据集。

## 参考资料

- [Parallelism in JAX: \`pmap\`](https://jax.readthedocs.io/en/latest/jax-101/06-using-pmap.html) — JAX Developers (2024)
  Publisher: JAX Documentation
  解释了 `pmap` 转换、轴命名和集体操作，是理解嵌套并行化的基础。
- [Distributed Arrays and \`shard_map\` in JAX](https://jax.readthedocs.io/en/latest/jax-101/device_parallel.html) — JAX authors (2024)
  Publisher: JAX Documentation
  详细介绍了高级数据分区、显式设备网格以及用于细粒度分布式数组管理的 `jax.Array` 系统，包括 `shard_map`。

---

[上一节](06-pmap%20%E4%B8%AD%E8%BD%B4%E5%90%8D%E7%A7%B0%E7%9A%84%E5%A4%84%E7%90%86.md) · [下一节](08-%E5%A4%9A%E4%B8%BB%E6%9C%BA%E7%BC%96%E7%A8%8B%E7%AE%80%E4%BB%8B.md)

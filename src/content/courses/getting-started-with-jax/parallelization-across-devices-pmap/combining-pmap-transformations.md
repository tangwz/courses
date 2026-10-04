---
course: "getting-started-with-jax"
chapter: "parallelization-across-devices-pmap"
lesson: "combining-pmap-transformations"
sourceId: 1788
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-5-parallelization-across-devices-pmap/combining-pmap-transformations"
title: "将 `pmap` 与其他变换结合使用"
description: "整合 `pmap` 与 `jit`、`grad` 和 `vmap`，用于复杂计算。"
order: 6
plots: []
sourceHash: "b25d6ae2b057e85cd6ff99d0925aa393054f67d984003f4d52b8934971c85773"
sourceCorrections: []
---

JAX 的能力在函数变换组合时显现其优势。虽然 `pmap` 提供在多个设备上分发计算的机制，但它很少独立运行。在每个设备上运行的计算通常需要编译以提高速度（`jit`），微分以优化（`grad`），或向量 (vector)化以提高效率（`vmap`）。幸好 JAX 变换被设计为可组合的，这使得您能够自然地分层使用这些功能。

本节说明 `pmap` 如何与 `jit`、`grad` 和 `vmap` 互相配合，使您能够构建精巧、高性能的分布式程序。

### `pmap` 与 `jit` 的结合

您可能想知道是否需要显式地将 `pmap` 和 `jit` 结合起来，例如通过编写 `pmap(jit(my_function))`。答案通常是不需要的，因为 `pmap` **已经包含了 JIT 编译**。

当您将 `pmap` 应用于一个函数时，JAX 会追踪该函数（类似于 `jit`），并使用 XLA 为目标设备（CPU、GPU 或 TPU）编译它。这种编译对于 `pmap` 的性能很重要，因为它专门优化代码以在设备间并行执行。

```python
import jax
import jax.numpy as jnp
import numpy as np
import time

# 假设本例中有 2 个可用设备
num_devices = 2
devices = jax.local_devices()[:num_devices]
print(f"使用 {len(devices)} 个设备: {devices}")

# 一个可以从 JIT 获得好处的简单函数
def complex_computation(x):
  y = jnp.sin(x) * jnp.cos(x)
  z = jnp.tanh(y) + jnp.sqrt(jnp.abs(x))
  return z * 2.0

# 直接应用 pmap
pmap_complex_computation = jax.pmap(complex_computation)

# 创建在设备间分片的数据
data = np.arange(8.0).reshape(num_devices, -1) # 形状 (2, 4)
sharded_data = jax.device_put(data, devices)

# 运行 pmap 化的函数（包含 JIT 编译）
start_time = time.time()
result = pmap_complex_computation(sharded_data)
result.block_until_ready() # 确保计算在计时前完成
end_time = time.time()
print(f"pmap 执行时间: {end_time - start_time:.6f} 秒")
print("结果形状:", result.shape)
# print("结果:\n", result) # 取消注释以查看结果

# 优先进行显式 JIT 通常没有额外好处
# jit_then_pmap = jax.pmap(jax.jit(complex_computation))
# start_time = time.time()
# result_jit_then_pmap = jit_then_pmap(sharded_data)
# result_jit_then_pmap.block_until_ready()
# end_time = time.time()
# print(f"jit -> pmap 执行时间: {end_time - start_time:.6f} 秒")
```

执行此代码表明 `pmap` 处理了编译。虽然 `jax.pmap(jax.jit(f))` 有效，但它通常不会比 `jax.pmap(f)` 带来更多好处，因为 `pmap` 会执行自己的 JIT 编译，专为多设备执行优化。您可能单独进行 JIT 编译的场景是，如果该函数是传递给 `pmap` 的主函数内部的一个组件，并且您希望独立控制其编译。

### `pmap` 与 `grad` 的结合：分布式梯度计算

`pmap` 在机器学习 (machine learning)中的一个基本应用是数据并行：通过在多个设备上分发数据批次来训练大型数据集上的模型。这需要根据每个设备上的数据分片计算梯度，然后汇集这些梯度。将 `pmap` 与 `grad`（或 `value_and_grad`）结合使用就能实现这一点。

典型的模式包括：

1. 定义一个损失函数 (loss function)，它将模型参数 (parameter)和一批数据作为输入。
2. 使用 `jax.grad` 或 `jax.value_and_grad` 创建一个函数，用于计算损失相对于参数的梯度。
3. 将 `pmap` 应用于这个梯度计算函数。`pmap` 将在每个设备上使用其本地数据分片执行梯度计算。
4. 在经过 `pmap` 处理的函数内部，使用**集体操作**，例如 `jax.lax.pmean`，来平均所有设备上计算的梯度。这确保所有设备获得相同的、全局平均的梯度，从而一致地更新模型参数。

我们用一个简化例子来举例说明：

```python
import jax
import jax.numpy as jnp
import numpy as np

# 假设有 2 个设备
num_devices = 2
devices = jax.local_devices()[:num_devices]

# 简单模型和损失函数
def predict(params, x):
  # 一个简单的线性模型: y = w*x + b
  w, b = params
  return w * x + b

def loss_fn(params, x_batch, y_batch):
  predictions = predict(params, x_batch)
  error = predictions - y_batch
  return jnp.mean(error**2) # 均方误差

# 计算值（损失）和梯度的函数
value_and_grad_fn = jax.value_and_grad(loss_fn)

# 待 pmap 处理的函数: 计算每个设备的梯度并平均它们
def parallel_update_step(params, x_shards, y_shards):
  # 在每个设备的数据分片上局部计算梯度
  loss, grads = value_and_grad_fn(params, x_shards, y_shards)

  # 平均所有设备上的梯度
  # 'axis_name' 必须与 pmap 中提供的一致
  avg_grads = jax.lax.pmean(grads, axis_name='devices')

  # 也可选择平均损失 (用于日志记录)
  avg_loss = jax.lax.pmean(loss, axis_name='devices')

  return avg_loss, avg_grads

# 应用 pmap, 指定映射轴和集合操作轴名称
# 参数被复制，数据沿轴 0 分片
pmap_update_step = jax.pmap(
    parallel_update_step,
    axis_name='devices', # 集合操作的名称
    in_axes=(None, 0, 0), # 复制参数，将 x 和 y 沿轴 0 映射
    out_axes=(None, None) # 返回平均后的损失/梯度（在所有设备上相同）
)

# 示例参数和数据
params = (jnp.array(2.0), jnp.array(1.0)) # (w, b)
# 为 2 个设备创建数据，每个设备 4 个样本
x_data = np.arange(8.0).reshape(num_devices, -1) # 形状 (2, 4)
y_data = (3 * x_data + 2 + np.random.randn(*x_data.shape) * 0.5) # y = 3x + 2 + 噪声

# 将数据放入设备
sharded_x = jax.device_put(x_data, devices)
sharded_y = jax.device_put(y_data, devices)

# 执行并行梯度计算
avg_loss, avg_grads = pmap_update_step(params, sharded_x, sharded_y)

print(f"设备间的平均损失: {avg_loss:.4f}")
print(f"平均梯度 (dw, db): ({avg_grads[0]:.4f}, {avg_grads[1]:.4f})")

# avg_grads 现在可以用于更新参数 'params'
# 注意: 返回的 avg_loss 和 avg_grads 是常规的 JAX 数组，
# 未分片，因为我们在 pmean 后使用了 out_axes=None。
```

在此示例中：

- `value_and_grad_fn` 创建用于计算损失和梯度的函数。
- `parallel_update_step` 封装了此过程，添加了 `jax.lax.pmean` 集合操作，以平均参与 `pmap` 的设备上的梯度（和损失）。`axis_name='devices'` 将 `pmean` 操作与 `pmap` 上下文 (context)关联起来。
- `jax.pmap` 配置为 `in_axes=(None, 0, 0)`，表示：
  - `params` 未映射（`None`），因此每个设备获得一个完整副本（复制）。
  - `x_shards` 和 `y_shards` 沿其第一个轴（`0`）映射，分发数据。
- `out_axes=(None, None)` 确保平均结果（在 `pmean` 之后在所有设备上相同）作为常规的、未复制的 JAX 数组返回。

这种 `pmap(value_and_grad(...))` 模式，与集合操作结合，是 JAX 中分布式数据并行训练的根本。

### `pmap` 与 `vmap` 的结合

`pmap` 与 `vmap` 的结合不如 `pmap` 与 `grad` 或 `jit` 的结合常见，主要是因为 `pmap` 本身就执行一种跨设备的映射。然而，`vmap` 在被 `pmap` 处理的函数*内部*仍然有用。

回想一下，`pmap` 实现了 SPMD（单程序多数据）：*相同*的函数在每个设备上运行，但处理不同的数据切片。`vmap` 对单个执行追踪*内部*的操作进行向量 (vector)化。

如果您需要额外的向量化 (quantization)层，独立处理每个设备的数据分片，您可以在 `pmap` 处理的函数内部使用 `vmap`。

考虑一个场景，其中每个设备处理一批图像（`pmap`），对于每张图像，您希望对从中提取的多个图像块应用相同的操作（`vmap`）。

```python
import jax
import jax.numpy as jnp
import numpy as np

# 假设有 2 个设备
num_devices = 2
devices = jax.local_devices()[:num_devices]

# 对单个项（例如，图像块）操作的函数
def process_item(item):
  return jnp.tanh(item) * 2.0

# 使用 vmap 处理多个项（例如，图像内的图像块）的函数
# 此函数*在每个设备上*运行
def process_batch_of_items(batch):
  # 在设备内部工作时使用 vmap 进行向量化
  vectorized_processor = jax.vmap(process_item)
  return vectorized_processor(batch)

# pmap 在设备间分发批次
pmap_process_batches = jax.pmap(
    process_batch_of_items,
    in_axes=0 # 将输入数据的第一个轴映射到设备间
)

# 示例: 2 个设备，每个设备处理 3 个大小为 4 的项的批次
# 总数据形状: (设备数量, 每个设备批次大小, 项大小) = (2, 3, 4)
data = np.arange(2 * 3 * 4.0).reshape(num_devices, 3, 4)
sharded_data = jax.device_put(data, devices)

# 执行: pmap 分发 (3, 4) 的批次，
# 内部的 vmap 在每个设备上并行处理这 3 个项。
result = pmap_process_batches(sharded_data)
result.block_until_ready()

print("每个设备的输入数据形状:", data.shape[1:]) # (3, 4)
print("输出结果形状:", result.shape) # (2, 3, 4)
# 每个设备获得 (3, 4) 的数据，vmap 在轴 0（3 个项）上操作，
# pmap 沿轴 0（2 个设备）连接结果
```

在这里，`pmap` 将外部维度（大小 2）分发到设备。在每个设备上，`process_batch_of_items` 接收形状为 `(3, 4)` 的切片。在此函数内部，`vmap(process_item)` 自动对 `process_item` 函数沿其接收数据的首要轴（大小 3）进行向量化。

虽然您技术上*可以*编写 `pmap(vmap(f))`，但这通常与 `pmap(f, in_axes=...)` 已经实现的功能重叠。在传递给 `pmap` 的函数*内部*使用 `vmap` 通常是更直观和常见的方式，可以在每个设备的并行计算中引入进一步的向量化。

### 变换的顺序

结合这些变换时的典型顺序反映了它们各自的角色：

1. **最内层变换（`vmap`、`grad`、`jit`）：** 这些定义了核心计算逻辑。`vmap` 进行向量 (vector)化，`grad` 计算导数，而 `jit`（通常由 `pmap` 隐式处理或应用于内部函数）为单设备效率进行编译。
2. **最外层变换（`pmap`）：** 这协调跨多个设备的执行，处理数据分发，在每个设备上启动（可能已经变换过的）内部函数，并管理集合通信。

因此，像 `pmap(grad(jit(my_loss)))` 或 `pmap(jit(vmap(my_kernel)))`（其中 `jit` 可能在 `pmap` 中是隐式的）这样的模式很常见，`pmap` 充当顶级分发器。

通过理解如何将 `pmap` 与 `jit`、`grad` 和 `vmap` 组合，您可以充分发挥 JAX 的能力，创建高效的程序，这些程序可以有效地扩展到现代硬件加速器。这种可组合性是 JAX 的一个标志性特点，它使分布式模型训练等复杂流程能够简洁高效地表达。

## 参考资料

- [JAX documentation](https://jax.readthedocs.io/en/latest/) — Google, LLC. (2024)
  详细说明JAX的函数变换，包括`pmap`、`jit`、`grad`和`vmap`，以及它们的可组合性。
- [JAX: Autodiff and XLA for high-performance ML](https://arxiv.org/abs/1810.04835) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Skye Wanderman-Milne, and Adam Paszke (2018)
  Journal: SysML Workshop; Publisher: SysML Workshop
  介绍JAX的基础设计，涵盖其函数变换、自动微分以及与XLA的编译。
- [XLA: Accelerated Linear Algebra](https://www.tensorflow.org/xla) — Google (2024)
  Publisher: Google
  描述XLA，JAX用于优化各种硬件加速器上计算的专用编译器。
- [Distributed JAX on Cloud TPUs](https://jax.readthedocs.io/en/latest/jax.distributed.html) — Google, LLC. (2024)
  Publisher: The JAX authors
  JAX中分布式计算的实用指南，包括使用`pmap`进行数据并行和集合操作。

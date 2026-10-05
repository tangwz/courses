# 集体操作（`lax.psum`、`lax.pmean`等）

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-5-parallelization-across-devices-pmap/pmap-collective-operations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当您使用 `jax.pmap` 将计算并行化到多个设备时，每个设备执行相同的函数（SPMD 中的“单一程序”），但处理其特有的数据片段（“多数据”）。尽管这种独立处理功能强大，但您通常需要一种机制，使这些并行执行能够通信并聚合结果。例如，在分布式机器学习 (machine learning)训练中，每个设备可能根据其本地数据批次计算梯度，但在更新模型参数 (parameter)之前，您需要将所有设备上的这些梯度进行组合。这就是集体操作发挥作用的地方。

集体操作是特定函数，通常位于 `jax.lax` 模块（JAX 的低级 API）中，专为在 `pmap` 内部运行而设计。它们协调参与并行执行的设备间的通信和计算。

### 聚合结果：`psum` 和 `pmean`

两种最常用的集体操作是 `lax.psum`（并行求和）和 `lax.pmean`（并行求平均）。

设想您有一个在每个设备上独立计算的值，例如 `local_value`。您希望计算 `pmap` 中所有设备上这些 `local_value` 的总和。您可以使用 `lax.psum` 实现这一点：

```python
import jax
import jax.numpy as jnp
from jax import lax

# 假设本例有4个可用设备
# 在实际场景中，JAX 会检测可用设备
num_devices = 4 

# 待 pmap 转换的示例函数
def calculate_local_sum(x):
  local_value = jnp.sum(x) # 每个设备计算其数据片段的和
  
  # 在参与 pmap 的所有设备上对 'local_value' 求和 
  # 轴名称为 'devices'
  total_sum = lax.psum(local_value, axis_name='devices') 
  
  return total_sum 

# 创建虚拟数据，在设备间分片
# 形状：(设备数量, 每个设备的数据量)
data = jnp.arange(num_devices * 3).reshape((num_devices, 3)) 
# 设备0上的数据：[0, 1, 2] -> local_sum = 3
# 设备1上的数据：[3, 4, 5] -> local_sum = 12
# 设备2上的数据：[6, 7, 8] -> local_sum = 21
# 设备3上的数据：[9, 10, 11] -> local_sum = 30

# 应用 pmap，将映射轴命名为 'devices'
# 此处的 axis_name 必须与 lax.psum 中使用的匹配
pmapped_calculate_sum = jax.pmap(calculate_local_sum, axis_name='devices')

# 执行 pmap 转换的函数
result = pmapped_calculate_sum(data)

# 结果将包含每个设备上的*总*和
# total_sum = 3 + 12 + 21 + 30 = 66
print(result) 
# 预期输出（在4个设备上）：[66 66 66 66] 
```

`lax.psum(value, axis_name)` 的要点：

1. **输入：** 它接收在每个设备上本地计算的 `value`。
2. **操作：** 它在由 `axis_name` 指定的轴上，对所有参与并行计算的设备上的这些值求和。
3. **`axis_name`：** 这个字符串标识符将集体操作与其所属的特定 `pmap` 执行关联起来。您在 `jax.pmap` 调用中定义此名称（例如 `axis_name='devices'`），并在集体操作内部使用*相同*的名称（`lax.psum(..., axis_name='devices')`）。这确保求和操作发生在正确的设备组上，如果您嵌套 `pmap` 调用，这一点尤其重要。
4. **输出：** 重要的是，`lax.psum` 将*相同*的总和返回给*每个*设备。这确保所有并行执行都能访问聚合结果。

`lax.pmean` 的工作方式类似，但计算的是平均值而非总和。它等同于 `lax.psum(value, axis_name) / N`，其中 $N$ 是映射轴上的设备数量。这在数据并行训练中对梯度求平均非常普遍。

```python
import jax
import jax.numpy as jnp
from jax import lax

# 假设有4个设备
num_devices = 4

def calculate_local_mean(x):
  local_value = jnp.mean(x) # 示例本地计算
  
  # 计算所有设备上 'local_value' 的平均值
  global_mean = lax.pmean(local_value, axis_name='devices')
  
  return global_mean

data = jnp.arange(num_devices * 3, dtype=jnp.float32).reshape((num_devices, 3))
# 本地平均值：1.0, 4.0, 7.0, 10.0

pmapped_calculate_mean = jax.pmap(calculate_local_mean, axis_name='devices')
result = pmapped_calculate_mean(data)

# 结果将包含每个设备上本地平均值的平均值
# global_mean = (1.0 + 4.0 + 7.0 + 10.0) / 4 = 22.0 / 4 = 5.5
print(result)
# 预期输出（在4个设备上）：[5.5 5.5 5.5 5.5] 
```

### 其他有用的集体操作

尽管 `psum` 和 `pmean` 是主要工具，`jax.lax` 还提供其他集体操作：

- `lax.pmax(value, axis_name)`：在设备间找到最大值。
- `lax.pmin(value, axis_name)`：在设备间找到最小值。
- `lax.all_gather(value, axis_name)`：从所有设备收集 `value` 并沿新的前导轴进行拼接。与 `psum` 或 `pmean` 返回单个聚合标量（如果输入 `value` 是标量）不同，`all_gather` 为每个设备提供来自*所有*其他设备的*完整*值集合。

> `lax.psum` 示意图。每个设备的本地值被求和，然后单一的总和被返回给所有参与的设备。

### 正确使用集体操作

在使用集体操作时，请记住以下几点：

1. **上下文 (context)：** 集体操作*必须*在由 `jax.pmap` 转换的函数内部调用。在此上下文之外调用它们将导致错误。
2. **`axis_name` 一致性：** 在集体函数（`lax.psum`、`lax.pmean`等）中使用的 `axis_name` 字符串*必须*与对应的 `jax.pmap` 调用中指定的 `axis_name` 匹配。这确保操作发生在预期的一组设备上。
3. **同步：** 集体操作充当同步点。所有设备都必须到达集体操作，计算才能继续并分发结果。
4. **数据类型：** 确保聚合值的设备间数据类型一致，并适合该操作（例如，`psum` 和 `pmean` 需要数值类型）。

集体操作是使用 `pmap` 编写有效多设备程序的基本工具，它们实现了协调和数据聚合，这对于分布式训练等算法非常重要。通过熟练使用 `psum`、`pmean` 并理解 `axis_name` 的作用，您可以有效地将 JAX 计算扩展到可用的硬件加速器上。

## 参考资料

- [Distributed JAX](https://jax.readthedocs.io/en/latest/jax.distributed.html) — JAX core contributors (2024)
  JAX官方文档，涵盖JAX的分布式编程模型，包括`pmap`和集合操作。
- [JAX: Accelerated Array Programming in Python](https://joss.theoj.org/papers/10.21105/joss.00843) — Roy Frostig, Matthew James Johnson, Chris Leary (2018)
  Journal: Journal of Open Source Software; Volume: 3; Pages: 843; DOI: [10.21105/joss.00843](https://doi.org/10.21105/joss.00843)
  介绍了JAX库，详细阐述了其可组合函数变换、自动微分和编译到XLA以实现加速器执行的设计原则，这些原则是`pmap`实现并行化的基础。
- [Horovod: fast and easy distributed deep learning with TensorFlow](https://arxiv.org/abs/1802.05799) — Alexander Sergeev, Mike Del Balso (2018)
  Journal: arXiv preprint arXiv:1802.05799; DOI: [10.48550/arXiv.1802.05799](https://doi.org/10.48550/arXiv.1802.05799)
  解释了分布式深度学习训练中高效集合通信操作（特别是all-reduce）的动机和实现，为JAX的`lax.psum`和`lax.pmean`提供了概念背景。
- [High Performance Computing: A Practical Approach](https://www.oreilly.com/library/view/high-performance-computing/9781491973680/) — Kevin Dowd, Charles Severance (2018)
  Publisher: O'Reilly Media
  提供了并行计算概念的实践概览，包括SPMD模型和集合通信操作的作用，为JAX的方法提供了更广阔的背景。

---

[上一节](04-%E8%AE%BE%E5%A4%87%E7%BD%91%E6%A0%BC%E4%B8%8E%E8%BD%B4%E5%90%8D%E7%A7%B0.md) · [下一节](06-%E5%B0%86%20%60pmap%60%20%E4%B8%8E%E5%85%B6%E4%BB%96%E5%8F%98%E6%8D%A2%E7%BB%93%E5%90%88%E4%BD%BF%E7%94%A8.md)

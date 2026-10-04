# 调试 `pmap` 化的函数

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-5-parallelization-across-devices-pmap/debugging-pmap)

[返回章节目录](README.md) · [返回课程目录](../README.md)

调试跨多个设备并行运行的代码，其复杂性超出单设备执行或标准 Python 代码。当你用 `jax.pmap` 包装一个函数时，你将从单一执行流转变为在单一程序、多数据（SPMD）模型下同时操作的多个流。错误可能不仅发生在函数逻辑内部，还可能发生在数据在设备间的分发、处理和收集方式上。此外，由于 `pmap` 使用 XLA 隐式编译你的函数（类似于 `jit`），你还需要考虑调试已编译代码所带来的困难。

以下是使用 `pmap` 时遇到的常见问题以及应对策略：

### 常见问题及如何识别它们

1. **形状不匹配**：这可能是最常见的错误原因。

   - **输入数据分割**：`pmap` 沿着 `in_axes` 指定的轴分割输入数组。该轴的大小*必须*等于 JAX 用于 `pmap` 执行的设备数量（通常是 `jax.local_device_count()`，除非使用多主机设置）。如果你有 4 个 GPU 并使用 `in_axes=0` 对一个数组进行映射，则该数组的第一个维度必须是 4。不匹配将在分发阶段引起错误。
   - **输出数据收集**：同样，每个设备的运行结果会沿着 `out_axes` 指定的轴堆叠。如果每个设备上的函数生成的输出形状与 `pmap` 预期的收集方式不一致，则可能发生错误。确保每个设备生成的形状在堆叠时是合理的。
   - **复制参数 (parameter)与分片参数**：`in_axes` 中对应 `None` 的参数会被复制，而不是分割。确保你正确区分应该在设备间分割的数据（分片）和应该在所有设备上相同的数据（复制）。将预期分片的数据设置为 `in_axes=None`（反之亦然）会导致每个设备计算内部的形状不正确。
2. **集合操作问题**：像 `lax.psum`、`lax.pmean`、`lax.all_gather` 这样的函数会协调 `pmap` 中所有设备间的操作。

   - **`axis_name` 不匹配**：集合函数中使用的 `axis_name`（例如 `lax.psum(x, axis_name='my_devices')`）必须与 `pmap` 调用中提供的名称（例如 `pmap(..., axis_name='my_devices')`）完全一致。拼写错误很常见。
   - **程序挂起**：如果并非所有设备都执行到集合操作，程序可能会无限期挂起等待其他设备。这通常发生在 `pmap` 化函数中的条件逻辑导致某些设备跳过集合调用时。请记住，在 SPMD 下，*所有*设备都执行相同的代码；如果预期并谨慎管理分歧，条件逻辑通常应仅依赖于复制的值或设备 ID (`lax.axis_index`)。
   - **结果不正确**：确保每个设备输入到集合操作的数据是你所期望的。如果设备为集合操作的输入数据类型或形状不一致，可能会导致错误或数学上不正确的结果。
3. **跟踪和编译错误**：由于 `pmap` 会编译函数，你可能会遇到与 `jit` 类似的错误。

   - **抽象追踪器**：在跟踪期间依赖输入*实际值*的操作（例如在标准 Python `if` 语句中根据 JAX 数组值进行条件判断）会引发问题。请回顾 `jit` 的调试技巧，例如使用 `jax.lax.cond` 或确保条件操作作用于静态参数。
   - **副作用**：期望函数是纯函数。`pmap` 化函数内部的副作用（例如修改外部变量或使用标准 Python `print` 打印）行为不可预测或导致错误。

### `pmap` 的调试策略

调试 `pmap` 通常涉及简化问题，并仔细检查数据在主机与设备之间或设备之间移动的边界。

1. **简化：首先移除 `pmap`**：在调试并行版本之前，确保底层函数在单个设备上使用*单个数据分片*能正常工作。

   - 取批量输入数据的一个切片（即用于单个设备的部分）。
   - 在这个单个切片上运行你的函数，*不使用* `pmap`（可能最初也不使用 `jit`）。
   - 首先在此处修复任何逻辑错误。这可以将核心函数问题与并行化问题区分开来。
2. **减少设备数量**：如果单切片版本正常工作，尝试使用尽可能少的设备运行 `pmap`（甚至只有 1 或 2 个，如果你的硬件允许且逻辑不依赖于特定数量）。如果错误只在多设备情况下出现，则更可能与数据分发、收集或集合操作有关。
3. **仔细检查形状**：在 `pmap` 化函数*外部*广泛使用 `.shape`。

   - 在调用 `pmap` 之前，验证*所有*输入参数 (parameter)的形状。`in_axes` 中指定的维度是否与 `jax.local_device_count()` 匹配？
   - 验证 `pmap` 返回的输出形状。它是否与你基于每个设备输出形状和 `out_axes` 规范的预期相符？

   ```python
   import jax
   import jax.numpy as jnp
   from jax import pmap

   # 示例：假设我们有 4 个设备
   num_devices = 4 # 实际使用中替换为 jax.local_device_count()

   # 旨在跨 4 个设备分割的数据
   sharded_data = jnp.arange(4 * 10).reshape((num_devices, 10)) 

   # 旨在复制到所有设备上的数据
   replicated_data = jnp.array(5.0) 

   def my_func(x, y):
       # x 是分片的（每个设备的形状是 (10,)），y 是复制的（标量）
       return x * y + lax.axis_index('batch') # 使用设备 ID

   pmapped_func = pmap(my_func, in_axes=(0, None), out_axes=0, axis_name='batch')

   print("分片数据形状:", sharded_data.shape)
   # 预期: (4, 10) - 检查第一维度是否与设备数量匹配
   assert sharded_data.shape[0] == num_devices 

   print("复制数据形状:", replicated_data.shape) 
   # 预期: () 对于标量，或者轴 0 不与设备数量匹配的形状

   output = pmapped_func(sharded_data, replicated_data)

   print("输出形状:", output.shape)
   # 预期: (4, 10) - 因为 out_axes=0 将每个设备的 (10,) 结果堆叠起来
   ```
4. **使用 `jax.debug` 检查内部值**：`pmap` 内部的标准 Python `print()` 会在*每个设备*上独立执行，并且通常异步打印到主机，导致输出交错且难以追踪。它还可能干扰执行。请改用 JAX 的调试工具：

   - `jax.debug.print`：打印编译/`pmap` 化函数中的值，并用设备源标记 (token)输出。它比 `print()` 更好地处理同步。你可以根据设备 ID (`lax.axis_index`) 有条件地使用它来减少干扰。

   ```python
   import jax
   import jax.numpy as jnp
   from jax import pmap, lax

   def func_with_debug_print(x):
       intermediate = x * 2
       # 仅从设备 0 打印中间值
       if lax.axis_index('data_axis') == 0:
           jax.debug.print("设备 0 中间值: {val}", val=intermediate)
       # 集合操作
       result = lax.psum(intermediate, axis_name='data_axis') 
       jax.debug.print("设备 {id} psum 后结果: {res}", id=lax.axis_index('data_axis'), res=result)
       return result

   # 假设有 2 个设备
   data = jnp.arange(2 * 3).reshape((2, 3)) 
   pmapped_func = pmap(func_with_debug_print, axis_name='data_axis')

   # 执行时，你将看到带标签的打印输出
   output = pmapped_func(data) 
   # 输出可能显示：
   # 设备 0 中间值: [0 2 4]
   # 设备 0 psum 后结果: [ 6  8 10]
   # 设备 1 psum 后结果: [ 6  8 10] 
   ```

   - `jax.debug.breakpoint()`：命中时会暂停*所有设备*上的执行，允许通过 pdb 进行检查。谨慎使用，因为它会停止所有操作。
5. **仔细测试集合操作**：如果你怀疑某个集合操作导致问题：

   - 创建一个最小的 `pmap` 化函数，该函数*只*使用简单输入数据执行该集合操作。
   - 使用 `jax.debug.print` 查看每个设备上集合操作的输入及其输出。
   - 仔细检查 `axis_name`。
6. **考虑暂时禁用 JIT**：尽管 `pmap` 需要编译，但如果你遇到在使用 `pmap` 时出现的模糊编译错误，而单独使用 `jit` 时不会出现这些错误，你*可以*在调试期间，在你调用 `pmap` 的周围有限范围内尝试使用 `jax.disable_jit()`。这会强制执行不同的路径，这*可能*会产生更容易理解的 Python 错误，但请注意，这*不是* `pmap` 正常运行的方式，并且可能会掩盖真实问题或引入不同的行为。将此作为诊断的最后手段，而非解决方案。

调试 `pmap` 需要耐心和系统的方法。通过简化问题、仔细检查数据形状以及使用 `jax.debug.print` 等适当的调试工具，你可以有效地定位并解决与多设备并行执行相关的问题。

## 参考资料

- [Parallelism with \`pmap\`](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html) — The JAX Developers (2024)
  JAX关于`pmap`多设备执行的官方教程。
- [Debugging JAX Code](https://jax.readthedocs.io/en/latest/debugging/index.html) — The JAX Developers (2024)
  JAX官方指南，介绍JAX调试工具和方法。
- [XLA: Optimizing Compilers for Machine Learning](https://www.tensorflow.org/xla) — Alexey Radul, Sam Gross, Chris Lattner (2023)
  Journal: Google AI Blog; Publisher: Google
  Google AI博客文章，介绍JAX所用编译系统XLA。
- [SPMD (Single Program, Multiple Data)](https://hpc-tutorials.llnl.gov/mpi/) — Lawrence Livermore National Laboratory (LLNL) ()
  Publisher: Lawrence Livermore National Laboratory
  SPMD并行编程模型的简明解释。

---

[上一节](06-%E5%B0%86%20%60pmap%60%20%E4%B8%8E%E5%85%B6%E4%BB%96%E5%8F%98%E6%8D%A2%E7%BB%93%E5%90%88%E4%BD%BF%E7%94%A8.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97.md)

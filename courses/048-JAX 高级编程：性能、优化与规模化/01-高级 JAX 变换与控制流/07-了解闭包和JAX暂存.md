# 了解闭包和JAX暂存

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow/closures-jax-staging)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当你构建更复杂的JAX函数时，特别是那些涉及嵌套函数定义或使用Flax或Haiku等库的构造时，你将不可避免地遇到Python闭包。了解JAX在其追踪或“暂存”过程中如何与闭包交互，有助于编写正确和高效的代码，尤其是在使用`jax.jit`等转换时。

### Python闭包：快速回顾

在Python中，当一个嵌套函数引用其包含（外层）函数作用域中的变量时，就会发生闭包。嵌套函数“捕获”这些变量，这意味着即使外部函数执行完毕后，它仍能访问它们。

考虑这个标准Python示例：

```python
def create_multiplier(factor):
  """定义因子的外部函数。"""
  def multiplier(x):
    """使用外部作用域中因子的内部函数。"""
    return x * factor # 'factor' 从外部作用域捕获
  return multiplier

# 分别创建乘以2和乘以10的函数
multiply_by_2 = create_multiplier(2)
multiply_by_10 = create_multiplier(10)

print(f"multiply_by_2(5) = {multiply_by_2(5)}") # 输出: 10
print(f"multiply_by_10(5) = {multiply_by_10(5)}") # 输出: 50
```

在这里，内部函数`multiplier`形成一个闭包。它从`create_multiplier`作用域捕获了`factor`变量。每次调用`create_multiplier`都会创建一个*新*闭包，并带有其自己捕获的`factor`值。

### JAX暂存和值捕获

`jax.jit`、`jax.vmap`或`jax.grad`等JAX转换不会每次都直接运行你的Python代码。相反，它们首先执行一个称为**暂存**或**追踪**的过程。在追踪期间，JAX使用表示输入的抽象值（追踪器）执行你的Python函数。它记录了对这些追踪器执行的基本操作序列，构建一个称为**jaxpr**的中间表示。然后这个jaxpr会被编译（例如，通过XLA用于`jit`）成针对目标加速器（GPU/TPU）的优化代码，或者用于计算梯度或向量 (vector)化操作。

那么，当JAX追踪一个包含闭包的函数时，会发生什么？当追踪器遇到捕获变量的内部函数时，JAX会*在追踪时*捕获该被捕获变量的*当前值*，并将其嵌入 (embedding)到jaxpr中，通常作为一个常量。

> 可视化显示了`factor`的值（5）在JAX追踪期间是如何被捕获的，并成为生成的jaxpr和编译代码中的一个常量。

### JIT编译和行为的影响

这种值捕获具有重要影响：

1. **编译代码中的常量**：被`jax.jit`装饰的函数所捕获的变量在编译代码中通常被视为常量。编译后的函数会针对追踪期间捕获的特定值进行专门化。
2. **陈旧值**：如果被捕获的变量在函数被JIT编译*之后*在Python环境中改变了其值，编译后的函数将*不会*看到这种变化。它会继续使用在初始追踪期间捕获的值。
3. **潜在的重新编译**：如果你使用不同的捕获值调用从像`create_multiplier`这样的闭包工厂派生出的JIT编译函数（例如，`jax.jit(create_multiplier(5))`然后`jax.jit(create_multiplier(10))`），JAX会为它遇到的每一个不同的捕获值追踪并编译一个新版本的函数。如果捕获的值是一个复杂的Python对象或以JAX无法作为静态追踪的方式变化，这可能导致频繁的重新编译，抵消`jit`的优点。

让我们看看这种“陈旧值”行为的实际表现：

```python
import jax
import jax.numpy as jnp

scale_factor = 2.0 # 一个全局范围内的变量

def apply_scale(x):
  # 此函数捕获了全局变量 'scale_factor'
  return x * scale_factor

# JIT编译此函数。在追踪期间，它捕获了 scale_factor=2.0
jitted_apply_scale = jax.jit(apply_scale)

# 第一次调用使用捕获的值
input_array = jnp.arange(3.)
print(f"Initial call: {jitted_apply_scale(input_array)}") # 预期结果: [0. 2. 4.]

# 现在，在编译*之后*改变全局变量
print("将 scale_factor 更改为 100.0")
scale_factor = 100.0

# 再次调用JIT编译的函数。它仍然使用*原始*捕获的值！
print(f"Second call: {jitted_apply_scale(input_array)}") # 预期结果: [0. 2. 4.] (不是 [0., 100., 200.])

# 要使用新值，你需要重新追踪和重新编译
jitted_apply_scale_new = jax.jit(apply_scale)
print(f"Call after re-jitting: {jitted_apply_scale_new(input_array)}") # 预期结果: [  0. 100. 200.]
```

这个示例清晰地表明，`jitted_apply_scale`函数为`scale_factor = 2.0`进行了专门化，并没有对后来全局变量的变化做出反应。

### 处理闭包的最佳实践

鉴于这种行为，这里有一些指导原则：

1. **优先使用显式参数 (parameter)处理动态值**：如果函数中使用的值需要在不同调用之间改变，将其作为显式函数参数传递，而不是依赖闭包。这使得依赖关系明确，并避免陈旧值或意外的重新编译。

```python
import jax
import jax.numpy as jnp

# 处理动态比例因子的优选方法
@jax.jit
def apply_scale_arg(x, factor):
  return x * factor

input_array = jnp.arange(3.)
print(f"Call with factor=2.0: {apply_scale_arg(input_array, 2.0)}")
print(f"Call with factor=100.0: {apply_scale_arg(input_array, 100.0)}")
```

JAX高效处理变化的参数，通常在数组参数只有*值*发生变化而形状和数据类型保持一致的情况下无需重新编译。

2. **将闭包用于静态配置**：闭包完全可以接受，并且通常很方便用于捕获在编译函数整个生命周期内保持不变的配置值，例如层大小、激活函数 (activation function)或固定超参数 (hyperparameter)。

```python
import jax
import jax.nn as nn

def create_dense_layer(output_size):
  # 'output_size' 是配置，在层创建后不太可能改变
  def apply_layer(params, x):
    # 假设params是一个字典 {'W': ..., 'b': ...}
    # 使用 params['W'], params['b'] 的实际层逻辑
    # 捕获的 'output_size' 可能用于形状断言或其他逻辑
    assert params['W'].shape[1] == output_size
    y = jnp.dot(x, params['W']) + params['b']
    return nn.relu(y) # 激活函数也隐式捕获了
  return apply_layer

# JIT编译 apply_layer 是可以的，output_size 成为专门化的一部分
# layer10 = create_dense_layer(10)
# jitted_layer10 = jax.jit(layer10)
```

3. **注意可变对象**：如果你打算在JAX函数追踪后修改可变Python对象（如列表或字典），请避免捕获它们。追踪的函数会在*追踪时*捕获对该对象的引用，但它通常不会像你在标准Python中可能预期那样看到对象内容的后续修改。从编译函数的角度来看，将捕获的值视为实际上不可变的。

了解JAX的暂存机制如何与Python的词法闭包交互，对于避免细微的错误和性能问题是基本的。通过认识到JAX在追踪时捕获值，你可以更有效地设计你的函数，主要通过将动态数据显式地作为参数传递，同时审慎地使用闭包进行静态配置。这种明确性可确保你的编译函数按预期运行并高效执行。

## 参考资料

- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056355/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: 1014
  对高级 Python 特性（包括闭包和词法作用域）进行了全面解释。
- [JAX: Autograd and XLA, Accelerated](https://ieeexplore.ieee.org/document/8693259) — Zachary Devito, Sam Gross, Matthew Johnson, Daniel J. Seita, Peter J. Battaglia, Alex Botev, Alexey Goldin, Siddharth Goel, George Hotz, Sherjil Ozair, James S. Smith, Balaji Sreenivasan, Austin Stone, Matthew Tucker, Charles R. Harris, Kevin J. Beaumont, Bart van Merriënboer, Joshua V. Dillon, David So, Ryan P. Adams, and Sander Dieleman (2019)
  Journal: 2019 USENIX Conference on Systems for Machine Learning (SysML); Publisher: IEEE; Pages: 1-13; DOI: [10.1109/SysML.2019.00003](https://doi.org/10.1109/SysML.2019.00003)
  描述了 JAX 的基础设计，详细说明了其追踪机制和与 XLA 的集成以实现加速计算。
- [JAX Documentation: JIT compilation](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-compilation) — JAX core contributors (2023)
  解释了 JAX 的 jit 转换如何工作，包括其追踪行为以及对 Python 值（例如闭包中捕获的值）的影响。

---

[上一节](06-%E9%AB%98%E7%BA%A7%E6%8E%A9%E7%A0%81%E6%8A%80%E6%9C%AF.md) · [下一节](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%A4%8D%E6%9D%82%E7%9A%84%E5%BE%AA%E7%8E%AF%E9%80%BB%E8%BE%91.md)

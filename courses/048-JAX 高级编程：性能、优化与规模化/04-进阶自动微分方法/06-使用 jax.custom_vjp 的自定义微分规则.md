# 使用 jax.custom_vjp 的自定义微分规则

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-4-advanced-automatic-differentiation/custom-vjp-rules)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 `jax.grad` 和底层的 `jax.vjp` 机制能够处理许多标准 JAX 运算的自动微分，但在某些情况下，您需要更多地控制梯度的计算方式。JAX 提供了 `jax.custom_vjp` 装饰器，让您可以为自己的 Python 函数定义定制的向量 (vector)-雅可比乘积 (VJP) 规则。

为什么会需要自定义 VJP？以下是一些常见情形：

1. **数值稳定性**：函数中标准的运算序列在自动微分时可能会导致数值不稳定的梯度，尤其是在边界条件附近或对于大的输入值。您可能知道一种替代的、数学上等效的梯度公式，该公式更为稳定。
2. **性能优化**：自动推导的 VJP 可能涉及比必要更多的计算或内存访问。如果您知道一个更高效的解析梯度，直接实现它可以加速反向传播 (backpropagation)。
3. **数学简化**：有时，解析梯度比自动微分所推导出的梯度明显更简单，可能涉及 JAX 无法自动识别的抵消或恒等式。
4. **隐式函数**：对于隐式定义的函数（例如，通过优化问题或求解器定义），您可能通过隐函数定理得到梯度，即使直接微分求解器步骤很困难或效率低下。
5. **近似**：您可能希望为特定目的使用近似梯度，例如在某些优化算法中或为了正则化 (regularization)行为（尽管 `jax.lax.stop_gradient` 通常用于简单地阻止梯度传播）。

### `jax.custom_vjp` 装饰器

`@jax.custom_vjp` 装饰器允许您将函数与自定义的前向和反向传播 (backpropagation)实现关联起来，以进行反向模式自动微分。

我们来分析一下它的工作原理。假设您有一个要自定义的函数 `f`：

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def f(x, y):
  # 原始实现
  result = x * jnp.sin(y)
  return result
```

为了实现这一点，您需要定义两个辅助函数：

1. `f_fwd(x, y)`：此函数定义了前向传播。它计算原始函数的输出（`primals_out`）以及反向传播所需的任何中间值（`residuals`）。它必须返回 `(primals_out, residuals)`。
2. `f_bwd(residuals, cotangents_in)`：此函数定义了反向传播（VJP）。它接收 `f_fwd` 保存的 `residuals` 以及传入的余切向量 (vector)（`cotangents_in`，它表示最终损失相对于函数 `f` *输出* 的梯度）。它必须返回一个包含相对于 `f` *原始输入* 的梯度的元组（在此情况下，为 `(dL/dx, dL/dy)`）。返回元组中的元素数量必须与 `f` 的原始输入数量匹配。

然后，您使用 `f.defvjp(f_fwd, f_bwd)` 将这些函数与原始的 `f` 关联起来。

以下是完整的结构：

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def f(x, y):
  # 这个原始实现通常很简单，
  # 代表了数学函数。
  # 如果 f_fwd 有效地计算了结果，
  # JAX 的 AD 系统甚至可能不会直接调用它。
  return x * jnp.sin(y)

# 前向传播：计算结果并保存必要的值
def f_fwd(x, y):
  # 计算原始输出
  primals_out = f(x, y) # 或重新计算: x * jnp.sin(y)
  # 保存反向传播所需的值
  # 在此情况，我们需要 x、y 和 sin(y) 来计算梯度
  residuals = (x, y, jnp.sin(y), jnp.cos(y)) # 可以在这里或在反向传播中计算 cos(y)
  return primals_out, residuals

# 反向传播：计算相对于输入的梯度
def f_bwd(residuals, cotangents_in):
  # 解包残差
  x, y, sin_y, cos_y = residuals
  # g 表示传入的梯度 dL/d(f(x,y))
  g = cotangents_in

  # 计算相对于 x 的梯度: dL/dx = dL/df * df/dx
  # df/dx = sin(y)
  grad_x = g * sin_y

  # 计算相对于 y 的梯度: dL/dy = dL/df * df/dy
  # df/dy = x * cos(y)
  grad_y = g * x * cos_y

  # 返回对应于输入 (x, y) 的梯度
  return (grad_x, grad_y)

# 向 f 注册前向和反向函数
f.defvjp(f_fwd, f_bwd)

# 使用示例：
key = jax.random.PRNGKey(0)
x_val = jnp.float32(2.0)
y_val = jnp.pi / 2.0

# 使用 f 计算函数的梯度
value_and_grad_fn = jax.value_and_grad(lambda x, y: f(x, y)**2, argnums=(0, 1))
value, grads = value_and_grad_fn(x_val, y_val)

print(f"函数值: {value}")
print(f"梯度 (dL/dx, dL/dy): {grads}")

# 对比自动微分进行验证（如果 f 的原始实现可微分）
value_and_grad_fn_auto = jax.value_and_grad(lambda x, y: (x * jnp.sin(y))**2, argnums=(0, 1))
value_auto, grads_auto = value_and_grad_fn_auto(x_val, y_val)

print(f"自动函数值: {value_auto}")
print(f"自动梯度 (dL/dx, dL/dy): {grads_auto}")
```

当 `jax.grad`（或 `jax.vjp`）遇到对 `f` 的调用时，它不会微分 `f` 的原始 Python 代码。相反，它在前向传播期间执行 `f_fwd`，并存储 `residuals`。在反向传播期间，它检索 `residuals` 并调用 `f_bwd` 传入 `cotangents_in`，以获得相对于 `x` 和 `y` 的梯度。

### 示例：数值稳定的 `log(1 + exp(x))`

自定义 VJP 有用的一种典型情况是函数 $f(x) = \log(1 + e^x)$，通常称为 `softplus`。对于大的正值 $x$， $e^x$ 可能会超出标准浮点表示的范围。即使 $e^x$ 没有溢出，使用自动微分计算梯度也涉及计算 $e^x / (1 + e^x)$，它趋近于 1。这可以重写为 $1 / (1 + e^{-x})$，即 sigmoid 函数 $\sigma(x)$。这种形式对于大的正值 $x$ 来说数值稳定。

我们来使用 `jax.custom_vjp` 实现它：

```python
import jax
import jax.numpy as jnp
import numpy as np # 用于比较

@jax.custom_vjp
def stable_log1pexp(x):
  # 原始函数实现
  return jnp.log1p(jnp.exp(x))

def stable_log1pexp_fwd(x):
  # 前向传播：计算输出并保存 x 用于反向传播
  exp_x = jnp.exp(x)
  output = jnp.log1p(exp_x)
  # 我们只需要 x 来计算稳定的梯度（sigmoid(x)）
  residuals = (x,)
  return output, residuals

def stable_log1pexp_bwd(residuals, cotangent_in):
  # 反向传播：使用稳定公式计算梯度
  (x,) = residuals
  # 梯度为 sigmoid(x) * cotangent_in
  # sigmoid(x) = 1 / (1 + exp(-x))
  grad_x = cotangent_in * (1. / (1. + jnp.exp(-x)))
  # 返回对应于输入 x 的梯度（作为元组）
  return (grad_x,)

# 注册自定义 VJP 规则
stable_log1pexp.defvjp(stable_log1pexp_fwd, stable_log1pexp_bwd)

# --- 比较 ---

# 使用标准 JAX 运算的函数
def unstable_log1pexp(x):
    return jnp.log(1.0 + jnp.exp(x))

# 梯度函数
grad_stable = jax.grad(stable_log1pexp)
grad_unstable = jax.grad(unstable_log1pexp)

# 测试值
x_small = jnp.float32(1.0)
x_large = jnp.float32(100.0) # exp(100) 会导致 float32 溢出
x_problematic = jnp.float32(35.0) # exp(35) 很大但可能不会使 float32 溢出，但 1+exp(x) 可能不准确

print(f"--- 输入: {x_small} ---")
print(f"稳定梯度:   {grad_stable(x_small)}")
print(f"不稳定梯度: {grad_unstable(x_small)}")

print(f"\n--- 输入: {x_problematic} ---")
print(f"稳定梯度:   {grad_stable(x_problematic)}")
print(f"不稳定梯度: {grad_unstable(x_problematic)}") # 可能准确性较低

print(f"\n--- 输入: {x_large} ---")
print(f"稳定梯度:   {grad_stable(x_large)}")
# print(f"不稳定梯度: {grad_unstable(x_large)}") # 这很可能会因溢出而产生 NaN

# 注意：对于非常大的输入，不稳定版本会溢出。
# 稳定版本正确计算了梯度，其值趋近于 1。
try:
    unstable_result = grad_unstable(x_large)
    print(f"不稳定梯度: {unstable_result}")
except FloatingPointError:
    print("不稳定梯度: NaN（预期溢出）")

# JAX 实际内置了一个数值稳定的版本：
grad_jax_builtin = jax.grad(jax.nn.softplus)
print(f"\n--- JAX 内置 softplus 梯度 ({x_large}) ---")
print(f"内置梯度: {grad_jax_builtin(x_large)}")
```

在此示例中，`stable_log1pexp_fwd` 计算结果并保存输入 `x`。`stable_log1pexp_bwd` 函数使用保存的 `x` 和传入的 `cotangent_in`（当 `jax.grad` 直接调用时为 1.0）通过数值稳定的 sigmoid 公式计算梯度。比较表明，对于大的输入，自定义 VJP 保持稳定，而自动推导的梯度可能会出现溢出或精度问题。

### 注意事项

- **正确性**：最重要的方面是确保您的自定义 VJP 在数学上是正确的。`f_bwd` 计算的梯度必须与原始函数 `f` 的真实梯度匹配。针对数值微分或已知结果彻底测试您的实现。
- **残差**：只在 `residuals` 中保存最少必要的数据。保存大的中间数组会增加内存消耗。有时，如果值计算成本低廉且保存它们会消耗大量内存，那么在 `f_bwd` 中重新计算它们会更好。
- **组合**：使用 `jax.custom_vjp` 的函数通常可以与其他 JAX 转换（如 `jit`、`vmap` 甚至嵌套微分）组合使用，前提是 `f_fwd` 和 `f_bwd` 函数本身是 JAX 可追踪的，并遵循标准的 JAX 函数规则。
- **不可微分输入**：如果 `f` 接受不应进行微分的参数 (parameter)（例如，静态参数、非浮点类型），`f_bwd` 函数应在输出元组的相应位置返回 `None`。

`jax.custom_vjp` 是获得反向模式微分精细控制的强大工具。虽然日常使用中不常需要，但在处理数值稳定性、性能瓶颈或高级模型和算法中的特定梯度计算时，它变得非常宝贵。另请记住，如果您需要控制前向模式微分，或者如果 JVP 规则对于您的函数更易于定义，也请考虑 `jax.custom_jvp`。

---

[上一节](05-%E8%AE%A1%E7%AE%97%E5%AE%8C%E6%95%B4%E7%9A%84%E9%9B%85%E5%8F%AF%E6%AF%94%E7%9F%A9%E9%98%B5%E5%92%8C%E6%B5%B7%E6%A3%AE%E7%9F%A9%E9%98%B5.md) · [下一节](07-jax.custom_jvp%20%E8%87%AA%E5%AE%9A%E4%B9%89%E6%B1%82%E5%AF%BC%E8%A7%84%E5%88%99.md)

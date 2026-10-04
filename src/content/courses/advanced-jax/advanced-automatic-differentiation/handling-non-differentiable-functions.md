---
course: "advanced-jax"
chapter: "advanced-automatic-differentiation"
lesson: "handling-non-differentiable-functions"
sourceId: 2890
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-4-advanced-automatic-differentiation/handling-non-differentiable-functions"
title: "处理不可微分函数"
description: "处理 JAX 计算中不可微分运算的方法，包括使用 `jax.lax.stop_gradient`。"
order: 9
plots: []
sourceHash: "c7d56a95623820e2554e08c20ad2a935b16c6aa08d546c02a402587244caf021"
sourceCorrections: []
---

自动微分通过对组成函数的原始运算递归应用链式法则来完成。此过程依赖于每个原始运算都有明确的方法计算其对总梯度的贡献，通常通过其雅可比向量 (vector)积（JVP）或向量雅可比积（VJP）规则。然而，并非所有运算都可数学微分，或者它们可能作用于（例如整数）类型，而这些类型通常不定义微分。此外，有时您可能出于建模或性能考量，*选择*阻止梯度通过计算的某些部分。JAX 提供了处理这些情况的机制。

### 微分可能失败的原因

以下几种运算可能给自动微分带来挑战：

1. **数学不连续性：** 具有跳变或尖角的函数在不连续点没有唯一的导数。包括：

   - 舍入函数（`jnp.round`、`jnp.floor`、`jnp.ceil`）
   - 符号函数（`jnp.sign`）在零点
   - 比较运算符（`>`、`<`、`>=`、`<=`、`==`、`!=`），它们产生布尔结果。虽然布尔结果本身不可微分，但将它们用于控制流（`lax.cond`、`lax.while_loop`）或算术运算（它们可能被转换为0/1）时，可能会在被微分的函数中产生不连续性。
   - `jnp.argmax` 或 `jnp.argmin`：这些函数返回索引（整数），并且运算本身是不连续的。输入值的微小变化可能导致结果索引跳变。
2. **整数运算：** 微分通常定义在连续域（实数或复数）上。本质上涉及整数的运算，例如整数类型转换或根据计算出的整数值对数组进行索引，对于这些整数没有标准导数。
3. **未定义的梯度规则：** 一些 JAX 原始运算可能根本没有实现 JVP 或 VJP 规则，特别是那些不常用或实验性的运算。尝试通过它们进行微分通常会导致错误。

### JAX 的默认行为

当 JAX 在微分过程中遇到一个没有定义 VJP 或 JVP 规则的运算时，它通常会引发 `TypeError`。

对于一些数学上不可微分但常见的运算（例如 `jnp.round` 或整数类型转换），JAX 通常定义一个返回零的梯度规则。这是一个实用选择：它避免了错误，但在微分过程中，它有效地将该运算的输出视为相对于其输入的常量。这可能是预期的行为，但了解这一点很重要。例如，如果 `y = jnp.round(x)` 并且您计算 `jax.grad(lambda x: jnp.round(x))(x_val)`，您很可能会得到 `0.0`。

### 使用 `jax.lax.stop_gradient` 明确停止梯度

JAX 在这些情况下控制梯度流的主要工具是 `jax.lax.stop_gradient`。此函数行为简单：

- **前向传播：** 它作为恒等函数；`jax.lax.stop_gradient(x)` 简单地返回 `x`。
- **反向传播 (backpropagation)（VJP）/ 前向传播（JVP）：** 它阻止梯度流过它。无论传入的梯度信号如何，其 VJP 始终返回适当形状和类型的零。同样，其 JVP 也始终为零。

本质上，`jax.lax.stop_gradient` 告诉 JAX：“使用此值计算前向传播，但在任何微分传播过程中将此值视为常量。”

让我们看看这在实践中如何运作。考虑一个函数，我们希望在计算中使用一个值，但阻止梯度通过该值的计算反向流动。

```python
import jax
import jax.numpy as jnp

# 梯度正常流动的函数
def f_normal(x):
  y = jnp.sin(x)
  z = jnp.cos(x)
  return y * z # 适用乘积法则：d/dx (sin(x)cos(x)) = cos^2(x) - sin^2(x)

# 阻止梯度通过 cos(x) 的函数
def f_stopped(x):
  y = jnp.sin(x)
  # 出于微分目的，将 cos(x) 视为常量
  z = jax.lax.stop_gradient(jnp.cos(x))
  # 梯度行为类似于 d/dx (sin(x) * K) = cos(x) * K，其中 K 是 z 的值
  return y * z

grad_f_normal = jax.grad(f_normal)
grad_f_stopped = jax.grad(f_stopped)

x_val = jnp.pi / 4.0 # 45 度

print(f"x = {x_val:.3f}")
print(f"f_normal(x) = {f_normal(x_val):.3f}")
print(f"f_stopped(x) = {f_stopped(x_val):.3f}") # 前向传播相同

print("\n梯度：")
# 正常梯度：cos(2*x) = cos(pi/2) = 0
print(f"梯度（正常） = {grad_f_normal(x_val):.3f}")

# 停止梯度：cos(x) * stop_gradient(cos(x)) = cos(x_val) * cos(x_val)
# 在 x=pi/4 处评估：cos(pi/4) * cos(pi/4) = (1/sqrt(2)) * (1/sqrt(2)) = 0.5
print(f"梯度（停止） = {grad_f_stopped(x_val):.3f}")
```

在 `f_stopped` 中，`stop_gradient` 调用确保在反向传播过程中，没有梯度信号*流入* `jnp.cos(x)` 的计算。由 `jnp.cos(x)` 计算出的值 `z` 在前向传播中使用，但从微分的角度来看，它被视为一个预先计算的常量。因此，梯度计算实际上变为 $d/dx (\sin(x) \times \text{常量}) = \cos(x) \times \text{常量}$，其中常量是 `z` 的值。

### 何时停止梯度

在以下几种情况下，使用 `jax.lax.stop_gradient` 是合适的：

1. **数学必要性：** 当通过一个真正不可微分的运算（例如 `jnp.round`）应用自动微分时，使用 `stop_gradient` 可以明确此选择，尽管 JAX 可能会通过定义零梯度隐式地为某些函数执行此操作。
2. **建模假设：** 您可能希望在特定的梯度计算中将某些输入或中间值视为固定常量，即使它们是从可微分运算得出的。例如，在强化学习 (reinforcement learning)中的目标网络等算法中，您可能会根据另一个网络的*固定*输出来更新一个网络。`stop_gradient` 可以实现这一点。
3. **中断循环依赖：** 在展开循环计算时，您可能希望梯度只回溯有限的步数。`stop_gradient` 可以修剪梯度路径。
4. **性能优化：** 如果您知道通过某个路径的梯度为零或可忽略不计，但计算开销大，`stop_gradient` 可以避免通过该路径进行不必要的反向计算（尽管 XLA 的死代码消除通常能有效处理零梯度）。
5. **数值稳定性：** 在极少数情况下，梯度流经计算的某个部分可能会变得数值不稳定（NaN 或 Inf）。`stop_gradient` 可以隔离这些部分，尽管通常最好解决不稳定性的根本原因。

### 不可微分运算的替代方法

尽管 `stop_gradient` 提供零梯度，但这并非总是预期的行为。如果您需要一个非零的“梯度信号”来进行优化，尽管存在数学上的不可微分性（在量化 (quantization)神经网络 (neural network)或训练具有离散步骤的网络等领域很常见），您可以考虑：

- **平滑近似：** 用一个数学上相似的平滑函数替换不可微分函数。例如，使用具有大 `k` 值的 sigmoid 函数 `jax.nn.sigmoid(k * x)` 来近似阶跃函数。梯度将是明确定义的。
- **自定义微分规则：** 使用 `jax.custom_vjp` 或 `jax.custom_jvp`（本章前面已介绍）来定义自定义梯度行为。这里的一种常见技术是*直通估计器（STE）*，其中前向传播使用不可微分函数（例如，舍入），但反向传播 (backpropagation)使用替代函数（通常是恒等函数）的梯度，有效地将传入梯度直接传递过去。

```python
import jax
import jax.numpy as jnp

@jax.custom_vjp
def round_straight_through(x):
  """前向传播使用 jnp.round，反向传播是恒等函数。"""
  return jnp.round(x)

# 为 custom_vjp 定义前向和反向传播函数
def round_straight_through_fwd(x):
  # 前向传播返回原始输出和用于反向传播的残差
  return round_straight_through(x), None

def round_straight_through_bwd(residuals, g):
  # 反向传播接收传出梯度 'g' 并返回
  # 相对于输入的梯度。这里，我们直接传递 'g'。
  return (g,)

# 向原始函数注册前向和反向函数
round_straight_through.defvjp(round_straight_through_fwd, round_straight_through_bwd)

def ste_example(x):
    rounded_x = round_straight_through(x)
    return rounded_x * x # d/dx = rounded_x * 1 + d/dx(rounded_x) * x = rounded_x + 1*x

grad_ste = jax.grad(ste_example)

x_val = 2.7
print(f"\n直通估计器示例，x={x_val}")
print(f"ste_example({x_val}) = {ste_example(x_val):.3f}") # 使用 round -> 3.0 * 2.7 = 8.1
# 梯度：round(x) + x = 3.0 + 2.7 = 5.7
print(f"梯度 grad_ste({x_val}) = {grad_ste(x_val):.3f}")
```

此示例使用 `jax.custom_vjp` 为 `jnp.round` 实现了一个 STE。前向传播计算 `jnp.round(x)`，但 VJP 规则被定义为仅将传入梯度 `g` 作为相对于 `x` 的梯度返回，有效地在反向传播中使用恒等函数的梯度。这使得基于梯度的优化能够“穿透”舍入运算。

处理不可微分函数需要理解其数学局限性以及 JAX 提供的工具。`jax.lax.stop_gradient` 提供了一种直接阻止梯度流的方法，而平滑近似或自定义微分规则等技术在零梯度不足时提供更多控制。选择正确的方法取决于具体问题和优化过程中期望的行为。

## 参考资料

- [JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) — JAX core team (2024)
  JAX官方文档，解释其自动微分系统，包含处理不可微分操作的`stop_gradient`和`custom_vjp`。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://books.google.com/books/about/Evaluating_Derivatives.html?id=lW7z0L3bLzYC) — Andreas Griewank, Andrea Walther (2008)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); DOI: [10.1137/1.9781611975571](https://doi.org/10.1137/1.9781611975571)
  关于算法微分的学术著作，覆盖JVP/VJP的理论基础以及与不可微分性相关的问题。
- [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/abs/1308.3432) — Yoshua Bengio, Nicholas Léonard, and Aaron Courville (2013)
  Journal: arXiv preprint arXiv:1308.3432; DOI: [10.48550/arXiv.1308.3432](https://doi.org/10.48550/arXiv.1308.3432)
  提出了直通估计器等方法，用于通过离散或随机单元反向传播梯度，与自定义VJP示例相关。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了深度学习背景下自动微分的介绍，讨论了梯度计算以及不可微分激活函数或操作可能产生的问题。

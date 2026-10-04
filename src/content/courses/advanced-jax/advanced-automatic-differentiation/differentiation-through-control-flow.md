---
course: "advanced-jax"
chapter: "advanced-automatic-differentiation"
lesson: "differentiation-through-control-flow"
sourceId: 2885
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-4-advanced-automatic-differentiation/differentiation-through-control-flow"
title: "通过控制流原语求导"
description: "了解自动求导如何与`lax.scan`、`lax.cond`和`lax.while_loop`交互。"
order: 8
plots: []
sourceHash: "ad7b5a0e9f3224a8e4cea65e4c84c930e3f670e93bd61b92392a7916de96851c"
sourceCorrections: []
---

JAX中的自动求导与其结构化控制流原语（如`lax.cond`、`lax.while_loop`和`lax.scan`）紧密结合。然而，了解求导如何与这些操作交互对于编写高效代码和解释结果十分必要，特别是JAX需要在执行*前*追踪计算图。

### 对`lax.cond`求导

回顾一下，`lax.cond(pred, true_fun, false_fun, operand)`根据`pred`的布尔值执行`true_fun(operand)`或`false_fun(operand)`。当JAX在被求导的函数中（例如使用`jax.grad`时）遇到`lax.cond`时，它必须生成能够计算梯度的代码，无论实际执行时会选择哪个分支。

为此，JAX在求导过程中会追踪`true_fun`和`false_fun`这两个分支。生成的求导函数将包含两个分支的梯度逻辑。在执行期间，原始正向传播确定使用哪个分支的结果。反向传播 (backpropagation)随后计算实际执行的分支对应的梯度。

考虑这个示例：

```python
import jax
import jax.numpy as jnp
from jax import lax

def conditional_computation(x, y):
  # 条件取决于输入'y'
  return lax.cond(y > 0,
                  lambda op: jnp.sin(op * 2.0),  # true_fun
                  lambda op: jnp.cos(op / 2.0),  # false_fun
                  x)                         # operand

grad_x = jax.grad(conditional_computation, argnums=0)
grad_y = jax.grad(conditional_computation, argnums=1) # 注意：关于pred的梯度通常为零

# 示例执行
x_val = jnp.pi / 4.0
y_val_pos = 1.0
y_val_neg = -1.0

print(f"f(x, y>0) = {conditional_computation(x_val, y_val_pos)}")
print(f"df/dx(x, y>0) = {grad_x(x_val, y_val_pos)}")
# sin(2x)的梯度是2*cos(2x)。在x=pi/4时，2*cos(pi/2) = 0.0

print(f"\nf(x, y<0) = {conditional_computation(x_val, y_val_neg)}")
print(f"df/dx(x, y<0) = {grad_x(x_val, y_val_neg)}")
# cos(x/2)的梯度是-0.5*sin(x/2)。在x=pi/4时，-0.5*sin(pi/8) 大约 -0.191

# 关于'y'（谓词条件）的梯度通常为零
# 因为谓词本身通常被视为离散的。
print(f"\ndf/dy(x, y>0) = {grad_y(x_val, y_val_pos)}")
print(f"df/dy(x, y<0) = {grad_y(x_val, y_val_neg)}")
```

重要提示：

- 两个分支的代码都包含在编译后的求导函数中。
- 传递给`true_fun`和`false_fun`的操作数必须具有相同的结构（形状和数据类型），因为JAX需要一致的类型签名。
- 通过谓词`pred`本身的求导通常没有实际意义或不受支持，除非谓词计算涉及可求导操作且其输出以允许梯度的方式使用（这对于典型的布尔条件并不常见）。JAX通常对仅影响谓词的参数 (parameter)给出零梯度。

### 对`lax.while_loop`求导

原语`lax.while_loop(cond_fun, body_fun, init_val)`只要`cond_fun`返回`True`就会重复应用`body_fun`。对`while_loop`求导涉及将链式法则反向应用于正向传播期间执行的迭代。

这类似于用于训练循环神经网络 (neural network)（RNN）的反向传播 (backpropagation)（BPTT）。初始状态`init_val`的梯度取决于状态通过`body_fun`在*所有*迭代中如何演变。

```python
import jax
import jax.numpy as jnp
from jax import lax

def loop_sum(max_val):
  # 求和从0到（但不包括）max_val的数字
  init_state = (0, 0.0) # (当前i, 当前和)

  def cond_fun(state):
    i, _ = state
    return i < max_val # 当i < max_val时继续

  def body_fun(state):
    i, current_sum = state
    return (i + 1, current_sum + jnp.sqrt(i.astype(jnp.float32))) # 示例操作

  final_state = lax.while_loop(cond_fun, body_fun, init_state)
  _, final_sum = final_state
  return final_sum

# 对循环结果关于初始'max_val'输入求导
# 注意：max_val影响迭代的*次数*，使其梯度变得复杂。
# 为了清晰起见，我们对循环*内部*的某些东西求导。

def loop_sum_param(scale, n_iters):
    init_state = (0, 0.0) # (i, 当前和)
    def cond_fun(state):
        i, _ = state
        return i < n_iters
    def body_fun(state):
        i, current_sum = state
        # 在循环中使用'scale'
        return (i + 1, current_sum + scale * i)
    _, final_sum = lax.while_loop(cond_fun, body_fun, init_state)
    return final_sum

grad_loop_sum = jax.grad(loop_sum_param, argnums=0) # 关于'scale'的梯度

scale_val = 2.0
iters = 5
print(f"Loop sum(scale={scale_val}, iters={iters}) = {loop_sum_param(scale_val, iters)}")
# 正向传播：2*0 + 2*1 + 2*2 + 2*3 + 2*4 = 0 + 2 + 4 + 6 + 8 = 20

# 预期梯度 d(sum)/d(scale) = 0 + 1 + 2 + 3 + 4 = 10
print(f"d(Sum)/d(scale) = {grad_loop_sum(scale_val, iters)}")
```

重要提示：

- 反向传播需要正向传播期间为循环的每次迭代计算的中间值（原值）。
- 对于有许多迭代的循环，存储所有中间值可能会占用大量内存。这是使用梯度检查点（`jax.checkpoint`）的主要原因，该技术在反向传播期间重新计算中间值而不是存储它们。这是用计算换取内存的做法。
- 类似于深度网络或RNN，通过许多循环迭代传播的梯度可能会消失或爆炸，尽管循环体内的细致初始化或归一化 (normalization)等方法可以缓解这种情况。

### 对`lax.scan`求导

`lax.scan(f, init, xs)`原语在序列`xs`上累积应用函数`f`，并携带状态`init`。它通常用于实现RNN或其他已知步数的顺序过程。

通过`lax.scan`的求导是定义明确且高效的。JAX自动处理梯度通过携带状态（`carry`）和每步输出（`y`）的传播。与`lax.while_loop`类似，反向传播 (backpropagation)类似于BPTT。

```python
import jax
import jax.numpy as jnp
from jax import lax

def simple_rnn_step(carry, x_t):
  # 一个非常基本的RNN单元：carry是隐藏状态 h_t-1
  # x_t 是时间 t 的输入
  # 输出是新的隐藏状态 h_t 和一个输出 y_t
  prev_h = carry
  weight_hh = 0.5 # 为简单起见，固定参数
  weight_xh = 1.5 # 为简单起见，固定参数

  # 简单线性更新
  new_h = jnp.tanh(prev_h * weight_hh + x_t * weight_xh)
  y_t = new_h * 2.0 # 基于隐藏状态的一些输出
  return new_h, y_t # 新携带状态, y_t

def run_rnn(initial_state, inputs):
  # 初始状态：h_0
  # 输入：x_t 值的序列
  final_state, outputs_y = lax.scan(simple_rnn_step, initial_state, inputs)
  return jnp.sum(outputs_y) # 返回输出总和作为标量损失

grad_rnn_params = jax.grad(run_rnn, argnums=0) # 关于初始状态的梯度
grad_rnn_inputs = jax.grad(run_rnn, argnums=1) # 关于输入的梯度

h0 = jnp.zeros(()) # 初始隐藏状态（标量）
xts = jnp.array([0.1, 0.2, -0.1, 0.3]) # 输入序列

total_output = run_rnn(h0, xts)
print(f"Total output = {total_output}")

# 关于初始状态和输入的梯度
dh0 = grad_rnn_params(h0, xts)
dxts = grad_rnn_inputs(h0, xts)
print(f"d(Sum)/dh0 = {dh0}")
print(f"d(Sum)/dxts = {dxts}")
```

重要提示：

- 对于固定长度的序列，`lax.scan`通常优于`lax.while_loop`，因为其结构更明确，通常会带来更简单的分析和XLA可能进行的更多优化。
- `scan`的反向传播通过反向遍历正向传播中执行的步骤，使用保存的中间值，从而高效地计算梯度。
- 对于非常长的序列，内存使用仍然是一个问题，`jax.checkpoint`也可以应用于`scan`中使用的函数`f`。

> `lax.scan`求导的数据流。正向传播按顺序计算状态（$h$）和输出（$y$）。反向传播（VJP）使用正向传播中的中间值，反向传播梯度（$dL/d\cdot$）通过展开的计算。

总之，JAX的自动求导系统旨在与结构化控制流原语正确协同工作。尽管从用户角度看求导“直接可用”，但知道`cond`追踪两个分支，以及`while_loop`和`scan`的求导类似于BPTT，有助于理解内存使用、潜在的数值问题以及梯度检查点等技术的适用性。

## 参考资料

- [JAX Control Flow Primitives](https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-primitives) — JAX core contributors (2024)
  JAX官方文档，说明了JAX自动微分与`lax.cond`、`lax.while_loop`和`lax.scan`的交互方式，包括追踪行为和相关影响。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717761) — Andreas Griewank and Andrea Walther (2008)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); DOI: [10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
  自动微分的奠基性著作，涵盖了其底层数学原理和计算图视角，对于理解自动微分如何处理控制流至关重要。
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) — Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin (2016)
  Journal: arXiv preprint arXiv:1604.06174; DOI: [10.48550/arXiv.1604.06174](https://doi.org/10.48550/arXiv.1604.06174)
  介绍了梯度检查点（gradient checkpointing）的实际应用，以减少深度神经网络训练期间的内存消耗，与处理长`lax.while_loop`和`lax.scan`操作中的内存问题直接相关。

---
course: "getting-started-with-jax"
chapter: "managing-state-in-jax"
lesson: "using-pytrees-structured-state"
sourceId: 1806
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-6-managing-state-in-jax/using-pytrees-structured-state"
title: "使用 PyTree 管理分层状态"
description: "处理嵌套的 Python 容器（列表、元组、字典）作为状态对象。"
order: 4
plots: []
sourceHash: "8c9da24c3be0982badfdfd0258d102872869fc8cf77a7ad66dc1385c6935e2e0"
sourceCorrections: []
---

"在上一节中，我们看到了通过显式地将状态传入函数并接收其更新版本作为输出的状态管理方式。这种方式对于简单状态，例如单个计数器，效果不错。但应用程序常涉及更复杂的状态。考虑神经网络 (neural network)的参数 (parameter)，它们通常包含多个权重 (weight)矩阵和偏置 (bias)向量 (vector)，并经常按层级排列。同样，优化器状态可能包含动量值或每个参数的自适应学习率。手动传递和返回几十个单独的数组会很繁琐且容易出错。"

JAX 提供了 PyTree 机制来高效管理复杂状态。PyTree 不是您可以从 JAX 导入的特定数据类型或类。相反，它只是 JAX 用来指代的一种由标准 Python 容器构建的树状结构。被识别为 PyTree 节点的最常见容器是列表、元组和字典。树的“叶子”通常是 JAX 数组或其他非容器对象。

看一个简单例子，表示两层线性模型的参数：

```python
import jax.numpy as jnp

params = {
    'layer1': {
        'weights': jnp.ones((3, 2)),
        'bias': jnp.zeros((2,))
    },
    'layer2': {
        'weights': jnp.ones((2, 1)),
        'bias': jnp.zeros((1,))
    }
}
```

这个嵌套字典 `params` 就是一个 PyTree。这些字典（`params`、`params['layer1']`、`params['layer2']`）充当内部节点，而 JAX 数组（`jnp.ones(...)`、`jnp.zeros(...)`）则是叶子。

> 嵌套字典 `params` 作为 PyTree 的视觉图示。字典是内部节点，JAX 数组是叶子。

PyTree 的重要性在于 JAX 的函数变换如何与它们协同工作。像 `jax.jit`、`jax.grad`、`jax.vmap` 和 `jax.pmap` 这样的函数都被设计为对 PyTree 进行操作。当你将这些变换应用于接受或返回 PyTree 的 Python 函数时，JAX 会自动遍历树结构，将核心逻辑应用于叶节点（如 JAX 数组），同时保持容器结构不变。

我们来设想一个简化的 `update_params` 函数，它接收 `params` PyTree 和（结构相同的）梯度，并执行一个梯度下降 (gradient descent)步骤：

```python
import jax
import jax.numpy as jnp

# 假设 'params' 如上文所示定义
# 假设 'grads' 是一个与 'params' 结构相同的 PyTree，包含梯度

def update_params(params, grads, learning_rate):
  # 此函数使用梯度更新权重和偏置
  # 我们需要将更新应用于 params 树中的每个叶子（数组）
  # JAX 提供了相关工具，例如 jax.tree_util.tree_map
  def sgd_update(param, grad):
    return param - learning_rate * grad

  # 将 sgd_update 函数应用于 params 和 grads 中的每个叶子对
  updated_params = jax.tree_util.tree_map(sgd_update, params, grads)
  return updated_params

# 示例用法（这里的梯度仅为占位符）
grads = {
    'layer1': {'weights': jnp.full((3, 2), 0.1), 'bias': jnp.full((2,), 0.01)},
    'layer2': {'weights': jnp.full((2, 1), 0.2), 'bias': jnp.full((1,), 0.02)}
}

learning_rate = 0.01
new_params = update_params(params, grads, learning_rate)

# 一个重要点是，我们可以直接 JIT 编译此函数
jitted_update_params = jax.jit(update_params)
new_params_jitted = jitted_update_params(params, grads, learning_rate)

# JAX 在编译和执行期间会自动处理 PyTree 结构。
# new_params 和 new_params_jitted 将具有相同的嵌套字典结构
# 并包含相同的更新数值。
```

在此示例中，`jax.tree_util.tree_map` 被显式使用，将 `sgd_update` 函数应用于 `params` 和 `grads` PyTree 中对应的叶子。`tree_map` 接收一个函数和一个或多个 PyTree，将函数逐元素地应用于 PyTree 的叶子，并返回一个具有相同结构、包含结果的新 PyTree。

请注意，我们可以直接将 `jax.jit` 应用于 `update_params`。JAX 知道 `params` 和 `grads` 是 PyTree。在追踪期间，它会识别叶节点（数组），并为这些叶子编译在 `sgd_update` 中定义的操作。容器结构（字典和键）会被保留并自动处理。您无需手动将参数展平为列表、执行更新，然后再重建嵌套字典。

这种透明性也适用于其他变换。如果您对以 `params` 为输入的损失函数 (loss function)使用 `jax.grad`，则生成的梯度将自动具有与 `params` 相同的 PyTree 结构。如果您使用 `jax.vmap` 处理一批数据，并且您的函数返回了一个激活值的 PyTree，那么 `vmap` 会处理激活值 PyTree 叶子上的批处理。

尽管 `jax.tree_util` 包含像 `tree_map`、`tree_leaves`（将所有叶子作为扁平列表获取）和 `tree_unflatten`（从叶子和结构定义重建树）这样的函数，但当您只是通过变换后的函数传递状态时，通常无需直接与它们交互。它们的主要作用是当您需要在自己的函数逻辑中显式地对 PyTree 的叶子进行操作时，如 `update_params` 示例所示。

使用 PyTree，您可以用标准的 Python 字典、列表和元组以自然易读的方式组织复杂状态，例如模型参数或优化器状态。JAX 在其变换系统中隐式处理这些结构的能力是一个重要的便利，它使得编写用于复杂计算的简洁、函数式代码变得容易得多，同时仍能从编译、自动微分和向量化 (quantization)中获益。它弥合了 Python 灵活数据结构与 JAX 高性能函数式核心之间的差异。

## 参考资料

- [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html) — JAX developers (2024)
  JAX官方文档，解释了PyTree的概念、其结构以及JAX变换如何与它们交互。
- [\`jax.tree_util\` API documentation](https://jax.readthedocs.io/en/latest/jax.tree_util.html) — JAX developers (2024)
  提供了用于操作和检查PyTree（例如`tree_map`、`tree_leaves`和`tree_unflatten`）函数的详细API参考。

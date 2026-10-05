# JAX 生态系统库（Flax, Haiku）简介

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-6-large-scale-model-training-jax/introduction-jax-ecosystem-libraries)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 JAX 提供了强大的核心原语（`jit`、`grad`、`vmap`、`pmap`），用于加速和分布式数值计算，但仅使用这些工具构建、训练和管理大型、复杂的机器学习 (machine learning)模型可能会变得繁琐。处理可能复杂的状，例如模型参数 (parameter)、优化器统计数据和随机数生成器（RNG）种子，需要仔细管理，尤其是在多个设备上分发计算时。

基于 JAX 构建的神经网络 (neural network)库便派上用场。它们提供更高层次的抽象，用于定义模型架构、管理状态和组织训练循环，让您更专注于模型设计和训练逻辑，而非低层次的实现细节。JAX 生态系统中有两个主要库：Flax 和 Haiku。它们提供不同的编程风格，但共同目标都是简化 JAX 框架内复杂模型的开发。

### Flax：结构化模块与显式状态

Flax 由 Google 开发，提供以 `linen` 模块（`flax.linen`）为核心的函数式方法。它强调显式状态管理，这意味着模型参数 (parameter)和其他状态变量（如批归一化 (normalization)统计数据）通常在模块方法之外处理。

Flax 的主要特点包括：

1. **模块系统（`flax.linen`）**：模型通过继承 `nn.Module` 定义。层和子模块在 `setup` 方法中定义。前向传播逻辑位于 `__call__` 方法（或其他命名方法）中。
2. **显式初始化**：模块参数并非在模块实例化时创建，而是在第一次调用时（或通过 `module.init` 显式创建），这需要一个示例输入和一个 RNG 密钥。这分离了模块结构定义与实际参数值。
3. **函数式 `apply`**：`apply` 方法用于使用特定参数和状态执行前向传播。这种函数式特性使其易于与 JAX 变换配合使用。
4. **PyTree 状态**：参数和其他状态通常存储在 JAX 识别为 PyTree 的嵌套 Python 结构中（如字典或自定义类）。这使它们与 `jax.jit`、`jax.grad` 等兼容。
5. **TrainState**：Flax 经常鼓励使用 `flax.training.train_state.TrainState` 等辅助类，将模型的 apply 函数、参数和优化器状态捆绑在一起，从而简化训练循环。

以下是一个简单的 Flax 模块示例：

```python
import jax
import jax.numpy as jnp
import flax.linen as nn

class SimpleMLP(nn.Module):
  features: list[int] # 定义层大小的列表，例如 [128, 64, 10]

  @nn.compact # 允许在 __call__ 中内联定义子模块
  def __call__(self, x):
    for i, feat in enumerate(self.features):
      x = nn.Dense(features=feat, name=f'dense_{i}')(x)
      if i != len(self.features) - 1: # 对除最后一层外的所有层应用 ReLU
        x = nn.relu(x)
    return x

# --- 用法 ---
key = jax.random.PRNGKey(0)
input_shape = (1, 28*28) # 示例：批大小为 1，扁平化的 MNIST 图像
dummy_input = jnp.ones(input_shape)
output_features = [128, 10] # 定义层大小

model = SimpleMLP(features=output_features)

# 初始化参数（需要随机数生成器和虚拟输入）
params = model.init(key, dummy_input)['params']
print(f"参数 PyTree 结构：\n{jax.tree_util.tree_map(lambda x: x.shape, params)}")

# 应用模型（前向传播）
output = model.apply({'params': params}, dummy_input)
print(f"\n输出形状：{output.shape}")
```

在大型模型的背景下，Flax 的结构化方法有助于组织复杂的架构。其显式状态管理与 JAX 的函数式方法非常契合，并简化了使用 `pmap` 在分布式设备间传递参数和状态。

### Haiku：面向对象风格与隐式状态管理

Haiku 由 DeepMind 开发，提供另一种编程模型，感觉更类似于 PyTorch 等面向对象的框架，同时仍与 JAX 的函数式特性保持根本兼容。

Haiku 的重要特点包括：

1. **模块系统（`hk.Module`）**：模型通过继承 `hk.Module` 定义。层通常在 `__init__` 方法中实例化，类似于 PyTorch。
2. **隐式参数 (parameter)创建**：参数是在变换函数中首次调用层时隐式创建的。Haiku 在内部管理参数和状态存储。
3. **`hk.transform`**：这是核心函数，它将纯粹的函数式计算（适用于 JAX 变换）与面向对象的模块定义分离。它将一个实例化并调用 Haiku 模块的函数转换为一对函数：`init`（用于初始化参数）和 `apply`（用于前向传播）。
4. **内部状态管理**：Haiku 在 `hk.transform` 上下文 (context)的幕后处理参数和状态（如 RNG 序列或批归一化 (normalization)统计数据）的传递和管理。

以下是一个使用 Haiku 的示例，类似于 Flax MLP：

```python
import jax
import jax.numpy as jnp
import haiku as hk

class SimpleMLP(hk.Module):
  def __init__(self, features: list[int], name: str | None = None):
    super().__init__(name=name)
    self.features = features

  def __call__(self, x):
    for i, feat in enumerate(self.features):
      # 在 Haiku 中，层通常在此处内联创建
      x = hk.Linear(output_size=feat, name=f'linear_{i}')(x)
      if i != len(self.features) - 1:
        x = jax.nn.relu(x)
    return x

# --- 用法 ---
# 定义使用 Haiku 模块的前向函数
def forward_fn(x):
  output_features = [128, 10]
  mlp = SimpleMLP(features=output_features)
  return mlp(x)

# 变换函数
model = hk.transform(forward_fn)

key = jax.random.PRNGKey(0)
input_shape = (1, 28*28)
dummy_input = jnp.ones(input_shape)

# 初始化参数（只需要 RNG 和虚拟输入）
params = model.init(key, dummy_input)
print(f"参数 PyTree 结构：\n{jax.tree_util.tree_map(lambda x: x.shape, params)}")

# 应用模型（前向传播，只需要参数和输入）
output = model.apply(params, key, dummy_input) # 注意：如果模块使用随机性，Haiku apply 通常也需要 RNG
print(f"\n输出形状：{output.shape}")
```

Haiku 的 `hk.transform` 机制巧妙地弥合了有状态的面向对象模块定义与 JAX 对纯函数的要求之间的差距。这可以让来自其他框架的用户感到更熟悉。对于大型模型，Haiku 提供清晰的状态管理，并与 JAX 变换和分布式原语良好集成。

### 为何大型模型要使用这些库？

尽管您*可以*在原生 JAX 中构建一切，但 Flax 和 Haiku 提供显著优势，尤其随着模型复杂性和规模的增加：

1. **组织性**：它们强制实行结构，使复杂的模型架构更易于定义、阅读和维护。
2. **状态管理**：它们提供系统化的方式来处理参数 (parameter)、优化器状态、批统计数据和 RNG 密钥，减少样板代码和潜在错误。在使用 `pmap` 处理跨多个设备的分布式状态时，这一点尤为重要。
3. **减少样板代码**：应用层、管理参数和与优化器集成等常见模式得到简化。
4. **可组合性**：模块可以轻松嵌套和重用。
5. **生态集成**：它们通常附带用于检查点、指标日志记录和管理训练循环的工具，这些对于大规模实验十分重要。

Flax 和 Haiku 之间的选择通常取决于风格偏好。Flax 更显式地函数化，需要手动传递状态；而 Haiku 使用 `hk.transform` 提供更面向对象的感受，并带有隐式状态管理。两者都是基于相同 JAX 核心构建的强大工具，能够在规模上构建和训练复杂模型。了解这些库如何组织代码和管理状态，是应用本章其余部分讨论的大规模技术（例如集成 `pmap` 进行数据并行或实现检查点策略）的根本。

> 原生 JAX、Flax 和 Haiku 中状态处理的比较。原生 JAX 需要手动管理。Flax 使用显式状态传递，通常捆绑在一起。Haiku 使用 `hk.transform` 在其 `apply` 函数中隐式管理状态。

## 参考资料

- [Flax Documentation](https://flax.readthedocs.io/en/latest/) — Google (2024)
  Publisher: Google
  Flax的官方文档，Flax是一个基于JAX构建的神经网络库，详细介绍了其模块系统、显式状态管理和用于训练循环的`TrainState`。
- [Haiku Documentation](https://dm-haiku.readthedocs.io/en/latest/) — DeepMind (2024)
  Publisher: DeepMind
  Haiku的官方文档，Haiku是DeepMind的JAX兼容神经网络库，解释了其通过`hk.transform`实现隐式状态管理的面向对象风格。
- [Haiku: JAX at DeepMind](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research) — DeepMind (2020)
  Publisher: DeepMind Blog
  DeepMind介绍Haiku的一篇文章，讨论了其设计原则、动机以及如何将面向对象编程与JAX的函数式范式结合。

---

[上一节](01-%E5%A4%A7%E5%9E%8B%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E4%B8%AD%E7%9A%84%E6%8C%91%E6%88%98%E6%A6%82%E8%BF%B0.md) · [下一节](03-%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0%E5%92%8C%E7%8A%B6%E6%80%81%E7%9A%84%E5%A4%84%E7%90%86.md)

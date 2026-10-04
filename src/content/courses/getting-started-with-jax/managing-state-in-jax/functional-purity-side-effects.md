---
course: "getting-started-with-jax"
chapter: "managing-state-in-jax"
lesson: "functional-purity-side-effects"
sourceId: 1798
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-6-managing-state-in-jax/functional-purity-side-effects"
title: "函数纯粹性与副作用"
description: "理解纯函数对 JAX 转换的意义。"
order: 1
plots: []
sourceHash: "a477aa8b2867fca957d9198d566170881e447db1b08cb17d03b6b5a98d602cd3"
sourceCorrections: []
---

JAX 主要通过函数转换（如 `jit`（即时编译）、`grad`（自动求导）、`vmap`（向量 (vector)化）和 `pmap`（并行化））来获得其性能和能力。这些转换会分析并重写您的 Python 代码，使其在加速器上高效运行。为了使这种分析和转换过程稳定且可预测地工作，JAX 最适合处理遵循函数式编程原则的函数，特别是**函数纯粹性**的理念。

### 什么是纯函数？

纯函数具有两个主要特点：

1. **确定性：** 对于相同的输入参数 (parameter)，函数*总是*产生相同的输出。其结果仅取决于其输入。
2. **无副作用：** 函数不修改其局部范围之外的任何状态。它不会原地改变输入参数、修改全局变量、执行 I/O 操作（例如打印到控制台或写入文件），也不会以任何隐藏方式与外部系统互动。

我们来看几个简单的 Python 例子：

```python
# 纯函数：对于相同的输入总是返回相同的输出，没有副作用。
def pure_add(a, b):
  return a + b

# 不纯函数：有副作用（打印到控制台）。
def impure_add_and_print(a, b):
  result = a + b
  print(f"Calculated {a} + {b} = {result}") # 副作用！
  return result

# 不纯函数：修改外部状态（一个全局变量）。
call_count = 0
def impure_increment_counter(x):
  global call_count
  call_count += 1 # 副作用！修改全局状态。
  return x + 1

# 不纯函数：原地修改输入参数。
def impure_append_to_list(data_list, value):
  data_list.append(value) # 副作用！修改输入列表。
  return data_list
```

在上面的例子中，`pure_add` 是纯函数。调用 `pure_add(2, 3)` 总是会返回 `5`。相反，`impure_add_and_print` 执行打印操作，`impure_increment_counter` 改变全局 `call_count`，而 `impure_append_to_list` 修改了传递给它的列表。这些行为都是副作用。

### 为什么函数纯粹性对 JAX 很重要

JAX 的函数转换很大程度上依赖于它们处理的函数是纯粹的这一假设。原因如下：

1. **追踪机制：** 像 `jax.jit` 这样的转换通过*追踪*您的 Python 函数来工作。JAX 使用抽象的占位值（tracer）执行函数一次，这些值记录了所执行的操作序列。这个被记录的序列（通常表示为称为 Jaxpr 的中间语言）随后被编译（例如，编译成 XLA 以在 GPU/TPU 上执行）。如果函数有副作用，这些副作用可能在初始追踪期间发生，但*不会*在编译代码的后续运行中发生，这会引起意料之外的行为。例如，`jit` 编译函数中的 `print` 语句通常只在追踪期间执行一次，而不是每次调用编译函数时都执行。
2. **缓存与优化：** JAX 会缓存函数的编译版本。当您使用相同形状和类型的参数 (parameter)调用一个 `jit` 编译的函数时，JAX 会重用已编译的代码。这种缓存机制假设函数的输出*只*取决于其输入（确定性）。如果函数的行为依赖于外部状态（如全局变量），那么当外部状态改变时，缓存版本可能会过时或产生错误的结果。纯粹性保证缓存的有效性。
3. **自动求导：** `jax.grad` 通过分析函数中的数学操作来计算梯度。它需要从输入到输出的清晰数据流路径。副作用引入的操作通常是不可微分的（如何计算 `print` 语句的梯度？）或者模糊了输入和输出之间的联系，使自动求导不稳定或无法进行。原地修改值会破坏自动求导所依赖的链式法则的应用。
4. **向量 (vector)化与并行化：** 像 `jax.vmap` 和 `jax.pmap` 这样的转换会在数据维度或硬件设备上复制函数执行。如果被映射的函数有副作用，特别是修改共享状态，可能导致竞态条件和不确定的结果。哪个并行执行会首先修改共享状态？纯函数确保每次执行都是独立的，并产生一致的结果，使并行化变得安全且可预期。

### 状态管理难题

许多标准编程模式，特别是在面向对象编程中，或在处理像训练机器学习 (machine learning)模型这样的迭代算法时，本身就涉及状态随时间变化。考虑在优化过程中更新模型权重 (weight)，管理优化器的状态（如动量值），甚至是简单的计数器。这些通常涉及原地修改对象属性或数据结构，这些模式直接与函数纯粹性的要求相悖。

```python
# 典型有状态模式（不纯）
class Counter:
    def __init__(self):
        self.n = 0

    def count(self):
        self.n += 1 # 原地修改（副作用）
        return self.n

    def reset(self):
        self.n = 0 # 原地修改（副作用）

# 使用不纯计数器
my_counter = Counter()
print(my_counter.count()) # 输出: 1
print(my_counter.count()) # 输出: 2
# 对像 count() 这样的方法应用 JAX 转换可能引起问题。
```

因为 JAX 转换最适合纯函数，我们需要替代模式来处理避免副作用的状态更新。我们接下来将说明的主要思路是使状态处理显式化：将当前状态作为参数 (parameter)*传入*函数，并让函数将*新的、更新后的*状态作为其输出的一部分返回。这种方法在保持函数纯粹性的同时，仍能让我们模拟有状态的计算。

## 参考资料

- [JAX: Autodiff and XLA, Accelerated NumPy](https://jax.readthedocs.io/en/latest/) — JAX core contributors (2024)
  Publisher: Google LLC
  JAX官方文档，详细介绍了其架构、`jit`等转换以及其运行所基于的函数式编程原则。
- [Structure and Interpretation of Computer Programs](https://mitpress.mit.edu/sites/default/files/sicp/full-text/book/book.html) — Harold Abelson, Gerald Jay Sussman, and Julie Sussman (1996)
  Publisher: MIT Press
  计算机科学经典教材，介绍了函数式编程的基本概念，包括纯函数和副作用的定义及重要性。
- [Functional Programming in Python](https://www.oreilly.com/library/view/functional-programming-in/9781491928561/) — David Mertz (2015)
  Publisher: O'Reilly Media
  一本将纯函数和不变性等函数式编程概念应用于Python开发的书籍。

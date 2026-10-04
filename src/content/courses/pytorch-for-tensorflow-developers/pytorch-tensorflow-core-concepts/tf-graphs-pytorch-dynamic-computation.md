---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-tensorflow-core-concepts"
lesson: "tf-graphs-pytorch-dynamic-computation"
sourceId: 5737
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-1-pytorch-tensorflow-core-concepts/tf-graphs-pytorch-dynamic-computation"
title: "TensorFlow 计算图与 PyTorch 动态计算的对比"
description: "对比 TensorFlow 的静态计算图和 PyTorch 的即时执行动态图。"
order: 2
plots: []
sourceHash: "a190936a5e548f8bd84e012bafc91d7f779b03f306b5723ed6c846b8077ac903"
sourceCorrections: []
---

TensorFlow，尤其是在其早期版本（2.0 之前）以及在 2.x 版本中使用 `tf.function` 时，以其对计算图的使用而广为人知。如果你使用过 TensorFlow，你可能熟悉“先定义后运行”的方法。在这种模型中，你首先构建一个操作图（“定义”阶段）。这个图表示你的模型将执行的计算，但它不会立即执行。这就像在启动机器之前，先为其绘制详细蓝图。

一旦定义好这个蓝图，也就是静态图，TensorFlow 就可以执行它（“运行”阶段），通常是在 `tf.Session` 中（在 TF1.x 中）或通过调用由 `tf.function` 装饰的 Python 函数。这种分离带来了独特的优点：

- **优化：** 整个计算路径都已规划好，TensorFlow 可以执行各种优化，例如融合操作、在不同设备间分配工作或消除冗余计算。
- **可移植性：** 静态图可以更容易地保存、加载并在不同平台（服务器、移动设备、嵌入 (embedding)式系统）上运行，因为它封装了整个模型结构和操作。

然而，这种静态特性也给开发者带来了一些挑战：

- **调试：** 当图执行期间发生错误时，在原始 Python 代码中找出源头可能很困难。堆栈跟踪可能指向内部图操作，而不是用户的模型定义。
- **动态结构：** 实现那些控制流依赖于中间张量值（例如，循环神经网络 (neural network) (RNN)中可变长度的序列，或条件执行路径）的模型，需要使用 `tf.cond` 和 `tf.while_loop` 等专门的 TensorFlow 图操作。与原生 Python 控制流相比，这可能会使代码直观性较差。

在 TensorFlow 2.x 中，即时执行成为了默认模式，允许操作立即运行，非常像标准 Python。然而，为了性能和部署，`tf.function` 被用来将 Python 代码转换为可调用的 TensorFlow 图。这充当了一个桥梁，最初提供了更具 Python 风格的开发体验，但仍然涉及到图编译步骤，通常在后台被称为“追踪”。当你用 `@tf.function` 装饰一个 Python 函数时，TensorFlow 会追踪其操作以构建一个静态图供后续调用。

## PyTorch 的动态计算图：即时执行

另一方面，PyTorch 采用“运行即定义”的理念。这意味着计算图是动态地、即时地在操作执行时构建的。没有单独的编译步骤，你需要先定义整个图再运行它。每一行执行张量操作的代码都会实时地构成图的一部分。

想象你正在用乐高积木搭建。在 TensorFlow 的静态图方法中，你会先在纸上设计整个结构，然后进行组装。在 PyTorch 的动态方法中，你拿起一块乐高积木，连接它，看看效果，然后决定下一块积木。结构（图）在你构建（执行操作）时就形成了。

这种动态特性带来了多项好处，特别是对于那些从更命令式编程风格过渡的开发者：

- **Python 风格的控制流：** 你可以直接在模型的正向传播中使用标准 Python 控制流语句（`if`、`for`、`while`），根据中间结果改变计算。图会适应这些动态条件。
- **更便捷的调试：** 由于操作是按顺序即时执行的（像标准 Python 代码一样），你可以使用 Python 的原生调试工具，例如 `pdb`，或者直接插入 `print()` 语句来在模型的任何点检查张量值。错误通常会直接追溯到导致它们的 Python 代码行。
- **复杂架构的灵活性：** 需要动态结构的模型，例如处理不同长度序列的循环神经网络 (neural network)（RNN），或涉及条件逻辑的模型（例如，根据状态做出决策的强化学习 (reinforcement learning)代理），通常更容易实现。

## 实际比较

让我们用一个简单例子来说明，其中计算路径取决于一个中间值。假设我们希望在和为正时执行一个操作，否则执行另一个操作。

在 PyTorch 中，这非常直接：

```python
import torch

def dynamic_behavior_pytorch(x, y, z):
    intermediate_sum = x + y
    # 使用 .item() 获取 Python 标量值，用于标准的 Python if 条件
    # 如果条件不需要成为自动求导图本身的一部分，则此方法适用。
    if intermediate_sum.item() > 0:
        result = intermediate_sum * z
    else:
        result = intermediate_sum + z
    return result

# 示例使用
a = torch.tensor(2.0)
b = torch.tensor(3.0)
c = torch.tensor(4.0)
# intermediate_sum = 5.0 (> 0), result = 5.0 * 4.0 = 20.0
print(f"PyTorch 结果（和为正）：{dynamic_behavior_pytorch(a, b, c)}")

d = torch.tensor(-2.0)
e = torch.tensor(-1.0)
# intermediate_sum = -3.0 (<= 0), result = -3.0 + 4.0 = 1.0
print(f"PyTorch 结果（和为负）：{dynamic_behavior_pytorch(d, e, c)}")
```

`dynamic_behavior_pytorch` 函数的图在每次被调用时都会根据 `x` 和 `y` 的值以不同方式构建。Python 的 `if` 语句直接控制哪些操作运行，从而决定哪些操作成为该特定执行图的一部分。

在 TensorFlow 的图模式下（例如，在由 `@tf.function` 装饰的函数内部），你传统上会为这类条件逻辑使用 `tf.cond`，以确保它明确地是静态图的一部分：

```python
import tensorflow as tf

@tf.function
def static_graph_conditional_tf(x, y, z):
    intermediate_sum = x + y
    # tf.cond 需要为真分支和假分支提供可调用函数
    result = tf.cond(intermediate_sum > 0,
                     lambda: intermediate_sum * z,  # 真分支
                     lambda: intermediate_sum + z)  # 假分支
    return result

# 示例使用
a_tf = tf.constant(2.0)
b_tf = tf.constant(3.0)
c_tf = tf.constant(4.0)
# intermediate_sum = 5.0 (> 0), result = 5.0 * 4.0 = 20.0
print(f"TensorFlow @tf.function 结果（和为正）：{static_graph_conditional_tf(a_tf, b_tf, c_tf)}")

d_tf = tf.constant(-2.0)
e_tf = tf.constant(-1.0) # 为保持一致性，将变量名从 'e' 更正为 'e_tf'
# intermediate_sum = -3.0 (<= 0), result = -3.0 + 4.0 = 1.0
print(f"TensorFlow @tf.function 结果（和为负）：{static_graph_conditional_tf(d_tf, e_tf, c_tf)}")
```

虽然 TensorFlow 2.x 的即时执行（在 `tf.function` 外部操作时）允许 Python 风格的 `if` 语句按预期工作，但将此代码封装在 `@tf.function` 中会导致 TensorFlow 的 AutoGraph 特性将 Python 控制流转换为 `tf.cond` 等图操作。这种转换功能强大，但如果 Python 代码的结构不是 AutoGraph 容易理解的方式，有时可能会导致意外行为或追踪错误。PyTorch 凭借其“运行即定义”的特性，避免了这种影响图结构的 Python 控制流的显式转换步骤。

一般的工作流程区别可以可视化如下：

> 此图展示了不同的工作流程。TensorFlow 在使用 `tf.function` 以提升性能时，会涉及一个独立的“追踪”或“编译”步骤来创建图。PyTorch 则将图的创建直接与 Python 代码的执行相结合。

## 对 TensorFlow 开发者的影响

对于一位过渡到 PyTorch 的 TensorFlow 开发者来说，这种从“先定义后运行”（或使用 `tf.function` 时的“先定义-追踪-后运行”）到“运行即定义”的转变是最重要的变化之一，并提供了多项优势：

- **调试便利性：** 你可能最快体会到的好处之一是直接的调试。如果你的 PyTorch 模型 `forward` 传递（定义计算的方法）中出现问题，你通常可以使用像 `pdb` 这样的 Python 调试器，或者直接在操作发生的地方插入 `print()` 语句。堆栈跟踪通常更偏向 Python，更容易追溯到你的代码。
- **自然的 Python 流程：** 复杂的控制流，例如迭代次数依赖于中间数据的循环，或改变计算路径的条件块，通常可以使用标准 Python 语法来表达。更少需要学习框架特定的控制流操作（如 TensorFlow 图模式下的 `tf.while_loop` 或 `tf.cond`）。如果你已经熟悉 Python，这会使代码感觉更直观。
- **思维模式调整：** 构造模型的方式会改变。你不再需要预先设想整个静态图，而是会更多地命令式地思考操作执行的序列，同时知道图正在后台动态组装。这会带来相当大的自由，尤其是在处理本身具有非静态架构的模型时（例如某些类型的循环神经网络 (neural network) (RNN)或图神经网络）。
- **TensorFlow 2.x 即时执行带来的熟悉感：** 如果你的主要经验是 TensorFlow 2.x 的即时执行（这是 `tf.function` 外部的默认模式），那么过渡到 PyTorch 的动态图会感觉非常自然。两者都允许操作即时执行，并且感觉是命令式的。主要区别在于考虑性能优化和部署时。PyTorch 的动态特性是其核心所在，而 TensorFlow 则依靠 `tf.function` 将操作捕获到图中以进行优化和序列化，这是一种即时 (JIT) 编译的形式。PyTorch 也通过 TorchScript 提供了 JIT 功能（我们将在课程后期涉及），但其标准执行模型默认是动态的。

理解图处理方式的这种区别是根本的。它影响着你在 PyTorch 中如何编写、调试和进行模型设计。随着你的学习，你将看到 PyTorch 的动态特性如何渗透到其 API 中，从使用 `torch.nn` 定义模型架构到编写自定义训练循环。

## 参考资料

- [PyTorch: An Imperative Style, High-Performance Deep Learning Library](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf) — Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, Soumith Chintala (2019)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 32; Pages: 8024-8035; DOI: [10.48550/arXiv.1912.01703](https://doi.org/10.48550/arXiv.1912.01703)
  这篇基础论文介绍了 PyTorch，阐述了其命令式编程风格和动态计算图设计，这是 'define-by-run' 理念的核心。
- [Better performance with tf.function](https://www.tensorflow.org/guide/function) — TensorFlow Team (2023)
  官方 TensorFlow `tf.function` 指南详细介绍了 TensorFlow 2.x 如何将 Python 代码编译成静态图，包括跟踪和 AutoGraph 在性能和部署中的作用。
- [Autograd: Automatic Differentiation](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Team (2017)
  这份 PyTorch 官方文档描述了 `autograd` 引擎，该引擎动态构建计算图以进行自动微分，提供了 'define-by-run' 方法的实用视角。
- [TensorFlow: A System for Large-Scale Machine Learning](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi) — Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G. Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, Xiaoqiang Zheng (2016)
  Journal: 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16); Publisher: USENIX Association; Pages: 265-283
  这篇重要论文介绍了 TensorFlow 的架构，主要聚焦其最初的静态图计算模型，为理解其演变提供了历史和架构背景。

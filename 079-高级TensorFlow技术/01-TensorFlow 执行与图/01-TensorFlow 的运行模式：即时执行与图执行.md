# TensorFlow 的运行模式：即时执行与图执行

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-1-tensorflow-execution-graphs/execution-modes-eager-graph)

[返回章节目录](README.md) · [返回课程目录](../README.md)

TensorFlow 的架构提供了两种不同的操作运行方式：**即时执行**和**图执行**。虽然 TensorFlow 2.x 默认采用即时执行，但要构建高性能、可扩展和易于部署的模型，了解这两种模式以及它们通过 `tf.function` 建立的关系是必不可少的。这两种基本方法将进行对比。

### 即时执行：Python 式方法

即时执行是一种命令式编程环境，它会立即评估操作，就像它们在 Python 代码中出现一样。当您在即时模式下运行 TensorFlow 操作时，其返回值会立即计算并变得可用。此行为与标准 Python 执行以及 NumPy 等库的行为高度一致。

```python
import tensorflow as tf

# 即时执行是 TensorFlow 2.x 的默认模式
print(f"即时执行已启用: {tf.executing_eagerly()}")

# 定义一些张量
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# 操作会立即执行
z = tf.matmul(x, y)
print(z)

# 轻松与 Python 控制流集成
if tf.reduce_sum(x) > 10.0:
    print("x 的和大于 10")
else:
    # 此分支将被执行
    result = x * 2.0
    print("x 的和不大于 10，x*2 的结果：")
    print(result)

# 值可以方便地作为 NumPy 数组使用
print(z.numpy())
```

**即时执行的优点：**

1. **直观的开发：** 即时反馈循环使调试变得简单。您可以直接使用标准 Python 调试器（如 `pdb`）和打印张量值来查看计算的中间状态。
2. **原生 Python 集成：** 可以将 Python 控制流（if 语句、循环、函数调用）和数据结构与 TensorFlow 操作一起使用。与 NumPy 等库的集成是自然而然的。
3. **更简单的原型设计：** 即时执行的交互性促进了快速迭代和实验。

**即时执行的缺点：**

1. **性能开销：** 每个 TensorFlow 操作都涉及与 Python 解释器来回通信。这种开销可能会变得明显，特别是对于涉及许多小操作的计算，或者在 GPU 或 TPU 等加速器上运行时，因为启动单个内核的代价很高。
2. **优化受限：** TensorFlow 无法跨多个步骤执行整个程序优化，因为它每次只看到一个操作。运算符合并（将多个操作合并到单个更高效的内核中）或静态图分析等技术是无法实现的。
3. **部署难题：** 即时执行的代码本质上是一个 Python 程序。将其部署到没有 Python 解释器的环境中（如移动设备、嵌入 (embedding)式系统或高性能服务基础设施），或导出独立于创建该模型的代码的模型是困难的。

### 图执行：构建和运行计算

图执行采用声明式方法。首先，您定义一个计算图，它是一个数据结构，表示操作序列以及它们之间流动的张量。图中的节点表示操作（例如 `tf.matmul`、`tf.add`），边表示这些操作消耗或生成的张量。一旦图构建完成，您就可以在 TensorFlow `Session`（在 TF 1.x 中）中高效地执行它，或者在 TF 2.x 中更常见的是，通过将您的 Python 逻辑封装在 `tf.function` 中来执行。

> 一个简单的计算图，表示 `z = (x + y) * y`。操作是节点，张量是边。

在 TF 2.x 中，您通常不手动构建图。相反，您使用 `tf.function` 装饰器，它会追踪您的 Python 代码，自动生成并缓存这些计算图。

```python
import tensorflow as tf

# 定义一个要转换为图的函数
@tf.function
def computation(x, y):
  print("追踪计算函数...") # 这只会在追踪期间打印
  a = tf.matmul(x, y)
  b = tf.reduce_sum(a)
  return b * 2.0

# 定义张量
x_data = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y_data = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# 第一次调用：追踪函数并执行图
result1 = computation(x_data, y_data)
print(f"结果 1: {result1}")

# 具有相同输入形状/类型的第二次调用：重用缓存的图
# “追踪计算函数...”将不会再次打印
result2 = computation(x_data, y_data)
print(f"结果 2: {result2}")

# 具有不同输入形状/类型的第三次调用：触发重新追踪
x_data_diff_shape = tf.constant([[1.0, 2.0, 3.0]])
y_data_diff_shape = tf.constant([[4.0], [5.0], [6.0]])
result3 = computation(x_data_diff_shape, y_data_diff_shape)
print(f"结果 3: {result3}")
```

**图执行的优点：**

1. **性能：** 图可以由 TensorFlow 运行时优化。它可以合并操作、消除公共子表达式、优化内存使用，并高效地调度计算，尤其是在多个设备（CPU、GPU、TPU）之间。这通常会带来相比即时执行的明显加速。
2. **可移植性：** 图可以序列化为一种语言无关、平台独立的格式，称为 `SavedModel`。这使您可以在 Python 中训练模型，并在不同环境（例如，使用 TensorFlow Serving、移动/边缘设备上的 TensorFlow Lite 或浏览器中的 TensorFlow.js）中部署它进行推理 (inference)，而无需原始 Python 代码。
3. **并行性：** 图的结构使 TensorFlow 更容易识别可以并行执行的操作。

**图执行的缺点：**

1. **调试复杂性：** 调试 `tf.function` 内的代码可能不如调试标准 Python 直观。错误可能在图执行期间发生，使得堆栈跟踪更难解释。您通常需要依赖 `tf.print` 等工具（它会插入到图中）或在 `tf.function` 之外进行逐步执行来找出问题。
2. **静态特性：** 图一旦构建完成，通常是静态的。动态控制流（数据相关的循环或条件）需要特殊的 TensorFlow 构造（`tf.cond`、`tf.while_loop`）或依赖 AutoGraph 转换 Python 控制流的能力（下一节将介绍）。输入签名（数据类型和形状）也可能触发重新追踪，这会产生额外的开销。

### 即时执行与图执行：概述

| 特性 | 即时执行 | 图执行（通过 `tf.function`） |
| --- | --- | --- |
| **执行** | 命令式，立即评估 | 声明式，延迟执行 |
| **TF2 默认** | 是 | 否（需要 `tf.function`） |
| **调试** | 简单（标准 Python 工具） | 更复杂（`tf.print`，追踪） |
| **性能** | 较低（Python 开销） | 较高（图优化） |
| **优化** | 受限（按操作） | 高效（全图分析） |
| **部署** | 困难（需要 Python 运行时） | 简单（通过 `SavedModel`） |
| **灵活性** | 高（原生 Python 控制流） | 需要图感知构造或 AutoGraph 转换 |

**何时使用哪种模式（或 `tf.function`）**

即时执行非常适合交互式开发、调试和简单脚本。然而，对于对性能要求高的训练循环、最大化硬件利用率以及准备模型进行部署，通过 `tf.function` 借助图执行是不可或缺的。

TensorFlow 2.x 旨在提供最佳体验：开发时的即时执行易用性，以及通过 `tf.function` 装饰器带来的图执行的性能和部署优势。下一节将展示 `tf.function` 如何使用 AutoGraph 实现这一点。

## 参考资料

- [Introduction to Graphs and tf.function](https://www.tensorflow.org/guide/intro_to_graphs) — TensorFlow Authors (2022)
  Publisher: Google LLC
  TensorFlow官方指南，解释了TensorFlow 2.x中的即时执行、图执行和`tf.function`的使用。
- [TensorFlow: A System for Large-Scale Machine Learning](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi) — Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G. Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng (2016)
  Journal: OSDI '16: Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation; Publisher: USENIX Association; Pages: 265--283; DOI: [10.5555/3026877.3026899](https://doi.org/10.5555/3026877.3026899)
  介绍TensorFlow系统及其基于图的计算模型的原始研究论文。
- [Effective TensorFlow 2.0](https://www.tensorflow.org/guide/effective_tf2) — TensorFlow team (2024)
  TensorFlow 2.x最佳实践指南，包含`tf.function`如何帮助实现高性能和部署能力。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125972/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本流行的教材，提供了TensorFlow 2.x的实践内容，解释了即时执行和图执行以及`tf.function`。

---

[下一节](02-%E7%90%86%E8%A7%A3%20tf.function%20%E5%92%8C%20AutoGraph.md)

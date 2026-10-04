---
course: "compiler-runtime-optimization-ml"
chapter: "jit-compilation-ml"
lesson: "practice-analyzing-jit-code"
sourceId: 4576
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-7-jit-compilation-ml/practice-analyzing-jit-code"
title: "实践操作：分析JIT编译代码"
description: "练习分析ML JIT编译器为简单模型生成的IR或汇编代码。"
order: 9
plots: []
sourceHash: "181e4f7881d3d629cf47f957ac8b36932f2489f2f2aa25fbc7a04c1c59777fd3"
sourceCorrections: []
---

理解即时（JIT）编译的理论基础，例如跟踪与脚本化或自适应优化，非常必要。但是，要获得实际体会，需要检查这些JIT编译器生成的实际输出。分析中间表示（IR）或编译后的代码，能够具体说明算子合并、常量折叠和专门化等优化是如何应用的，从而将高级Python代码与低级执行计划直接关联起来。这种实际分析对于调试性能问题、验证优化效果以及提高对JIT过程的理解具有很高价值。

在本实践部分，我们将逐步介绍如何使用PyTorch (TorchScript) 和 TensorFlow (XLA) 等常用框架对简单模型片段进行JIT编译，然后分析生成的IR。我们假设您已安装PyTorch和TensorFlow，并拥有一个可用的Python环境。

### 分析PyTorch TorchScript IR

PyTorch的JIT模块TorchScript提供了两种主要方法将Python代码转换为可优化的图表示：跟踪（`torch.jit.trace`）和脚本化（`torch.jit.script`）。我们从跟踪开始。

#### 示例设置：跟踪

考虑一个简单的操作序列：一个线性层，接着是ReLU激活，再是一个线性层。

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super().__init__()
        self.linear1 = nn.Linear(input_features, hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 实例化模型并创建示例输入
input_size = 128
hidden_size = 256
output_size = 64
model = SimpleModel(input_size, hidden_size, output_size)
dummy_input = torch.randn(32, input_size) # 批大小为32

# 跟踪模型
traced_model = torch.jit.trace(model, dummy_input)

print("模型跟踪成功。")
```

#### 检查跟踪图

跟踪会使用提供的示例输入执行模型，并记录所执行的操作。生成的`traced_model`包含一个专门针对输入形状（$32 \times 128$）的静态图表示。我们可以检查这个图：

```python
# 打印TorchScript图IR
print(traced_model.graph)
```

您将看到类似于以下内容的输出（细节可能因PyTorch版本略有不同）：

```
graph(%self.1 : __torch__.SimpleModel,
      %x : Float(32, 128, strides=[128, 1], requires_grad=0, device=cpu)):
  %linear1 : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="linear1"](%self.1)
  %3 : Float(256, 128, strides=[128, 1], device=cpu) = prim::GetAttr[name="weight"](%linear1)
  %4 : Float(256, strides=[1], device=cpu) = prim::GetAttr[name="bias"](%linear1)
  %5 : Tensor = aten::linear(%x, %3, %4) # <eval_with_key>.13:10:8
  %relu : __torch__.torch.nn.modules.activation.ReLU = prim::GetAttr[name="relu"](%self.1)
  %7 : Tensor = aten::relu(%5) # <eval_with_key>.14:8:8
  %linear2 : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="linear2"](%self.1)
  %9 : Float(64, 256, strides=[256, 1], device=cpu) = prim::GetAttr[name="weight"](%linear2)
  %10 : Float(64, strides=[1], device=cpu) = prim::GetAttr[name="bias"](%linear2)
  %11 : Tensor = aten::linear(%7, %9, %10) # <eval_with_key>.15:8:8
  return (%11)
```

**分析：**

- **输入**：图清楚地显示了模型（`%self.1`）和输入张量`%x`及其跟踪到的形状`Float(32, 128, ...)`。
- **操作**：每行通常对应一个操作（`aten::linear`、`aten::relu`）或属性访问（`prim::GetAttr`）。
- **属性**：权重 (weight)和偏置 (bias)（`%3`、`%4`、`%9`、`%10`）作为图输入或属性嵌入 (embedding)，通过`prim::GetAttr`获取。
- **专门化**：该图是静态的。它不包含Python控制流，如`if`语句（除非它们在跟踪期间是常量）。它专门针对形状为`(32, 128)`的输入。如果操作支持，使用不同形状运行`traced_model`可能有效，但图本身并未明确表示动态形状逻辑。

#### 观察优化（合并）

TorchScript会自动应用优化过程，包括合并。虽然简单的`线性 -> ReLU`合并并不总是保证发生，或在这种高级图转储中不总是直接可见（它通常发生在后续的降低阶段），但我们可以思考如何寻找它。更复杂的模式，尤其是在卷积网络中（Conv-BN-ReLU），是常见的合并目标。

我们可以检查潜在优化*之后*的图，尽管默认的打印输出通常显示的是初始图。要查看优化的图或执行计划，您可能需要分析工具或内部API（谨慎使用，因为它们可能会更改）：

```python
# 使用内部API的示例（可能会更改）
# 这提供了不同的视图，有时会显示优化/合并的操作
# print(torch._C._jit_get_profiling_executor_graph(traced_model.graph))
```

另外，分析JIT编译前后（例如，使用`torch.profiler.profile`）的性能特点，可以体现合并等优化的影响，即使IR可视化没有明确显示合并操作。

#### 示例设置：脚本化

现在，我们使用脚本化，它直接分析Python字节码。

```python
import torch
import torch.nn as nn

# 假设SimpleModel类如上定义

# 脚本化模型
scripted_model = torch.jit.script(model)

print("模型脚本化成功。")

# 打印TorchScript图IR
print(scripted_model.graph)
```

对于这个线性模型，输出图看起来会与跟踪图非常相似，因为它不包含跟踪会遗漏的Python控制流。但是，如果模型包含数据依赖的控制流（例如，基于张量值的`if`语句），脚本化图将使用`prim::If`节点明确表示此控制流，而跟踪图将只包含跟踪期间所取的路径。

**分析（`脚本化` vs `跟踪`）：**

- **控制流**：脚本化直接在图中捕获Python控制流（`if`、`for`），使其对具有动态行为的模型更具灵活性。跟踪仅捕获针对*特定*跟踪输入执行的操作。
- **灵活性 vs. 优化**：脚本化可能更难进行激进优化，因为编译器必须考虑所有潜在的控制流路径。跟踪根据示例输入提供静态图，通常能够实现更直接的优化，例如针对特定形状的代码生成，但如果输入与跟踪输入显著偏离，则可能失败。

### 分析TensorFlow XLA输出

TensorFlow使用XLA（加速线性代数）作为其优化编译器，通常通过`tf.function(jit_compile=True)`调用。XLA在其自己的IR，即HLO（高级优化器IR）上进行操作。

#### 示例设置：XLA JIT

```python
import tensorflow as tf

# 定义一个简单函数
@tf.function(jit_compile=True)
def simple_computation(x, w1, b1, w2, b2):
    y = tf.matmul(x, w1) + b1
    y = tf.nn.relu(y)
    z = tf.matmul(y, w2) + b2
    return z

# 创建一些示例输入
input_shape = (32, 128)
hidden_shape = 256
output_shape = 64

x_in = tf.random.normal(input_shape)
w1_in = tf.random.normal((input_shape[1], hidden_shape))
b1_in = tf.random.normal((hidden_shape,))
w2_in = tf.random.normal((hidden_shape, output_shape))
b2_in = tf.random.normal((output_shape,))

# 执行JIT编译的函数
result = simple_computation(x_in, w1_in, b1_in, w2_in, b2_in)

print("XLA JIT函数执行。")
# print(result.numpy()) # 查看输出
```

#### 检查HLO图

检查XLA生成的HLO通常涉及使用环境变量或TensorFlow日志/分析工具。一种常用方法是使用TensorBoard进行分析，它可以可视化XLA编译步骤和生成的HLO图。另一种方法是在运行脚本之前设置环境变量：

```bash
export TF_XLA_FLAGS="--tf_xla_dump_to=/path/to/dump/folder"
# 现在运行您的Python脚本
python your_script.py
```

此命令指示XLA将其编译过程的各个阶段，包括HLO图（通常是`.hlo.dot`文件或文本proto格式），转储到指定目录中。然后您可以检查这些文件。

例如，`.dot`文件可以使用Graphviz工具（`dot -Tpng input.dot -o output.png`）进行可视化。HLO图可能看起来像这样（简化版）：

> simple\_computation函数的简化HLO图。节点表示操作（dot表示矩阵乘法，add表示加法，maximum表示relu），边表示数据流，参数 (parameter)表示输入。

**分析：**

- **HLO操作**：图中使用HLO操作，如`dot`（用于矩阵乘法）、`add`、`maximum`（常用于ReLU）、`broadcast`和参数。
- **合并**：XLA在算子合并方面尤其激进。在转储的HLO图中（尤其是在“HloFusion”等优化通过后），您可能会看到多个操作合并为一个`fusion`节点。例如，`MatMul + BiasAdd + ReLU`序列是合并为一个单一优化内核的主要候选，这将在优化后的HLO图中显示为一个节点。
- **布局分配**：XLA执行布局分配（在类似NCHW/NHWC的格式之间进行决定），这在HLO中是可见的。
- **后端特定**：HLO最终会降低到LLVM IR或特定目标代码（例如，用于NVIDIA GPU的PTX）。查看这些更低的层级需要不同的工具（部分在第5章和第9章中有所提及）。

### 解释与后续步骤

这种实践分析演示了如何访问和解释ML JIT编译器生成的中间表示。通过检查TorchScript图或XLA HLO转储，您可以：

1. **验证正确性**：确保JIT捕获了预期的计算。
2. **理解专门化**：了解跟踪如何固化特定输入形状或值。
3. **辨识优化**：观察常量折叠、死代码消除，尤其是算子合并的迹象。比较跟踪图与脚本化图，或未优化与优化后的HLO。
4. **调试性能**：将性能观察与编译图的结构联系起来。合并是否按预期发生？是否存在意外操作或数据依赖？

分析JIT输出是从事ML框架的性能工程师的一项基本技能。它弥合了高级模型代码与优化后的、硬件特定的执行计划之间的差距，为实现最高性能提供了重要信息。使用这些方法来研究您自己模型的JIT行为，并考察本章中讨论的不同JIT策略的影响。

## 参考资料

- [PyTorch JIT Documentation](https://pytorch.org/docs/stable/jit.html) — PyTorch Authors (2024)
  官方文档，提供了关于TorchScript跟踪和脚本模式的详细信息，包括如何检查生成的图IR。
- [XLA: Optimized TensorFlow Compilation](https://www.tensorflow.org/xla/) — TensorFlow Authors (2024)
  Publisher: Google
  官方文档，概述了XLA在TensorFlow中的作用、其JIT编译方法以及如何启用和与XLA编译的计算进行交互。
- [Compilers: Principles, Techniques, and Tools](https://www.informit.com/store/compilers-principles-techniques-and-tools-9780321486813) — Alfred V. Aho, Monica S. Lam, Ravi Sethi, Jeffrey D. Ullman (2006)
  Publisher: Addison-Wesley
  一本经典的编译器设计教材，全面涵盖了中间表示、静态分析和与JIT编译相关的各种优化技术。

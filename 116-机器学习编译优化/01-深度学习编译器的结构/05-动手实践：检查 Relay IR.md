# 动手实践：检查 Relay IR

来源：[原文](https://apxml.com/zh/courses/compiler-optimizations-machine-learning/chapter-1-anatomy-deep-learning-compilers/hands-on-inspecting-relay-ir)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Apache TVM 提供一个实用平台，用于检查高级中间表示 (IR) 并了解静态单赋值 (SSA) 形式在生产环境中的作用。TVM 使用 Relay，这是一种专门为机器学习 (machine learning)设计的高级函数式中间表示。与传统计算图（可能只表示操作的有向无环图 (DAG)）不同，Relay 是一种完整的编程语言，支持控制流、递归和复杂数据结构，尽管大多数深度学习 (deep learning)模型使用这些特性的静态子集。

在本节中，我们将从一个标准框架中引入一个模型，将其转换为 Relay IR，并检查生成的文本格式。此过程显示了编译器如何在任何硬件特定优化发生之前获取形状信息、数据类型和运算符语义。

### 将模型引入 TVM

为了观察 IR，我们首先需要一个源模型。我们将在 PyTorch 中定义一个简单的卷积块，它作为典型深度学习 (deep learning)工作负载的代表性示例。此块包含卷积层、批归一化 (normalization)层和 ReLU 激活层。

```python
import torch
import torch.nn as nn
import tvm
from tvm import relay

# 定义一个标准卷积块
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# 实例化并追踪模型
model = ConvBlock().eval()
input_shape = (1, 3, 224, 224)
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data)
```

追踪步骤将动态 PyTorch 执行转换为静态图表示 (TorchScript)。TVM 需要此静态定义才能准确导入图结构。

接下来，我们调用 TVM 前端将此 TorchScript 图转换为 Relay `IRModule`。`IRModule` 是 TVM 中主要的容器，用于保存程序的函数和类型定义。

```python
# 将输入名称映射到形状
shape_list = [("input_0", input_shape)]

# 将图导入 Relay
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

print(mod)
```

### 分析文本表示

当您打印 `mod` 对象时，TVM 会输出 Relay IR 的文本格式。输出将类似于以下结构。这段文本不仅仅是一个调试字符串；它是 Relay 语言中语法有效的代码。

```text
def @main(%input_0: Tensor[(1, 3, 224, 224), float32]) {
  %0 = nn.conv2d(%input_0, meta[relay.Constant][0], padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3]);
  %1 = nn.batch_norm(%0, meta[relay.Constant][1], meta[relay.Constant][2], meta[relay.Constant][3], meta[relay.Constant][4]);
  %2 = %1.0;
  nn.relu(%2)
}
```

此输出明确体现了前面部分讨论过的几个编译器工程原理：

1. **函数定义**：整个图被封装在一个 `main` 函数中。这与函数式编程方法一致，其中模型只是一个将输入转换为输出的函数。
2. **强类型**：请看 `%input_0`。它被标注了一个具体的形状 `(1, 3, 224, 224)` 和数据类型 `float32`。与 Python 不同，Relay 在编译期间严格执行这些类型。如果卷积层和输入之间发生维度不匹配，编译器会在此处捕获它，远在代码生成之前。
3. **SSA 形式**：变量 `%0`、`%1` 和 `%2` 表示中间张量。每个变量都只被赋值一次。这种结构简化了数据流分析，使编译器能够轻松追踪数据的生成和使用位置。
4. **属性**：像 `nn.conv2d` 这样的操作直接在调用中携带 `padding`、`channels` 和 `kernel_size` 等属性。这些对于降低阶段选择正确的实现方案是必不可少的。

### 可视化数据流

尽管文本 IR 精确，但将依赖关系可视化为图有助于理解数据流，特别是对于复杂的拓扑结构。`IRModule` 描述了一种结构，其中数据从输入参数 (parameter)通过一系列变换流动。

以下图表显示了我们刚刚生成的 Relay 程序的结构。注意权重 (weight)（表示为常量）如何沿着数据路径输入到运算符中。

> 该图表描绘了从 Relay IR 派生的依赖图。蓝色节点表示输入张量，黄色节点表示学习参数（权重/偏置 (bias)），绿色节点表示计算运算符。

IR 中有一个具体细节经常让新编译器工程师感到困惑，那就是 `TupleGetItem` 节点（在文本中显示为 `%2 = %1.0`，在图中显示为紫色节点）。Relay 中的 `batch_norm` 运算符返回一个包含三个元素的元组：归一化 (normalization)张量、移动均值和移动方差。由于我们只关心前向传播中的归一化数据，因此 `%1.0` 指令提取该元组的第零个元素。这种对多个返回值的显式处理是 Relay 基于表达式设计的特性。

### 使用 Pass 操作 IR

检查 IR 是被动的；编译器的强大之处在于对其进行变换。在 TVM 中，变换通过“Pass”应用。一个 Pass 接收一个 `IRModule` 并返回一个新的、已优化的 `IRModule`。

为了看到实际效果，我们可以应用一个简单的优化 Pass：**常量折叠**。如果我们的图包含对常量值的数学操作（例如 `3 + 4`），编译器应该在编译时而不是运行时计算结果 (`7`)。尽管我们当前的 ConvBlock 大部分是动态的，但应用一个 Pass 显示了基础设施如何修改图。

```python
# 应用一个变换 Pass
seq = tvm.transform.Sequential([
    relay.transform.SimplifyInference(),
    relay.transform.FoldConstant(),
    relay.transform.DeadCodeElimination()
])

# 优化在 PassContext 中进行
with tvm.transform.PassContext(opt_level=3):
    opt_mod = seq(mod)

print("--- 优化后的 IR ---")
print(opt_mod)
```

运行此代码可能会改变 `batch_norm` 的结构。在推理 (inference)模式下，批归一化 (normalization)通常可以通过更新卷积层的权重 (weight)和偏置 (bias)，折叠到前面的卷积层中。`SimplifyInference` Pass 尝试移除 `batch_norm` 运算符，通过用更简单的算术运算替换它或将其合并。当您检查 `opt_mod` 时，您可能会注意到 `nn.batch_norm` 调用已消失，被显式广播、乘法和加法运算取代，或者如果后端支持，则完全融合到 `conv2d` 中。

这种动手检查证实 IR 是编译器的真实依据。它是高级框架和低级代码生成器之间的契约。了解如何阅读和可视化此 IR 是编写自定义编译器 Pass 或调试深度学习 (deep learning)模型中性能退化的先决条件。

---

[上一节](04-%E9%AB%98%E7%BA%A7IR%E4%B8%8E%E4%BD%8E%E7%BA%A7IR.md) · [下一节](../02-%E5%9B%BE%E7%BA%A7%E8%BD%AC%E6%8D%A2/01-%E7%AE%97%E5%AD%90%E8%9E%8D%E5%90%88%E7%AD%96%E7%95%A5.md)

# 表示高层ML图（例如TF, TOSA）

来源：[原文](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-2-advanced-ml-intermediate-representations/representing-ml-graphs-mlir)

[返回章节目录](README.md) · [返回课程目录](../README.md)

传统的编译器IR通常表达能力不足，难以充分捕捉机器学习 (machine learning)计算图的高层语义。这些IR操作的层级过低，无法直接理解诸如卷积、批归一化 (normalization)或张量布局之类的构思。多层IR，特别是MLIR，通过允许使用*方言*来定制针对特定抽象范围的表示，展现出显著的优点。

### 方言作为语义载体

MLIR的主要设计理念认为，没有单一的IR层级足以应对整个编译流程。它提供了一个定义多个*方言*的框架，每个方言都包含一组特定的操作、类型和属性，与某个特定范围或抽象层级关联。为表示高层ML图，方言是连接源框架（如TensorFlow、PyTorch、JAX）或标准化格式（如TOSA）与编译器内部表示的主要手段。

在此阶段常用的方言包括：

- **`tf`方言：** 直接反映TensorFlow生态系统中的操作和类型。这使得TensorFlow GraphDef或SavedModel可以近乎一对一地导入到MLIR中，从而保留TensorFlow特有的语义和属性。
- **`tosa`方言：** 表示张量算子集架构，这是一组标准化的张量操作，旨在作为不同框架的通用目标。相比`tf`方言，它给出一种更与框架无关的表示。
- **`mhlo` / `stablehlo`方言：** 最初来自XLA（加速线性代数），这些方言在适合进行如操作合并等高层优化的层级上表示操作，常作为从特定框架方言导入后的中间步骤。

首先选择使用哪种高层方言，通常取决于编译器的前端策略和源模型格式。重要的是，这些方言至少在初始阶段让编译器能够“理解”该ML框架的语言。

### 表示操作和属性

在一种高层方言中，每个MLIR*操作*对应于原始计算图中的一个节点。例如，TensorFlow中的2D卷积操作可能在`tf`方言中表示为`tf.Conv2D`操作，或在`tosa`方言中表示为`tosa.conv2d`操作。

重要的是，这些MLIR操作带有*属性*，捕获了高层配置细节，这些细节对于保留原始框架操作的精确语义非常必要。这些属性通常无法在像LLVM IR这样的低层IR中表示。例如：

- 填充模式（`SAME`, `VALID`）
- 步长（高，宽）
- 膨胀系数
- 数据布局规范（`NHWC`, `NCHW`）
- 融合到操作中的激活函数 (activation function)
- 量化 (quantization)参数 (parameter)（缩放因子，零点，如果此层级适用）

考虑一个简化的TensorFlow `Conv2D`操作，其后跟`BiasAdd`和`Relu`激活：

> 一个涉及卷积、偏置 (bias)加法和ReLU激活的TensorFlow图片段。

此序列在MLIR `tf`方言中可能表示如下（为清晰起见，语法已简化）：

```mlir
// %input, %filter, %bias 是代表张量的MLIR SSA值

%conv_output = "tf.Conv2D"(%input, %filter) {
    strides = [1, 1, 1, 1],
    padding = "SAME",
    data_format = "NHWC",
    dilations = [1, 1, 1, 1]
  } : (tensor<1x28x28x1xf32>, tensor<5x5x1x32xf32>) -> tensor<1x28x28x32xf32>

%bias_add_output = "tf.BiasAdd"(%conv_output, %bias) {
    data_format = "NHWC"
  } : (tensor<1x28x28x32xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>

%output = "tf.Relu"(%bias_add_output)
  : (tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
```

请注意，MLIR操作（`tf.Conv2D`、`tf.BiasAdd`、`tf.Relu`）如何直接反映TensorFlow的构思。属性（`strides`、`padding`、`data_format`）附加到相关操作上，保留重要元数据。张量类型包括形状和元素类型信息。

### 使用SSA建模数据流

MLIR遵循静态单赋值（SSA）形式，这在现代编译器中很常见。在这种形式中，每个变量（表示为MLIR*值*）只定义一次，可以多次使用。这自然地表示计算图中固有的数据流依赖关系。

- 一个MLIR操作的结果是SSA值（例如，上面例子中的`%conv_output`、`%bias_add_output`、`%output`）。
- MLIR操作的输入（操作数）是由前置操作（或函数参数 (parameter)/常量）产生的SSA值。

这种使用-定义链结构清晰地表示了原始计算图的有向边，使数据依赖关系清晰，并有助于分析和转换。MLIR表示，即使使用高层方言，本质上也是一种以SSA形式表示的数据流图。

### 高层图表示的优点

在编译器中使用这些高层方言表示ML模型很重要，因为它允许优化在熟悉、特定范围的构思上进行，避免在降低层级时丢失重要信息。例如：

1. **特定框架的操作合并：** 诸如“Conv2D -> BiasAdd -> Relu”的序列很常见，且常有优化的核实现。当`tf.Conv2D`、`tf.BiasAdd`和`tf.Relu`等操作明确时，识别这种序列相对简单。低层IR会将这种序列隐藏在通用的内存访问和算术操作背后。
2. **布局优化：** 关于最佳数据布局（例如NHWC与NCHW）的决定通常取决于操作序列和目标硬件特性。当高层操作及其布局（如`data_format`属性）可见时，这些决定最好做出。
3. **代数简化：** 高层代数恒等式（例如，消除冗余的转置操作）在此层级更容易识别和应用。

通过保留源图的高层语义，MLIR方言（如`tf`和`tosa`）提供了进行强大图级优化所需的抽象层。这种表示作为渐进式降低过程的入口点，其中这些高层操作将逐步转换并分解为低层方言，最终变为硬件特定指令，我们将在后续小节和章节中进行讨论。

## 参考资料

- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://dl.acm.org/doi/abs/10.1145/3441558.3446095) — Chris Lattner, Mehdi Amini, River Riddle, Albert Cohen, Alan Daynes, John Field, Georges-Axel Jaloyan, Artem Krassovsky, Joshua Lee, Roman Popov, Siegfried Puchbauer, Tatiana Shpeisman, Nicolas Vasilache (2021)
  Journal: Proceedings of the 2021 International Symposium on Code Generation and Optimization (CGO); Publisher: ACM; Pages: 250–262; DOI: [10.1145/3441558.3446095](https://doi.org/10.1145/3441558.3446095)
  介绍了MLIR的基础架构和设计原则，包括其多级IR方法、方言系统和SSA形式，对于理解其在表示高级ML图中的作用至关重要。
- [TOSA (Tensor Operator Set Architecture) Specification](https://mlplatform.org/tosa/tosa_spec.html) — Khronos Group (2023)
  Publisher: Khronos Group
  定义了用于机器学习的张量算子集架构，与TOSA方言直接相关。
- [MLIR for TensorFlow: The Compiler Stack for the Next Generation of ML Hardware](https://mlir.llvm.org/presentations/2019-10-24-MLIR-for-TensorFlow.pdf) — River Riddle, Nicolas Vasilache, Chris Lattner, Mehdi Amini, Albert Cohen, Alan Daynes, John Field, Georges-Axel Jaloyan, Artem Krassovsky, Joshua Lee, Roman Popov, Siegfried Puchbauer, Tatiana Shpeisman (2019)
  Journal: LLVM Developers' Meeting (Presentation)
  由MLIR和TensorFlow核心开发者进行的演示，详细介绍了TensorFlow与MLIR集成的动机和设计，特别是TF方言在高层图表示和优化方面的作用。
- [StableHLO: An Open Standard for ML Models](https://openxla.dev/stablehlo/) — OpenXLA Project (2023)
  Publisher: OpenXLA Project
  StableHLO的官方网站和规范，它建立在MHLO之上，提供了一种可移植且稳定的ML模型表示，与`stablehlo`方言相关。

---

[上一节](03-MLIR%EF%BC%9A%E6%96%B9%E8%A8%80%E5%92%8C%E6%93%8D%E4%BD%9C.md) · [下一节](05-MLIR%20%E4%B8%AD%E7%9A%84%E4%B8%8B%E6%B2%89%E8%B7%AF%E5%BE%84.md)

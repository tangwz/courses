# 量化感知训练 (QAT) 的编译器处理过程

来源：[原文](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-8-quantization-low-precision-optimizations/compiler-passes-qat)

[返回章节目录](README.md) · [返回课程目录](../README.md)

量化 (quantization)感知训练 (QAT) 提供了一种方法，可以减少训练后量化 (PTQ) 常常导致的精度损失。通过在模型训练阶段模拟量化效果，模型参数 (parameter)能够适应降低的精度，这通常会使量化模型获得更高的最终精度。虽然QAT的核心过程在训练期间发生于机器学习 (machine learning)框架（如TensorFlow或PyTorch）内部，但编译器在将QAT训练后的模型转换为高效可执行的部署格式方面发挥着重要作用。这涉及专门的编译器处理过程，旨在解析、优化和转化QAT过程中引入的图结构成分。

### QAT后编译器的作用

在QAT期间，框架通常会在计算图中插入“伪量化 (quantization)”或“量化模拟”节点。这些节点在前向传播过程中模拟量化对权重 (weight)和激活值的舍入和钳制效果，使得梯度能够传递（通常使用直通估计器等方法）进行反向传播 (backpropagation)。常见的例子包括TensorFlow的`tf.quantization.fake_quant_with_min_max_vars`或PyTorch的`torch.quantization.FakeQuantize`模块。

QAT的输出是一个标准模型文件（例如SavedModel或TorchScript对象），其中包含嵌入 (embedding)在图结构中的这些显式伪量化操作。编译器的任务从这里开始：它必须处理此图，理解伪量化节点的意图，并将其转换为适合目标硬件的真实低精度计算。

### 在编译器IR中表示QAT成分

编译器的中间表示 (IR)，如MLIR，需要构建机制来表示这些QAT成分。伪量化 (quantization)节点及其相关参数 (parameter)（训练期间学习到的最小/最大范围）必须被准确捕获。这可能包括：

1. **专门的QAT操作：** 在IR方言中定义特定操作（例如MLIR量化方言中的`quant.fake_quant`操作），直接对应框架的操作。这些操作将携带存储量化参数（最小/最大范围、比特数、对称/非对称）的属性。
2. **带属性的标准操作：** 使用标准操作上的属性或通过IR中与张量相关的元数据来表示伪量化的 *效果*。

核心在于，IR必须保持量化参数以及在图中应用量化模拟的精确位置。

### 处理QAT图的编译器处理过程

有几个编译器处理过程对于处理QAT训练模型是必要的：

#### 1. 识别与标准化

第一步是可靠识别框架插入的伪量化 (quantization)节点。不同版本或框架可能使用略有不同的操作。标准化处理会将这些表示统一为编译器IR中的一致形式。这简化了后续的优化处理，使其能够针对伪量化的单一、明确定义的表示进行操作。

#### 2. 量化参数 (parameter)提取

需要有处理过程来提取嵌入 (embedding)在伪量化节点中的量化参数（最小/最大范围或导出的缩放因子/零点）。这些在QAT期间学习到的参数，对于在推理 (inference)阶段执行实际量化十分必要。此信息通常作为元数据或属性附加到IR中相应的张量或操作上。对于权重 (weight)张量 $W$，QAT过程可能会得到 $min_W$ 和 $max_W$。编译器处理会提取这些值并计算仿射量化映射所需的缩放因子 $s_W$ 和零点 $z_W$：
$W_{int8} = \text{取整}(W / s_W) + z_W$
其中 $s_W = (max_W - min_W) / (2^N - 1)$，而 $z_W$ 取决于量化是对称的还是非对称的 (N为比特数，通常是8)。

#### 3. 伪量化融合与合并

这也许是最重要的优化。伪量化节点主要用于训练模拟。为了高效推理，它们的计算效果应直接合并到消耗或生成量化张量的操作中。

- **权重量化：** `Constant -> FakeQuant -> Conv2D` 这样的模式应被转换。应用于常量权重张量的 `FakeQuant` 操作被消除。编译器会修改 `Conv2D` 操作本身，使其成为一个“量化Conv2D”操作，直接使用已静态量化的权重张量（存储为INT8），并在计算期间应用相应的缩放因子。
- **激活量化：** `Conv2D -> FakeQuant -> ReLU` 这样的模式可能被转换。模拟卷积后激活张量量化的 `FakeQuant` 被向前融合到后续操作或向后融合到 `Conv2D` 中。目标是指定（可能已量化的）`Conv2D` 的输出应以量化格式生成或存储，从而消除显式模拟节点。

> QAT融合处理对计算图片段的转换示意。训练期间模拟量化的伪量化节点被吸收到主要操作（Conv2D）中，形成一个直接处理低精度数据和参数的量化操作（QuantizedConv2D）。

融合消除了伪量化节点的运行时开销，并使得编译器后端能够生成使用高度优化的低精度硬件指令（例如，CPU上的INT8点积或GPU上的张量核心操作）的代码。

#### 4. 冗余量化操作消除

融合后，图可能包含数据被去量化后立即再量化的序列，可能发生在可以在量化域中保留的操作之间。例如：`QuantizedConv2D -> Dequantize -> Quantize -> QuantizedAdd`。处理会分析这些模式，并消除不必要的去量化/量化对，尽可能保持数据流在低精度域中，以最小化精度转换开销。

#### 5. 转化到底层量化核

最后，高级量化操作（如融合后的 `QuantizedConv2D`）需要被转化到具体的实现。这包括：

- 映射到供应商特定的量化库（例如cuDNN、MIOpen、oneDNN），这些库提供高度优化的低精度核函数。
- 直接生成目标特定的代码，利用硬件指令进行整数算术、带展宽累加器的乘加操作，并处理缩放因子。例如，在Intel CPU上生成VNNI指令，或在NVIDIA GPU上生成DP4A/IMMA指令。

这一转化步骤将逻辑量化操作转换为可在目标硬件上执行的具体、高效的代码。

### 处理训练残留

QAT图可能包含训练过程的残留，例如与伪量化 (quantization)节点相关的自定义梯度计算，或用于阻止梯度流过量化参数 (parameter)的操作。编译器处理必须识别并剪除这些推理 (inference)无关的操作，确保最终优化图中只保留必要的前向计算。

通过精心实现这些处理过程，编译器可以有效地衔接包含模拟节点的QAT训练模型与高度优化的推理就绪模型，后者能够充分发挥低精度硬件执行的潜力，在保留通过QAT获得的精度优势的同时，获得性能提升。

## 参考资料

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 2704-2713; DOI: [10.48550/arXiv.1712.05877](https://doi.org/10.48550/arXiv.1712.05877)
  本文介绍了量化感知训练（QAT）的原理，包括伪量化及其在实现深度神经网络高效纯整数推理中的作用。
- [MLIR Quantization Dialect](https://mlir.llvm.org/docs/Dialects/QuantDialect/) — The MLIR Authors (2024)
  Publisher: LLVM Project
  官方文档，描述了MLIR编译器框架中用于表示量化计算的中间表示（IR）构造和操作。
- [Quantization for PyTorch Models](https://pytorch.org/docs/stable/quantization.html) — The PyTorch Authors (2019)
  Publisher: PyTorch Foundation
  PyTorch官方量化指南，包含量化感知训练（QAT）在框架内的实现细节和使用方法。

---

[上一节](02-%E5%9C%A8IR%E4%B8%AD%E8%A1%A8%E7%A4%BA%E9%87%8F%E5%8C%96%E6%93%8D%E4%BD%9C.md) · [下一节](04-%E8%AE%AD%E7%BB%83%E5%90%8E%E9%87%8F%E5%8C%96%20%28PTQ%29%20%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B.md)

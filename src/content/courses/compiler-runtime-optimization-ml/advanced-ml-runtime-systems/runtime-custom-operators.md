---
course: "compiler-runtime-optimization-ml"
chapter: "advanced-ml-runtime-systems"
lesson: "runtime-custom-operators"
sourceId: 4554
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-6-advanced-ml-runtime-systems/runtime-custom-operators"
title: "整合定制算子与核函数"
description: "了解注册和调用用户定义或外部编译核函数的机制。"
order: 6
plots: []
sourceHash: "996142027ecc297890ec94dfd0eba756e47eb475f1c50abeee72c0b85a26e531"
sourceCorrections: []
---

尽管机器学习 (machine learning)编译器擅长优化流行框架中常见的标准操作，但应用和研究经常需要一些不被原生支持的操作。这些操作可能包括新颖的激活函数 (activation function)、专门的数据预处理步骤、利用独特硬件功能的操作，或在主编译器流程之外开发的高度优化核函数。高级机器学习运行时系统必须提供机制来整合这些*定制算子*和*核函数*。

### 定制算子的动机

整合定制算子变得必要，原因有以下几点：

1. **性能：** 手动调优的核函数，可能用 CUDA、汇编或特定内部函数编写，在目标硬件上对特定操作的执行可能明显优于编译器生成的代码。
2. **新颖性：** 研究人员和工程师经常尝试使用新的层类型或算法，这些尚不属于标准库或编译器方言。
3. **硬件专用性：** 定制算子可以直接使用小众加速器或特定硬件指令的功能，这些通常不是编译器后端的目标。
4. **专有逻辑：** 将现有、可能是闭源的优化库或函数作为定制算子进行整合。
5. **框架桥接：** 使用定制算子作为桥梁，调用来自其他库或系统的函数。

### 整合流程

整合定制算子通常需要编译器前端、编译器优化遍和运行时系统之间的配合。

1. **表示：** 定制操作需要在机器学习 (machine learning)模型的图表示中体现（例如，作为 TensorFlow/PyTorch 图中的特定节点类型或 MLIR 方言中的定制操作）。这种表示向编译器表明该节点需要特殊处理。
2. **编译器处理：** 编译器的优化遍通常将定制算子节点视为不透明单元。虽然可能发生涉及定制操作*输出*的常规图优化（如常量折叠），但定制操作的*内部*通常不会被标准优化流水线转换（例如，合并可能在定制操作边界处被阻止）。编译器在后端代码生成阶段对定制操作的主要作用是发出代码，调用运行时执行该特定定制操作的机制，并传递所需输入和为输出分配空间。
3. **运行时注册：** 整合的核心在于运行时。定制算子的实现（实际代码/核函数）必须让运行时系统知晓。这通常通过注册 API 实现。
4. **运行时调度：** 在执行期间，当运行时遇到执行定制算子的指令时，它会使用算子的标识符（例如名称或类型）查找已注册的实现并调用它，传递必要的上下文 (context)和张量数据。

> 整合并执行定制算子的流程。编译器保留定制算子节点并生成调用运行时的代码，运行时随后查找并调用已注册的核函数实现。

### 运行时注册机制

主要方面包括：

- **算子名称/标识符：** 标识定制算子的唯一字符串或枚举（例如，`MyCustomAttention`、`SpecialPreprocessing`）。这必须与图表示中使用的标识符匹配。
- **实现指针：** 指向执行该操作的实际 C++/CUDA 等代码的函数指针（或等效机制，如函数对象）。
- **设备专用性：** 注册通常需要是设备专用的（例如，为同一逻辑算子注册独立的 CPU 和 GPU 实现）。运行时会根据张量放置选择合适的实现。
- **元数据（可选）：** 一些运行时允许注册元数据，例如预期的输入/输出类型和形状（或推断它们的函数），这有助于验证或与定制操作谨慎交互的图优化遍。

一个简化的注册 API 可能如下所示（C++）：

```cpp
// 核函数的前向声明
Status my_custom_op_cpu_kernel(KernelContext* context);

// 注册函数（通常在库加载时调用）
void register_ops() {
  OpRegistry* registry = Runtime::GetGlobalOpRegistry();

  registry->Register("MyCustomOp")
      .Device(DeviceType::CPU)
      .Implementation(my_custom_op_cpu_kernel)
      .Input("input_tensor", DataType::FLOAT32) // 可选元数据
      .Output("output_tensor", DataType::FLOAT32); // 可选元数据

  // 也可能在此处注册 GPU 版本
  // registry->Register("MyCustomOp").Device(DeviceType::GPU)...
}
```

实现可以静态链接到主应用程序中，也可以动态加载（例如，从共享对象 `.so` 或动态链接库 `.dll`）。动态加载提供了灵活性，允许用户在不重新编译整个运行时系统的情况下添加定制算子。

### 核函数接口与上下文 (context)

定制核函数的函数签名非常重要。运行时需要传递所有必要信息：

- **执行上下文：** 提供对运行时资源的访问，例如计算流（例如，用于 GPU 的 `cudaStream_t`）、临时内存分配器，以及可能的性能分析工具。
- **输入张量：** 关于每个输入张量的信息，包括：
  - 数据指针（`void*` 或类型化指针），位于正确的设备上。
  - 数据类型（例如，`float32`、`int8`）。
  - 形状（维度）。
  - 步长（针对非连续张量）。
- **输出张量：** 指向预分配内存缓冲区的指针，核函数应将结果写入此处。运行时通常根据形状推断（如果可能）或在注册或图构建期间提供的形状信息来处理分配。
- **属性：** 与图中算子节点关联的任何编译时属性（例如，`dilation_rate`、`epsilon`）。

一个典型的核函数签名可能如下所示：

```cpp
// 简化的核函数上下文结构
struct KernelContext {
  void* stream; // 例如，cudaStream_t 或等效物
  Allocator* temp_allocator;
  // ... 其他上下文信息
};

// 简化的张量信息结构
struct TensorInfo {
  void* data;
  DataType dtype;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DeviceType device;
};

// 定制核函数的示例签名
Status my_custom_op_gpu_kernel(
    KernelContext* context,
    const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs,
    const std::map<std::string, AttributeValue>& attributes
) {
  // 实现时使用 context->stream, inputs[0].data, 等。
  // 检查属性、输入形状/类型。
  // 启动 GPU 核函数。
  // 返回 Status::OK 或错误代码。
}
```

### 数据管理与同步

运行时负责确保输入张量数据在定制核函数预期其存在的设备上可用。如果用 CPU 张量输入调用定制 GPU 核函数，运行时必须管理数据传输（可能是异步的）。类似地，在设备上产生的输出可能需要传回。

定制核函数，特别是 GPU 核函数，通常异步执行。核函数实现必须正确使用提供的执行流（示例中的 `context->stream`）来排队其工作。运行时需要管理依赖关系，确保定制核函数启动与先行操作同步，并且后续操作在必要时等待定制核函数完成（例如，通过记录和等待与流关联的事件）。不正确的同步是整合定制核函数时常见的错误源。

### 挑战与考量

- **ABI 稳定性：** 如果定制核函数动态加载，运行时与定制核函数之间的接口（KernelContext、TensorInfo 结构、函数签名）必须在不同版本之间保持稳定，或必须存在版本控制机制。破坏应用程序二进制接口（ABI）可能导致崩溃或不正确的行为。
- **性能开销：** 调度机制本身（查找并调用函数指针）相比于完全编译和内联的代码增加了一些开销。数据编组（打包张量信息）也造成了影响。
- **调试：** 调试跨越运行时和定制核函数边界的代码（可能使用不同语言或在不同编译单元中）具有挑战性。标准调试器可能难以在这两者之间进行单步调试。
- **内存管理：** 定制核函数必须正确地与运行时的内存管理器交互，特别是在分配临时缓冲区时。使用提供的 `temp_allocator` 可确保缓冲区在运行时的内存计划内管理；独立分配内存可能会干扰运行时的优化和跟踪。
- **构建系统复杂性：** 将定制算子的构建过程（例如，编译 CUDA 代码）整合到主应用程序构建系统需要仔细配置。

有效支持定制算子是灵活强大的机器学习 (machine learning)运行时系统的一个显著特点，它使用户能够突破性能限制，尝试新颖的模型架构，这些能力是仅靠标准编译器优化难以实现的。

## 参考资料

- [TensorFlow Custom Operations Guide](https://www.tensorflow.org/guide/create_op) — TensorFlow Team (2024)
  该官方指南提供了在TensorFlow中创建和集成自定义操作的全面说明和示例，包括内核实现、C++操作注册和Python API绑定。
- [PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/docs/stable/cpp_extension.html) — PyTorch Team (2024)
  官方文档详细说明了如何使用自定义C++和CUDA操作扩展PyTorch，包括设置、编译以及与Python前端的集成。
- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://dl.acm.org/doi/10.1145/3477174) — Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, Oleksandr Zinenko (2021)
  Journal: ACM Transactions on Programming Languages and Systems; Publisher: ACM; Volume: 43; Pages: 1-84; DOI: [10.1145/3477174](https://doi.org/10.1145/3477174)
  这篇基础论文介绍了MLIR，它是现代ML编译器中使用的关键中间表示，解释了其可扩展的方言系统，该系统允许有效表示和集成自定义操作。

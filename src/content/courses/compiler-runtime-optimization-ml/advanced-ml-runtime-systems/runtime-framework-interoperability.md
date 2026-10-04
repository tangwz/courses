---
course: "compiler-runtime-optimization-ml"
chapter: "advanced-ml-runtime-systems"
lesson: "runtime-framework-interoperability"
sourceId: 4556
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-6-advanced-ml-runtime-systems/runtime-framework-interoperability"
title: "机器学习框架的互操作性"
description: "分析独立运行时如何与TensorFlow和PyTorch等高级框架交互。"
order: 7
plots: []
sourceHash: "e7aaf4df19065b10b85374abf91ff5a63818bdf52b96332254f52c53acab3966"
sourceCorrections: []
---

尽管专用编译器和运行时在为目标硬件优化特定模型或子图方面表现出色，但这些优化组件很少独立存在。它们必须平稳地融入到模型开发、训练和部署的更大体系中，通常涉及TensorFlow或PyTorch等高级框架。确保有效的互操作性是运行时系统设计的一个主要方面。这涉及在宿主框架和专用运行时之间定义清晰的控制传输、数据交换和资源管理接口。

主要目的是让开发者在其偏好的框架中工作时，能受益于运行时提供的优化执行，通常只需对现有代码进行极少改动。这需要弥合框架的高级、通常动态的执行模型与运行时更底层、可能静态优化的执行环境之间的差异。

### 集成机制

框架通常提供多个专用运行时可使用的扩展点：

1. **自定义操作 (Op)：** 这是一个常见方法。运行时将其功能（例如，执行编译后的子图）封装在自定义操作符定义中。该操作符在框架中注册（例如，使用TensorFlow中的`tf.RegisterOp`或PyTorch的C++/CUDA扩展机制）。从框架的角度来看，它只是计算图中的另一个节点或即时执行中的另一个函数调用。此自定义操作的实现涉及调用运行时的API来加载编译好的产物、准备输入、执行并获取输出。
2. **后端/设备插件：** 更复杂的框架提供集成替代计算后端或虚拟设备的机制。示例包括TensorFlow的PluggableDevice接口或PyTorch的分发器可扩展性（`__torch_dispatch__`，外部后端）。在此，运行时充当特定设备或计算后端的实现。框架会拦截指向此后端的操作，并将其转发到运行时对应的核函数实现。这使得集成程度可能比自定义操作更深，使运行时能够更直接地管理自己的设备内存和执行流。
3. **JIT编译器集成：** 框架通常包含自己的JIT编译器（例如XLA、TorchScript）。专用运行时可能作为框架JIT的后端。框架JIT执行初始图捕获和高级优化，然后将图的部分内容降级为专用运行时编译器可处理的中间表示。运行时随后处理这些特定子图的优化、代码生成和执行的最后阶段。

以下图表说明了这些交互点：

> 高级机器学习 (machine learning)框架与专用运行时系统之间的交互点。控制和数据通过自定义操作符或后端插件等已定义接口流动。

### 数据交换协议

重要考量包括：

- **内存空间感知：** 框架张量可能驻留在CPU内存或特定设备（例如GPU）上。运行时需要从这些位置接收数据，并将其输出放置到框架期望的位置。这通常涉及理解设备指针，并可能管理数据传输（例如`memcpy`、`cudaMemcpyAsync`）。
- **零拷贝机制：** 复制大型张量代价高昂。在可能的情况下，运行时应追求零拷贝共享。这可能涉及：
  - 如果运行时可以访问框架分配的内存空间，则直接对其进行操作（例如，对PyTorch分配的GPU缓冲区进行操作）。
  - 使用标准化的缓冲区交换协议，例如`dlpack`，它提供了一种通用方式来描述张量元数据（形状、步长、数据类型、设备）并共享底层内存指针，即使在不同库之间也无需复制。
  - 在宿主（CPU）端进行内存锁定，以实现向运行时管理的设备（GPU/加速器）进行高效的异步DMA传输。
- **数据布局转换：** 框架通常默认为特定布局（例如NCHW），而运行时可能已为其核函数优化了不同的布局（例如NHWC或平铺格式）。接口层必须处理这些潜在的布局转换，无论是通过自定义操作/后端内明确处理，还是通过通知运行时编译器生成接受框架布局的代码。
- **数据类型映射：** 运行时必须正确解析框架数据类型（例如`float32`、`bfloat16`、`int8`）并处理任何必要的转换，特别是在处理量化 (quantization)模型时，还需要传递比例因子和零点信息。

### 控制流与同步

框架通常驱动整体执行流程。调用运行时时：

- **调用：** 框架调用运行时的入口点（例如，自定义操作的`Compute`方法，一个后端函数）。此调用通常传递输入张量和任何必要的配置参数 (parameter)。
- **异步执行：** 为了最大化并行度，运行时执行（特别是在加速器上）应相对于框架的主控制线程是异步的。框架管理自己的执行流（例如CUDA流）。运行时必须正确地将其操作与框架的流同步。这通常涉及：
  - 接受一个框架流句柄作为输入。
  - 将自己的核函数和内存传输排队到该流上。
  - 使用框架提供的事件或运行时特定的同步原语（例如`cudaEventRecord`、`cudaStreamWaitEvent`）来建立框架操作与运行时操作之间的依赖关系。
- **错误处理：** 运行时错误（例如，核函数启动失败、内存不足）必须被捕获并以框架可理解的方式传回框架，通常通过返回特定的错误代码或引发跨语言边界（例如C++到Python）的异常。

### 生命周期与配置管理

运行时通常需要初始化并管理其自身状态和资源。

- **初始化：** 运行时可能需要一次性初始化，包括发现可用硬件、分配内部资源（如内存池或线程池）或加载预编译的产物。这种初始化可能在首次调用时延 (latency)迟进行，或通过框架或用户代码的配置调用明确进行。
- **配置：** 用户可能需要传递运行时特定的配置（例如，选择特定的加速器核心、设置内存限制、启用/禁用特定的优化）。接口必须提供传递这些设置的机制。
- **资源管理：** 运行时必须适当地释放其资源（设备内存、宿主内存、文件句柄）。这通常与框架内自定义操作对象或后端插件的生命周期相关联。适当的引用计数或明确的清理函数对于防止资源泄漏是必要的。

设计互操作性需要仔细考虑目标框架的扩展机制、数据处理约定和执行模型。良好定义的接口允许专用运行时接入这些框架，在不扰乱用户现有开发流程的情况下提供优化的性能。

## 参考资料

- [TensorFlow PluggableDevice](https://www.tensorflow.org/install/gpu_device_plugins) — TensorFlow Authors (2024)
  TensorFlow官方文档，描述了集成定制硬件设备和替代计算后端的接口，是实现更深层运行时互操作性的关键机制。
- [Extending PyTorch with C++ and CUDA](https://pytorch.org/docs/stable/cpp_extension.html) — PyTorch Authors (2018)
  Publisher: PyTorch Foundation
  PyTorch官方指南，介绍如何创建定制的C++和CUDA操作，以及如何通过外部运行时扩展框架以实现专用、高性能的内核或子图。
- [DLPack: A Standard for Tensor Exchange](https://github.com/dmlc/dlpack) — DMLC Community (2024)
  DLPack的官方GitHub仓库和规范。DLPack是跨框架的零拷贝张量数据交换标准，是ML框架与专用运行时之间高效数据交换的基础。
- [TVM: An End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594; DOI: [10.5555/3295222.3295267](https://doi.org/10.5555/3295222.3295267)
  TVM的奠基性论文，TVM是一个开源的深度学习编译器和运行时，常作为专用后端与高级框架集成，展示了JIT编译器集成和优化执行的原理。

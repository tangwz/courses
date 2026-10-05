# 选择框架 (PyTorch, TensorFlow, JAX)

来源：[原文](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/framework-choice-transformers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

实现和优化强大的 Transformer 模型需要考虑实际操作事项。此过程中的一个重要首步是选择合适的深度学习 (deep learning)框架。这项选择显著影响开发速度、调试便捷性、性能提升途径以及部署方案。当前用于认真构建 Transformer 的主要选择是 PyTorch、TensorFlow 和 JAX，它们各自提供不同的优势和理念。

### PyTorch

PyTorch 主要由 Meta AI 开发，获得了相当大的关注，尤其是在研究群体中。它的吸引力常源于其“Python风格”的设计。调试通常更直接，因为它默认采用即时执行模式，能立即执行操作，这与标准 Python 程序流程相仿。这使得检查中间张量和使用 `pdb` 等标准 Python 调试工具相对简单。

Transformer 开发的主要优势包括：

- **灵活性与控制力：** PyTorch 提供一个相对低级的接口（与 Keras 相比），对模型架构和训练循环提供细致的控制，这对于实现自定义 Transformer 变体或高级优化方法很有益处。
- **完善的生态：** 它与流行的 Hugging Face `transformers` 库集成，方便使用大量的预训练 (pre-training)模型和分词 (tokenization)器 (tokenizer)。像 `Accelerate` 这样的库简化了分布式训练和混合精度使用。
- **研究群体：** 它在研究领域的普遍使用意味着新方法和模型架构常首先在 PyTorch 中实现。
- **性能：** 尽管传统上以易用性著称，PyTorch 凭借 `torch.compile` 在性能优化方面取得了显著进展，它能融合操作并利用 Triton 等后端加速模型执行，性能常接近编译图。`Torch Distributed` 提供数据和模型并行化的工具。

### TensorFlow

TensorFlow 最初由 Google Brain 开发，是一个成熟的框架，非常注重生产部署和可扩展性。其高级 API Keras 现在是与 TensorFlow 交互的标准方式，提供用户友好的界面来定义模型和训练流程。

与 Transformer 有关的重要方面：

- **生产部署：** TensorFlow 提供一套完整的部署工具，包括用于高性能推理 (inference)服务器的 TensorFlow Serving、用于移动和边缘设备的 TensorFlow Lite，以及用于端到端 MLOps 流水线的 TensorFlow Extended (TFX)。
- **图编译：** TensorFlow 主要通过首先构建计算图（`tf.function` 装饰器）然后执行它来运行。这允许通过其 XLA (加速线性代数) 编译器进行广泛的图级优化，可能带来高性能，尤其是在 Google TPU 等硬件加速器上。
- **可扩展性：** TensorFlow 对分布式训练策略有成熟的支持，能够训练跨多个 GPU 或 TPU pod 的大型模型。
- **生态系统：** 它受益于 TensorBoard 等可视化工具和一个提供支持和扩展的庞大社区，尽管 PyTorch 在最新研究思想的快速采纳方面可以说已经超越它。

### JAX

JAX 同样由 Google Brain 开发，是一个较新的库，旨在进行高性能数值计算，特别适合涉及大型模型和硬件加速器的机器学习 (machine learning)研究。它不像 PyTorch 或 TensorFlow 那样是一个完整的深度学习 (deep learning)框架，但为 NumPy 代码提供了可组合的函数变换。

Transformer 工作的重要特征：

- **函数变换：** JAX 的核心优势在于其变换：
  - `grad`：自动微分。
  - `jit`：使用 XLA 进行即时编译，显著提升速度。
  - `vmap`：自动向量 (vector)化（批处理）。
  - `pmap`：跨多个设备（GPU/TPU）的自动并行化，简化数据和模型并行化实现。
- **性能侧重：** JAX 常被选择用于提升性能和规模的极限，尤其是在 TPU 上，因为 `pmap` 与硬件架构天然契合。
- **函数式方法：** JAX 鼓励函数式编程风格（纯函数），这可以带来更清晰、更可预测的代码，但对于习惯面向对象框架的人来说可能是一个学习曲线。状态管理（如模型参数 (parameter)和优化器状态）是明确处理的。
- **成长中的生态：** Flax 和 Haiku 等库在 JAX 之上提供了更高级的抽象，使 Transformer 实现更有条理。

### Transformer 的框架考量

在这些框架中选择取决于项目需求、团队专长和目标基础设施。以下是比较汇总：

| 特性 | PyTorch | TensorFlow (带 Keras) | JAX |
| --- | --- | --- | --- |
| **主要 API** | 命令式 (Eager), Python 风格 | 声明式 (Keras), 基于图 (`tf.function`) | 函数式, 类似 NumPy, 变换 |
| **调试** | 通常较容易 (Eager 模式) | 可能较难 (图模式), Keras 简化 | 需理解 JIT/变换 |
| **性能** | 卓越 (尤其使用 `torch.compile`) | 卓越 (尤其使用 XLA) | 潜力最高 (尤其在 TPU 上, `pmap`) |
| **灵活性** | 高, 控制良好 | 中等 (Keras), 高 (低级 TF) | 很高 (低级, 函数式) |
| **生态系统** | 强大的研究社群, Hugging Face 集成 | 成熟的生产环境, TFX, TensorBoard | 快速发展, 注重研究 |
| **部署** | 良好 (TorchServe, ONNX) | 卓越 (TF Serving, TFLite) | 需要更多专业/定制化 |
| **分布式** | (`DistributedDataParallel`, FSDP) | (`MirroredStrategy`, DTensor) | 集成 (`pmap`) |
| **学习曲线** | 中等 | 中等 (Keras), 更陡峭 (TF Core) | 更陡峭 (函数式, 变换) |

**建议：**

- **对于快速原型开发、研究以及获取最新模型：** PyTorch 常是一个强大的选择，因为其灵活性以及与 Hugging Face 生态系统的紧密结合。
- **对于具有不同部署目标（服务器、移动、边缘）的生产流水线：** TensorFlow 成熟的部署工具提供显著优势。
- **对于尖端性能、大规模训练（尤其在 TPU 上）以及偏好函数式编程：** JAX 提供强大的工具，尽管它可能需要更多的初始投入来掌握其思想。

最终，所有这三个框架都能高效实现复杂的 Transformer 架构。熟悉团队现有的技能和基础设施常起决定作用。如果可行，在不同框架中尝试小型 Transformer 实现，可获得关于各自工作流程和取舍的宝贵见解。对于任何认真从事现代大型语言模型工作的工程师而言，至少精通其中一个框架是必不可少的。

## 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/) — PyTorch contributors (2024)
  解释了 PyTorch 框架的核心特性、API 设计、如 `torch.compile` 等性能优化以及分布式训练工具。
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs) — Google and the TensorFlow Authors (2024)
  Publisher: Google
  详细介绍了 TensorFlow 的架构、Keras API、使用 `tf.function` 和 XLA 的图编译以及其广泛的部署功能。
- [JAX Documentation](https://jax.readthedocs.io/en/latest/) — Google and JAX contributors (2024)
  介绍了 JAX 的函数式编程模型、自动微分（`grad`）、JIT 编译（使用 `jit` 的 XLA）以及 `pmap` 等并行化原语。
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  描述了如何有效使用 `transformers` 库在 PyTorch、TensorFlow 和 JAX 中构建、预训练和微调 Transformer 模型。

---

[上一节](../06-%E9%AB%98%E7%BA%A7%E6%9E%B6%E6%9E%84%E5%8F%98%E4%BD%93%E4%B8%8E%E5%88%86%E6%9E%90/10-%E5%8F%82%E6%95%B0%E6%95%88%E7%8E%87%E4%B8%8E%E5%85%B1%E4%BA%AB%E6%8A%80%E6%9C%AF.md) · [下一节](02-%E6%9D%83%E9%87%8D%E5%88%9D%E5%A7%8B%E5%8C%96%E7%AD%96%E7%95%A5.md)

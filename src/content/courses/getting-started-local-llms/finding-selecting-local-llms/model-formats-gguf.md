---
course: "getting-started-local-llms"
chapter: "finding-selecting-local-llms"
lesson: "model-formats-gguf"
sourceId: 4236
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-3-finding-selecting-local-llms/model-formats-gguf"
title: "模型格式：GGUF及其他"
description: "理解GGUF等流行的模型文件格式，它们为在CPU和GPU上运行而优化。"
order: 3
plots: []
sourceHash: "82bd1b4b07851ac3d2dc73560394507b55c2875d4692877cd3b367ae7695c4b0"
sourceCorrections: []
---

理解模型的文件格式是使用大型语言模型的一个主要方面。可以把模型格式想象成电脑上的不同文件类型，比如`.txt`、`.docx`或`.pdf`。每种格式都有其自身的结构和用途，并非所有软件都能读取所有格式。对于大型语言模型，尤其是在本地运行时，文件格式会很大程度上影响其使用便捷性和在您机器上的运行表现。

模型通常使用大型、复杂的软件框架（如PyTorch或TensorFlow）进行训练。尽管这些框架在模型开发和训练方面功能强大，但其生成的模型文件（通常带`.pth`、`.pt`等扩展名，或存储在`SavedModel`等特定目录结构中）并不总是针对普通台式机或笔记本电脑硬件（特别是CPU）进行高效运行的优化。它们可能需要安装大型软件库才能加载和运行模型。

### GGUF登场：为本地使用优化

为了在您自己的电脑上高效运行大型语言模型，一种名为**GGUF**（Georgi Gerganov通用格式）的格式变得非常流行。在寻找本地运行的模型时，您会经常碰到以`.gguf`结尾的文件。

GGUF是早期格式GGML的后续版本。它被专门开发出来，旨在让大型模型在消费级硬件上运行变得可行。以下是它适合初学者和本地配置的原因：

1. **为CPU效率而设计：** 尽管它能发挥GPU的作用，但GGUF经过高度优化，即使您只有标准CPU也能良好运行。这使得更多用户可以方便使用它。
2. **集成量化 (quantization)：** GGUF与量化（我们将在下一节讨论）紧密结合。量化能减小模型大小和计算需求。GGUF文件通常是预量化的，这意味着大小缩减已经为您完成。
3. **基本自足：** GGUF文件旨在将模型参数 (parameter)（其“知识”）和所需的元数据（关于模型的信息，如其架构和文本处理方式）捆绑到一个文件中。这简化了模型的加载和运行。
4. **广泛的工具支持：** 许多旨在方便本地运行大型语言模型的流行工具，如Ollama、LM Studio以及底层`llama.cpp`库，都直接支持GGUF格式。这意味着您通常可以下载一个`.gguf`文件并轻松运行它。

> 一个简化的视图，展示了在PyTorch或TensorFlow等框架中训练的模型，如何经常被转换成GGUF格式，以便与本地大型语言模型运行工具更方便地配合使用。

### 您可能还会看到其他格式

尽管GGUF在本地易于运行的模型中很普遍，但您偶尔可能会看到其他格式的提及：

- **PyTorch (`.pt`, `.pth`)：** 这些是使用PyTorch框架训练的模型原生的格式。运行它们通常需要安装PyTorch，并可能需要编写Python代码来加载和操作模型。
- **TensorFlow (`SavedModel`, `.pb`)：** 与PyTorch格式类似，这些是TensorFlow框架原生的格式，通常需要安装TensorFlow库。
- **Safetensors (`.safetensors`)：** 这是一种新的格式，渐受关注，旨在安全高效地保存和加载模型权重 (weight)。它常与PyTorch等框架一起使用，但比一些旧格式（如Python的`pickle`，`.pth`文件有时使用）更安全。尽管更安全，但它可能仍需要底层框架（如PyTorch）来运行模型逻辑。

对于初学者，如果不想安装大型开发框架或编写代码，那么通常最直接的途径是专注于GGUF格式的模型。Ollama和LM Studio等工具简化了大部分复杂性，它们通常在后台依赖GGUF模型。随着您进一步的学习，您可能会接触其他格式，但GGUF为在个人电脑上运行大型语言模型提供了一个很好的起点。

## 参考资料

- [GGUF File Format](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov and llama.cpp contributors (2024)
  llama.cpp项目中为高效本地LLM推理开发的GGUF文件格式的官方规范和说明。
- [safetensors Documentation](https://github.com/huggingface/safetensors) — Hugging Face and safetensors contributors (2024)
  `safetensors`库的官方文档，提供了一种安全高效保存和加载模型权重的方法，是传统序列化方法的替代方案。
- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) — Matthew Inkawhich (2024)
  Publisher: PyTorch Foundation
  PyTorch官方关于保存和加载模型的教程，展示了如何使用原生框架格式存储和检索PyTorch训练的模型。
- [Save and load Keras models](https://www.tensorflow.org/tutorials/keras/save_and_load) — TensorFlow Team and contributors (2024)
  TensorFlow官方关于使用Keras API保存和加载模型的指南，涵盖了TensorFlow生态系统中的SavedModel等原生格式。

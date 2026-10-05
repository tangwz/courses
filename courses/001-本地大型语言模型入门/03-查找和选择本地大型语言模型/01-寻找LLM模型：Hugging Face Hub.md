# 寻找LLM模型：Hugging Face Hub

来源：[原文](https://apxml.com/zh/courses/getting-started-local-llms/chapter-3-finding-selecting-local-llms/hugging-face-hub)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要运行本地大型语言模型，首要步骤是获取模型文件。可以把LLM看作是生成文本的“引擎”；没有它，我们稍后讨论的工具就无法运行。但是，去哪里找到这些“引擎”呢？幸运的是，机器学习 (machine learning)社区有广泛的合作，并且线上有一些集中的地方供大家分享模型。

寻找AI模型，包括适合本地使用的LLM，最主要、使用最广的平台就是**Hugging Face Hub**。

可以想象一个庞大的数字图书馆，或者一个满是工具和部件的工坊，但专门用于人工智能。这与Hugging Face提供的很像。它是一个集中的交流点，供研究人员、开发者和机构共享和找到AI模型、数据集及相关软件工具。虽然它有多种类型的AI资源，但它已经成为寻找大型语言模型的主要场所。

### Hugging Face Hub是什么？

Hugging Face Hub（通常简称“Hugging Face”）是一个可通过网页浏览器访问的在线平台。它的主要功能是托管：

- **模型：** 用于各种任务的预训练 (pre-training)模型，包括文本生成（LLM）、图像识别、翻译等。
- **数据集：** 用于训练或评估模型的数据集。
- **Spaces：** 使用Hub中模型构建的演示和应用。
- **库：** 帮助开发者处理这些模型和数据集的软件工具（例如 `transformers`、`diffusers`）。

对我们本地运行LLM的目的而言，我们主要关注Hub的**模型**部分。这一部分包含了个人、研究团队以及Meta、Google、Mistral AI等公司贡献的数千个模型。

### 在Hub上寻找LLM

当你访问Hugging Face网站 (huggingface.co) 时，你会看到一个搜索栏和浏览模型的选项。通常可以这样寻找LLM：

1. **使用搜索栏：** 如果你知道具体的模型名称（例如“Llama 2”、“Mistral 7B”），可以直接搜索。你也可以搜索与适合本地使用的模型格式相关的词汇，比如“GGUF”，我们很快会讨论它。
2. **按任务筛选：** Hugging Face根据模型执行的任务进行分类。对于LLM，相关的任务通常是“文本生成”，有时是“对话”。按这些任务筛选有助于缩小选择范围。
3. **查看合集：** 有时，模型会被归入合集，这有助于找到相关模型或特定类型。

例如，搜索“Mistral 7B GGUF”可能会显示流行模型Mistral 7B的不同版本，它们以GGUF格式打包，这种格式常用于在消费级硬件上高效运行模型。

### 模型页面会看到什么

当你点击Hub上的一个模型时，会进入它的专属页面，通常被称为“模型卡片”。这个页面就像模型的使用手册或规格表。它通常包括：

- **模型描述：** 关于模型的信息、能力、训练方式和预期用途。
- **文件和版本：** 可供下载的模型文件列表。你可能会在这里看到不同的版本或格式。你会在这个地方寻找像 `.gguf` 这样的文件。
- **使用示例：** 使用模型的代码片段或说明（通常面向使用Hugging Face库的开发者，但仍有参考价值）。
- **许可信息：** 关于模型如何使用的详细信息（例如，仅用于研究，或商业用途）。我们将在本章后面讨论许可。
- **社区选项卡：** 用户的讨论、问题和贡献。

可以把Hugging Face Hub看作你浏览和找到你可能想在本地运行的LLM的主要起点。它的集中管理特性使得寻找模型比搜索分散的网站要容易得多。在接下来的部分，我们将讨论你在查看Hub上的模型时应注意的特点，例如它们的大小、格式和许可，以帮助你选择一个适合你的需求和硬件的模型。

## 参考资料

- [The Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index) — Hugging Face team (2024)
  Publisher: Hugging Face
  Hugging Face Hub的官方文档，说明了其功能以及如何与模型、数据集和空间进行交互。
- [llama.cpp: Port of Facebook's LLaMA model in C/C++](https://github.com/ggml-org/llama.cpp) — ggml-org and llama.cpp contributors (2024)
  Publisher: ggml-org
  `llama.cpp`的官方存储库，它实现了GGUF格式，用于LLM的高效本地推理。这有助于理解内容中提到的“GGUF”格式。

---

[上一节](../02-%E5%87%86%E5%A4%87%E6%9C%AC%E5%9C%B0%E7%8E%AF%E5%A2%83/07-%E5%91%BD%E4%BB%A4%E8%A1%8C-%E7%BB%88%E7%AB%AF%E7%AE%80%E4%BB%8B.md) · [下一节](02-%E7%90%86%E8%A7%A3%E6%A8%A1%E5%9E%8B%E5%A4%A7%E5%B0%8F%E4%B8%8E%E5%8F%82%E6%95%B0.md)

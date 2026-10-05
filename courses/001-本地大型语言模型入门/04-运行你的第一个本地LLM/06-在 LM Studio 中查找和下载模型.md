# 在 LM Studio 中查找和下载模型

来源：[原文](https://apxml.com/zh/courses/getting-started-local-llms/chapter-4-running-first-local-llm/downloading-models-lm-studio)

[返回章节目录](README.md) · [返回课程目录](../README.md)

LM Studio 提供内置图形界面，可以直接查找和下载模型，省去手动查找和管理模型文件的麻烦。通过 LM Studio 可获取的大多数模型来自 Hugging Face Hub，这是一个广受欢迎的机器学习 (machine learning)模型共享平台。

### 前往模型查找界面

打开 LM Studio 后，找到查找或模型浏览部分。这通常由放大镜图标表示，或标记 (token)为“查找”或“发现”之类的字样，在应用程序的主导航区域，通常位于左侧边栏。点击此处会进入模型浏览界面。

### 查找模型

你会看到模型浏览页面顶部有一个醒目的查找栏。你可以输入感兴趣的模型名称（例如，“Mistral”、“Llama 3”、“Phi-3”）或与其功能相关的关键词。当你输入时，LM Studio 通常会推荐 Hugging Face 上可用的匹配模型。

结果通常以模型列表的形式呈现。每个条目通常显示：

- **模型名称：** 通常采用 `组织/模型名称` 的格式（例如，`microsoft/Phi-3-mini-4k-instruct-gguf`）。
- **创建者/组织：** 上传该模型的实体。
- **简要说明：** 模型的简短概述。
- **指标：** 有时会显示下载次数或点赞数，以表明流行程度。

### 了解模型文件和量化 (quantization)选项

当你从查找结果中选择一个模型时，LM Studio 通常会显示详细信息和可用的下载选项，通常在屏幕右侧的面板中。此处一个重要特点是列出了多个模型文件，通常是 `.gguf` 格式。

你可能会在同一个模型名称下看到多个文件，例如：

- `phi-3-mini-4k-instruct-q4_k_m.gguf`
- `phi-3-mini-4k-instruct-q5_k_m.gguf`
- `phi-3-mini-4k-instruct-q8_0.gguf`
- `phi-3-mini-4k-instruct-f16.gguf`

这些不同的文件代表了同一基础模型的各种**量化级别**。正如第 3 章所述，量化是一个减少模型大小和计算需求的过程，使其可以在消费级硬件上运行。

- **量化标识符：** `Q4_K_M`、`Q5_K_M`、`Q8_0` 等术语表示具体的量化方法和级别。数字越小（如 Q2、Q3、Q4）通常意味着文件尺寸更小，RAM 使用量更低，但可能会轻微降低响应质量或准确性。数字越大（Q5、Q6、Q8）或未量化版本（`F16` 代表 16 位浮点数）能保留更多质量，但需要更多资源。
- **选择文件：** 对于初学者，通常选择像 `Q4_K_M` 或 `Q5_K_M` 这样的均衡量化级别是个不错的选择。它们在性能、资源使用和输出质量之间提供了合理的平衡。如果 LM Studio 中有推荐或默认建议，请查看它们。

### 下载前查看文件详情

在开始下载之前，请注意每个具体 `.gguf` 文件提供的详情：

- **所需估计内存：** LM Studio 通常会提供加载和运行该特定模型文件所需的内存估计。请确保此估计值在您系统可用内存的舒适范围内（您在第 2 章学习过如何查看）。
- **文件大小：** 留意下载大小。文件越大，下载所需时间越长，占用的磁盘空间也越多。
- **兼容性说明：** 有时可能会有关于特定文件兼容性或推荐设置的说明。
- **量化 (quantization)类型：** 确认量化级别与您的硬件能力和质量预期相符。

### 下载模型文件

一旦您确认了要下载的特定 `.gguf` 文件（根据量化 (quantization)、大小和内存需求），请在其列表中查找“下载”按钮。

点击此按钮将开始下载过程。LM Studio 通常会在应用程序内显示下载进度，通常以百分比或进度条的形式。您通常可以在专用部分或面板中查看进行中的下载。

### 查找已下载的模型

模型文件成功下载后，它将在 LM Studio 中可用，可以加载到聊天界面。已下载的模型通常列在应用程序的特定区域，可能标记 (token)为“我的模型”、“本地模型”或类似名称。您将在下一步中使用此下载的模型开始与您的第一个本地 LLM 进行交互。

## 参考资料

- [LM Studio: Run Local LLMs](https://lmstudio.ai/) — LM Studio Team (2024)
  用于在消费级硬件上本地下载、管理和运行大型语言模型的官方平台。
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Hub的官方指南，这是一个机器学习模型、数据集和演示的中心存储库。
- [GGUF File Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) — Georgi Gerganov and the llama.cpp community (2024)
  GGUF文件格式的详细规范，旨在本地系统上高效存储和推理大型语言模型。
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2023)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍了一种用于大型语言模型的著名训练后量化方法，可在最小性能下降的情况下减少内存和计算需求。

---

[上一节](05-%E8%AE%BE%E7%BD%AE%20LM%20Studio.md) · [下一节](07-%E5%9C%A8LM%20Studio%E4%B8%AD%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%BF%9B%E8%A1%8C%E8%81%8A%E5%A4%A9.md)

# 语音识别API与库简介

来源：[原文](https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-6-building-your-first-speech-recognition-app/speech-recognition-apis-and-libraries)

[返回章节目录](README.md) · [返回课程目录](../README.md)

前面的章节构建了语音识别的理论体系。我们了解到声学模型如何识别声音，以及语言模型如何提供语境，它们共同作用，以找到音频信号中最可能的词语序列。从头开始构建这样的系统是一项大工程，需要大量数据集和可观的计算能力。幸运的是，对于大多数应用来说，这并非必需。

您无需自行构建这些组件，而是可以使用现成的**库**和**应用程序编程接口 (API)**。这使您可以借鉴现有成果，仅需几行代码即可将复杂的语音识别功能集成到您的程序中。这里将介绍您会遇到的主要工具类别，以及我们将用于构建第一个应用的工具。

### 工具包：库和API

从宏观上看，向应用程序添加语音识别的工具可分为几类。理解它们之间的区别对于为您的项目选择合适的工具有帮助。

**库**是您安装并运行在您自己的计算机上的一组代码。对于ASR，这通常包含预训练 (pre-training)模型。您对过程有直接控制权，一旦安装，它可以完全离线工作。这就像在家里拥有专业级厨房电器；您拥有完全控制权，但其性能取决于您自己的配置。

另一方面，**API**是在远程服务器上运行的服务，通常由像Google、Amazon或Microsoft这样的公司管理。您的应用程序通过互联网向API发送音频文件，服务会返回文本转录。这就像从餐馆点餐；您无需管理厨房即可获得高质量结果，但您需要与餐馆建立连接，并且服务会有费用。

### 三种转录途径

集成ASR有三种常见方法，每种方法在复杂性、成本和控制方面各有取舍。

1. **云API：** 您可以直接使用诸如Google Cloud Speech-to-Text或Amazon Transcribe等服务。这种方法使您可以使用极度准确、大规模的模型。主要不足之处在于依赖互联网连接、基于使用量的潜在费用，以及需要将数据发送给第三方服务。
2. **本地开源库：** 您可以使用Hugging Face `transformers`等强大的开源库，直接在您的机器上运行OpenAI的Whisper等模型。这使您对数据有完全控制权，可以离线工作，并且通常免费。然而，它可能需要更多设置和一台性能较好的计算机才能高效运行。
3. **封装库：** 这些库为多种不同的ASR服务提供了简化的统一接口，包括云API和本地模型。它们非常适合学习和快速原型开发，因为它们为您处理了大部分复杂性。

> 使用ASR工具从应用程序到最终文本转录的不同路径。

### 我们选择的工具

对于本课程，我们将专注于为入门提供最大简便性和灵活性的途径：封装库。

#### `SpeechRecognition` 库

我们的主要工具将是Python `SpeechRecognition` 库。它是初学者的绝佳选择，原因有几点：

- **简洁性：** 它提供了一个单一、易用的ASR执行接口。
- **灵活性：** 它支持多种ASR引擎和API，包括Google Web Speech API、Sphinx（用于离线识别），以及来自Google、Microsoft等公司的API。
- **易用性：** 它允许您免费使用Google Web Speech API进行个人项目和学习，非常符合我们的需求。您无需注册云账户或提供信用卡即可转录音频。

这个库充当一个有用的管理器，使我们能以最少的代码将音频发送到能干的后端服务。

#### 关于现代开源模型的一点说明

语音识别领域发展迅速，强大的开源模型正变得普遍可用。一个突出的例子是**OpenAI的Whisper**，它在多种语言中提供了出色的准确性。您可以通过Hugging Face的`transformers`等库使用Whisper等模型。尽管直接使用这些工具提供更强的能力，但它也涉及更陡峭的学习曲线，包括管理更大的模型下载和潜在的复杂软件依赖。

通过从 `SpeechRecognition` 库开始，您将学习语音转文本应用程序的基本工作流程。您获得的技能将为后续使用更高级的、直接与模型交互的库提供坚实的基础。

在接下来的章节中，我们将安装 `SpeechRecognition` 并编写我们的第一个Python脚本，将口语转换为文本。

## 参考资料

- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever (2022)
  Journal: arXiv preprint arXiv:2212.04356; DOI: [10.48550/arXiv.2212.04356](https://doi.org/10.48550/arXiv.2212.04356)
  本文介绍了OpenAI的Whisper模型，该章节指出这是一个重要的现代开源ASR模型。
- [SpeechRecognition Library Documentation](https://github.com/Uberi/speech_recognition) — Anthony Zhang and contributors (2014)
  Publisher: Anthony Zhang (Uberi)
  Python SpeechRecognition 库的官方文档，该章节将其作为初学者的主要工具介绍。
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/en/index) — Hugging Face and contributors (N/A)
  Publisher: Hugging Face
  Hugging Face transformers 库的官方文档，该库被提及为在本地运行像Whisper这样的当代开源ASR模型的方法。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  一本权威教材，涵盖了语音识别和自然语言处理，为ASR方法提供了基础背景知识。

---

[上一节](../05-%E8%A7%A3%E7%A0%81%E4%B8%8E%E7%B3%BB%E7%BB%9F%E9%9B%86%E6%88%90/07-%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB%E4%B8%AD%E7%9A%84%E5%B8%B8%E8%A7%81%E6%8C%91%E6%88%98.md) · [下一节](02-%E9%85%8D%E7%BD%AEPython%E7%8E%AF%E5%A2%83.md)

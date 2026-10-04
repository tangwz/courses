---
course: "introduction-to-speech-recognition"
chapter: "building-your-first-speech-recognition-app"
lesson: "using-a-pretrained-model-for-transcription"
sourceId: 7142
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-6-building-your-first-speech-recognition-app/using-a-pretrained-model-for-transcription"
title: "使用预训练模型进行转录"
description: "学习如何加载和使用预训练ASR模型，从而避免从零开始训练自己的模型。"
order: 3
plots: []
sourceHash: "8bb28d83c92f06cb3ea01580129c01e748a6f0ed6b638163607005f93ad01520"
sourceCorrections: []
---

从零开始训练语音识别模型是一项艰巨的任务。它需要大量、精心整理的音频和对应文本数据集，通常长达数千小时。此过程还需要大量计算资源，涉及在专用硬件上进行数周甚至数月的训练。对于大多数开发者和项目而言，这种做法根本不切实际。

幸运的是，我们可以站在巨人的肩膀上。我们不必从头构建自己的模型，而是可以**使用预训练 (pre-training)模型**。这些模型已由研究机构或大型公司在海量数据集上训练完成。它们包含了数千小时音频中所有复杂的学习成果，可供您在自己的应用中使用。

可以这样想：您不是从字母表和基本语法开始学习一门语言，而是雇佣了一位流利且专业的翻译。您的任务不再是教导模型，而是向其提供音频，然后获得文本。

### 模型中心的功用

为了使预训练 (pre-training)的语音识别模型易于获取，社区通常依赖于中央存储库，或称“中心”。Hugging Face Hub 就是一个著名的例子，它托管了数千个用于各种任务的预训练模型，包括自动语音识别。我们可以使用像 `transformers` 这样的 Python 库，仅需几行代码即可轻松下载和使用这些模型。

当您从中心加载预训练的 ASR 模型时，您通常不仅仅获得模型权重 (weight)。您会得到一个完整、即用的软件包，其中包括：

1. **特征提取器**：此组件处理第2章中讨论的音频预处理步骤，将原始音频波形转换为模型所需的数字格式，例如梅尔频谱图。
2. **模型**：这是核心神经网络 (neural network)，经过训练可以将音频特征映射到音素序列或其他语言单位。
3. **分词 (tokenization)器 (tokenizer)**：此组件接收模型的输出并将其解码为人类可读的文本字符串，将语言单位转换为单词。

`transformers` 库方便地将这些组件捆绑到一个 `pipeline` 对象中，抽象化了底层复杂性。

> 预训练ASR流程的组成部分。`pipeline` 对象管理从音频输入到最终文本转录的流程，并在内部处理特征提取、模型处理和解码。

### 在代码中使用 `pipeline`

加载整个转录系统非常简单。您可以使用 `pipeline()` 函数并指定任务“automatic-speech-recognition”以及您想从中心使用的模型名称。

```python
# 确保您已安装所需的库：
# pip install transformers torch

from transformers import pipeline

# 使用特定的预训练模型加载整个 ASR 流程
# "facebook/wav2vec2-base-960h" 是一个流行的模型，在 960 小时英语语音上训练得到
transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# 'transcriber' 对象现在已准备好使用。
# 它包含特征提取器、模型和分词器。
print(transcriber)
```

执行此代码将下载模型文件（如果您尚未拥有）并初始化 `transcriber` 对象。此对象现在是一个功能齐全的语音识别引擎。在接下来的章节中，我们将传递一个音频文件给它，并看它如何运行。

### 选择合适的模型

我们选择的模型 `facebook/wav2vec2-base-960h` 是一个非常适合英语的通用选择。然而，该中心包含数千个模型。在选择时，您可以考虑以下几点：

- **语言：** 模型几乎总是语言专用的。一个在英语上训练的模型将无法理解西班牙语。您必须选择为您的目标语言训练的模型。
- **训练数据：** 模型的表现受其训练数据的影响很大。一个在清晰有声读物录音上训练的模型（如我们选择的）在类似任务中会表现出色，但在嘈杂的电话对话中可能表现不佳。还有一些模型专门针对这些不同场景进行了“微调 (fine-tuning)”。
- **大小和速度：** 更大的模型通常提供更高的准确性，但需要更多内存且运行速度较慢。对于初学者的应用来说，“基础”模型是一个极好的起点，它在性能和资源使用之间提供了良好的平衡。

通过使用预训练 (pre-training)模型，您可以绕过机器学习 (machine learning)中最耗资源的部分，并立即开始构建您的应用。既然我们的 `transcriber` 已准备就绪，那么让我们给它一些音频进行处理吧。

## 参考资料

- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://openreview.net/forum?id=Q5Gq5eBfE8g) — Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); Publisher: Neural Information Processing Systems (NeurIPS); Volume: 33; DOI: [10.48550/arXiv.2006.11477](https://doi.org/10.48550/arXiv.2006.11477)
  介绍了像 Wav2Vec 2.0 这样模型中使用的语音表示自监督学习方法。
- [Automatic Speech Recognition with 🤗 Transformers](https://huggingface.co/docs/transformers/main/en/task_summary#automatic-speech-recognition) — Hugging Face team (2024)
  Publisher: Hugging Face
  解释如何使用 `transformers` 库进行 ASR 的官方文档，涵盖 `pipeline` API 和模型选择。
- [Speech and Language Processing (3rd edition draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  一本涵盖语音识别基础知识的教材，包括声学特征、隐马尔可夫模型和神经网络方法。

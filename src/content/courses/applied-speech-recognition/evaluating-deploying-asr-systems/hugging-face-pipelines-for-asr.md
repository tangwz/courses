---
course: "applied-speech-recognition"
chapter: "evaluating-deploying-asr-systems"
lesson: "hugging-face-pipelines-for-asr"
sourceId: 7160
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-6-evaluating-deploying-asr-systems/hugging-face-pipelines-for-asr"
title: "使用Hugging Face流水线进行ASR"
description: "一份关于如何使用高级Hugging Face `pipeline` API进行快速简便的语音转文本推理的指南。"
order: 4
plots: []
sourceHash: "e7e64c08cc3c3b5bf3b460eb9ca3fefa0f9d9b25d8d617591e857bc45fc87524"
sourceCorrections: []
---

将模型用于推理 (inference)，例如ASR模型生成预测结果，是其应用的一个主要阶段。模型性能通常可以使用词错率等指标进行评估。虽然您可以手动加载模型、预处理音频并解码输出，但Hugging Face的`transformers`库通过其`pipeline` API提供了一种更便捷的方式。这个高级工具将整个推理过程封装成一个函数调用，使得从预训练 (pre-training)模型中获取转录结果变得非常容易。

`pipeline`处理了您在前面章节中学到的必要步骤：它接收原始音频源，应用适当的特征提取，将特征传递给模型，并将模型的输出logits解码成人类可读的文本。

### 流水线工作流程

当您使用`automatic-speech-recognition`流水线时，它会协调一系列您应该很熟悉的操作。它将特征提取器、模型和分词 (tokenization)器 (tokenizer)（用于解码）捆绑成一个便捷对象。该过程隐藏了底层复杂性，让您可以专注于输入和输出。

> 此图示了ASR流水线中的自动化流程。它从音频输入开始，最终得到文本转录，并在内部处理所有中间步骤。

### 开始简单转录

首先，请确保您已安装所需的库。您将需要`transformers`和一个深度学习 (deep learning)框架，例如PyTorch。我们还将使用`datasets`以便为本示例轻松加载一个示例音频文件。

```bash
pip install transformers torch datasets
```

安装好库后，您只需一行代码即可实例化ASR流水线。默认情况下，它将下载并缓存一个通用ASR模型（在撰写本文时，此模型为`openai/whisper-base`）。

```python
from transformers import pipeline
from datasets import load_dataset

# 1. 实例化ASR流水线
# 这将下载默认模型和分词器
transcriber = pipeline("automatic-speech-recognition")

# 2. 从数据集中加载示例音频文件
# 我们使用LibriSpeech数据集中的一个小样本
dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
sample = next(iter(dataset))

# 3. 执行推理
result = transcriber(sample["audio"]["array"])

print(result)
```

运行此代码将生成一个包含转录文本的字典。

```text
{'text': ' A man said to the universe Sir I exist.'}
```

该流水线正确处理了数据集样本中的原始音频数组并返回了转录结果。它自动处理了音频的采样率，并将其转换为模型所需的特征格式。

### 使用特定预训练 (pre-training)模型

默认模型是一个不错的起点，但您通常会希望使用Hugging Face Hub中的特定模型，也许是您自己微调 (fine-tuning)过的模型，或者针对特定语言或大小进行优化的模型。您可以通过将模型的仓库ID传递给`model`参数 (parameter)来指定模型。

让我们使用Whisper模型中更小、更快的版本：`openai/whisper-tiny`。

```python
from transformers import pipeline

# 使用特定模型实例化流水线
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# 您现在可以在音频文件上使用此转录器
# 例如，加载本地文件（请确保它是.wav或.flac文件）
# result = transcriber("path/to/your/audio.wav")
# print(result)
```

流水线会自动下载指定模型及其相应的特征提取器和分词 (tokenization)器 (tokenizer)，确保所有组件兼容。这使得尝试不同的先进ASR模型变得非常简单。

### 实际考量：批处理和GPU加速

对于转录多个文件，您可以向流水线传入一个音频输入列表。这支持批处理，与逐个文件迭代相比，批处理效率更高，尤其是在使用GPU时。

此外，如果您有兼容的GPU可用，您可以通过指定`device`参数 (parameter)来指示流水线使用它。这可以显著加速推理 (inference)。

```python
# 假设您有一个音频文件路径列表
audio_files = ["speech_01.wav", "speech_02.wav", "speech_03.wav"]

# 实例化用于GPU的流水线（device=0表示第一个GPU）
# 并指定处理的批大小
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=0,  # 使用第一个启用CUDA的GPU
    batch_size=4 # 一次最多处理4个文件
)

# 将文件列表传递给转录器
transcriptions = transcriber(audio_files)

for i, result in enumerate(transcriptions):
    print(f"File {i+1}: {result['text']}")
```

这种方法对于处理整个评估数据集或音频文件目录更具扩展性。流水线处理批处理并将计算发送到GPU，为您提供显著的性能提升。

Hugging Face `pipeline`是用于快速推理和集成的出色工具。它简化了对强大模型的访问，让您可以直接从模型选择转向生成结果。在下一节中，我们将更进一步，通过使用Gradio构建一个简单的Web应用程序，为我们的转录模型提供一个交互式界面。

## 参考资料

- [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) — Hugging Face (2024)
  Publisher: Hugging Face
  `transformers`管道API的官方指南，说明其结构和在包括ASR在内的各种机器学习任务中的应用。
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever (2022)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2212.04356](https://doi.org/10.48550/arXiv.2212.04356)
  介绍Whisper模型架构和训练方法，该模型常用于Hugging Face ASR管道。
- [The Hugging Face Course: Chapter 2 - Using a 🤗 Transformers model for inference](https://huggingface.co/learn/nlp-course/chapter2/1) — Hugging Face (2023)
  Publisher: Hugging Face
  一个介绍`pipeline` API的教育资源，用于使用预训练模型在不同任务上执行推理。
- [The 🤗 Datasets library: Quick tour](https://huggingface.co/docs/datasets/en/quicktour) — Hugging Face (2024)
  Publisher: Hugging Face
  `datasets`库的官方指南，介绍如何加载和处理数据，包括示例中使用的音频数据。

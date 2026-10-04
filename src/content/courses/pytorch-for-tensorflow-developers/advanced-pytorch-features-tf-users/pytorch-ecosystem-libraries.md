---
course: "pytorch-for-tensorflow-developers"
chapter: "advanced-pytorch-features-tf-users"
lesson: "pytorch-ecosystem-libraries"
sourceId: 5778
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-6-advanced-pytorch-features-tf-users/pytorch-ecosystem-libraries"
title: "PyTorch 生态系统概述：torchvision、torchaudio、torchtext"
description: "简要介绍用于视觉、音频和文本处理的 PyTorch 主要领域库。"
order: 5
plots: []
sourceHash: "e227e572f209c820782f184b5e015d3182a736aec61bd860d0694f3154f6deef"
sourceCorrections: []
---

PyTorch 的核心组件提供构建几乎任何机器学习 (machine learning)模型的基本工具，而计算机视觉、音频处理和自然语言处理等领域的特定任务常受益于专用工具。PyTorch 通过一系列领域专用库来支持这些方面。这些库作为 PyTorch 官方项目的一部分开发，提供预打包数据集、预训练 (pre-training)模型架构和常用变换函数，使您的开发流程更高效。其中三个主要的库是 `torchvision`、`torchaudio` 和 `torchtext`。

> 概述 PyTorch 核心及其与领域专用库的关系：torchvision、torchaudio 和 torchtext。

这些库不仅仅是工具的集合；它们旨在与 PyTorch 的张量和神经网络 (neural network)模块平稳集成，让您能够轻松地将它们的功能整合到您的自定义模型和训练循环中。

### TorchVision：用于计算机视觉任务

如果您在处理图像或视频，`torchvision` 是一个不可或缺的库。它提供工具来简化许多常见的计算机视觉任务。TensorFlow 开发者可能会发现它与 `tf.keras.applications` （用于模型）和 `tf.image` （用于某些变换）有相似之处。

主要功能包括：

- **数据集：** `torchvision.datasets` 提供对 MNIST、CIFAR10、ImageNet 和 COCO 等流行计算机视觉数据集的便捷访问。您只需极少的样板代码即可下载和加载这些数据集。

  ```python
  from torchvision import datasets
  from torchvision.transforms import ToTensor

  # 示例：加载 MNIST 训练数据集
  # train_data = datasets.MNIST(
  #     root="data", # 数据下载目录
  #     train=True,
  #     download=True,
  #     transform=ToTensor() # 将 PIL 图像或 numpy.ndarray 转换为张量
  # )
  ```
- **模型：** `torchvision.models` 包含 ResNet、VGG、AlexNet、MobileNet 和 Vision Transformer 等许多知名模型架构的定义。其中许多都附带在 ImageNet 上预训练 (pre-training)的权重 (weight)。

  ```python
  import torchvision.models as models

  # 加载预训练的 ResNet18 模型
  resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  resnet18.eval() # 将模型设置为评估模式
  ```

  使用枚举（例如 `ResNet18_Weights.DEFAULT` 或 `ResNet18_Weights.IMAGENET1K_V1`）作为 `weights` 参数 (parameter)是加载预训练权重的现代方法，提供更好的清晰度和版本管理。
- **变换：** `torchvision.transforms` 提供一套常用的图像变换方法。这些对于数据预处理和增强非常重要。示例包括 `Resize`、`CenterCrop`、`RandomHorizontalFlip`、`ToTensor` 和 `Normalize`。您可以使用 `transforms.Compose` 将它们串联起来。

  ```python
  from torchvision import transforms

  # 为输入图像定义一系列变换
  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  # img_tensor = preprocess(input_pil_image)
  ```
- **实用工具：** `torchvision.utils` 包含辅助函数，例如用于创建图像网格（有助于批次可视化）的 `make_grid` 和用于将张量保存为图像文件的 `save_image`。

通过使用 `torchvision`，您可以显著减少数据加载、模型实例化和图像预处理所需的代码量，让您能更专注于计算机视觉项目的独特性。

### TorchAudio：用于音频处理

对于涉及音频数据的任务，`torchaudio` 提供必要的工具。音频处理有其自身的挑战和所需的变换，`torchaudio` 旨在使这些在 PyTorch 环境中更易于使用。

其功能包括：

- **I/O：** 加载音频文件 (`torchaudio.load`) 和保存它们 (`torchaudio.save`) 是基本操作。`torchaudio` 支持多种音频格式，并与不同的后端（如 SoX 和 SoundFile）集成以处理音频数据。

  ```python
  import torchaudio

  # 示例：加载音频文件（假设 'audio.wav' 存在）
  # waveform, sample_rate = torchaudio.load("audio.wav")
  # print(f"Waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
  ```
- **数据集：** `torchaudio.datasets` 包含 LibriSpeech、SpeechCommands、VCTK 和 YesNo 等常见音频数据集，有助于研究和模型开发。
- **变换：** `torchaudio.transforms` 提供了一系列音频专用变换。这些常用于特征提取。一些广泛使用的变换包括：
  - `MelSpectrogram`：从波形计算梅尔频谱图。梅尔频谱图是频率频谱随时间变化的视觉表示，并根据梅尔刻度进行调整，常用于语音和音乐分析。
  - `MFCC`（梅尔频率倒谱系数）：计算梅尔频率倒谱系数 (MFCCs)，这些特征广泛用于自动语音识别。
  - `Resample`：改变音频信号的采样率。
  - `Spectrogram`：计算标准频谱图。

  ```python
  import torchaudio.transforms as T

  # 示例：定义梅尔频谱图变换
  # Assuming sample_rate is known, e.g., 16000 Hz
  # mel_spectrogram_transform = T.MelSpectrogram(
  #     sample_rate=16000,
  #     n_fft=400,      # FFT 大小
  #     hop_length=160, # STFT 窗口之间的跳长
  #     n_mels=80       # 梅尔滤波器组数量
  # )
  # mel_spec = mel_spectrogram_transform(waveform) # 应用于加载的波形
  ```
- **函数：** `torchaudio.functional` 提供低级音频操作，如滤波、心理声学计算和效果（例如 `lowpass_biquad`、`contrast`）。

`torchaudio` 对于任何想要构建基于音频的机器学习 (machine learning)系统的人来说都是不可或缺的，从语音识别和音乐生成到声学场景分类。

### TorchText：用于自然语言处理

自然语言处理 (NLP) 涉及理解和生成人类语言。`torchtext` 提供实用工具帮助准备文本数据用于深度学习 (deep learning)模型。虽然更广泛的 NLP 生态系统常涉及 Hugging Face Transformers 等库以获取最先进的模型，但 `torchtext` 提供基本的数据处理能力。TensorFlow 用户可能会将其与 `tf.keras.layers.TextVectorization` 或 `tf.text` 中的模块进行比较。

`torchtext` 侧重于：

- **数据集：** `torchtext.datasets` 提供对 IMDb（情感分析）、AG\_NEWS（文本分类）和 SST（斯坦福情感树库）等几个标准 NLP 数据集的访问。
- **数据处理实用工具：** 为模型准备文本通常涉及几个步骤：
  - **分词 (tokenization)：** `torchtext.data.utils.get_tokenizer` 可用于获取分词器 (tokenizer)函数（例如“basic\_english”、“spacy”），将句子分成单词或子词 (subword)单元。

    ```python
    from torchtext.data.utils import get_tokenizer

    tokenizer = get_tokenizer("basic_english")
    # tokens = tokenizer("This is an example sentence from TorchText.")
    # print(tokens) # Output: ['this', 'is', 'an', 'example', 'sentence', 'from', 'torchtext', '.']
    ```
  - **词汇表 (vocabulary)：** `torchtext.vocab.build_vocab_from_iterator` 允许您创建词汇表对象，将 token 映射到数字索引。这是在将文本数据馈送到嵌入 (embedding)层之前是必要步骤。

    ```python
    from torchtext.vocab import build_vocab_from_iterator

    # 示例迭代器，生成 token 列表
    # def yield_tokens(data_iterator):
    #     for text in data_iterator:
    #         yield tokenizer(text)

    # sample_texts = ["hello world", "hello pytorch"]
    # vocab = build_vocab_from_iterator(yield_tokens(sample_texts), specials=["<unk>", "<pad>"])
    # vocab.set_default_index(vocab["<unk>"]) # 处理未知词
    # indexed_tokens = vocab(tokenizer("hello unknown"))
    # print(indexed_tokens) # 例如，[vocab['hello'], vocab['<unk>']]
    ```
  - **DataPipes (`torchdata.datapipes`)：** 现代 `torchtext` 鼓励使用 DataPipes（来自 `torchdata` 库，这是一个用于可组合数据加载的 PyTorch 领域库）来构建高效且灵活的数据加载管道。这种方法与 PyTorch 的 `DataLoader` 理念相符，允许将分词、数值化和批处理等操作定义为一系列可组合的步骤。

虽然 `torchtext` 本身已不再直接提供许多预构建模型（因为 NLP 领域发展迅速，大型语言模型常来源于其他中心），但其数据处理工具对于自定义 NLP 任务以及为任何 PyTorch 模型准备数据仍然有价值。

这些领域库显著增强 PyTorch 的功能，提供专业化、优化的工具，使在计算机视觉、音频和 NLP 领域处理问题变得更容易。作为一名转向 PyTorch 的 TensorFlow 开发者，熟悉这些库将使您能够快速在这些常见的机器学习 (machine learning)应用领域中高效工作。它们抽象掉许多领域专用样板代码，让您专注于模型创新和训练。

## 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch框架的通用指南，提供其核心组件的基础信息，并作为其专业领域库的入口。
- [TorchVision Documentation](https://pytorch.org/vision/stable/index.html) — PyTorch Contributors (2024)
  `torchvision`库的官方参考，详细说明其用于计算机视觉任务的组件，包括数据集、模型和图像转换。
- [TorchAudio Documentation](https://pytorch.org/audio/stable/index.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  `torchaudio`库的官方参考，概述其用于音频处理的功能，例如I/O操作、数据集和音频特定转换。
- [TorchText Documentation](https://pytorch.org/text/stable/index.html) — PyTorch Contributors (2024)
  `torchtext`库的官方参考，解释其用于自然语言处理的工具，包括分词和词汇构建等数据处理实用工具。

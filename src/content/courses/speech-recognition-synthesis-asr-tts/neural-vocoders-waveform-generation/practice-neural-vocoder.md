---
course: "speech-recognition-synthesis-asr-tts"
chapter: "neural-vocoders-waveform-generation"
lesson: "practice-neural-vocoder"
sourceId: 3213
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-5-neural-vocoders-waveform-generation/practice-neural-vocoder"
title: "动手实践：使用神经声码器"
description: "使用预训练神经声码器（例如HiFi-GAN）从TTS模型生成的声学特征合成音频。"
order: 8
plots: []
sourceHash: "1dbd917d94a063820d848e192489a222112e6e7771ed7afca2f83f8fe6e51526"
sourceCorrections: []
---

神经声码器架构，包括WaveNet等自回归 (autoregressive)模型和HiFi-GAN等高效的基于GAN的方法，在将TTS系统生成的中间声学特征（通常是梅尔频谱图）转换为最终可听波形方面扮演着核心角色。本实践指南将介绍如何使用预训练 (pre-training)的神经声码器来执行此合成步骤。主要关注点在于获取现有梅尔频谱图并将其转换为音频，模拟现代TTS流程的最后阶段。

本次练习中，我们将使用`TTS`库（一个流行的开源工具包）和一个预训练的HiFi-GAN声码器模型。HiFi-GAN以其高保真度和计算效率而闻名，使其成为许多TTS系统中的常见选择。

### 环境设置

首先，请确保已安装所需库。我们主要需要`TTS`，它捆绑了`torch`和其他依赖项。您可能还需要`soundfile`来保存音频。

```bash
# 从其仓库安装TTS库
pip install TTS

# 如果您尚未安装soundfile，请安装
pip install soundfile numpy
```

您还需要一个预计算的梅尔频谱图文件作为声码器的输入。在此示例中，我们假设您有一个名为`sample_mel_spectrogram.npy`的NumPy文件。此文件通常是前一个TTS阶段中声学模型（如Tacotron 2或FastSpeech 2）的输出，代表特定话语的声学特征。

*注：生成此梅尔频谱图文件本身涉及运行一个独立的TTS声学模型，这已在第4章中介绍。本次练习中，请侧重于声码器的作用，假设梅尔频谱图已可用。*

### 加载预训练 (pre-training)声码器

`TTS`库提供了一个方便的接口来加载各种预训练模型。我们将加载一个在LJSpeech数据集上训练的HiFi-GAN模型。

```python
import torch
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

# 定义下载模型的路径（或它们将被下载到的位置）
# 替换为您偏好的路径
path = "~/.local/share/tts/"
manager = ModelManager(path)

# 列出可用的声码器模型（可选，用于查看）
# print(manager.list_models())

# 下载并加载预训练的HiFi-GAN声码器模型
# 示例：使用通用HiFi-GAN模型
vocoder_model_name = "vocoder_models/universal/libri-tts/wavegrad"
# 或者，如果可用且偏好，使用特定的LJSpeech HiFi-GAN：
# vocoder_model_name = "vocoder_models/en/ljspeech/hifigan_v2"

try:
    vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_model_name)
except ValueError as e:
    print(f"下载模型出错: {e}")
    print(f"请检查模型名称或您的互联网连接。")
    # 如有需要，提供查找正确模型名称的指导
    print("您可以使用manager.list_models()列出可用模型")
    exit() # 如果模型下载失败则退出

# 检查CUDA（GPU）是否可用，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {device}")

# 仅用声码器初始化合成器
# 我们此处不需要TTS模型，因为我们直接提供梅尔频谱图
syn = Synthesizer(
    tts_checkpoint=None, # 无TTS模型检查点
    tts_config_path=None, # 无TTS模型配置
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    use_cuda=(device == "cuda"),
)

print("神经声码器模型加载成功。")
# syn.vocoder对象包含已加载的声码器模型。
# 示例：syn.vocoder.model是实际的HiFi-GAN生成器网络
```

此代码片段初始化`ModelManager`以处理模型下载，然后使用`Synthesizer`类，仅配置声码器模型细节。如果指定的HiFi-GAN模型尚未在本地存在，库将下载它。它还会自动检测GPU是否可用，以进行更快的处理。

### 加载声学特征（梅尔频谱图）

现在，从`.npy`文件中加载预计算的梅尔频谱图。此文件应包含一个二维NumPy数组，其中一个维度代表梅尔频率箱，另一个维度代表时间帧。

```python
import numpy as np

# 从文件中加载梅尔频谱图
# 将'sample_mel_spectrogram.npy'替换为您文件的实际路径
mel_file = 'sample_mel_spectrogram.npy'

try:
    mel_spectrogram = np.load(mel_file)
    print(f"从 {mel_file} 加载梅尔频谱图")
    print(f"形状: {mel_spectrogram.shape}") # 示例形状：(80, 250) -> 80个梅尔频率箱，250帧
except FileNotFoundError:
    print(f"错误：未在 {mel_file} 找到梅尔频谱图文件")
    print("请确保文件存在或提供正确路径。")
    # 如果需要，创建一个用于演示的虚拟频谱图
    print("正在创建用于演示的虚拟梅尔频谱图。")
    mel_spectrogram = np.random.rand(80, 250).astype(np.float32) # 80个梅尔频率箱，250帧
except Exception as e:
    print(f"加载梅尔频谱图出错: {e}")
    exit()

# TTS合成器期望梅尔频谱图为Torch张量
# 它还应添加一个批次维度。
# 许多声码器期望的形状：[批次大小, 梅尔数, 帧数]
mel_tensor = torch.tensor(mel_spectrogram).unsqueeze(0).to(device)
print(f"已将梅尔频谱图转换为形状为: {mel_tensor.shape} 的张量")
```

此处，我们加载NumPy数组并将其转换为PyTorch张量。重要的是，我们添加了一个批次维度（`unsqueeze(0)`），因为大多数深度学习 (deep learning)模型期望批次输入，即使批次大小只有一个。我们还将张量移动到相应的设备（CPU或GPU）。

### 生成波形

声码器加载完毕且输入梅尔频谱图准备就绪后，我们可以执行推理 (inference)步骤。`Synthesizer`对象提供了一个方便的方法（`tts`或直接访问声码器）来生成波形。由于我们绕过了文本到梅尔阶段，我们使用底层声码器的推理功能。

```python
# 使用合成器的声码器将梅尔频谱图转换为波形
# `vocoder.inference`方法通常处理此过程
# 注意：具体方法可能因TTS库版本而略有不同。
# 如有需要，请查阅文档。Synthesizer通常会封装此功能。

print("正在从梅尔频谱图生成波形...")
# 我们将mel_tensor直接传递给声码器的推理方法
# 确保张量位于正确设备上
outputs = syn.vocoder.inference(mel_tensor)

# 输出通常是包含原始音频波形样本的张量。
# 它可能在GPU上，因此请将其移至CPU并转换为NumPy数组。
# 输出张量形状可能是[批次大小, 1, 样本数]或[批次大小, 样本数]
waveform = outputs.squeeze().cpu().numpy()

print(f"已生成形状为: {waveform.shape} 的波形") # 示例形状：(55125,) -> 音频样本数量
print("波形生成完成。")
```

已加载声码器模型的`inference`方法将梅尔频谱图张量作为输入，并输出相应的音频波形张量。然后，我们将在CPU上将此张量转换回NumPy数组，以便于处理和保存。

### 保存与聆听

最后，将生成的波形保存为标准音频文件（如WAV格式）并聆听。您需要与预训练 (pre-training)声码器模型相关联的采样率。这通常存储在模型的配置中。

```python
import soundfile as sf

# 从合成器的配置中获取采样率
# 这确保音频被正确保存和回放
output_sample_rate = syn.vocoder_config.get('audio', {}).get('sample_rate', 22050) # 如果未找到，默认使用22050
print(f"正在使用采样率: {output_sample_rate} Hz")

# 定义输出文件路径
output_wav_file = 'generated_audio_hifigan.wav'

# 将波形保存为WAV文件
try:
    sf.write(output_wav_file, waveform, output_sample_rate)
    print(f"音频已成功保存到 {output_wav_file}")
except Exception as e:
    print(f"保存音频文件出错: {e}")

print("\n实践完成。您现在可以聆听生成的音频文件。")
```

此代码从声码器配置中获取正确的采样率（对于正确的回放速度必不可少），并使用`soundfile`库将包含波形样本的NumPy数组写入`.wav`文件。

聆听`generated_audio_hifigan.wav`文件。将其质量与您可能听过的传统声码器（如Griffin-Lim）的示例进行比较。它听起来自然吗？是否有明显的伪影（例如嗡嗡声或嘶嘶声）？此动手体验直接呈现了现代神经声码器（如HiFi-GAN）所带来的质量提升，您已在本章前面了解过它们。您可以获取不同句子的梅尔频谱图，或使用工具包中可用的不同预训练声码器模型（例如WaveGrad、MelGAN）来比较它们的输出，从而进行进一步试验。

## 参考资料

- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) — Jungil Kong, Jaehyeon Kim, Jaekyoung Bae (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2010.05646](https://doi.org/10.48550/arXiv.2010.05646)
  介绍了HiFi-GAN架构，阐述了其从声学特征生成高保真、计算高效波形的能力，与本实践内容直接相关。
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) — Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu (2016)
  Journal: arXiv preprint arXiv:1609.03499; DOI: [10.48550/arXiv.1609.03499](https://doi.org/10.48550/arXiv.1609.03499)
  介绍了WaveNet，一种用于原始音频生成的自回归模型，被本节引用为神经声码器的理论基础。
- [TTS Library Documentation](https://tts.readthedocs.io/en/latest/) — Coqui (2024)
  Publisher: Coqui
  `TTS`库的官方文档，提供了详细的指南和API参考，用于实践语音合成组件（包括本实践中使用的声码器）的实现。

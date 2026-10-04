---
course: "applied-speech-recognition"
chapter: "foundations-digital-audio-speech"
lesson: "practice-visualizing-audio"
sourceId: 7104
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-1-foundations-digital-audio-speech/practice-visualizing-audio"
title: "实践操作：加载与可视化音频波形"
description: "一次关于在 Python 中加载音频文件并生成波形图和频谱图的动手操作练习。"
order: 7
plots: ["plots/7104-0.json", "plots/7104-1.json"]
sourceHash: "c24e3388be8e737403fe567d726e155a59dce87356a1cd795659819506746a03"
sourceCorrections: []
---

将编写 Python 代码来加载音频文件并可视化其波形和频谱图。这项实践练习说明了模拟声波如何转换为数字信号，以及时域和频域表示之间的区别。

这项动手操作练习是任何语音处理任务的核心能力。可视化音频数据不仅仅是为了制作图示；它也是调试、理解数据集以及培养机器对不同声音感知的一种直观认知的重要步骤。我们将使用 `Librosa` 库，它是 Python 中音频分析的实际标准库。

### 环境准备

在开始之前，请确保您已安装 `Librosa` 和 `Matplotlib`。`Librosa` 将负责音频的加载和处理，而 `Matplotlib` 则提供可视化的支持。您可以使用 pip 安装它们：

```bash
pip install librosa matplotlib numpy
```

### 使用 Librosa 加载音频文件

我们的第一步是将音频文件从磁盘加载到一种可操作的格式。`librosa.load()` 函数是完成此项任务的主要工具。它会返回两个重要内容：

1. 一个 NumPy 数组表示音频时间序列。按照惯例，我们将其称为 `y`。这就是我们之前讨论的数字波形 $x[n]$。Librosa 会自动将信号转换为单声道（一个通道）并将其重采样到默认的 22,050 Hz 速率。
2. 时间序列的采样率，我们将其称为 `sr`。这个值表示数组 `y` 中有多少个样本对应一秒钟的音频。

让我们来看看实际操作。以下代码片段将加载一个音频文件，并打印结果数组的形状及其采样率。

```python
import librosa

# 定义音频文件路径
# 请将其替换为您系统上的 .wav 或 .mp3 文件
audio_path = 'path/to/your/audio.wav'

# 加载音频文件
y, sr = librosa.load(audio_path)

print(f"Audio time series shape: {y.shape}")
print(f"Sampling rate: {sr} Hz")
```

如果您使用一个 5 秒的音频文件运行此代码，您可能会看到类似 `Audio time series shape: (110250,)` 和 `Sampling rate: 22050 Hz` 的输出。这很容易理解：5 秒乘以每秒 22,050 个样本等于总共 110,250 个样本。

### 时域波形可视化

原始时间序列 `y` 包含音频在每个离散时间点的振幅。最直接的显示方式是绘制振幅与时间的关系图。这个图就是**波形图**。它能让我们感知音频的动态，显示出响亮和安静的部分。

我们可以创建一个简单的波形图。尽管我们可以直接使用 `Matplotlib`，但以下示例生成了一个自包含的图表，表示典型的语音波形。

```python
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# 假设 'y' 和 'sr' 已从上一步加载
# 为了演示，我们创建一个虚拟信号
sr = 22050
t = np.linspace(0., 5., int(sr*5.0))
amplitude = np.i0(np.sin(2. * np.pi * t * 1.2))
y = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude)) * 0.8 - 0.4
y = y * np.sin(2 * np.pi * t * 220) * (np.sin(2 * np.pi * t * 0.5) + 0.2)

# --- 绘制波形图 ---
# 此代码仅用于说明；直接绘制的图表如下所示。
fig, ax = plt.subplots(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr, ax=ax)
ax.set_title('音频波形')
ax.set_xlabel('时间 (s)')
ax.set_ylabel('振幅')
plt.tight_layout()
plt.show()
```

该代码生成一个图，其中 x 轴表示时间，y 轴表示振幅。绝对振幅高的区域对应于声音较响的部分，而接近零的区域则表示静音或安静。



![时域波形](plots/7104-0.json)



> 波形图显示了信号随时间变化的振幅。密集、高振幅的部分表示说话内容，而接近零的平坦部分则表示静音。

### 从时域到频域：生成频谱图

波形图很有用，但它没有显示信号的频率成分。为了查看每个时刻存在哪些频率，我们需要转向频域表示。针对语音信号，标准做法是计算**频谱图**。

频谱图是通过应用短时傅里叶变换（STFT）创建的。STFT 将音频信号分成小的重叠窗口，并为每个窗口计算傅里叶变换。这为我们提供了连续短时间间隔内的频率成分。

1. **计算 STFT**：我们在时间序列 `y` 上使用 `librosa.stft()`。这会返回一个二维复值矩阵，其中行对应于频率分段，列对应于时间帧。
2. **获取幅度**：STFT 输出包含幅度和相位。对于可视化和大多数 ASR 模型，我们只需要幅度。我们可以通过对 STFT 矩阵使用 `np.abs()` 来获取。
3. **转换为分贝 (dB)**：所得的幅度频谱图具有非常宽的动态范围。人耳的听觉是呈对数关系的，因此通常做法是使用 `librosa.amplitude_to_db()` 将振幅转换为分贝 (dB) 尺度。这会压缩范围，并使结果的可视化更具信息量。

以下是执行这些步骤的代码：

```python
import numpy as np

# 计算短时傅里叶变换 (STFT)
D = librosa.stft(y)

# 将复值 STFT 输出转换为幅度频谱图
S_mag = np.abs(D)

# 将幅度频谱图转换为分贝 (dB) 尺度
S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

# --- 绘制频谱图 ---
fig, ax = plt.subplots(figsize=(10, 4))
img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB', label='幅度 (dB)')
ax.set_title('梅尔频谱图')
plt.tight_layout()
plt.show()
```

最终生成的图表，即频谱图，是一个热力图。x 轴是时间，y 轴是频率，任意点 `(t, f)` 的颜色强度表示频率 `f` 在时间 `t` 时的能量。



![频谱图（幅度单位为dB）](plots/7104-1.json)



> 频谱图显示了频率内容随时间的变化。较亮的区域表示能量较高。能量的水平条带是语音的典型特征，对应于共振峰，它们是人类声道的声学共振。

完成本次练习，您就掌握了任何音频处理流程中最为基础的步骤。您现在可以加载任何音频文件，并检查其在时域和频域中的结构。这些可视化结果构成了我们将在下一章中介绍的自动特征提取方法的基础，届时我们将把这些丰富的视觉表示转换为深度学习 (deep learning)模型所需的紧凑数值特征。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本关于语音处理的综合资源，涵盖数字音频、时频域和语音识别相关的频谱图。
- [Discrete-Time Signal Processing](https://www.pearson.com/us/higher-education/program/Oppenheim-Discrete-Time-Signal-Processing-3rd-edition/PGM215167.html) — Alan V. Oppenheim, Ronald W. Schafer (2009)
  Publisher: Pearson
  一本关于数字信号处理的基础教科书，讨论采样、傅里叶变换和短时傅里叶变换等主题。

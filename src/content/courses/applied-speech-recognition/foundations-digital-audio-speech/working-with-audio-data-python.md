---
course: "applied-speech-recognition"
chapter: "foundations-digital-audio-speech"
lesson: "working-with-audio-data-python"
sourceId: 7088
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-1-foundations-digital-audio-speech/working-with-audio-data-python"
title: "在 Python 中使用 Librosa 处理音频数据"
description: "Python 中 Librosa 库用于加载、操作和分析音频文件的使用指南。"
order: 4
plots: ["plots/7088-0.json"]
sourceHash: "a14431c5c46693c189352082455cfa2549f3e00c361427f9ecb21fa38570fb72"
sourceCorrections: []
---

Python 自带处理文本和数字的功能，但处理音频需要专门的工具。原始音频数据是表示信号振幅随时间变化的数字序列，需要从文件中加载到程序可以操作的格式，最常见的是 NumPy 数组。这时，`Librosa`——一个功能强大且广受欢迎的 Python 音乐和音频分析包——就派上用场了。它提供加载、保存和分析音频的核心功能，作为我们语音识别流程的构建块。

### 安装 Librosa

在使用音频文件之前，你需要安装 `Librosa` 库。因为它在 Python 包索引 (PyPI) 上可用，所以你可以使用 `pip` 进行安装。还建议安装 `soundfile`，`Librosa` 用它作为后端来写入音频文件。

```bash
pip install librosa soundfile
```

### 加载音频文件

加载音频文件的主函数是 `librosa.load()`。这个单一函数处理很多复杂情况，例如解码不同的音频格式（如 .wav、.mp3）并将信号转换为标准化的数值格式。

当你调用 `librosa.load()` 时，它会返回两个重要值：

1. **时间序列 `y`**：一个 NumPy 数组，包含音频信号的振幅值。默认情况下，`Librosa` 会对数据进行归一化 (normalization)，使值范围在 -1.0 到 1.0 之间。这就是我们之前讨论的数字信号 $x[n]$。
2. **采样率 `sr`**：每秒音频样本的数量。

让我们看看它的作用。假设我们有一个名为 `sample_utterance.wav` 的音频文件。我们可以这样加载它：

```python
import librosa

# 定义音频文件路径
audio_path = 'sample_utterance.wav'

# 加载音频文件
y, sr = librosa.load(audio_path)

print(f"音频时间序列形状: {y.shape}")
print(f"采样率: {sr} Hz")
```

如果你运行这段代码，你可能会看到类似这样的输出：

```
Audio time series shape: (110250,)
Sampling rate: 22050 Hz
```

这告诉我们音频信号 `y` 是一个包含 110,250 个样本的一维数组。采样率 `sr` 是 22,050 Hz，这是 `librosa.load()` 的默认值。这意味着 `Librosa` 在加载过程中已自动将音频重采样到此速率。音频的时长就是样本数量除以采样率：$110250 / 22050 = 5$ 秒。

### 控制采样率

对于语音识别，在所有数据中使用一致的采样率是很重要的。虽然 22,050 Hz 是音频处理中常见的默认值，但 16,000 Hz（或 16 kHz）的采样率是 ASR 的普遍标准。这个速率足以捕捉人类语音所需的频率，并有助于减少数据处理的计算成本。

你可以使用 `librosa.load()` 中的 `sr` 参数 (parameter)来控制采样率：

- `sr=16000`：加载音频并将其重采样到 16,000 Hz。
- `sr=None`：以原始的、固有的采样率加载音频，避免任何重采样。

让我们再次加载文件，但这次保留其固有采样率，然后将其重采样到 16 kHz。

```python
# 以固有采样率加载
y_native, sr_native = librosa.load(audio_path, sr=None)
print(f"固有采样率: {sr_native} Hz")
print(f"固有采样率下的形状: {y_native.shape}")

# 加载并重采样到 16 kHz
y_16k, sr_16k = librosa.load(audio_path, sr=16000)
print(f"新采样率: {sr_16k} Hz")
print(f"16kHz 采样率下的形状: {y_16k.shape}")
```

请注意，当我们重采样音频时，数组的形状是如何变化的。如果原始文件以 44.1 kHz 采样，将其重采样到 16 kHz 会得到一个更小的数组，因为表示相同音频时长所需的样本数更少。

### 基本音频操作

`Librosa` 不仅用于加载数据；它还提供一套用于音频操作的工具。一个常见的预处理步骤是修剪语音中的开头和结尾静音。这很有用，因为它移除了信号中不包含语音的部分，使 ASR 模型关注于相关的音频。`librosa.effects.trim` 函数正是执行此操作。

```python
import librosa.effects

# 加载音频文件
y, sr = librosa.load(audio_path, sr=16000)

# 修剪开头和结尾的静音
y_trimmed, _ = librosa.effects.trim(y)

print(f"原始长度: {len(y)}")
print(f"修剪后长度: {len(y_trimmed)}")
```

### 音频波形可视化

我们从 `Librosa` 获得的音频时间序列数组 `y` 直接表示音频波形。将此数组对时间绘制，可以获得声音的可视化表示，这非常有启发性。x 轴表示时间，y 轴表示信号的振幅。



![音频波形](plots/7088-0.json)



> 音频波形的可视化。信号的振幅在零点附近波动，声音越大的部分对应于更高的绝对振幅值。

此图显示了信号的原始时域表示。在接下来的部分中，我们将了解如何将这个一维信号转换为更具信息量的二维表示——频谱图，它更适合语音识别模型。

## 参考资料

- [Discrete-Time Signal Processing](https://library.rice.edu/catalog/98050398) — Alan V. Oppenheim, Ronald W. Schafer, John R. Buck (1999)
  Publisher: Prentice Hall
  数字信号处理的经典教材，解释了采样、离散时间信号及其表示等核心概念。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  语音识别的综合教材，为音频预处理步骤和ASR系统特定采样率的重要性提供了背景知识。

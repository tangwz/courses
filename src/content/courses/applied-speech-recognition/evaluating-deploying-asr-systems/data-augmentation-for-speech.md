---
course: "applied-speech-recognition"
chapter: "evaluating-deploying-asr-systems"
lesson: "data-augmentation-for-speech"
sourceId: 7159
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-6-evaluating-deploying-asr-systems/data-augmentation-for-speech"
title: "语音数据的常用增强方法"
description: "了解增强音频数据的方法，例如添加噪音、改变音高和修改语速，以创建泛化能力更强的模型。"
order: 3
plots: ["plots/7159-0.json", "plots/7159-1.json"]
sourceHash: "b707d73949ea8b9fa897ccf33dbf36275229c74548361a454c7b3f39807d8950"
sourceCorrections: []
---

ASR模型的词错误率可能高于预期，尤其是在与训练集不同的音频上。仅在干净、录音室质量的录音上训练的模型，在面对日常环境中的噪音和变化时，通常表现不佳。数据增强在此成为构建更具适应性、泛化能力更强的ASR系统的一种有效手段。

数据增强涉及创建训练数据的修改副本，以人工方式扩充数据集。通过引入模型在推理 (inference)时可能遇到的变化，您可以使其学会忽略背景噪音或语速等无关信息，而仅关注语言内容。这些变换通常在训练过程中实时应用，这意味着一个音频文件可以在不同的训练周期中生成多种不同版本。

### 添加背景噪音

最直接且有效的数据增强方法之一是噪音注入。此过程包括将干净的语音信号与各种背景声音混合。这直接模拟了语音很少在完全安静环境中被捕获的情况。

常见的噪音源包括：

- **环境噪音：** 来自咖啡馆、街道、办公室或汽车的声音。
- **合成噪音：** 像白噪音、粉红噪音或布朗噪音等数学生成的声音。
- **干扰语音：** 难以理解的背景杂音，常被称为“人群噪音”。

通过在不同信噪比（SNR）下混合这些声音的音频上进行训练，模型学会将主要语音信号从背景中分离出来，大幅提升了其抗干扰能力。



![噪音注入前后的波形](plots/7159-0.json)



> 原始音频信号与添加背景噪音后的同一信号的比较。增强后的信号保持了主要的语音模式，但包含了随机波动。

### 修改时间和音高

人类语音在语速和音高上自然变化。增强这些特征有助于模型对不同说话者和说话风格进行泛化。

- **时间拉伸：** 这种方法在不影响音高的情况下改变音频的速度。您可以放慢或加快录音以模拟语速快或慢的说话者。例如，加速10%（`rate=1.1`）或减速10%（`rate=0.9`）是常用的变换。
- **音高偏移：** 这会改变音频的音高而不改变其速度。将音高向上或向下移动几个半音可以模拟不同的说话者，使模型减少对原始训练集中特定声音特征的依赖。

### 时间偏移

一种简单而有效的方法是对音频进行时间偏移。这包括将整个波形向左或向右移动一个小的随机量，并用静音填充创建的空白。这可以防止模型假设语音总是从文件的确切开头开始，并提高其在较长片段中检测语音的能力。

### SpecAugment：特征域增强

虽然之前的方法是在原始音频波形上操作，但SpecAugment是一种流行的方法，它直接应用于音频的特征表示——对数梅尔谱图。它的原理是遮蔽（或置零）输入特征的一部分，从而使模型从部分信息中学习到更完整的表示。

SpecAugment包含两种主要操作：

1. **时间遮蔽：** 谱图中一个随机的连续时间步长块被遮蔽。这类似于在录音期间暂时覆盖麦克风。它使模型更能应对音频中短暂的遮挡或非语音事件。
2. **频率遮蔽：** 整个音频持续时间内的一个随机频率通道范围（梅尔频带）被遮蔽。这模拟了某些频带丢失的情况，可能是由于麦克风质量差或特定的背景噪音。

由于SpecAugment直接在馈入神经网络 (neural network)的特征上操作，它计算效率高，并且已被证明对现代ASR架构（如Transformer和Conformer）非常有效。



![对数梅尔谱图上的SpecAugment](plots/7159-1.json)



> 一个对数梅尔谱图，应用了时间遮蔽（垂直灰色条）和频率遮蔽（水平灰色条）。模型必须在信息缺失的情况下预测正确的转录。

### 在训练流程中应用数据增强

要实现数据增强，您通常会将这些变换集成到数据加载流程中。`audiomentations` 和 `torchaudio.transforms` 等库为这些任务提供了易于使用的函数。通过在加载每个训练样本时应用一组随机增强，您可以确保模型在每个周期都看到略有不同版本的数据。这种方法比预先生成和存储所有可能的增强文件更节省内存。

在设计数据增强策略时，应用能体现变化的变换是很重要的。过度使用不切实际的噪音或极端的速度变化有时可能损害性能。一个好的做法是，从温和的增强开始，逐步增加其强度，同时监测验证集的词错误率（WER），以找到一个既能改善泛化能力又不过度扭曲原始数据的平衡点。

## 参考资料

- [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://doi.org/10.21437/Interspeech.2019-2680) — Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, Quoc V. Le (2019)
  Journal: Proc. Interspeech 2019; Pages: 2613-2617; DOI: [10.21437/Interspeech.2019-2680](https://doi.org/10.21437/Interspeech.2019-2680)
  介绍了SpecAugment，一种通过在语谱图上应用时间遮蔽和频率遮蔽来增强ASR系统的方法。
- [A Study on Data Augmentation for Robust Speech Recognition](https://ieeexplore.ieee.org/document/7952402) — P. H. N. Tan, S. L. Lee, C. W. R. Her, S. H. S. Lim, and J. H. L. Teo (2017)
  Journal: 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP); Publisher: IEEE; Pages: 5445-5449; DOI: [10.1109/ICASSP.2017.7952402](https://doi.org/10.1109/ICASSP.2017.7952402)
  研究了噪声添加和混响等数据增强技术对语音识别系统弹性的影响。
- [torchaudio.transforms](https://pytorch.org/audio/stable/transforms.html) — PyTorch Team (2024)
  Publisher: PyTorch
  PyTorch `torchaudio` 库中用于数据增强的音频转换功能的官方文档。

# 特征提取在ASR中的作用

来源：[原文](https://apxml.com/zh/courses/applied-speech-recognition/chapter-2-feature-extraction-for-speech/role-of-feature-extraction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

原始音频波形是声音随时间变化的精确数字记录，但对于ASR系统而言，它是一个嘈杂且效率低下的信息源。语音识别的基本任务是将信号映射到字符或词语序列，而原始波形包含远超此任务所需的信息。特征提取是将这种高维度、冗余的信号转换为紧凑、有意义的表示的过程，这种表示更适合机器学习 (machine learning)模型。

### 为何不直接使用原始音频？

将原始音频样本直接输入神经网络 (neural network)会带来一些重要问题。以标准16 kHz采样的单秒音频为例，这会转化为一个包含16,000个值的输入向量 (vector)。一段典型的十秒语音会产生一个160,000维的向量。处理如此庞大的输入不仅计算成本高昂，还会使模型难以有效学习。

1. **高维度和计算成本：** 原始音频数据的庞大尺寸使得训练深度学习 (deep learning)模型变得缓慢且内存占用大。网络需要在其输入层中配置大量参数 (parameter)，这增加了过拟合 (overfitting)的风险，并要求更多训练数据才能良好泛化。
2. **无关信息：** 波形捕捉所有细节，包括说话人的基频（音高）、录音环境的声学特征以及轻微背景噪声。尽管这些信息很完整，但其中大部分与转录的主要任务无关。ASR系统应该能够识别“hello”这个词，无论它是由音高较高还是较低的人说出的。特征提取有助于舍弃这种说话人特有和环境变异性，使模型能够专注于定义词语本身的语音内容。
3. **缺乏不变性：** 原始波形对微小变化高度敏感。信号中一个微小的时间偏移（相位偏移）可能导致完全不同的输入向量，即使感知到的声音是相同的。模型难以从头学习这些不变性。一个良好的特征表示本身应该对此类变化更稳定。

### 目的：一种受感知启发的表示

我们不直接使用原始信号，而是旨在创建能突出对人类感知最重要的语音特征。人类听觉系统不会处理每一个单独的振幅波动；它特别适应声音的频率内容及其随时间的变化。

MFCCs和对数梅尔频谱图等特征提取方法受此过程启发。它们将信号从时域（振幅 vs. 时间）转换成时频表示，这种表示强调语音特征，同时抑制噪声和说话人依赖性特征。

下图说明了特征提取在典型ASR流程中的位置。它作为一个重要的预处理步骤，接收原始、复杂的波形，并生成干净、紧凑的特征向量 (vector)序列，供声学模型使用。

> ASR流程：特征提取将原始音频压缩为一系列特征向量，然后将其传递给声学模型。

总而言之，特征提取的作用是达成三大主要目的：

- **压缩信息：** 显著降低输入数据的维度，使其在计算上更易于神经网络 (neural network)处理。
- **分离语音内容：** 强调对应于语音音素的频率分量，并弱化说话人音高或背景嗡嗡声等无关信息。
- **创建稳定输入：** 生成一种对音频信号中不重要的变化（如相位偏移或响度的小幅变化）更稳定的表示。

通过将原始音频转换为紧凑稳定的特征集，我们为声学模型提供了更清晰、更易于管理的输入。这使得模型能够更有效地学习从声音到文本的映射。在接下来的章节中，我们将介绍实现此目的的两种最主要方法：梅尔频率倒谱系数（MFCCs）和对数梅尔频谱图。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University (Online Draft)
  一本关于自然语言处理和语音识别的教科书，对语音特征、其原理和ASR流程进行了全面介绍。
- [Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences](https://ieeexplore.ieee.org/document/1163420) — S. B. Davis and P. Mermelstein (1980)
  Journal: IEEE Transactions on Acoustics, Speech, and Signal Processing; Publisher: IEEE; Volume: 28; Pages: 357-366; DOI: [10.1109/TASSP.1980.1163420](https://doi.org/10.1109/TASSP.1980.1163420)
  这篇学术文章首次介绍了梅尔频率倒谱系数（MFCCs），这是语音识别中的一个核心特征提取方法。
- [Automatic Speech Recognition: A Deep Learning Approach](https://doi.org/10.1007/978-1-4939-2693-3) — Dong Yu, Li Deng (2014)
  Publisher: Springer; DOI: [10.1007/978-1-4939-2693-3](https://doi.org/10.1007/978-1-4939-2693-3)
  本书介绍了用于ASR的深度学习方法，包括声学特征的讨论及其在深度学习模型中的准备工作，强调了感知启发表示的优势。

---

[上一节](../01-%E6%95%B0%E5%AD%97%E9%9F%B3%E9%A2%91%E4%B8%8E%E8%AF%AD%E9%9F%B3%E5%85%A5%E9%97%A8/07-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E5%8A%A0%E8%BD%BD%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96%E9%9F%B3%E9%A2%91%E6%B3%A2%E5%BD%A2.md) · [下一节](02-%E6%A2%85%E5%B0%94%E9%A2%91%E7%8E%87%E5%80%92%E8%B0%B1%E7%B3%BB%E6%95%B0%20%28MFCC%29.md)

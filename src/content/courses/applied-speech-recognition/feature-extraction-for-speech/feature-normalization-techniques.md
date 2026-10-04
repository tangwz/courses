---
course: "applied-speech-recognition"
chapter: "feature-extraction-for-speech"
lesson: "feature-normalization-techniques"
sourceId: 7115
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-2-feature-extraction-for-speech/feature-normalization-techniques"
title: "特征归一化技术"
description: "了解倒谱均值和方差归一化（CMVN）等特征归一化方法，以提高模型的稳定性。"
order: 5
plots: ["plots/7115-0.json"]
sourceHash: "80364d3f2d109e648f6c02f8459fadd6aad6cd1ffe250913461386b1c6a8afd5"
sourceCorrections: []
---

语音识别系统依赖于MFCC或对数梅尔频谱等已提取的特征。然而，这些原始特征值可能因与语音内容无关的因素（如麦克风类型、录音距离和背景噪音水平）而显著不同。特征归一化 (normalization)是一个预处理步骤，用于处理这种变异性，确保模型学习语音的语音内容，而不是偶然的录音条件。通过将特征缩放到一致的范围，归一化有助于稳定训练过程，并提高模型对新的、未见过的数据的泛化能力。

### 倒谱均值和方差归一化 (normalization)（CMVN）

语音识别中最常用的归一化技术是倒谱均值和方差归一化，简称CMVN。CMVN 的目标是转换特征向量 (vector)，使其均值为零，方差为一。这个过程，也称为标准化，消除了由录音通道或说话人特征引起的特征均值和方差的变化。

应用 CMVN 涉及计算一组帧中每个特征系数的均值和标准差，然后使用这些统计量来归一化每个帧。根据统计量计算的范围，这可以通过几种不同的方式完成。

- **语句级 CMVN：** 均值和方差按语句计算和应用。这对于消除单次录音特有的变化（例如音量突然变化）非常有效。它应用简单，因为每个语句都是独立的。
- **说话人级 CMVN：** 统计量是根据单个说话人的所有语句计算的。这有助于适应个体独特的发声特征，但您的数据需要说话人标签，这可能并非总是可用。
- **全局 CMVN：** 均值和方差是在整个训练数据集上计算的。这些固定统计量随后被保存并应用于所有数据，包括验证集和测试集。这是 ASR 管道中非常常见的方法，因为它确保项目所有部分的一致缩放。

下图展示了语句级 CMVN 的过程，其中每个语句的特征矩阵都独立归一化。

> 语句级倒谱均值和方差归一化过程。

### 归一化 (normalization)公式

对于给定的特征系数维度 $d$（例如，第4个MFCC系数），均值 $\mu_d$ 是通过计算一个语句或一组语句的所有 $T$ 个时间帧得到的：


$$
\mu_d = \frac{1}{T} \sum_{t=1}^{T} c_t(d)
$$


这里，$c_t(d)$ 是时间帧 $t$ 上第 $d$ 个系数的值。标准差 $\sigma_d$ 通过方差的平方根计算得到：


$$
\sigma_d = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (c_t(d) - \mu_d)^2}
$$


最后，每个系数 $c_t(d)$ 被归一化以产生新的特征值 $\hat{c}_t(d)$：


$$
\hat{c}_t(d) = \frac{c_t(d) - \mu_d}{\sigma_d}
$$


这种计算独立地对特征向量 (vector)中的每个系数执行。例如，如果您使用13个MFCC，您将计算13个独立的均值和13个独立的标准差。

下图展示了一个单个特征系数在CMVN前后的直方图。归一化之前，特征分布的中心均值约为 5.0。应用CMVN后，分布中心在0，标准差为1。



![特征系数归一化前后的分布](plots/7115-0.json)



> CMVN 对单个特征系数分布的影响。归一化后的特征中心位于零。

### 实际考量

在实现特征提取管道时，归一化 (normalization)策略的选择具有实际影响。

如果您选择**全局CMVN**，则极为重要的一点是，*仅*从训练数据计算均值和标准差统计量。保存这些统计量并将其应用于验证集和测试集，可以防止测试集信息“泄露”到训练过程中，这会使您对模型性能的评估过于乐观。

对于**语句级CMVN**，过程更简单，因为每个文件都是独立处理的。这对于录音条件预期会频繁变化的应用有益。

尽管现代神经网络 (neural network)架构通常包含内部归一化层，如 `BatchNorm` 或 `LayerNorm`，但将CMVN直接应用于输入特征仍然是一种常见且有益的做法。它提供了一个干净、标准化的输入，可以减轻网络初始层的负担，通常导致更快的收敛和更好的整体性能。此步骤确保声学模型可以专注于学习从语音相关模式到文本的映射，而不会被外部声学变异性分散注意力。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Prentice Hall
  一本经典教科书，全面介绍了语音处理基础知识，包括MFCC等特征提取技术以及倒谱归一化的原理和方法。
- [The Kaldi Speech Recognition Toolkit](https://www.danielpovey.com/files/2011_asru_kaldi.pdf) — Daniel Povey, Arnab Ghoshal, Gilles Boulianne, Luka Burget, Ondřej Glembek, Nagendra Kumar, Mirko Hannemann, Gabriel Audette, John Bastien, Ken Church, Jong Hun Park, Yuval Shakhnarovich, David Snyder, Tanel Alumäe, Emre Yilmaz, Francine Chen, and Casey Kennington (2011)
  Journal: IEEE Automatic Speech Recognition and Understanding (ASRU) Workshop; Pages: 1-4
  这篇论文介绍了Kaldi工具包，其中倒谱均值方差归一化(CMVN)作为其鲁棒语音识别流程中的标准组件。它展示了该技术的实际应用。

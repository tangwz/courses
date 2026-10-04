---
course: "applied-speech-recognition"
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-2-feature-extraction-for-speech"
sourceId: 1241
chapter: "feature-extraction-for-speech"
title: "语音识别的特征提取"
order: 2
description: "学习从音频信号中提取有意义的特征，包括MFCC和对数梅尔谱图，用于ASR模型。"
hasQuiz: false
---

原始音频波形，表现为一系列随时间变化的振幅值，维度很高，且包含对语音识别并非直接有用的信息。将这些每秒数万个采样点的原始数据直接输入神经网络，计算效率低，并且会掩盖模型需要学习的语音模式。因此，特征提取是ASR（语音识别）流程中的一个常规步骤。目的是将原始音频压缩成维度更低、信息量更大的形式。

本章中，我们将实践这些特征的生成方法。您将学会构建两种广泛使用的表示形式：梅尔频率倒谱系数（MFCC）和对数梅尔谱图。我们将分步演示它们的计算过程，比较它们的特点，并讨论在现代深度学习模型中，何种情况下一种形式会比另一种更受青睐。我们还将讲解归一化方法，例如倒谱均值方差归一化（CMVN），以使特征更具一致性。本章以一个实践练习作结，在此您将编写代码来处理整个音频数据集，形成一个可用于训练的特征集合。

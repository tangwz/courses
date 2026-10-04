---
course: "applied-speech-recognition"
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-1-foundations-digital-audio-speech"
sourceId: 1238
chapter: "foundations-digital-audio-speech"
title: "数字音频与语音入门"
order: 1
description: "学习数字音频信号、语音学和时频分析的核心内容，用于语音识别应用。"
hasQuiz: false
---

要构建能理解语音的系统，我们必须先了解语音本身的结构及其数字表示方式。本章提供必要的背景知识，从声波的特性开始，到它在计算机上可供处理的数字表示形式。

我们将介绍从语音话语到机器可读格式的完整过程。你将学习人类语音的基本特性，以及将其数字化的技术步骤。一个表示为 $x(t)$ 的连续模拟音频信号，必须通过采样和量化等过程，转换为离散的数字序列 $x[n]$。

学完本章后，你将能够：

*   描述一个典型的自动语音识别（ASR）系统的高层架构。
*   辨认语音的基本语言单位，例如音素和音位变体。
*   解释模拟音频如何通过采样和量化转换为数字信号。
*   使用 Python `Librosa` 库加载和处理音频数据。
*   区分信号的时域和频域表示。
*   生成并理解语谱图，作为可视化语音的一种方式。

本章以一个动手练习作为结尾，你将应用这些技能来加载并可视化音频波形和语谱图，为后续的特征提取方法做准备。

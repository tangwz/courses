---
course: "introduction-to-speech-recognition"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-2-processing-audio-signals"
sourceId: 1239
chapter: "processing-audio-signals"
title: "处理音频信号"
order: 2
description: "学习如何将原始声波转换为机器学习模型能理解的MFCCs等特征。"
hasQuiz: false
---

在计算机理解语音之前，原始音频信号必须转换为结构化的数字形式。机器学习模型不直接处理声波；它们需要能提取与语音相关的声学属性的特征。本章介绍执行这种转换的常规方法。

我们将从连续声音如何通过采样和量化进行数字化的基本原理讲起。接着，你将学习如何将这些信号可视化为波形图和频谱图，以便检查其内容。本章的主要部分侧重于特征提取，逐步讲解生成梅尔频率倒谱系数（MFCCs）的步骤，MFCCs是语音识别系统的一种常见输入。

具体来说，你将学会：
*   使用采样和量化以数字方式表示声音。
*   用波形图和频谱图将音频数据可视化。
*   应用预加重、分帧和加窗等预处理步骤。
*   从原始音频信号中计算MFCCs。
*   使用Python将音频文件处理为特征。

本章结束时，你将能够将标准音频文件转换为适合与声学模型一起使用的特征矩阵。

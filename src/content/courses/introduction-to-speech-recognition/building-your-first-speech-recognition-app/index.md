---
course: "introduction-to-speech-recognition"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-6-building-your-first-speech-recognition-app"
sourceId: 1246
chapter: "building-your-first-speech-recognition-app"
title: "构建你的第一个语音识别应用程序"
order: 6
description: "一个实用且动手的章节，引导你使用 Python 和预训练模型构建一个简单的语音转文本应用程序。"
hasQuiz: false
---

前面几章已经详细说明了自动语音识别（ASR）系统的组成部分，从音频处理到声学模型和语言模型的结合。基本目标是找到对于给定的音频观测 $O$ 使概率最大化的词序列 $W$，通常表示为：

$$
\hat{W} = \underset{W}{\arg\max} \, P(O|W)P(W)
$$

本章将从该方程的原理转向其在实践中的运用。重点转向使用已实现这些组成部分的成熟工具，让你能够无需从头训练模型即可构建功能性应用程序。

你将使用常用Python库来构建一个完整的语音转文本程序。我们将介绍环境设置、加载预训练模型以及编写转录音频的脚本。你将用到预先录制的音频文件以及来自麦克风的实时输入。本章以一个动手练习结束，该练习旨在构建一个简单的声控命令工具，巩固你将语音识别集成到应用程序中的能力。

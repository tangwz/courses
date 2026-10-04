---
course: "speech-recognition-synthesis-asr-tts"
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-5-neural-vocoders-waveform-generation"
sourceId: 692
chapter: "neural-vocoders-waveform-generation"
title: "神经网络声码器与波形生成"
order: 5
description: "学习并实现先进的神经网络声码器，将声学特征转换为高质量音频波形。"
hasQuiz: false
---

之前的章节侧重于使用文本转语音（TTS）模型从文本生成中间声学表示，例如梅尔频谱图。然而，这些表示并非直接可听的声音。为了生成最终的语音波形，我们需要一个能将这些声学特征转换为高保真音频信号的组件。这个组件被称为声码器。

传统的声码器方法，通常基于格里芬-利姆（Griffin-Lim）等信号处理技术，可以合成可理解的语音，但常出现伪影并缺乏自然度。本章主要讲解现代神经网络声码器，它们采用深度学习来生成明显更高质量的音频。

您将学习几种类型的神经网络声码器：
*   **自回归模型**，如WaveNet，这类模型顺序生成音频样本。
*   **基于流的模型**，如WaveGlow，可以实现并行波形生成。
*   **基于GAN的模型**，例如MelGAN和HiFi-GAN，它们采用对抗训练来实现高效、高质量的合成。
*   **扩散模型**，一种新兴的波形生成方法。

我们还将介绍这些模型如何以声学特征为条件，并讨论评估生成音频质量的方法。本章包含一个实践环节，您将使用预训练的神经网络声码器来合成音频。

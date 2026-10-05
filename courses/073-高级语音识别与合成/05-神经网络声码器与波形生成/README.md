# 第 5 章：神经网络声码器与波形生成

来源：[原章节](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-5-neural-vocoders-waveform-generation)

[返回课程目录](../README.md)

之前的章节侧重于使用文本转语音（TTS）模型从文本生成中间声学表示，例如梅尔频谱图。然而，这些表示并非直接可听的声音。为了生成最终的语音波形，我们需要一个能将这些声学特征转换为高保真音频信号的组件。这个组件被称为声码器。

传统的声码器方法，通常基于格里芬-利姆（Griffin-Lim）等信号处理技术，可以合成可理解的语音，但常出现伪影并缺乏自然度。本章主要讲解现代神经网络声码器，它们采用深度学习来生成明显更高质量的音频。

您将学习几种类型的神经网络声码器：
*   **自回归模型**，如WaveNet，这类模型顺序生成音频样本。
*   **基于流的模型**，如WaveGlow，可以实现并行波形生成。
*   **基于GAN的模型**，例如MelGAN和HiFi-GAN，它们采用对抗训练来实现高效、高质量的合成。
*   **扩散模型**，一种新兴的波形生成方法。

我们还将介绍这些模型如何以声学特征为条件，并讨论评估生成音频质量的方法。本章包含一个实践环节，您将使用预训练的神经网络声码器来合成音频。

## 小节

- 1. [传统声码器的不足之处](01-%E4%BC%A0%E7%BB%9F%E5%A3%B0%E7%A0%81%E5%99%A8%E7%9A%84%E4%B8%8D%E8%B6%B3%E4%B9%8B%E5%A4%84.md)
- 2. [自回归波形模型（WaveNet, WaveRNN）](02-%E8%87%AA%E5%9B%9E%E5%BD%92%E6%B3%A2%E5%BD%A2%E6%A8%A1%E5%9E%8B%EF%BC%88WaveNet%2C%20WaveRNN%EF%BC%89.md)
- 3. [基于流的声码器 (WaveGlow, FloWaveNet)](03-%E5%9F%BA%E4%BA%8E%E6%B5%81%E7%9A%84%E5%A3%B0%E7%A0%81%E5%99%A8%20%28WaveGlow%2C%20FloWaveNet%29.md)
- 4. [基于GAN的声码器（MelGAN, HiFi-GAN）](04-%E5%9F%BA%E4%BA%8EGAN%E7%9A%84%E5%A3%B0%E7%A0%81%E5%99%A8%EF%BC%88MelGAN%2C%20HiFi-GAN%EF%BC%89.md)
- 5. [用于声码器的扩散模型](05-%E7%94%A8%E4%BA%8E%E5%A3%B0%E7%A0%81%E5%99%A8%E7%9A%84%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B.md)
- 6. [神经网络声码器的条件化](06-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%A3%B0%E7%A0%81%E5%99%A8%E7%9A%84%E6%9D%A1%E4%BB%B6%E5%8C%96.md)
- 7. [合成音频质量评估](07-%E5%90%88%E6%88%90%E9%9F%B3%E9%A2%91%E8%B4%A8%E9%87%8F%E8%AF%84%E4%BC%B0.md)
- 8. [动手实践：使用神经声码器](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%E7%A5%9E%E7%BB%8F%E5%A3%B0%E7%A0%81%E5%99%A8.md)

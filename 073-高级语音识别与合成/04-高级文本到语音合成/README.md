# 第 4 章：高级文本到语音合成

来源：[原章节](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-4-advanced-text-to-speech-synthesis)

[返回课程目录](../README.md)

本章将重点转向语音生成，详细介绍构建现代文本到语音（TTS）系统所用的方法。目的是从合成的初步认识开始，逐步达到能生成高保真、听起来自然且可控制的人工语音的技术水平。

您将学习当前几种先进声学模型的架构和训练过程：

*   **自回归模型：** 分析 Tacotron 和基于 Transformer 的 TTS 等序列到序列的方法。
*   **非自回归模型：** 学习用于加速推理的并行生成技术，如 FastSpeech 及其变体。
*   **基于流和基于 GAN 的模型：** 考察应用于声学特征生成的不同生成建模方式。

除了核心模型架构，我们还将介绍以下方法：

*   对语音韵律（节奏、语调）进行建模和控制。
*   生成具有不同风格或情感的富有表现力的语音。
*   实现语音克隆和转换系统。

本章包含一个动手实践部分，侧重于使用现代工具包训练高级 TTS 模型。

## 小节

- 1. [自回归声学模型 (Tacotron, Transformer TTS)](01-%E8%87%AA%E5%9B%9E%E5%BD%92%E5%A3%B0%E5%AD%A6%E6%A8%A1%E5%9E%8B%20%28Tacotron%2C%20Transformer%20TTS%29.md)
- 2. [非自回归声学模型 (FastSpeech, ParaNet)](02-%E9%9D%9E%E8%87%AA%E5%9B%9E%E5%BD%92%E5%A3%B0%E5%AD%A6%E6%A8%A1%E5%9E%8B%20%28FastSpeech%2C%20ParaNet%29.md)
- 3. [基于流的文本到语音合成模型](03-%E5%9F%BA%E4%BA%8E%E6%B5%81%E7%9A%84%E6%96%87%E6%9C%AC%E5%88%B0%E8%AF%AD%E9%9F%B3%E5%90%88%E6%88%90%E6%A8%A1%E5%9E%8B.md)
- 4. [生成对抗网络（GANs）在文本到语音中的应用](04-%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%EF%BC%88GANs%EF%BC%89%E5%9C%A8%E6%96%87%E6%9C%AC%E5%88%B0%E8%AF%AD%E9%9F%B3%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md)
- 5. [韵律建模与控制](05-%E9%9F%B5%E5%BE%8B%E5%BB%BA%E6%A8%A1%E4%B8%8E%E6%8E%A7%E5%88%B6.md)
- 6. [富有表现力的语音合成](06-%E5%AF%8C%E6%9C%89%E8%A1%A8%E7%8E%B0%E5%8A%9B%E7%9A%84%E8%AF%AD%E9%9F%B3%E5%90%88%E6%88%90.md)
- 7. [声音克隆与转换](07-%E5%A3%B0%E9%9F%B3%E5%85%8B%E9%9A%86%E4%B8%8E%E8%BD%AC%E6%8D%A2.md)
- 8. [动手实践：训练高级TTS模型](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83%E9%AB%98%E7%BA%A7TTS%E6%A8%A1%E5%9E%8B.md)

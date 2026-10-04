---
course: "introduction-to-speech-recognition"
chapter: "processing-audio-signals"
lesson: "understanding-audio-formats"
sourceId: 7093
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-2-processing-audio-signals/understanding-audio-formats"
title: "了解音频格式（WAV、MP3、FLAC）"
description: "一份关于WAV、MP3、FLAC等常用音频文件格式及其特点的实用指南。"
order: 2
plots: []
sourceHash: "7a90f0a93d162ae3f01ba01fa81346f0337f030209e3cf0101405b5e5176a17d"
sourceCorrections: []
---

数字音频通过采样和量化 (quantization)从连续声波中转换而来，需要存储在特定的文件格式中。如同图片有多种格式（如`.jpg`、`.png`、`.gif`），音频也采用多种格式，每种格式在文件大小、音频质量和处理需求之间都有独特的权衡。

对于语音识别，格式的选择很重要，因为它决定了模型将接收到的数据质量。下面我们来考察你将遇到的三种最常用格式：WAV、MP3和FLAC。

### 未压缩音频：WAV

波形音频文件格式，即WAV（`.wav`），是存储数字音频最简单、最直接的方式。可以将WAV文件视为采样和量化 (quantization)后音频数据的原始容器。它依次存储每个采样的振幅值，文件开头有一个小小的头部，会指定采样率和位深度等元数据。

- **质量：** 由于是未压缩格式，WAV文件能以完美的保真度表示原始数字音频。不会丢失任何信息。
- **文件大小：** 这种完美的保真度是有代价的。WAV文件非常大。一分钟CD音质的立体声音频（44.1 kHz，16位）大约占用10 MB的空间。
- **在ASR中的使用：** 由于其无损特性，WAV文件是训练和分析语音的首选格式。在编程环境中处理音频时，无论原始文件格式如何，您几乎总是先将其解码成内存中原始的、类似WAV的表示。

这种关系很直接：


$$
\text{文件大小 (WAV)} = \text{采样率} \times \text{位深度} \times \text{声道数} \times \text{持续时间（秒）}
$$


WAV文件的一个类比是位图图像文件（`.bmp`）。它存储了每个像素的颜色值，从而得到一个完美但非常大的图像文件。

### 有损压缩：MP3

MP3（`.mp3`）格式旨在解决文件过大的问题。它采用**有损压缩**，这意味着它通过永久丢弃部分音频数据来减小文件大小。

这个过程并非随机。MP3编码器采用心理声学原理——即人耳感知声音的研究。它们会去除普通人难以听到的音频频率和声音，例如极高频率或与响亮声音同时出现的微弱声音。

- **质量：** 由于数据已被永久去除，其音频质量低于WAV。尽管对于人耳听起来通常“足够好”，但对于机器学习 (machine learning)模型而言，去除的信息有时可能很重要。
- **文件大小：** MP3文件比WAV文件明显小得多，通常小10倍或更多。这使其非常适合流媒体音乐和在消费设备上存储大量内容。
- **在ASR中的使用：** 尽管您通常不会使用MP3来*训练*高性能ASR模型（因为这将是在不完美数据上进行训练），但您经常会遇到它们作为模型的*输入*。在处理流程中，常见的第一个步骤是在处理MP3文件之前，将其解压缩回原始波形。

MP3文件就像JPEG图像（`.jpg`）。它通过丢弃细微的视觉细节来达到较小的文件大小，从而得到一个在人眼看来效果很好但并非原始完美复制的图像。

### 无损压缩：FLAC

自由无损音频编解码器，即FLAC（`.flac`），在WAV的巨型文件和MP3的降质压缩之间提供了一个中间方案。它采用**无损压缩**。

这意味着FLAC文件虽然比WAV文件小，但它保留了原始音频信息的每一个比特。它通过找到更高效的数据表示方式来实现这一点，类似于ZIP文件在不改变内容的情况下压缩文档。当您解压缩FLAC文件时，您会得到一个与原始未处理音频逐比特相同的副本。

- **质量：** 完美，与WAV相同。不会丢失任何信息。
- **文件大小：** 通常比同等的WAV文件小40%到60%。
- **在ASR中的使用：** FLAC是存储高质量语音数据用于训练的绝佳选择。与WAV相比，它节省了大量存储空间，同时不牺牲任何音频质量。与MP3类似，FLAC文件必须解压缩成原始波形表示才能用于特征提取。

FLAC文件类似于ZIP文件（`.zip`）。它使内容在存储和传输时更小，但当您“解压缩”它时，会恢复完全原始的文件。

### 语音识别中选择合适的格式

主要的结论是，所有音频必须转换为原始的、未压缩的波形，然后才能由ASR系统处理。文件格式，无论是WAV、MP3还是FLAC，仅仅告知您音频的存储方式和来源。

- 对于**训练ASR模型**，始终使用无损源（WAV或FLAC），以确保模型从可用的最高质量数据中学习。
- **使用ASR模型时**，您可能会收到任何格式的输入。您的应用程序必须能够将MP3和FLAC等格式解压缩为原始波形，然后才能将数据发送给特征提取器。

下图说明了不同格式在ASR处理前如何被处理。

> 无论音频的存储格式如何，在用于为语音识别模型创建特征之前，它都会转换为原始波形。

了解这些格式使您能够做出明智的决策，关于数据存储和构建能够处理各种音频输入的流程。在下一节中，我们将学习如何可视化这些原始波形，以查看语音的呈现方式。

## 参考资料

- [Principles of Digital Audio](https://www.vitalsource.com/products/principles-of-digital-audio-sixth-edition-ken-c-pohlmann-v9780071663472) — Ken C. Pohlmann (2010)
  Publisher: McGraw-Hill Education
  全面概述数字音频基础知识、采样、量化和WAV等非压缩音频存储方式。
- [Audio Signal Processing and Coding](https://www.wiley.com/en-us/Audio+Signal+Processing+and+Coding-p-9780471791478) — Andreas Spanias, Ted Painter, Venkatraman Atti (2007)
  Publisher: John Wiley & Sons; Pages: 464
  解释了音频压缩原理，包括有损编解码器（MP3）的心理声学和无损压缩（FLAC）的方法。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition (3rd Edition Draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  涵盖语音识别的标准教材，包括音频预处理和音频质量对ASR系统性能的影响。

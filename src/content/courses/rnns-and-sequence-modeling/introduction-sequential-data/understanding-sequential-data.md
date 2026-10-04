---
course: "rnns-and-sequence-modeling"
chapter: "introduction-sequential-data"
lesson: "understanding-sequential-data"
sourceId: 2512
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-1-introduction-sequential-data/understanding-sequential-data"
title: "理解序列数据"
description: "查看序列数据的例子，包括文本、时间序列、音频和视频。"
order: 1
plots: ["plots/2512-0.json"]
sourceHash: "af39937b120d6ebc38cc157716dc5c6a4655b2f2d9fd6958b006a72189faef71"
sourceCorrections: []
---

许多类型的数据并非仅仅是独立测量值的集合；它们具有固有的顺序或结构，其中元素的位置提供了重要上下文 (context)。这种有序的特性是序列数据的本质。严格来说，我们可以将序列视为一个有序的元素列表，$x = (x_1, x_2, ..., x_T)$，其中 $x_t$ 是时间步或位置 $t$ 上的元素，$T$ 是序列的总长度。

一个重要特点是顺序很重要。交换元素会完全改变数据的含义或性质。下面是一些常见例子：

### 文本数据

自然语言可能是最直观的例子。思考句子“The quick brown fox jumps over the lazy dog.”（敏捷的棕色狐狸跳过懒狗）。其含义源于词语的特定排列。将顺序改为“Lazy jumps dog the over brown quick fox the”会导致语无伦次。每个词相对于其他词的位置都非常重要。应用包括：

- 情感分析（评论分类）
- 机器翻译（英译法）
- 文本生成（撰写文章、代码）

> 文本序列的简单表示，其中每个词语占据一个特定位置。

### 时间序列数据

随时间记录的数据点构成序列。例子包括：

- 每日股票价格
- 每小时温度读数
- 每月销售额
- 心电图（ECG）信号

在这些情况中，顺序很重要。今天的股票价格值取决于其先前的值。天气模式从随时间变化的温度、气压和湿度读数序列中产生。预测未来值（预报）很大程度上取决于对这些过去时间依赖性的把握。



![时间序列例子：温度读数](plots/2512-0.json)



> 每小时温度数据显示序列依赖性；某一小时的读数与前一小时的读数相关联。

### 音频数据

声音是固有的序列。音频波形表示为以规则间隔采样（例如，CD音质音频每秒44,100次）的振幅值序列。这些振幅值的顺序定义了频率、节奏，并最终定义了我们感知到的声音，无论是语音、音乐还是环境噪声。打乱样本会破坏音频信息。

### 视频数据

视频可以看作是图像（帧）的序列。虽然每帧都是一个像素的空间网格，但随时间变化的帧序列产生了运动和叙事。视频中的动作识别等任务取决于分析物体或场景如何在连续帧之间变化。

### 生物序列

在生物信息学中，DNA和蛋白质序列是基本的。DNA链是核苷酸（A、C、G、T）的序列，蛋白质是氨基酸的序列。特定的顺序决定了其生物学功能、结构和性质。识别基因或预测蛋白质折叠取决于对这些有序序列的分析。

在所有这些例子中，主要特点是元素之间的关系取决于它们的位置。元素 $x_t$ 的值或含义通常受其前面元素 $(x_1, ..., x_{t-1})$ 的影响，并可能影响其后面元素 $(x_{t+1}, ..., x_T)$。这与每个样本都独立的数据集形成鲜明对比，例如仅仅基于像素值对图像进行分类，而不考虑来自其他图像的任何时间顺序或上下文 (context)（除非是视频！）。

识别您的数据是否具有这种序列属性，是选择合适建模技术的第一步。正如我们稍后将看到的，标准前馈神经网络 (neural network)难以处理这种有序结构，因此需要像循环神经网络（RNN）这样专门设计用于按顺序处理信息的模型。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本综合性教材，提供了深度学习的基础理解，其中包括关于序列模型和循环神经网络的专门章节。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Online manuscript released August 24, 2025
  一本关于自然语言处理的权威教材，详细介绍了文本作为序列数据的特性及其各种建模技术。第四版草稿持续更新中。
- [Sequence Models (Course 5 of the Deep Learning Specialization)](https://www.coursera.org/learn/nlp-sequence-models) — Andrew Ng, Kian Katanforoosh, Younes Bensouda Mourri (2021)
  Publisher: Coursera / DeepLearning.AI
  一个在线课程，为序列数据及其使用循环神经网络、LSTM 和其他高级技术进行建模提供了实用介绍。

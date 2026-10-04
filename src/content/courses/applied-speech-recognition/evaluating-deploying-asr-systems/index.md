---
course: "applied-speech-recognition"
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-6-evaluating-deploying-asr-systems"
sourceId: 1248
chapter: "evaluating-deploying-asr-systems"
title: "ASR系统的评估与部署"
order: 6
description: "学习采用词错误率 (WER) 评估ASR模型，进行数据扩充，并部署一个简单的语音转文本应用。"
hasQuiz: false
---

在构建了多个声学模型之后，接下来需要衡量它们的效能并将其投入使用。本章主要讲解ASR开发周期的这些收尾阶段：评估与部署。

首先，你将学习如何定量评估一个ASR系统的表现。我们会讲解行业标准指标，即词错误率 (WER) 和字符错误率 (CER)。你会看到WER是如何根据替换次数 ($S$)、删除次数 ($D$) 和插入次数 ($I$) 相对于参考文本中的总词数 ($N$) 计算得出的：

$$
\text{WER} = \frac{S + D + I}{N}
$$

完成评估后，我们将了解一种通过音频数据扩充来提高模型通用性的常用方法。本章随后会从理论转向实践。你将使用Hugging Face `pipeline` 用于简便的推断，然后使用Gradio库为你的模型构建一个交互式网页界面。最后，我们将讨论处理流式音频系统的架构要求。

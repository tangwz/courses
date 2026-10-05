# 第 6 章：ASR系统的评估与部署

来源：[原章节](https://apxml.com/zh/courses/applied-speech-recognition/chapter-6-evaluating-deploying-asr-systems)

[返回课程目录](../README.md)

在构建了多个声学模型之后，接下来需要衡量它们的效能并将其投入使用。本章主要讲解ASR开发周期的这些收尾阶段：评估与部署。

首先，你将学习如何定量评估一个ASR系统的表现。我们会讲解行业标准指标，即词错误率 (WER) 和字符错误率 (CER)。你会看到WER是如何根据替换次数 ($S$)、删除次数 ($D$) 和插入次数 ($I$) 相对于参考文本中的总词数 ($N$) 计算得出的：

$$
\text{WER} = \frac{S + D + I}{N}
$$

完成评估后，我们将了解一种通过音频数据扩充来提高模型通用性的常用方法。本章随后会从理论转向实践。你将使用Hugging Face `pipeline` 用于简便的推断，然后使用Gradio库为你的模型构建一个交互式网页界面。最后，我们将讨论处理流式音频系统的架构要求。

## 小节

- 1. [ASR 性能评估指标：词错误率 (WER) 和字符错误率 (CER)](01-ASR%20%E6%80%A7%E8%83%BD%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87%EF%BC%9A%E8%AF%8D%E9%94%99%E8%AF%AF%E7%8E%87%20%28WER%29%20%E5%92%8C%E5%AD%97%E7%AC%A6%E9%94%99%E8%AF%AF%E7%8E%87%20%28CER%29.md)
- 2. [计算词错率](02-%E8%AE%A1%E7%AE%97%E8%AF%8D%E9%94%99%E7%8E%87.md)
- 3. [语音数据的常用增强方法](03-%E8%AF%AD%E9%9F%B3%E6%95%B0%E6%8D%AE%E7%9A%84%E5%B8%B8%E7%94%A8%E5%A2%9E%E5%BC%BA%E6%96%B9%E6%B3%95.md)
- 4. [使用Hugging Face流水线进行ASR](04-%E4%BD%BF%E7%94%A8Hugging%20Face%E6%B5%81%E6%B0%B4%E7%BA%BF%E8%BF%9B%E8%A1%8CASR.md)
- 5. [使用 Gradio 构建语音转文本应用](05-%E4%BD%BF%E7%94%A8%20Gradio%20%E6%9E%84%E5%BB%BA%E8%AF%AD%E9%9F%B3%E8%BD%AC%E6%96%87%E6%9C%AC%E5%BA%94%E7%94%A8.md)
- 6. [实时流式ASR的考虑事项](06-%E5%AE%9E%E6%97%B6%E6%B5%81%E5%BC%8FASR%E7%9A%84%E8%80%83%E8%99%91%E4%BA%8B%E9%A1%B9.md)
- 7. [实践：评估和构建演示应用](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AF%84%E4%BC%B0%E5%92%8C%E6%9E%84%E5%BB%BA%E6%BC%94%E7%A4%BA%E5%BA%94%E7%94%A8.md)

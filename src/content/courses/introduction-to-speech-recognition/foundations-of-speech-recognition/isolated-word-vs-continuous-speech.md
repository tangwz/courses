---
course: "introduction-to-speech-recognition"
chapter: "foundations-of-speech-recognition"
lesson: "isolated-word-vs-continuous-speech"
sourceId: 7087
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-1-foundations-of-speech-recognition/isolated-word-vs-continuous-speech"
title: "语音识别的分类：孤立词识别与连续语音识别"
description: "了解识别单个词语与转录流畅口语的区别。"
order: 5
plots: ["plots/7087-0.json"]
sourceHash: "539243d7890ea48e6fc534d8169c35067ec584b0b388cd718d0c80c8d493b7d7"
sourceCorrections: []
---

并非所有语音识别任务都相同。旨在理解“播放”等单个语音指令的系统，其运作方式与转录完整口语句子的系统大不相同。这种区分产生了自动语音识别（ASR）系统的两大主要分类：孤立词识别和连续语音识别。弄清它们之间的不同，是理解语音转文本所涉难度的一个起始点。

### 孤立词识别

孤立词识别是这两种任务中较为简单的一种。这类系统旨在识别一个单词或一个简短、固定的短语，这些词语或短语前后带有特意的停顿。系统所能识别的词汇（或词汇集）通常较少且预先设定。

想象一下自动化电话系统中的语音控制菜单。当它提示你“说‘账单’进行账单查询或‘支持’寻求技术帮助”时，它希望你清晰地单独说出其中一个特定词语。

这种方法的首要优点是其简易性。词语周围的静音为需要分析的音频提供了清晰的起始和结束信号。系统不必解决判断一个词语在哪里结束、下一个词语在哪里开始的难题。

**孤立词识别的常见应用有：**

- **命令控制系统：** 软件或设备的简单命令，如“开始”、“停止”、“下一个”或“打印”。
- **自动化电话菜单：** 导航交互式语音应答（IVR）系统。
- **简单语音拨号：** 说出单个名字，如“给妈妈打电话”。



![孤立语音与连续语音波形](plots/7087-0.json)



> 孤立命令的音频波形（上方）有清晰的静音间隔，而连续语音（下方）则形成一个不间断的信号。

### 连续语音识别

连续语音识别是一项更具难度且更复杂的任务。它旨在理解并转录自然、流畅的人类语音，其中词语之间没有强制停顿。亚马逊Alexa等虚拟助手或谷歌文档语音输入等听写软件便是以此方式运作的。

此处面临的挑战明显更大。

1. **词语切分：** 系统首先必须在持续的声音流中分辨出每个词语的起始和结束。与孤立词不同，这里没有整齐的静音间隔可供参考。
2. **协同发音：** 在自然语音中，一个音的读法会受到其前后音的影响。例如，你在“did you”（听起来常像“did-joo”）和“did that”这两个短语中发“d”音的方式略有不同。系统必须能够处理这些差异。
3. **歧义：** 连续语音会引入歧义，这需要通过语言上下文 (context)来解决。例如，“recognize speech”（识别语音）和“wreck a nice beach”（毁了一个好海滩）这两个短语听起来可能几乎一样。自动语音识别系统不只需要声学信息来做出正确判断；它还需要知道“recognize speech”是一个更可能出现的词语序列。

### 并列比较

这两种识别类型之间的区分，对于理解自动语音识别系统尝试解决问题的范围非常重要。

| 特点 | 孤立词识别 | 连续语音识别 |
| --- | --- | --- |
| **输入方式** | 单个词语，有明显停顿 | 自然流畅的句子，无停顿 |
| **复杂度** | 较低 | 较高 |
| **主要难题** | 从列表中正确识别词语 | 寻找词语边界并解决歧义 |
| **词汇量** | 通常较少且固定 | 通常非常大且开放 |
| **应用示例** | 设备语音命令（例如，“下一个”） | 听写邮件或向虚拟助手提问 |

总而言之，孤立词识别在于*识别*单个项目，而连续语音识别在于*转录*一系列连贯的项目。尽管孤立词系统对特定应用有其用处，但多数现代自动语音识别技术致力于处理连续语音这一复杂且多样的难题。我们将在后续章节中讲解的技术主要针对这项更具挑战性的任务。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  这本权威教材全面介绍了语音识别基础知识，详细阐述了孤立词和连续语音识别系统的特点及区别。
- [Fundamentals of Speech Recognition](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEoqcP3BUS9JTWBEmc7KIX-DlM-zyKu7ZZf-NsudnQNoYcl3Yf_NcNpOTAmvA-7XAF1nEkO_rdAWfyU0nwoVkkGMM4RQU3EkjS8gAFjRXGhpgK_17dsdzLzyAgL5NLr9V2zd9X7b4ADxKU2KtaUCd0uupOf0P9stmFOGYvNUob5zmhKIPnHX9tHOFuozlQVzH8V1NdiX6r) — Lawrence R. Rabiner, Biing-Hwang Juang (1993)
  Publisher: Prentice Hall
  这是一部开创性著作，为语音识别奠定了理论和算法基础，其中包括对不同语音识别类型核心区别的解释。

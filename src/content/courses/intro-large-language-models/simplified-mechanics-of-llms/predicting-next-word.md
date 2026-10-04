---
course: "intro-large-language-models"
chapter: "simplified-mechanics-of-llms"
lesson: "predicting-next-word"
sourceId: 3688
sourceUrl: "https://apxml.com/zh/courses/intro-large-language-models/chapter-2-simplified-mechanics-of-llms/predicting-next-word"
title: "预测下一个词：核心理念"
description: "了解大型语言模型如何通过预测后续词汇来生成文本的基本理念。"
order: 2
plots: []
sourceHash: "ab729ed2dd3873918be0dc4f4fe133d78ba89f8376039f53d9f58b1ed24636db"
sourceCorrections: []
---

大型语言模型在生成文本时，其核心是执行一个高度复杂的预测任务。想象一下你在输入短信时，手机会建议下一个词。大型语言模型的工作原理与此类似，但规模更大，并且具有更强的语境理解能力。

其基本理念是**预测序列中的下一个词元 (token)**（通常对应一个词或词的一部分）。给定一系列前序词元（通常称为**上下文 (context)**），模型会计算其整个词汇表 (vocabulary)中下一个词元应该是什么的概率分布。

### 预测如何顺序进行

可以将其理解为一次构建句子的一部分。

1. \*\*输入上下文 (context)：\*\*模型接收一个初始的词元 (token)序列。这可以是您提供的提示，也可以是它迄今为止生成的文本。例如，上下文可能代表“The cat sat on the”这些词元。
2. \*\*概率计算：\*\*基于此上下文，模型分析它在训练过程中学习到的模式。然后，它会计算其词汇表 (vocabulary)中*每个可能*的词元在下一个位置出现的概率。它可能会得出：
   - “mat”的概率是 0.6（或 60%）
   - “roof”的概率是 0.2（或 20%）
   - “chair”的概率是 0.1（或 10%）
   - “computer”的概率是 0.0001（或 0.01%）
   - ……以此类推，还有成千上万个其他可能的词元。从数学角度看，您可以将此视为模型在估计 $P(\text{下一个词元} \mid \text{上下文})$。
3. **词元选择：**模型需要选择下一个词元。最简单的策略通常是**贪婪解码**，即直接选择概率最高的词元（在我们的例子中是“mat”）。更复杂的策略可能涉及从概率最高的前几个词元中进行采样，以引入多样性，但其核心理念仍基于这些计算出的概率。
4. \*\*更新上下文：\*\*选定的词元（“mat”）被附加到序列中。上下文现在变为“The cat sat on the mat”。
5. \*\*重复：\*\*此过程重复进行。模型接收新的、更长的上下文（“The cat sat on the mat”），并预测其后的*下一个*词元（可能是“and”），计算概率，选择词元，然后再次附加。

这个循环持续进行，每次添加一个词元，直到模型达到停止条件，例如生成一个预定义的序列结束词元或满足提示中指定的长度要求。

> 文本生成过程涉及基于当前上下文迭代预测下一个词元，然后附加所选词元，并重复此循环。

### 预测的依据：训练数据

模型如何“知道”在“The cat sat on the”之后“mat”比“computer”更可能出现？这种知识完全来自它所训练的海量文本数据。在训练期间，模型学习了词元 (token)之间的统计关系。它看到了无数类似“sat on the mat”、“sat on the chair”这样的序列示例，而类似“sat on the computer”的序列则极少（甚至没有）。这种接触使其能够构建语言模式的内部表示，并用其进行这些预测。

尽管我们常常将此简化为预测“下一个词”，但请记住，在上一节中提到，大型语言模型实际操作的是**词元**。原理相同，但预测的单位可能是整个词、词的一部分或标点符号，这取决于所用的词元化方法。

这种顺序的、概率驱动的预测机制是大型语言模型生成连贯且与上下文 (context)相关的文本背后的基本运作原理。预测的质量和复杂程度在很大程度上取决于模型的架构、训练数据集的大小以及其参数 (parameter)数量，我们将在后面提及这些。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NIPS 2017); Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.5555/3295222.3295349](https://doi.org/10.5555/3295222.3295349)
  介绍Transformer架构的基石论文，该架构支撑了大多数现代大型语言模型，并使其能够进行复杂的上下文理解和下一个词元预测。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本全面的教材，涵盖语言模型、序列预测以及自然语言处理的统计基础，这些是理解大型语言模型机制的。第三章（“N-gram 语言模型”）和深度学习在自然语言处理中的应用等相关章节尤为切合主题。
- [The Hugging Face Course: How do Large Language Models work?](https://huggingface.co/learn/llm_course/) — Hugging Face (2023)
  Publisher: Hugging Face
  对大型语言模型如何通过顺序词元预测、概率分布和贪婪解码等策略生成文本，提供了易懂且详细的说明。

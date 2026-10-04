# 定义大型语言模型（LLMs）

来源：[原文](https://apxml.com/zh/courses/intro-large-language-models/chapter-1-understanding-large-language-models/defining-large-language-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型，通常缩写为LLMs，是人工智能（AI）和自然语言处理（NLP）的一部分。它们是自然语言处理方面的一项重要进展。

那么，大型语言模型究竟是什么？LLM本质上是一种AI模型，专门用于理解、生成和与人类语言文本交互。可以将其视为一个经过训练、能处理单词、句子和段落的复杂系统。

让我们来分析一下这个名称：

- \*\*模型：\*\*在AI和机器学习 (machine learning)中，“模型”是一个经过数据训练的系统，用于识别模式或进行预测。它不是一个物理实体，而是一个复杂的数学和计算结构。
- \*\*语言：\*\*这表明了模型的处理对象。它专门处理人类语言——阅读、处理和生成。这使其与训练于图像、声音或数字数据的AI模型有所不同。
- \*\*大型：\*\*这个词非常重要，指代两个主要方面：
  1. \*\*用于训练的文本数据量之大：\*\*LLMs 在庞大的数据集上进行训练，这些数据通常包含互联网上可用的大部分文本、数字化书籍、文章及其他来源。这种广泛的接触使它们能够学到文本中复杂的模式、语法、事实，甚至推理 (inference)方式。
  2. \*\*模型拥有的参数 (parameter)数量：\*\*参数是模型在训练过程中调整的内部变量或“旋钮”。它们存储从数据中学到的模式。现代LLMs可以拥有数十亿，甚至数万亿的这些参数。如此多的参数有助于它们掌握复杂的语言细节，并生成在广泛主题下连贯、相关的文本。

简单来说，LLM接收输入文本（常称为“提示词 (prompt)”），并根据其训练中学到的统计模式生成输出文本。其基本运作方式通常是根据前面的一串文本预测最有可能的下一个词（或词的一部分）。通过反复预测下一个元素，它可以生成完整的句子、段落或文档。

将LLMs与早期的NLP系统区分开来很重要。较旧的方法可能依赖预设的语法规则或在较小数据集上的简单统计计算，LLMs则从它们处理的大量数据中隐式地学习这些模式。这种数据驱动的学习使它们能够处理种类多得多的任务，并展现出更灵活、更像人类的语言能力，而无需为每条特定语言规则进行显式编程。

然而，同样重要的是要记住，它们的能力源于识别训练数据中的模式，而不是源于真正的理解、意识或感知。它们是极其复杂的模式匹配和预测引擎。它们如何通过训练获得这些模式是下一部分的主要内容。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构，这是现代大型语言模型的基础组成部分。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  阐述模型大小、数据集规模和计算预算如何影响语言模型的性能。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson
  一本基础教材，全面介绍自然语言处理，包含现代语言模型及其演进的章节。

---

[上一节](02-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%20%28NLP%29%20%E7%AE%80%E4%BB%8B.md) · [下一节](04-LLM%E5%A6%82%E4%BD%95%E4%BB%8E%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%AD%A6%E4%B9%A0.md)

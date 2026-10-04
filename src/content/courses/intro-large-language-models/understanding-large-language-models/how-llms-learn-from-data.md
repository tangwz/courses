---
course: "intro-large-language-models"
chapter: "understanding-large-language-models"
lesson: "how-llms-learn-from-data"
sourceId: 3677
sourceUrl: "https://apxml.com/zh/courses/intro-large-language-models/chapter-1-understanding-large-language-models/how-llms-learn-from-data"
title: "LLM如何从文本数据中学习"
description: "对LLM使用海量文本数据进行训练过程的简要说明。"
order: 4
plots: []
sourceHash: "18bfa2dfa613f0e231d5eadb20b902b09569d6af671bbfa73cb61b094c55a8b8"
sourceCorrections: []
---

让我们了解这些大型语言模型是怎样获得语言能力的。这并非魔法；这是一个从海量文本数据中学习的过程。

可以把训练前的LLM想象成一个空的“大脑”，准备学习一门语言，但目前一无所知。为了教它，我们向它提供一个大量的数字图书馆——可能包含数十亿个网页、书籍、文章、代码库以及来自互联网和数字化收藏的其他文本资料。这个集合被称为**训练数据集**。

这个学习过程（常被称为**训练**或**预训练 (pre-training)**）背后的主要思想，从宏观上看非常简单：模型学习预测一段文本中接下来会出现什么。它会不断地接收来自训练数据的文本序列，其中一部分被隐藏起来，它的任务是猜出被隐藏的部分，最常见的是下一个词。

例如，模型可能会看到：

`"The quick brown fox jumps over the lazy..."`

它的任务是预测下一个词，在这个常见短语中，这个词是`"dog"`。

最初，模型的预测是随机的，通常不正确。然而，每当它做出预测时，它会将其猜测与训练数据中的实际文本进行比较。如果预测错误，模型会稍微调 (fine-tuning)整其内部配置，使其在下次遇到类似情境时，更有可能预测出正确的词（或一个类似合理的词）。

这些内部调整发生在数十亿甚至数万亿个内部值上，这些值被称为**参数 (parameter)**或**权重 (weight)**。你可以将这些参数看作是控制模型学习的不同想法之间连接强度的“旋钮”。当模型犯错时，训练过程会计算如何转动这些“旋钮”来改进未来的预测。

> LLM训练过程的简化视图。文本数据输入到训练算法中，该算法迭代预测文本的某些部分，并根据正确性调整模型的内部参数，最终生成一个能够理解和生成文本的训练好的LLM。

这个过程重复无数次，不断处理来自庞大训练数据集的序列。随着时间的推移，通过简单地学习预测各种情境中的下一个词，模型间接地学习到：

- **语法和句法:** 管理句子结构的规则。
- **事实和知识:** 文本数据中包含的信息（尽管它不像人类那样“知道”事物，但它会学习关联）。
- **推理 (inference)模式:** 思想连接或论证形成的常见方式（同样，基于数据中的模式）。
- **风格和语气:** 源材料中不同的写作方式。

数据的庞大规模非常重要。接触数万亿个词汇使模型能够吸收细微的语言使用模式，而这些模式在较小的数据集中无法显现。这就是它们被称为“大型”语言模型的原因——模型的大小（参数数量）和训练数据的大小是决定它们惊人能力的重要特点。

重要的是要记住，模型不像人类那样理解意义。它完全基于训练它的文本来学习词语和想法之间的统计关系。这对其能力和局限性都有影响，我们将在后面提到。目前，主要观点是LLM通过处理海量文本并调整其内部参数来学习，从而非常擅长预测接下来出现什么文本。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，该架构是现代大型语言模型及其学习机制的基础。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  详细介绍了GPT-3的训练和能力，展示了通过在海量数据集上扩展下一个词预测如何实现强大的语言理解和生成。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); DOI: [10.48550/arXiv.1810.04805](https://doi.org/10.48550/arXiv.1810.04805)
  描述了使用掩码语言建模等自监督任务在大规模文本语料库上预训练大型Transformer模型的方法，这对学习广泛的语言表示至关重要。
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  提供了自然语言处理的全面学术介绍，包括语言模型和神经网络训练的基础理论。

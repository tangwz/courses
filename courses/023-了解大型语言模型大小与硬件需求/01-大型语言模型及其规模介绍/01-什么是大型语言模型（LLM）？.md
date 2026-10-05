# 什么是大型语言模型（LLM）？

来源：[原文](https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-1-intro-llms-model-size/what-is-llm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

设想一下，你能和一个电脑程序对话，向它提问，让它写故事，概括长文档，甚至翻译语言，它都能以一种听起来非常像人类的方式理解并回应。这就是大型语言模型（LLM）的特点。

从根本上说，LLM是一种人工智能（AI）程序，专门用于理解、处理和生成人类语言（文本）。把它看作一个极其精密的模式匹配机器。它已经通过海量的文本数据进行了训练，比如维基百科、大量的书籍、文章以及互联网上的其他文本资料。

为什么叫“大型”？“大型”这个词主要指两个方面：

1. **它所训练的庞大数据量。** 这种充分的训练使模型能够学习语法、事实、推理 (inference)能力、不同的写作风格以及语言的精微之处。
2. **它用于存储和处理这些已学信息的内部“参数 (parameter)”的庞大数量。** 我们将在下一节具体介绍参数，但现在，请先明白，通常更多的参数能让模型掌握更复杂的模式和知识。

LLM并非以人类意识或经验的方式“理解”语言。相反，它们学习词语和想法之间的统计关系。当你给LLM一个提示（一段文本输入）时，它会根据训练期间学到的模式，预测最有可能出现的词语序列。这种预测过程使其能完成以下任务：

- **生成文本：** 编写电子邮件、代码、诗歌、营销文案等。
- **回答问题：** 根据其训练数据提供信息。
- **翻译：** 将文本从一种语言转换到另一种语言。
- **概括：** 将长文本 (long context)浓缩成更短的摘要。
- **对话：** 进行互动，比如聊天机器人或虚拟助手。

例如，如果你输入“泰国的首都是”，LLM会利用其学到的模式预测下一个最可能的词是“曼谷”。它会逐字继续这个过程，生成连贯且符合语境的回应。

这些模型构成了许多你可能已经在使用的人工智能工具的基础，比如高级聊天机器人、搜索引擎改进和内容创建辅助。本章以及本课程将帮助你弄明白这些模型“有多大”（就参数而言）与实际运行它们所需的计算机硬件之间的根本关联。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  描述了Transformer架构，该架构是现代大型语言模型的基础，解释了其模式匹配能力背后的机制。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  介绍了GPT-3，展示了参数和训练数据规模的增加如何使大型语言模型能够以最少的特定任务数据执行各种任务。
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu (2019)
  Journal: Journal of Machine Learning Research (JMLR); DOI: [10.48550/arXiv.1910.10683](https://doi.org/10.48550/arXiv.1910.10683)
  详细介绍了文本到文本传输Transformer (T5) 模型，对迁移学习技术进行了全面研究，并展示了如何将各种自然语言处理任务构建为文本到文本问题。
- [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) — Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Dilara Bakar, Percy Liang, et al. (2021)
  Journal: arXiv; Publisher: Stanford Institute for Human-Centered Artificial Intelligence (HAI); DOI: [10.48550/arXiv.2108.07258](https://doi.org/10.48550/arXiv.2108.07258)
  介绍了基础模型的概念，其中大型语言模型是突出类型，讨论了它们在各种应用中的共享能力和影响。

---

[下一节](02-%E4%BA%86%E8%A7%A3%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0.md)

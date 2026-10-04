# 回顾：预训练语言模型和Transformer

来源：[原文](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-1-llm-adaptation-foundations/recap-pretrained-transformers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型 (LLM) 适应方法的基础概念包括预训练 (pre-training)语言模型和Transformer架构。一个概述涵盖这些主要方面，侧重于与微调 (fine-tuning)最相关的部分。熟悉深度学习 (deep learning)、自然语言处理 (NLP) 和LLM的基础知识将会有所帮助。

### 预训练 (pre-training)的效用

现代大型语言模型（如GPT、Llama、Claude、BERT及其变体）的生命周期始于一个资源密集型的**预训练**阶段，而非针对特定任务。在此阶段，模型接触海量数据集，这些数据通常包含公共互联网文本和数字化书籍的大部分内容，可能达到数TB的数据量和数万亿个token。

预训练过程中的学习通常是**自监督的**。模型不依赖于人类为特定任务（如情感分析或翻译）生成的标签，而是从语言的内在结构中获取知识。常见的自监督目标包括：

1. **下一个Token预测**：给定文本序列，预测紧随其后的token（词或子词 (subword)）。这是GPT等自回归 (autoregressive)模型的特点。损失函数 (loss function)鼓励模型为训练数据中实际观察到的下一个token赋予高概率。
2. **掩码语言建模 (MLM)**：随机掩盖输入序列中一定比例的token，并训练模型根据周围上下文 (context)预测这些被掩盖的token。BERT等模型采用此方法。

通过这些目标，模型被要求学习语言中复杂的统计模式，包括语法、句法、词语间的语义关联 (semantic relationship)、常识推理 (inference)，甚至训练语料库中内含的一定程度的事实知识。此阶段的成果是一个**基础模型**，它具备广泛的语言理解和生成能力，但尚未专门用于任何特定下游应用。

### Transformer架构：实现规模与上下文 (context)处理

这些预训练 (pre-training)模型取得显著成果，与论文《Attention Is All You Need》中提出的**Transformer**架构紧密相关。这种架构摆脱了以往在序列建模中占主导地位的循环（RNN、LSTM）或卷积（CNN）方法。

Transformer的主要创新是**自注意力 (self-attention)机制 (attention mechanism)**。该机制使得模型在处理序列中特定位置的token时，能够动态地衡量序列中*所有*其他token（包括自身）的重要性，并从中获取信息。它为每个token计算查询 (Q)、键 (K) 和值 (V) 向量 (vector)，并根据查询和键之间的兼容性计算注意力分数。这使得模型能够比早期架构更有效地捕获长距离依赖和上下文关联。

> Transformer单块的简化视图（所示为解码器风格，编码器类似）。输入表示通过多头自注意力和逐位置前馈网络，每个子层之后应用残差连接和层归一化 (normalization)。

使Transformer适合大规模预训练的主要特性有：

- **上下文嵌入 (embedding)**：与静态词嵌入（如Word2Vec）不同，Transformer生成的嵌入取决于周围上下文。“bank”在“river bank”（河岸）和“investment bank”（投资银行）中的表示有所不同。
- **并行处理**：Transformer层内的计算，特别是自注意力机制，可以在token之间进行大量并行处理，使得在大型数据集和硬件加速器（GPU/TPU）上进行训练非常高效。
- **处理长距离依赖**：自注意力直接关联层内整个序列长度的token，克服了RNN在捕获超长距离依赖时梯度消失的问题。

多头注意力 (multi-head attention)（并行运行多次具有不同学习投影的自注意力）、位置编码 (positional encoding)（注入token顺序信息）、层归一化和逐位置前馈网络等标准构成部分在堆叠的层中共同作用，以构建深度、功能强大的模型。

这些预训练的Transformer模型代表了一项重大学习进展，它们从网络规模数据中获得了大量的通用知识。然而，这种通用性也是它们在特定应用中的局限。它们的原始输出可能不够聚焦，在特定领域可能存在事实不符，或者不符合所需的格式或风格。这正是微调 (fine-tuning)和适应方法变得不可或缺之处，它们使我们能够调整这些强大的基础模型以满足特定需求，这也是本课程的主旨内容。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  提出了Transformer架构，包括自注意力机制，这使得语言模型的大规模预训练成为可能。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2018)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); DOI: [10.48550/arXiv.1810.04805](https://doi.org/10.48550/arXiv.1810.04805)
  介绍了基于Transformer的双向编码器表示模型（BERT），并提出了用于预训练的掩码语言建模目标。
- [Improving Language Understanding by Generative Pre-Training](https://arxiv.org/pdf/1803.05380.pdf) — Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (2018)
  Journal: arXiv preprint; Pages: 1-17; DOI: [10.48550/arXiv.1803.05380](https://doi.org/10.48550/arXiv.1803.05380)
  描述了GPT-1模型以及使用下一词元预测的生成式预训练方法，是自回归语言模型的基础。
- [CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) — Diyi Yang, Tatsunori Hashimoto (2025)
  Publisher: Stanford University
  一门学术课程，提供丰富的讲义和作业，内容涵盖NLP的深度学习方法，包括Transformer和大型语言模型的详细介绍。

---

[下一节](02-%E5%BE%AE%E8%B0%83%E5%92%8C%E9%80%82%E9%85%8D%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7.md)

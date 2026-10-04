# 第 2 章：大型语言模型的工作原理（简化版）

来源：[原章节](https://apxml.com/zh/courses/intro-large-language-models/chapter-2-simplified-mechanics-of-llms)

[返回课程目录](../README.md)

在对大型语言模型有了初步了解后，本章将简要介绍其内部运作方式。我们将了解这些模型如何处理和生成文本。

你将学到：

*   如何将文本分解为称为**分词**（tokens）的单元，并通过**嵌入**（embeddings）进行数值表示。
*   作为文本生成依据的**预测下一个词**的基本理念。
*   **训练数据量**和**模型参数**（$P$）在决定大型语言模型能力方面的重要性。
*   **Transformer架构**的简要概览，它是这些模型的一种常见结构。
*   前置文本，或称**上下文**，如何影响模型的输出。

本章为大型语言模型的运作方式提供了理论基础，无需深厚的数学或编程知识。

## 小节

- 1. [词语表示：分词和嵌入](01-%E8%AF%8D%E8%AF%AD%E8%A1%A8%E7%A4%BA%EF%BC%9A%E5%88%86%E8%AF%8D%E5%92%8C%E5%B5%8C%E5%85%A5.md)
- 2. [预测下一个词：核心理念](02-%E9%A2%84%E6%B5%8B%E4%B8%8B%E4%B8%80%E4%B8%AA%E8%AF%8D%EF%BC%9A%E6%A0%B8%E5%BF%83%E7%90%86%E5%BF%B5.md)
- 3. [训练数据规模的作用](03-%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE%E8%A7%84%E6%A8%A1%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 4. [理解模型参数](04-%E7%90%86%E8%A7%A3%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0.md)
- 5. [Transformer架构（高层）简介](05-Transformer%E6%9E%B6%E6%9E%84%EF%BC%88%E9%AB%98%E5%B1%82%EF%BC%89%E7%AE%80%E4%BB%8B.md)
- 6. [语境如何影响生成](06-%E8%AF%AD%E5%A2%83%E5%A6%82%E4%BD%95%E5%BD%B1%E5%93%8D%E7%94%9F%E6%88%90.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-large-language-models/chapter-2-simplified-mechanics-of-llms/quiz)

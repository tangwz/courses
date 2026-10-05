# 第 6 章：高级架构变体与分析

来源：[原章节](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis)

[返回课程目录](../README.md)

标准Transformer架构虽然有效，却带来了计算上的难题，主要体现在自注意力机制与输入序列长度$N$相关的二次复杂度$O(N^2)$。这种复杂性限制了Transformer在处理非常长序列时的实际使用。

本章审视这些局限，并介绍了几种旨在提升效率和性能的架构改进。我们将分析普通自注意力机制的计算成本，然后研究替代方案，包括：

*   **稀疏注意力机制：** 限制注意力计算到特定模式的方法，减少查询-键比较的数量。
*   **线性注意力近似：** 例如Linformer和Performer之类的方法，它们使用低秩投影或核方法等技术来近似注意力矩阵，目标是达到线性$O(N)$复杂度。
*   **Transformer-XL：** 引入循环机制，以更有效地处理更长的上下文。
*   **相对位置编码：** 一种表示序列顺序的替代方式，基于成对距离。
*   **归一化放置位置：** 比较Pre-LN和Post-LN变体以及它们对训练动态的影响。
*   **缩放定律：** 关于模型大小、数据集大小、计算量和性能之间关系的实证观察。
*   **参数效率：** 减少模型参数数量而不明显牺牲性能的技术。

通过研究这些变体，您将了解当前旨在使Transformer模型更具可扩展性和效率、以适应多种应用场景的研究与发展。

## 小节

- 1. [自注意力机制的计算复杂度](01-%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E7%9A%84%E8%AE%A1%E7%AE%97%E5%A4%8D%E6%9D%82%E5%BA%A6.md)
- 2. [稀疏注意力机制](02-%E7%A8%80%E7%96%8F%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
- 3. [近似注意力机制：线性Transformer](03-%E8%BF%91%E4%BC%BC%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%EF%BC%9A%E7%BA%BF%E6%80%A7Transformer.md)
- 4. [基于核的注意力近似 (Performers模型)](04-%E5%9F%BA%E4%BA%8E%E6%A0%B8%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E8%BF%91%E4%BC%BC%20%28Performers%E6%A8%A1%E5%9E%8B%29.md)
- 5. [低秩投影方法（Linformer）](05-%E4%BD%8E%E7%A7%A9%E6%8A%95%E5%BD%B1%E6%96%B9%E6%B3%95%EF%BC%88Linformer%EF%BC%89.md)
- 6. [Transformer-XL：分段循环](06-Transformer-XL%EF%BC%9A%E5%88%86%E6%AE%B5%E5%BE%AA%E7%8E%AF.md)
- 7. [相对位置编码](07-%E7%9B%B8%E5%AF%B9%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)
- 8. [预归一化与后归一化 (预LN与后LN)](08-%E9%A2%84%E5%BD%92%E4%B8%80%E5%8C%96%E4%B8%8E%E5%90%8E%E5%BD%92%E4%B8%80%E5%8C%96%20%28%E9%A2%84LN%E4%B8%8E%E5%90%8ELN%29.md)
- 9. [神经网络语言模型的缩放法则](09-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E7%BC%A9%E6%94%BE%E6%B3%95%E5%88%99.md)
- 10. [参数效率与共享技术](10-%E5%8F%82%E6%95%B0%E6%95%88%E7%8E%87%E4%B8%8E%E5%85%B1%E4%BA%AB%E6%8A%80%E6%9C%AF.md)

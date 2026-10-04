# 第 13 章：位置编码的变体

来源：[原章节](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-13-positional-encoding-variations)

[返回课程目录](../README.md)

Transformer模型完全依赖于注意力机制，缺乏循环神经网络（RNN）固有的序列感知能力。因此，注入位置信息是不可或缺的。虽然前面介绍过的标准正弦和学习型绝对位置编码提供了一个基础，但它们也存在一些局限性，尤其是在推广至更长序列以及明确表示token间相对距离方面。

本章将探讨旨在更有效或高效地编码位置信息的其他方法。我们将学习：

*   不再局限于绝对位置编码的理由。
*   相对位置编码背后的原理，即编码的是位置*之间*的关系。
*   具体的实现方法，例如将相对位置偏差直接添加到注意力分数中（Shaw等人的工作）。
*   Transformer-XL中采用的相对编码方案。
*   旋转位置编码（RoPE），这是一种根据位置对查询和键向量进行旋转的处理方法。

学完本章，您将掌握这些更高级的位置编码技术的运作原理，并了解它们在哪些场景下比标准绝对编码更具优势。

## 小节

- 1. [绝对位置编码的局限性](01-%E7%BB%9D%E5%AF%B9%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 2. [相对位置编码的原理](02-%E7%9B%B8%E5%AF%B9%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E7%9A%84%E5%8E%9F%E7%90%86.md)
- 3. [Shaw 等人的相对位置实现](03-Shaw%20%E7%AD%89%E4%BA%BA%E7%9A%84%E7%9B%B8%E5%AF%B9%E4%BD%8D%E7%BD%AE%E5%AE%9E%E7%8E%B0.md)
- 4. [Transformer-XL 相对位置编码](04-Transformer-XL%20%E7%9B%B8%E5%AF%B9%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)
- 5. [旋转位置编码 (RoPE)](05-%E6%97%8B%E8%BD%AC%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%20%28RoPE%29.md)

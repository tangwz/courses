---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-13-positional-encoding-variations"
sourceId: 1091
chapter: "positional-encoding-variations"
title: "位置编码的变体"
order: 13
description: "了解对标准正弦位置编码的替代方案和改进，其中包括相对和旋转嵌入。"
hasQuiz: false
---

Transformer模型完全依赖于注意力机制，缺乏循环神经网络（RNN）固有的序列感知能力。因此，注入位置信息是不可或缺的。虽然前面介绍过的标准正弦和学习型绝对位置编码提供了一个基础，但它们也存在一些局限性，尤其是在推广至更长序列以及明确表示token间相对距离方面。

本章将探讨旨在更有效或高效地编码位置信息的其他方法。我们将学习：

*   不再局限于绝对位置编码的理由。
*   相对位置编码背后的原理，即编码的是位置*之间*的关系。
*   具体的实现方法，例如将相对位置偏差直接添加到注意力分数中（Shaw等人的工作）。
*   Transformer-XL中采用的相对编码方案。
*   旋转位置编码（RoPE），这是一种根据位置对查询和键向量进行旋转的处理方法。

学完本章，您将掌握这些更高级的位置编码技术的运作原理，并了解它们在哪些场景下比标准绝对编码更具优势。

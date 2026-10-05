# 第 10 章：从零开始实现Transformer

来源：[原章节](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-10-implementing-transformer-from-scratch)

[返回课程目录](../README.md)

在前面章节中，我们已经了解了Transformer架构的理论原理，本章将重点把这些理论转化为可运行的代码。我们将使用一个常用的深度学习框架，一步步地构建Transformer模型的重要组成部分。

你将学习实现：
*   缩放点积注意力，这种基本的注意力机制。
*   多头注意力，它将多种注意力视角结合起来。
*   逐位置前馈网络。
*   编码器和解码器层的结构，包括残差连接和层归一化。
*   通过组装这些部分来构建完整的Transformer模型。

本章结束后，你将拥有一个清晰、可操作的Transformer实现，这将使你对这些模型在代码层面如何运作有清晰的认识，并为你后续关于扩展和优化的章节做好准备。我们将建立一个基本的项目配置，并按逻辑顺序逐步实现每个架构元素。

## 小节

- 1. [设置项目环境](01-%E8%AE%BE%E7%BD%AE%E9%A1%B9%E7%9B%AE%E7%8E%AF%E5%A2%83.md)
- 2. [实现缩放点积注意力](02-%E5%AE%9E%E7%8E%B0%E7%BC%A9%E6%94%BE%E7%82%B9%E7%A7%AF%E6%B3%A8%E6%84%8F%E5%8A%9B.md)
- 3. [构建多头注意力层](03-%E6%9E%84%E5%BB%BA%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%B1%82.md)
- 4. [实现位置感知前馈网络](04-%E5%AE%9E%E7%8E%B0%E4%BD%8D%E7%BD%AE%E6%84%9F%E7%9F%A5%E5%89%8D%E9%A6%88%E7%BD%91%E7%BB%9C.md)
- 5. [构建编码器层和解码器层](05-%E6%9E%84%E5%BB%BA%E7%BC%96%E7%A0%81%E5%99%A8%E5%B1%82%E5%92%8C%E8%A7%A3%E7%A0%81%E5%99%A8%E5%B1%82.md)
- 6. [组装完整的Transformer模型](06-%E7%BB%84%E8%A3%85%E5%AE%8C%E6%95%B4%E7%9A%84Transformer%E6%A8%A1%E5%9E%8B.md)

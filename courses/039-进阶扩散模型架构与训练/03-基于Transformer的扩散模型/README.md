# 第 3 章：基于Transformer的扩散模型

来源：[原章节](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models)

[返回课程目录](../README.md)

尽管基于卷积神经网络（CNN）的U-Net架构已成为许多成功扩散模型的标准骨干，但Transformer架构（最初在自然语言处理中占据主导地位）已在包括图像合成在内的生成任务中显示出良好的应用前景。Transformer擅长建模长距离依赖，与CNN固有的局部性偏置相比，它提供了一种捕获数据中关联的不同方法。

本章将研究如何在扩散模型框架内有效应用Transformer架构。我们将涵盖：

*   使用Transformer进行生成建模的动机，以及如何借鉴如Vision Transformer (ViT)等思路，使用补丁嵌入等技术使其适应图像数据。
*   Diffusion Transformers (DiTs) 的具体架构，其用Transformer块取代了U-Net骨干。
*   将类别标签或文本嵌入等条件信息融入DiT模型的方法。
*   对U-Net和Transformer骨干在性能、可扩展性和计算需求方面的权衡比较。
*   实现和训练基于Transformer的扩散模型的实际考量。

本章结束时，您将理解基于Transformer的扩散模型的结构和功能，并能够分析和实现它们。

## 小节

- 1. [生成式建模中采用Transformer模型的缘由](01-%E7%94%9F%E6%88%90%E5%BC%8F%E5%BB%BA%E6%A8%A1%E4%B8%AD%E9%87%87%E7%94%A8Transformer%E6%A8%A1%E5%9E%8B%E7%9A%84%E7%BC%98%E7%94%B1.md)
- 2. [使Transformer适应图像数据 (ViT, 图像块嵌入)](02-%E4%BD%BFTransformer%E9%80%82%E5%BA%94%E5%9B%BE%E5%83%8F%E6%95%B0%E6%8D%AE%20%28ViT%2C%20%E5%9B%BE%E5%83%8F%E5%9D%97%E5%B5%8C%E5%85%A5%29.md)
- 3. [扩散变换器 (DiT)：架构概述](03-%E6%89%A9%E6%95%A3%E5%8F%98%E6%8D%A2%E5%99%A8%20%28DiT%29%EF%BC%9A%E6%9E%B6%E6%9E%84%E6%A6%82%E8%BF%B0.md)
- 4. [扩散Transformer模型中的条件作用](04-%E6%89%A9%E6%95%A3Transformer%E6%A8%A1%E5%9E%8B%E4%B8%AD%E7%9A%84%E6%9D%A1%E4%BB%B6%E4%BD%9C%E7%94%A8.md)
- 5. [U-Net与Transformer在扩散模型中的比较](05-U-Net%E4%B8%8ETransformer%E5%9C%A8%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%AD%E7%9A%84%E6%AF%94%E8%BE%83.md)
- 6. [DiT 的实现考量](06-DiT%20%E7%9A%84%E5%AE%9E%E7%8E%B0%E8%80%83%E9%87%8F.md)
- 7. [动手实践：构建一个简单的 DiT 模块](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%20DiT%20%E6%A8%A1%E5%9D%97.md)

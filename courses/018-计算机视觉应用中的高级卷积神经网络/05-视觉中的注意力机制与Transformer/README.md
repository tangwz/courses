# 第 5 章：视觉中的注意力机制与Transformer

来源：[原章节](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-5-attention-transformers-vision)

[返回课程目录](../README.md)

卷积神经网络通过空间受限的感受野在学习局部模式方面表现出色，但要掌握图像中的整体信息和长距离依赖，则需要不同的方法。本章介绍注意力机制和Transformer架构，作为提升视觉模型捕获这些更广泛关联的方法。

您将学习自注意力机制如何与CNN框架结合，以使网络能够有选择地关注更具信息量的特征。我们将介绍具体例子，例如Squeeze-and-Excitation (SE) 模块和非局部网络。接下来，我们研究视觉Transformer (ViT)，这是一种通过处理图像块序列的方式，将成功的Transformer架构直接应用于图像数据的模型。我们将学习ViT的主要构成部分，包括图像块嵌入和多头自注意力层。最后，我们将讨论结合了卷积和Transformer元素的混合模型，并比较CNN和ViT的运行特点和数据需求。

## 小节

- 1. [CNN中的自注意力机制](01-CNN%E4%B8%AD%E7%9A%84%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
- 2. [非局部神经网络](02-%E9%9D%9E%E5%B1%80%E9%83%A8%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
- 3. [视觉Transformer简介](03-%E8%A7%86%E8%A7%89Transformer%E7%AE%80%E4%BB%8B.md)
- 4. [ViT 架构：图像块、嵌入和 Transformer 编码器](04-ViT%20%E6%9E%B6%E6%9E%84%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%9D%97%E3%80%81%E5%B5%8C%E5%85%A5%E5%92%8C%20Transformer%20%E7%BC%96%E7%A0%81%E5%99%A8.md)
- 5. [混合CNN-Transformer模型](05-%E6%B7%B7%E5%90%88CNN-Transformer%E6%A8%A1%E5%9E%8B.md)
- 6. [CNN与Transformer在视觉任务中的比较](06-CNN%E4%B8%8ETransformer%E5%9C%A8%E8%A7%86%E8%A7%89%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84%E6%AF%94%E8%BE%83.md)
- 7. [在CNN中实现注意力模块的实践](07-%E5%9C%A8CNN%E4%B8%AD%E5%AE%9E%E7%8E%B0%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%A8%A1%E5%9D%97%E7%9A%84%E5%AE%9E%E8%B7%B5.md)

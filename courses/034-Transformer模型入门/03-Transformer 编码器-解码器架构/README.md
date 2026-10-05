# 第 3 章：Transformer 编码器-解码器架构

来源：[原章节](https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-3-transformer-encoder-decoder-architecture)

[返回课程目录](../README.md)

在上一章了解了核心注意力机制后，我们将这些部件组合起来，以构建完整的 Transformer 架构。本章将详细介绍模型的结构，解释编码器和解码器堆栈如何在序列到序列任务中配合工作。

您将了解以下内容：

*   整体编码器-解码器布局。
*   输入处理，包括词元嵌入以及使用正弦和余弦函数等位置编码技术加入位置信息。
*   编码器层的结构，包含多头自注意力模块和逐位置前馈网络。
*   解码器层的结构，包括遮蔽多头自注意力、编码器-解码器注意力和前馈网络。
*   残差连接和层归一化（`Add & Norm`）在稳定网络和改善梯度流动中的作用。
*   解码器输出如何通过线性层和 Softmax 函数转换为最终词元概率。
*   一个实践实现部分，重点在于构建单个编码器层。

在本章结束时，您将理解这些不同部分如何组合形成完整的 Transformer 模型，以及每个组件设计背后的原理。

## 小节

- 1. [整体架构概览](01-%E6%95%B4%E4%BD%93%E6%9E%B6%E6%9E%84%E6%A6%82%E8%A7%88.md)
- 2. [输入嵌入层](02-%E8%BE%93%E5%85%A5%E5%B5%8C%E5%85%A5%E5%B1%82.md)
- 3. [位置信息的必要性](03-%E4%BD%8D%E7%BD%AE%E4%BF%A1%E6%81%AF%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7.md)
- 4. [位置编码说明](04-%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E8%AF%B4%E6%98%8E.md)
- 5. [编码器层堆叠](05-%E7%BC%96%E7%A0%81%E5%99%A8%E5%B1%82%E5%A0%86%E5%8F%A0.md)
- 6. [加法与归一化层 (残差连接)](06-%E5%8A%A0%E6%B3%95%E4%B8%8E%E5%BD%92%E4%B8%80%E5%8C%96%E5%B1%82%20%28%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5%29.md)
- 7. [逐位置前馈网络](07-%E9%80%90%E4%BD%8D%E7%BD%AE%E5%89%8D%E9%A6%88%E7%BD%91%E7%BB%9C.md)
- 8. [解码器堆栈](08-%E8%A7%A3%E7%A0%81%E5%99%A8%E5%A0%86%E6%A0%88.md)
- 9. [带掩码的多头自注意力](09-%E5%B8%A6%E6%8E%A9%E7%A0%81%E7%9A%84%E5%A4%9A%E5%A4%B4%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md)
- 10. [编码器-解码器注意力机制](10-%E7%BC%96%E7%A0%81%E5%99%A8-%E8%A7%A3%E7%A0%81%E5%99%A8%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
- 11. [最终线性层和Softmax](11-%E6%9C%80%E7%BB%88%E7%BA%BF%E6%80%A7%E5%B1%82%E5%92%8CSoftmax.md)
- 12. [动手实践：构建编码器层](12-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E7%BC%96%E7%A0%81%E5%99%A8%E5%B1%82.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-3-transformer-encoder-decoder-architecture/quiz)

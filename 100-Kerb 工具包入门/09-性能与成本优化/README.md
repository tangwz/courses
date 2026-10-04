# 第 9 章：性能与成本优化

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-9-performance-and-cost-optimization)

[返回课程目录](../README.md)

在构建常用的大语言模型（LLM）应用时，会遇到两个实际问题：延迟和运行成本。每一次调用语言模型或嵌入服务都需要时间处理，并产生费用，这些费用通常根据token数量计算。单次API调用的总成本$C$可以表示为：

$$C = (P_{prompt} \times N_{prompt}) + (P_{completion} \times N_{completion})$$

其中$P$是每个token的价格，$N$是prompt和completion的token数量。对于处理重复查询的应用，这些成本会迅速增加。

本章介绍缓存作为解决这些问题的一种主要方法。我们将首先确定大语言模型驱动系统中的常见性能瓶颈。接着，您将学习如何使用`cache`模块实现两种具体的缓存策略：

*   **大语言模型响应缓存：** 存储并复用完整的模型生成内容，用于相同的prompt。
*   **嵌入缓存：** 避免对相同文本片段重复调用API来生成向量嵌入。

最后，我们将介绍缓存失效的实际操作，以确保应用数据保持最新。通过学习这些内容，您将学会如何使您的应用更快、更经济。

## 小节

- 1. [定位性能瓶颈](01-%E5%AE%9A%E4%BD%8D%E6%80%A7%E8%83%BD%E7%93%B6%E9%A2%88.md)
- 2. [实现LLM响应缓存](02-%E5%AE%9E%E7%8E%B0LLM%E5%93%8D%E5%BA%94%E7%BC%93%E5%AD%98.md)
- 3. [缓存嵌入以减少API调用](03-%E7%BC%93%E5%AD%98%E5%B5%8C%E5%85%A5%E4%BB%A5%E5%87%8F%E5%B0%91API%E8%B0%83%E7%94%A8.md)
- 4. [缓存失效策略](04-%E7%BC%93%E5%AD%98%E5%A4%B1%E6%95%88%E7%AD%96%E7%95%A5.md)

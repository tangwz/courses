# 第 8 章：建立和管理大规模数据集

来源：[原章节](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-8-building-managing-large-scale-datasets)

[返回课程目录](../README.md)

在获取并预处理了潜在的数太字节乃至拍字节文本数据后，下一个工程步骤是有效地存储、组织和访问这些海量数据集合。本章介绍管理数据集所需的基础设施和技术，以适应大型语言模型训练的规模需求。

我们将讨论实际的考量，例如：
*   选择合适的数据存储格式（如文本、Apache Arrow 或 Parquet）。
*   使用分布式文件系统，例如 HDFS 或云对象存储。
*   实施数据索引以实现高效检索。
*   建立数据集版本控制实践以保证可复现性。
*   设计流式数据加载器，将数据有效地输入到分布式训练流程中。

## 小节

- 1. [数据存储格式（文本、Arrow、Parquet）](01-%E6%95%B0%E6%8D%AE%E5%AD%98%E5%82%A8%E6%A0%BC%E5%BC%8F%EF%BC%88%E6%96%87%E6%9C%AC%E3%80%81Arrow%E3%80%81Parquet%EF%BC%89.md)
- 2. [分布式文件系统 (HDFS, S3)](02-%E5%88%86%E5%B8%83%E5%BC%8F%E6%96%87%E4%BB%B6%E7%B3%BB%E7%BB%9F%20%28HDFS%2C%20S3%29.md)
- 3. [数据索引用于高效检索](03-%E6%95%B0%E6%8D%AE%E7%B4%A2%E5%BC%95%E7%94%A8%E4%BA%8E%E9%AB%98%E6%95%88%E6%A3%80%E7%B4%A2.md)
- 4. [数据集版本管理与复现性](04-%E6%95%B0%E6%8D%AE%E9%9B%86%E7%89%88%E6%9C%AC%E7%AE%A1%E7%90%86%E4%B8%8E%E5%A4%8D%E7%8E%B0%E6%80%A7.md)
- 5. [用于训练的流式数据加载器](05-%E7%94%A8%E4%BA%8E%E8%AE%AD%E7%BB%83%E7%9A%84%E6%B5%81%E5%BC%8F%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E5%99%A8.md)

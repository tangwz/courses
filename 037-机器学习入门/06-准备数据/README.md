# 第 6 章：准备数据

来源：[原章节](https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-6-preparing-your-data)

[返回课程目录](../README.md)

机器学习模型高度依赖于其训练数据的质量。通常，您会遇到的数据集在被算法有效使用之前，需要大量的准备工作。实际数据通常是不完整、不一致的，或者格式不适合处理。

本章介绍数据预处理的基本技术。您将学习以下实用方法：

*   识别并处理数据集中缺失的值。
*   应用特征缩放，例如归一化 (Normalization)
    $$
    x' = \frac{x - \min(x)}{\max(x) - \min(x)}
    $$
    和标准化 (Standardization)
    $$
    x' = \frac{x - \mu}{\sigma}
    $$
    以将数值特征调整到统一的范围。
*   将分类（非数值）特征编码为模型可理解的数值表示，使用独热编码 (One-Hot Encoding) 等方法。
*   实现将数据划分为训练集和测试集。

学完本章后，您将明白为什么这些步骤是必需的，以及如何执行基本的数据清洗和转换任务，为机器学习模型准备数据。

## 小节

- 1. [数据预处理的重要性](01-%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86%E7%9A%84%E9%87%8D%E8%A6%81%E6%80%A7.md)
- 2. [处理缺失值](02-%E5%A4%84%E7%90%86%E7%BC%BA%E5%A4%B1%E5%80%BC.md)
- 3. [特征缩放介绍](03-%E7%89%B9%E5%BE%81%E7%BC%A9%E6%94%BE%E4%BB%8B%E7%BB%8D.md)
- 4. [编码分类特征](04-%E7%BC%96%E7%A0%81%E5%88%86%E7%B1%BB%E7%89%B9%E5%BE%81.md)
- 5. [再次谈谈：将数据划分为训练集和测试集](05-%E5%86%8D%E6%AC%A1%E8%B0%88%E8%B0%88%EF%BC%9A%E5%B0%86%E6%95%B0%E6%8D%AE%E5%88%92%E5%88%86%E4%B8%BA%E8%AE%AD%E7%BB%83%E9%9B%86%E5%92%8C%E6%B5%8B%E8%AF%95%E9%9B%86.md)
- 6. [动手实践：基本数据清洗步骤](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%9F%BA%E6%9C%AC%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97%E6%AD%A5%E9%AA%A4.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-6-preparing-your-data/quiz)

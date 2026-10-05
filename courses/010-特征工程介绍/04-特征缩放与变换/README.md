# 第 4 章：特征缩放与变换

来源：[原章节](https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation)

[返回课程目录](../README.md)

既然我们有了处理类别数据的方法，我们接下来看数值特征。原始数值特征的范围和分布会直接影响对特征尺度敏感的算法的有效性，例如基于距离的方法或使用梯度下降的方法。本章主要介绍通过调整数值数据的尺度和分布来准备数据进行模型训练的技术。

我们将介绍常见的缩放方法，包括标准化（$Z$-score scaling），它使数据具有零均值和单位方差（$Z = \frac{x - \mu}{\sigma}$），以及归一化（Min-Max scaling），它将数值限制在 [0, 1] 等特定区间。我们也将介绍适用于含有异常值数据的鲁棒缩放（Robust Scaling）。此外，你还将学习对数、Box-Cox 和 Yeo-Johnson 等变换方法，用来改变偏斜分布，并使数据更适合某些模型假设。本章将指导你如何有效地选择和应用这些技术，使用 Scikit-learn。

## 小节

- 1. [特征缩放的必要性](01-%E7%89%B9%E5%BE%81%E7%BC%A9%E6%94%BE%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7.md)
- 2. [标准化 (Z-score 缩放)](02-%E6%A0%87%E5%87%86%E5%8C%96%20%28Z-score%20%E7%BC%A9%E6%94%BE%29.md)
- 3. [归一化（最小-最大值缩放）](03-%E5%BD%92%E4%B8%80%E5%8C%96%EF%BC%88%E6%9C%80%E5%B0%8F-%E6%9C%80%E5%A4%A7%E5%80%BC%E7%BC%A9%E6%94%BE%EF%BC%89.md)
- 4. [处理异常值的缩放](04-%E5%A4%84%E7%90%86%E5%BC%82%E5%B8%B8%E5%80%BC%E7%9A%84%E7%BC%A9%E6%94%BE.md)
- 5. [对偏斜数据的对数变换](05-%E5%AF%B9%E5%81%8F%E6%96%9C%E6%95%B0%E6%8D%AE%E7%9A%84%E5%AF%B9%E6%95%B0%E5%8F%98%E6%8D%A2.md)
- 6. [Box-Cox 变换](06-Box-Cox%20%E5%8F%98%E6%8D%A2.md)
- 7. [Yeo-Johnson 变换](07-Yeo-Johnson%20%E5%8F%98%E6%8D%A2.md)
- 8. [分位数变换](08-%E5%88%86%E4%BD%8D%E6%95%B0%E5%8F%98%E6%8D%A2.md)
- 9. [选择合适的缩放/转换方法](09-%E9%80%89%E6%8B%A9%E5%90%88%E9%80%82%E7%9A%84%E7%BC%A9%E6%94%BE-%E8%BD%AC%E6%8D%A2%E6%96%B9%E6%B3%95.md)
- 10. [动手实践：特征缩放与转换](10-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%89%B9%E5%BE%81%E7%BC%A9%E6%94%BE%E4%B8%8E%E8%BD%AC%E6%8D%A2.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation/quiz)

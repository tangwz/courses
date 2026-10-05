# 第 5 章：模型评估的假设检验

来源：[原章节](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-5-hypothesis-testing-model-evaluation)

[返回课程目录](../README.md)

在处理数据时，尤其是在机器学习中，我们经常需要根据样本做出判断或验证假定。新模型是否明显优于旧模型？某个特定特征是否有可测量的影响？假设检验提供了一个有结构的统计体系来回答此类问题。

本章涵盖了假设检验的基本知识。您将学习如何：
*   建立零假设 ($H_0$) 和备择假设 ($H_1$)。
*   理解第一类 ($\alpha$) 和第二类 ($\beta$) 错误之间的权衡。
*   解读 p 值以做出统计判断。
*   应用常用检验方法，包括用于比较均值的 t 检验和用于分类数据分析的卡方检验。
*   对方差分析 (ANOVA) 有一个概览，用于比较多个组。

我们还将演示如何使用 Python 的 SciPy 库高效地实现这些检验，提供实用的工具用于模型评估和数据分析。

## 小节

- 1. [制定零假设与备择假设](01-%E5%88%B6%E5%AE%9A%E9%9B%B6%E5%81%87%E8%AE%BE%E4%B8%8E%E5%A4%87%E6%8B%A9%E5%81%87%E8%AE%BE.md)
- 2. [理解第一类和第二类错误](02-%E7%90%86%E8%A7%A3%E7%AC%AC%E4%B8%80%E7%B1%BB%E5%92%8C%E7%AC%AC%E4%BA%8C%E7%B1%BB%E9%94%99%E8%AF%AF.md)
- 3. [P值说明](03-P%E5%80%BC%E8%AF%B4%E6%98%8E.md)
- 4. [T检验简介](04-T%E6%A3%80%E9%AA%8C%E7%AE%80%E4%BB%8B.md)
- 5. [卡方检验介绍](05-%E5%8D%A1%E6%96%B9%E6%A3%80%E9%AA%8C%E4%BB%8B%E7%BB%8D.md)
- 6. [方差分析 (ANOVA) 概述](06-%E6%96%B9%E5%B7%AE%E5%88%86%E6%9E%90%20%28ANOVA%29%20%E6%A6%82%E8%BF%B0.md)
- 7. [使用 Python 进行假设检验](07-%E4%BD%BF%E7%94%A8%20Python%20%E8%BF%9B%E8%A1%8C%E5%81%87%E8%AE%BE%E6%A3%80%E9%AA%8C.md)
- 8. [实践：将T检验应用于样本数据](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%B0%86T%E6%A3%80%E9%AA%8C%E5%BA%94%E7%94%A8%E4%BA%8E%E6%A0%B7%E6%9C%AC%E6%95%B0%E6%8D%AE.md)

章节测验：[在线测验](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-5-hypothesis-testing-model-evaluation/quiz)

# 第 5 章：统计推断入门

来源：[原章节](https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-5-intro-statistical-inference)

[返回课程目录](../README.md)

迄今为止，我们一直专注于使用描述性统计来描述已有的数据，并通过概率了解随机性规律。然而，通常情况下，我们在机器学习和数据分析中的目标不仅仅是描述所收集的数据；更重要的是，要基于有限的数据（样本）对更大的群体（总体）做出明智的推测或判断。从样本中得出关于总体的结论的这一过程被称为统计推断。

本章将讲解统计推断的基本原理。你将学习到：

*   从样本数据对总体特征（例如平均值）进行估计的方法，也就是**点估计**。
*   如何使用**置信区间**量化这些估计值的不确定性。
*   **假设检验**的原理：这是一种对总体相关论断做出判断的正式方式，包括建立**原假设与备择假设**以及解释**p值**。
*   这些推断方法如何与评估机器学习结果的重要性相关联。

我们将以前几章的描述性统计和概率知识为基础，以理解如何超出当前数据进行归纳。

## 小节

- 1. [从数据中得出结论](01-%E4%BB%8E%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BE%97%E5%87%BA%E7%BB%93%E8%AE%BA.md)
- 2. [点估计](02-%E7%82%B9%E4%BC%B0%E8%AE%A1.md)
- 3. [区间估计：置信区间](03-%E5%8C%BA%E9%97%B4%E4%BC%B0%E8%AE%A1%EF%BC%9A%E7%BD%AE%E4%BF%A1%E5%8C%BA%E9%97%B4.md)
- 4. [假设检验：基本思路](04-%E5%81%87%E8%AE%BE%E6%A3%80%E9%AA%8C%EF%BC%9A%E5%9F%BA%E6%9C%AC%E6%80%9D%E8%B7%AF.md)
- 5. [零假设与备择假设](05-%E9%9B%B6%E5%81%87%E8%AE%BE%E4%B8%8E%E5%A4%87%E6%8B%A9%E5%81%87%E8%AE%BE.md)
- 6. [理解P值](06-%E7%90%86%E8%A7%A3P%E5%80%BC.md)
- 7. [统计推断与机器学习评估的联系](07-%E7%BB%9F%E8%AE%A1%E6%8E%A8%E6%96%AD%E4%B8%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AF%84%E4%BC%B0%E7%9A%84%E8%81%94%E7%B3%BB.md)
- 8. [练习：解释统计结果](08-%E7%BB%83%E4%B9%A0%EF%BC%9A%E8%A7%A3%E9%87%8A%E7%BB%9F%E8%AE%A1%E7%BB%93%E6%9E%9C.md)

章节测验：[在线测验](https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-5-intro-statistical-inference/quiz)

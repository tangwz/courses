# 第 3 章：监督学习：回归

来源：[原章节](https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-3-supervised-learning-regression)

[返回课程目录](../README.md)

在掌握了机器学习的基础和核心思想后，我们现在转向一个特定类别：监督学习。本章侧重于回归任务，其目标是预测连续的数值输出。例如，根据大小和位置等特征预测房价，或根据历史数据估计温度。

你将学到：
*   如何定义和识别回归问题。
*   线性回归的基本原理，它是一种常用于建模变量间线性关系的算法。
*   成本函数的思想，特别是它们如何用于衡量回归模型的误差或不准确性。
*   梯度下降的概述，这是一种用于最小化成本函数并为模型找到最佳参数的优化算法。
*   如何解释简单线性回归模型的结果。

我们将通过一个实际例子来阐明这些思想，指导您完成应用简单线性回归的步骤。

## 小节

- 1. [了解回归问题](01-%E4%BA%86%E8%A7%A3%E5%9B%9E%E5%BD%92%E9%97%AE%E9%A2%98.md)
- 2. [线性回归介绍](02-%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E4%BB%8B%E7%BB%8D.md)
- 3. [线性回归如何学习](03-%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E5%A6%82%E4%BD%95%E5%AD%A6%E4%B9%A0.md)
- 4. [成本函数：衡量误差](04-%E6%88%90%E6%9C%AC%E5%87%BD%E6%95%B0%EF%BC%9A%E8%A1%A1%E9%87%8F%E8%AF%AF%E5%B7%AE.md)
- 5. [梯度下降：寻找最佳拟合](05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%EF%BC%9A%E5%AF%BB%E6%89%BE%E6%9C%80%E4%BD%B3%E6%8B%9F%E5%90%88.md)
- 6. [简单线性回归示例](06-%E7%AE%80%E5%8D%95%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E7%A4%BA%E4%BE%8B.md)
- 7. [实践：实现简单线性回归](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E7%AE%80%E5%8D%95%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-3-supervised-learning-regression/quiz)

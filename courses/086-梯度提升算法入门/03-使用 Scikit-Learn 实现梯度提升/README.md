# 第 3 章：使用 Scikit-Learn 实现梯度提升

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-3-gradient-boosting-with-scikit-learn)

[返回课程目录](../README.md)

在掌握了梯度提升机 (GBM) 的工作原理后，我们现在转向其实际应用。本章主要介绍如何使用 Scikit-Learn 实现这些模型。Scikit-Learn 是 Python 机器学习技术栈中的一个标准库。其统一的 API 为您构建第一个梯度提升模型提供了直接途径。

您将使用 Scikit-Learn 的两种主要实现：用于分类任务的 `GradientBoostingClassifier` 和用于回归问题的 `GradientBoostingRegressor`。我们将介绍将这些模型拟合到数据、生成预测结果以及理解决定模型行为的主要参数的标准流程，例如提升阶段的数量 ($n\_estimators$) 和学习率。

构建模型只是过程的一部分；模型解释也同等重要。我们还将说明如何理解训练好的模型。您将学习提取特征重要性分数，以识别哪些变量对预测结果影响最大，并使用偏依赖图来呈现单个特征与模型输出之间的关系。

## 小节

- 1. [Scikit-Learn的GradientBoostingClassifier](01-Scikit-Learn%E7%9A%84GradientBoostingClassifier.md)
- 2. [Scikit-Learn 的 GradientBoostingRegressor](02-Scikit-Learn%20%E7%9A%84%20GradientBoostingRegressor.md)
- 3. [梯度提升模型的拟合与预测](03-%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8B%9F%E5%90%88%E4%B8%8E%E9%A2%84%E6%B5%8B.md)
- 4. [解读模型参数](04-%E8%A7%A3%E8%AF%BB%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0.md)
- 5. [GBM中的特征贡献度](05-GBM%E4%B8%AD%E7%9A%84%E7%89%B9%E5%BE%81%E8%B4%A1%E7%8C%AE%E5%BA%A6.md)
- 6. [偏依赖图用于模型解释](06-%E5%81%8F%E4%BE%9D%E8%B5%96%E5%9B%BE%E7%94%A8%E4%BA%8E%E6%A8%A1%E5%9E%8B%E8%A7%A3%E9%87%8A.md)
- 7. [实战演练：构建预测模型](07-%E5%AE%9E%E6%88%98%E6%BC%94%E7%BB%83%EF%BC%9A%E6%9E%84%E5%BB%BA%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B.md)

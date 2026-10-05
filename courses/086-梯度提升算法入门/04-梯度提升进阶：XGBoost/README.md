# 第 4 章：梯度提升进阶：XGBoost

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-4-advanced-gradient-boosting-xgboost)

[返回课程目录](../README.md)

在 Scikit-Learn 中实践了梯度提升机之后，我们现在转向 XGBoost，它是“极端梯度提升”的缩写。该库是一个优化过的、分布式的梯度提升框架，旨在实现高效率和准确性。它在机器学习竞赛中持续取得优异表现，并在生产环境中被广泛应用，使其成为任何实践者不可或缺的工具。

本章主要介绍有助于 XGBoost 性能的特有属性。我们将考察它在架构方面对标准 GBM 的改进，其中包含更规范化的正则化方法。XGBoost 的目标函数明确加入了对模型复杂度的惩罚，通常表示为：

$$Obj(\Theta) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$

其中，$l$ 代表损失函数，$\Omega$ 是惩罚树 $f_k$ 复杂程度的正则化项。我们还将讲解其内置的处理机制，用于处理缺失值，这可以简化数据预处理过程。本章最后通过实践介绍 XGBoost Python API，包括安装、特有的 `DMatrix` 数据结构、模型训练和预测。

## 小节

- 1. [为何选择XGBoost？速度与性能](01-%E4%B8%BA%E4%BD%95%E9%80%89%E6%8B%A9XGBoost%EF%BC%9F%E9%80%9F%E5%BA%A6%E4%B8%8E%E6%80%A7%E8%83%BD.md)
- 2. [相较于标准GBM的架构改进](02-%E7%9B%B8%E8%BE%83%E4%BA%8E%E6%A0%87%E5%87%86GBM%E7%9A%84%E6%9E%B6%E6%9E%84%E6%94%B9%E8%BF%9B.md)
- 3. [XGBoost 中的正则化 (L1 和 L2)](03-XGBoost%20%E4%B8%AD%E7%9A%84%E6%AD%A3%E5%88%99%E5%8C%96%20%28L1%20%E5%92%8C%20L2%29.md)
- 4. [自动处理缺失值](04-%E8%87%AA%E5%8A%A8%E5%A4%84%E7%90%86%E7%BC%BA%E5%A4%B1%E5%80%BC.md)
- 5. [安装与设置XGBoost](05-%E5%AE%89%E8%A3%85%E4%B8%8E%E8%AE%BE%E7%BD%AEXGBoost.md)
- 6. [XGBoost API：使用指南](06-XGBoost%20API%EF%BC%9A%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)
- 7. [动手实践：训练XGBoost模型](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83XGBoost%E6%A8%A1%E5%9E%8B.md)

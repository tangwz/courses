# 第 6 章：CatBoost：梯度提升决策树

来源：[原章节](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-6-catboost-gradient-boosting)

[返回课程目录](../README.md)

在考察了XGBoost和LightGBM之后，我们现在转向CatBoost。这是一个梯度提升库，专门为解决一个特定但普遍的难题而优化：即如何有效处理类别特征。传统的做法通常涉及预处理步骤，这些步骤可能不够理想，或导致诸如目标泄漏之类的问题。CatBoost将其处理类别数据的创新方案直接整合到算法中。

本章内容包括：

*   传统提升方法在处理类别变量时遇到的困难。
*   有序目标统计（Ordered TS）：CatBoost在编码类别特征时，如何最大限度地减少目标泄漏的方法。
*   有序提升（Ordered Boosting）：一种在训练过程中抵消预测漂移的技术。
*   自动特征组合：CatBoost如何生成类别特征之间的关联。
*   无偏树（Oblivious Trees）：CatBoost采用的对称决策树。
*   CatBoost API的使用：主要参数和具体的实现细节。

完成本章后，你将理解CatBoost的独特方法，并能够应用它们，尤其是在处理包含大量类别数据的问题时。

## 小节

- 1. [动机：分类数据面临的难题](01-%E5%8A%A8%E6%9C%BA%EF%BC%9A%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9D%A2%E4%B8%B4%E7%9A%84%E9%9A%BE%E9%A2%98.md)
- 2. [有序目标统计 (Ordered TS)](02-%E6%9C%89%E5%BA%8F%E7%9B%AE%E6%A0%87%E7%BB%9F%E8%AE%A1%20%28Ordered%20TS%29.md)
- 3. [处理预测偏差：有序提升](03-%E5%A4%84%E7%90%86%E9%A2%84%E6%B5%8B%E5%81%8F%E5%B7%AE%EF%BC%9A%E6%9C%89%E5%BA%8F%E6%8F%90%E5%8D%87.md)
- 4. [处理特征组合](04-%E5%A4%84%E7%90%86%E7%89%B9%E5%BE%81%E7%BB%84%E5%90%88.md)
- 5. [遗忘树](05-%E9%81%97%E5%BF%98%E6%A0%91.md)
- 6. [GPU训练加速](06-GPU%E8%AE%AD%E7%BB%83%E5%8A%A0%E9%80%9F.md)
- 7. [CatBoost API: 参数与配置](07-CatBoost%20API-%20%E5%8F%82%E6%95%B0%E4%B8%8E%E9%85%8D%E7%BD%AE.md)
- 8. [动手实践：实现 CatBoost](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%20CatBoost.md)

# 第 2 章：局部可解释模型无关解释 (LIME)

来源：[原章节](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-2-lime-local-interpretability)

[返回课程目录](../README.md)

了解机器学习模型的整体行为表现是有帮助的，但很多时候我们需要弄清楚为什么会做出某个具体的预测。对于单个样本，我们如何说明复杂、不透明模型的输出呢？本章将介绍局部可解释模型无关解释 (LIME)，这是一种专门为此目的而设计的方法。

LIME 的工作方式是，通过采用一个更简单、可解释的模型，在您想要解释的预测附近局部地模拟复杂模型。由于它将原始模型视为黑箱，因此几乎可以应用于任何分类器或回归器。

在本章中，您将学到：

*   LIME 的核心思路及其生成解释的方式。
*   LIME 的运作机制，包括数据扰动和局部替代模型的使用。
*   如何应用 LIME 来说明在表格数据和文本数据上训练的模型的预测结果。
*   理解 LIME 生成的特征重要性分数和可视化结果的方法。
*   使用 LIME Python 库进行实际操作。
*   使用 LIME 时需要注意的重要考量和局限性。

最后，我们将通过一个动手练习结束本章，您将在其中应用 LIME 为预训练模型生成并分析解释。

## 小节

- 1. [LIME 的基本思想](01-LIME%20%E7%9A%84%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3.md)
- 2. [LIME 的工作原理：扰动与替代模型](02-LIME%20%E7%9A%84%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86%EF%BC%9A%E6%89%B0%E5%8A%A8%E4%B8%8E%E6%9B%BF%E4%BB%A3%E6%A8%A1%E5%9E%8B.md)
- 3. [将 LIME 应用于表格数据](03-%E5%B0%86%20LIME%20%E5%BA%94%E7%94%A8%E4%BA%8E%E8%A1%A8%E6%A0%BC%E6%95%B0%E6%8D%AE.md)
- 4. [LIME 在文本数据上的应用](04-LIME%20%E5%9C%A8%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE%E4%B8%8A%E7%9A%84%E5%BA%94%E7%94%A8.md)
- 5. [解读 LIME 解释](05-%E8%A7%A3%E8%AF%BB%20LIME%20%E8%A7%A3%E9%87%8A.md)
- 6. [使用 Python 实现 LIME](06-%E4%BD%BF%E7%94%A8%20Python%20%E5%AE%9E%E7%8E%B0%20LIME.md)
- 7. [LIME的局限性与注意事项](07-LIME%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7%E4%B8%8E%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9.md)
- 8. [动手实践：生成 LIME 解释](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%94%9F%E6%88%90%20LIME%20%E8%A7%A3%E9%87%8A.md)

章节测验：[在线测验](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-2-lime-local-interpretability/quiz)

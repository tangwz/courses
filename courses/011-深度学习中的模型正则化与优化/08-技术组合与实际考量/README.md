# 第 8 章：技术组合与实际考量

来源：[原章节](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-8-combining-techniques-practical)

[返回课程目录](../README.md)

之前的章节分别介绍了各种正则化方法（如L1/L2、Dropout）、归一化技术（如批量归一化）和优化算法（如SGD、Adam等）。基于这些前期内容，本章侧重于如何在实际的深度学习工作流程中有效地结合这些方法。

你将学到：
*   不同正则化和优化技术之间的相互影响。
*   构建一个运用这些方法的典型训练过程。
*   使用损失曲线和指标来监控训练进展。
*   将早期停止作为一种额外的正则化策略来应用。
*   结合Dropout和批量归一化时的具体考虑。
*   数据增强如何提升模型泛化能力。
*   选择合适技术组合的指导原则。
*   解决与优化和正则化相关的常见训练问题。

本章最后将通过一个实践练习，让你使用这些组合策略中的几种来构建和调整模型。

## 小节

- 1. [正则化与优化之间的联动](01-%E6%AD%A3%E5%88%99%E5%8C%96%E4%B8%8E%E4%BC%98%E5%8C%96%E4%B9%8B%E9%97%B4%E7%9A%84%E8%81%94%E5%8A%A8.md)
- 2. [典型的深度学习训练流程](02-%E5%85%B8%E5%9E%8B%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.md)
- 3. [训练监控：损失曲线与指标](03-%E8%AE%AD%E7%BB%83%E7%9B%91%E6%8E%A7%EF%BC%9A%E6%8D%9F%E5%A4%B1%E6%9B%B2%E7%BA%BF%E4%B8%8E%E6%8C%87%E6%A0%87.md)
- 4. [提前停止作为正则化](04-%E6%8F%90%E5%89%8D%E5%81%9C%E6%AD%A2%E4%BD%9C%E4%B8%BA%E6%AD%A3%E5%88%99%E5%8C%96.md)
- 5. [结合 Dropout 与批归一化](05-%E7%BB%93%E5%90%88%20Dropout%20%E4%B8%8E%E6%89%B9%E5%BD%92%E4%B8%80%E5%8C%96.md)
- 6. [数据增强作为隐式正则化](06-%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA%E4%BD%9C%E4%B8%BA%E9%9A%90%E5%BC%8F%E6%AD%A3%E5%88%99%E5%8C%96.md)
- 7. [选择恰当的技术组合](07-%E9%80%89%E6%8B%A9%E6%81%B0%E5%BD%93%E7%9A%84%E6%8A%80%E6%9C%AF%E7%BB%84%E5%90%88.md)
- 8. [调试与优化/正则化相关的训练问题](08-%E8%B0%83%E8%AF%95%E4%B8%8E%E4%BC%98%E5%8C%96-%E6%AD%A3%E5%88%99%E5%8C%96%E7%9B%B8%E5%85%B3%E7%9A%84%E8%AE%AD%E7%BB%83%E9%97%AE%E9%A2%98.md)
- 9. [动手实践：构建与调整正则化/优化模型](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%8E%E8%B0%83%E6%95%B4%E6%AD%A3%E5%88%99%E5%8C%96-%E4%BC%98%E5%8C%96%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-8-combining-techniques-practical/quiz)

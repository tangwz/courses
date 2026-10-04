# 第 4 章：数据预处理与特征工程

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-4-data-preprocessing-feature-engineering)

[返回课程目录](../README.md)

机器学习模型很少能直接处理原始数据并取得好效果。实际数据集通常包含不一致、缺失值以及度量尺度差异大或非数值格式的特征。许多算法要求数据干净、数值化并进行适当缩放以达到最佳表现。例如，计算点之间距离的算法（如K近邻）或使用梯度下降优化的算法（如带正则化的线性回归）对输入特征的尺度很敏感。同样，大多数算法需要数值输入，因此有必要将分类文本数据转换为合适的格式。

本章侧重介绍使用Scikit-learn工具进行数据准备的重要技术。您将学习如何：

*   **缩放数值特征**：使用标准化（$$(X - \mu) / \sigma$$）和归一化（缩放到 $[0, 1]$ 范围）等方法，确保特征对模型训练有恰当的作用。
*   **编码分类特征**：使用独热编码和序数编码等策略，将分类特征转换为数值表示。
*   **处理缺失值**：通过填充策略，用估计值或统计值替换缺失项。

掌握这些预处理步骤，对于构建有效的机器学习模型非常重要。我们将学习Scikit-learn的变换器API，以便高效地应用这些技术。

## 小节

- 1. [数据预处理的重要性](01-%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86%E7%9A%84%E9%87%8D%E8%A6%81%E6%80%A7.md)
- 2. [特征缩放技术](02-%E7%89%B9%E5%BE%81%E7%BC%A9%E6%94%BE%E6%8A%80%E6%9C%AF.md)
- 3. [在Scikit-learn中应用缩放器](03-%E5%9C%A8Scikit-learn%E4%B8%AD%E5%BA%94%E7%94%A8%E7%BC%A9%E6%94%BE%E5%99%A8.md)
- 4. [分类特征编码](04-%E5%88%86%E7%B1%BB%E7%89%B9%E5%BE%81%E7%BC%96%E7%A0%81.md)
- 5. [在Scikit-learn中应用编码器](05-%E5%9C%A8Scikit-learn%E4%B8%AD%E5%BA%94%E7%94%A8%E7%BC%96%E7%A0%81%E5%99%A8.md)
- 6. [处理缺失值](06-%E5%A4%84%E7%90%86%E7%BC%BA%E5%A4%B1%E5%80%BC.md)
- 7. [在Scikit-learn中使用填充器](07-%E5%9C%A8Scikit-learn%E4%B8%AD%E4%BD%BF%E7%94%A8%E5%A1%AB%E5%85%85%E5%99%A8.md)
- 8. [动手实践：数据预处理](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-4-data-preprocessing-feature-engineering/quiz)

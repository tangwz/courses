# 第 6 章：构建管道

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-6-building-pipelines)

[返回课程目录](../README.md)

在之前的章节中，我们将数据预处理和模型训练作为独立的步骤进行处理。顺序执行这些操作，尤其是在交叉验证循环中，可能会很麻烦，并且存在无意中将测试折叠中的信息泄露到训练过程中的风险。

本章介绍 Scikit-learn 的 `Pipeline` 对象，这是一个旨在将多个处理步骤（例如缩放器、编码器和缺失值填充器）与一个最终估计器（例如分类器或回归器）链接起来的工具。您将学习如何构建这些管道，以创建一个代表您整个建模流程的单一对象。我们将介绍如何使用 `GridSearchCV` 将管道与交叉验证和超参数调优有效地结合起来，确保预处理在每个折叠中正确应用。最后，我们将讨论如何使用 `ColumnTransformer` 来构建更复杂的管道，这些管道可以对数据集中的不同列子集应用不同的转换。

## 小节

- 1. [使用管道的理由](01-%E4%BD%BF%E7%94%A8%E7%AE%A1%E9%81%93%E7%9A%84%E7%90%86%E7%94%B1.md)
- 2. [创建简单管道](02-%E5%88%9B%E5%BB%BA%E7%AE%80%E5%8D%95%E7%AE%A1%E9%81%93.md)
- 3. [访问流水线步骤](03-%E8%AE%BF%E9%97%AE%E6%B5%81%E6%B0%B4%E7%BA%BF%E6%AD%A5%E9%AA%A4.md)
- 4. [使用管道结合交叉验证](04-%E4%BD%BF%E7%94%A8%E7%AE%A1%E9%81%93%E7%BB%93%E5%90%88%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81.md)
- 5. [使用管道进行网格搜索](05-%E4%BD%BF%E7%94%A8%E7%AE%A1%E9%81%93%E8%BF%9B%E8%A1%8C%E7%BD%91%E6%A0%BC%E6%90%9C%E7%B4%A2.md)
- 6. [使用 ColumnTransformer 构建复杂管道](06-%E4%BD%BF%E7%94%A8%20ColumnTransformer%20%E6%9E%84%E5%BB%BA%E5%A4%8D%E6%9D%82%E7%AE%A1%E9%81%93.md)
- 7. [实战：流水线的构建与调优](07-%E5%AE%9E%E6%88%98%EF%BC%9A%E6%B5%81%E6%B0%B4%E7%BA%BF%E7%9A%84%E6%9E%84%E5%BB%BA%E4%B8%8E%E8%B0%83%E4%BC%98.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-6-building-pipelines/quiz)

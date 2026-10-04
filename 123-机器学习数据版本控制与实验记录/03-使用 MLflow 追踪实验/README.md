# 第 3 章：使用 MLflow 追踪实验

来源：[原章节](https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow)

[返回课程目录](../README.md)

前一章主要介绍如何使用 DVC 管理数据集，而开发机器学习模型则会遇到另一类管理挑战。模型训练通常需要多次迭代，期间会调整超参数、代码或进行特征工程。如果没有条理分明的方法，就很难回溯是哪些设置产生了特定的结果，也难以准确复现过去的实验。

本章将介绍 MLflow Tracking，它是一种系统记录模型开发工作的方案。我们将讲解如何：
*   设置 MLflow 环境。
*   修改训练脚本以记录重要信息：参数、指标以及模型或图表等工件。
*   将相关的训练尝试归类到实验中。
*   使用 MLflow 界面查看、搜索和比较不同运行的结果。

采纳这些做法能为您的开发过程提供清晰的记录，有助于提高模型的复现性及迭代效率。

## 小节

- 1. [实验跟踪的重要性](01-%E5%AE%9E%E9%AA%8C%E8%B7%9F%E8%B8%AA%E7%9A%84%E9%87%8D%E8%A6%81%E6%80%A7.md)
- 2. [MLflow 追踪功能介绍](02-MLflow%20%E8%BF%BD%E8%B8%AA%E5%8A%9F%E8%83%BD%E4%BB%8B%E7%BB%8D.md)
- 3. [配置 MLflow](03-%E9%85%8D%E7%BD%AE%20MLflow.md)
- 4. [记录参数和指标](04-%E8%AE%B0%E5%BD%95%E5%8F%82%E6%95%B0%E5%92%8C%E6%8C%87%E6%A0%87.md)
- 5. [记录工件（模型、图表、文件）](05-%E8%AE%B0%E5%BD%95%E5%B7%A5%E4%BB%B6%EF%BC%88%E6%A8%A1%E5%9E%8B%E3%80%81%E5%9B%BE%E8%A1%A8%E3%80%81%E6%96%87%E4%BB%B6%EF%BC%89.md)
- 6. [使用实验管理运行](06-%E4%BD%BF%E7%94%A8%E5%AE%9E%E9%AA%8C%E7%AE%A1%E7%90%86%E8%BF%90%E8%A1%8C.md)
- 7. [使用 MLflow 用户界面](07-%E4%BD%BF%E7%94%A8%20MLflow%20%E7%94%A8%E6%88%B7%E7%95%8C%E9%9D%A2.md)
- 8. [比较实验运行](08-%E6%AF%94%E8%BE%83%E5%AE%9E%E9%AA%8C%E8%BF%90%E8%A1%8C.md)
- 9. [实践：追踪训练运行](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%BF%BD%E8%B8%AA%E8%AE%AD%E7%BB%83%E8%BF%90%E8%A1%8C.md)

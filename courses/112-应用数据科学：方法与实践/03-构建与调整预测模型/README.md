# 第 3 章：构建与调整预测模型

来源：[原章节](https://apxml.com/zh/courses/applied-data-science/chapter-3-building-tuning-predictive-models)

[返回课程目录](../README.md)

准备数据并构建有意义的特征后，重心将转向建立能够学习规律并进行预测的模型。本章着重讲解监督学习，即模型从带有标签的数据中学习。

我们将首先对常见的回归和分类算法进行简要回顾，随后使用Python的scikit-learn库进行实际操作。您将应用线性模型（如线性回归和逻辑回归）、基于树的方法（如决策树和随机森林），以及集成技术（包括梯度提升机XGBoost和LightGBM）。

构建有效模型的一个重要环节是正确评估并优化其设置。我们将介绍适用于不同任务的评估指标，不局限于基本准确率，还会涵盖精确度、召回率、$F_1$ 分数和ROC AUC。您将学习实施可靠的验证策略，例如$k$折交叉验证。最后，我们将讨论超参数调优，运用网格搜索和随机搜索等系统方法来寻找最佳模型配置。

学完本章，您将能够训练、严格评估和调整多种标准的监督学习模型，以应对预测任务。

## 小节

- 1. [常用监督学习算法回顾](01-%E5%B8%B8%E7%94%A8%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E5%9B%9E%E9%A1%BE.md)
- 2. [实施线性回归与逻辑回归](02-%E5%AE%9E%E6%96%BD%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E4%B8%8E%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92.md)
- 3. [应用基于树的模型](03-%E5%BA%94%E7%94%A8%E5%9F%BA%E4%BA%8E%E6%A0%91%E7%9A%84%E6%A8%A1%E5%9E%8B.md)
- 4. [梯度提升机介绍](04-%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E6%9C%BA%E4%BB%8B%E7%BB%8D.md)
- 5. [使用网格搜索和随机搜索进行超参数调优](05-%E4%BD%BF%E7%94%A8%E7%BD%91%E6%A0%BC%E6%90%9C%E7%B4%A2%E5%92%8C%E9%9A%8F%E6%9C%BA%E6%90%9C%E7%B4%A2%E8%BF%9B%E8%A1%8C%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E4%BC%98.md)
- 6. [模型准确度评估](06-%E6%A8%A1%E5%9E%8B%E5%87%86%E7%A1%AE%E5%BA%A6%E8%AF%84%E4%BC%B0.md)
- 7. [交叉验证策略](07-%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81%E7%AD%96%E7%95%A5.md)
- 8. [动手实践：模型训练与超参数优化](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%B6%85%E5%8F%82%E6%95%B0%E4%BC%98%E5%8C%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/applied-data-science/chapter-3-building-tuning-predictive-models/quiz)

---
course: "building-ml-recommendation-system"
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-5-evaluating-recommendation-systems"
sourceId: 1296
chapter: "evaluating-recommendation-systems"
title: "评估推荐系统"
order: 5
description: "学习使用精确率、召回率、MAP 和 NDCG 等离线评估指标来衡量推荐系统的表现。"
hasQuiz: false
---

你已经构建了从基于内容过滤到矩阵分解的多种推荐模型。接下来的一个自然且必要的问题是：这些模型的效果如何？模型的有效性并非绝对，其表现受应用具体目标的影响。为了做出明智的决策并改进系统，你需要一套正式的方法来衡量并对比它们的输出。

本章介绍相关的评估技术。我们首先区分使用历史数据的离线评估，以及 A/B 测试等在线评估方法。重点在于离线指标的实际应用，这能让你在部署前对模型进行迭代和测试。

你将学习如何实现并解读针对不同评估任务的多种行业标准指标：

*   **预测准确性：** 对于以准确预测用户评分为目标的情景，我们将介绍均方根误差 (RMSE) 和平均绝对误差 (MAE) 等指标。
*   **排序质量：** 对于大多数应用而言，头部推荐物品的质量和顺序才是核心。我们将分析一系列排序指标，包括 $k$ 处的精确率 (Precision at $k$) 和召回率 (Recall at $k$)、平均精度均值 (MAP) 以及归一化折损累计增益 (NDCG)。

学完本章后，你将拥有一套量化推荐模型表现的实用框架，从而能够对比不同的算法并有效地调整其参数。

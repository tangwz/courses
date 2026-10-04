---
course: "federated-learning"
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-2-advanced-aggregation-algorithms"
sourceId: 734
chapter: "advanced-aggregation-algorithms"
title: "高级聚合算法"
order: 2
description: "学习高级联邦学习聚合算法，例如FedProx、SCAFFOLD、FedNova以及FedAvg标准方法之外的拜占庭鲁棒方法。"
hasQuiz: false
---

标准的联邦平均（FedAvg）算法是一种聚合多客户端模型更新的基本方法。然而，它的有效性可能受限，尤其是在处理客户端间非独立同分布（Non-IID）数据或面对不可靠的参与者时。本章介绍旨在在这些具有挑战性但常见的情况下表现更佳的聚合算法。

您将学习到：

*   标准FedAvg方法的具体弱点和失效模式。
*   像FedProx这样的算法，通过在本地客户端目标函数中添加一个近端项（表示为 $$ \frac{\mu}{2} ||w - w^t||^2 $$），旨在减轻统计异质性的负面影响。
*   SCAFFOLD，一种采用控制变量以减少客户端漂移和方差，从而提高收敛速度的方法。
*   FedNova，通过在聚合前对本地更新进行归一化来处理系统异质性。
*   诸如Krum、Median和Trimmed Mean等技术，它们提供针对提交潜在恶意更新的拜占庭客户端的鲁棒性。
*   这些高级方法的理论动机和收敛特性。
*   将这些算法整合到联邦学习工作流程中的实际实施考量。

完成本章学习后，您将能够选择、实施和分析FedAvg之外的聚合策略，从而在多样化环境中开发出更可靠、性能更优的联邦学习系统。

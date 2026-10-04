---
course: "intro-feature-engineering"
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-2-handling-missing-data"
sourceId: 443
chapter: "handling-missing-data"
title: "处理缺失数据"
order: 2
description: "学习在机器学习数据集中识别和填补缺失值的策略。"
hasQuiz: true
---

实际数据集经常不完整。机器学习算法通常无法直接处理缺失值，因此数据预处理是必需的第一步。本章介绍管理缺失数据的方法。

您将学习使用 Pandas 识别数据中缺失项的方法。我们将考察导致数据缺失的常见原因，例如完全随机缺失 (MCAR)、随机缺失 (MAR) 和非随机缺失 (MNAR)。然后着重介绍实际的填补策略，从均值、中位数和众数填补等简单方法开始。您还将学习创建指示特征，以保留缺失值原始位置的信息。更高级的方法，包括 Scikit-learn 中的 K-近邻 (KNN) 填补器和迭代填补器，将用于多元填补。最后，我们将比较这些不同的方法，以帮助您选择适用于您具体情况的策略。本章包含应用这些方法的实践练习。

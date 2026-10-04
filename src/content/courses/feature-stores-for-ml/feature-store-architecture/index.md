---
course: "feature-stores-for-ml"
sourceUrl: "https://apxml.com/zh/courses/feature-stores-for-ml/chapter-1-feature-store-architecture"
sourceId: 711
chapter: "feature-store-architecture"
title: "特征平台架构"
order: 1
description: "了解特征平台中精巧的在线/离线存储、元数据管理及注册表实现架构模式。"
hasQuiz: false
---

有效的特征平台实现，很大程度上取决于良好的架构设计。本章侧重于构建可扩展且具备韧性的系统所需的结构方面。我们将考察注册表、在线存储和离线存储等核心组成部分如何相互配合。

您将了解到针对低延迟在线服务的具体架构模式，这些模式通常涉及最大限度地缩短检索时间 $T_{retrieval}$ 的策略，以及适用于大规模批处理的可扩展离线存储设计。我们还将介绍对组织和可发现性很重要的元数据管理技术。此外，我们分析集成式与解耦式系统设计之间的权衡取舍，并处理针对多区域或多云部署的考量。本章包含一个实践练习，用于将这些知识点应用于一个设计问题。

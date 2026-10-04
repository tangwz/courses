---
course: "planning-optimizing-ai-infrastructure"
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-2-designing-on-premise-ai-infrastructure"
sourceId: 1223
chapter: "designing-on-premise-ai-infrastructure"
title: "设计本地AI基础设施"
order: 2
description: "学习设计本地AI基础设施，涵盖从选择服务器硬件和GPU到配置存储和网络，以实现最佳性能。"
hasQuiz: false
---

下章将讨论云平台，而本章将侧重于构建和管理您自己的硬件的过程。本地部署提供对性能、安全性及配置的直接控制，但它也需要清楚了解物理组件如何整合为一个协调运作的系统。这种方法需要在初始资本支出（CapEx）与长期运营效率之间取得平衡。

在本章中，您将学习如何将机器学习工作负载需求转化为具体的硬件规格。我们将涵盖服务器机箱、主板和图形处理器（GPU）的选择，并密切关注NVLink等互连技术及其对多GPU性能的影响。互连方式的选择直接影响分布式任务中的通信开销，这是总训练时间公式 $T_{total} = T_{compute} + T_{communication}$ 中的一个重要因素。我们还将讨论重要的支持系统，包括使用NVMe硬盘的高速存储配置、用于快速数据访问的网络，以及对电力和散热的大量需求。

本章结束时，您将能够评估您的需求，选择合适的组件，并规划专用AI服务器的物理部署。您将通过为特定工作负载场景创建详细的硬件规格表来实践这些知识。

---
course: "intro-ml-compiler-optimization"
sourceUrl: "https://apxml.com/zh/courses/intro-ml-compiler-optimization"
sourceId: 248
title: "机器学习编译器优化入门"
description: "学习机器学习编译器如何将计算图转换为高效、硬件专用的机器码。"
category: "Machine Learning"
level: 2
duration: 12
order: 128
prerequisites: "基础机器学习和编程知识"
color: "pink"
chapterCount: 5
lessonCount: 27
outcomes: [{"topic": "编译器架构", "description": "理解ML编译器栈的组成部分，包括前端、中间表示和后端代码生成。"}, {"topic": "图优化", "description": "将算子融合、常量折叠和死代码消除等高级转换应用于计算图。"}, {"topic": "循环调度", "description": "实施分块、向量化和循环重排等低级优化，以最大限度地提升硬件利用率。"}, {"topic": "自动调优", "description": "配置并运行自动化搜索过程，以找到特定硬件目标的最优参数。"}]
hasProject: false
---

机器学习模型，特别是深度神经网络，需要大量计算资源。尽管PyTorch和TensorFlow等框架为模型设计提供了高级抽象，但要在特定硬件上高效运行这些模型，需要在编译器层面进行优化。本课程介绍机器学习编译器的架构和运行机制，重点讲述如何将高级计算图转换为高效的机器码。

您将学习ML模型从图捕获到代码生成的整个生命周期。课程内容包含中间表示（IR）、图级转换以及低级循环优化。您将了解编译器如何进行算子融合、内存布局重写以及硬件专用指令映射。本课程还介绍自动调优策略，这些策略用于在无需手动干预的情况下找到最优执行方案。通过学习这些内容，您将获得使用现代编译器栈检查、调试和提升模型性能所需的技术能力。

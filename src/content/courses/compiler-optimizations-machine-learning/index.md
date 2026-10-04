---
course: "compiler-optimizations-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/compiler-optimizations-machine-learning"
sourceId: 247
title: "机器学习编译优化"
description: "应用编译器技术，转换并优化深度学习模型，以实现高性能硬件运行。"
category: "Machine Learning"
level: 3
duration: 28
order: 116
prerequisites: "扎实的Python基础，熟悉机器学习基础知识。"
color: "pink"
chapterCount: 6
lessonCount: 30
outcomes: [{"topic": "中间表示", "description": "设计并操作用于张量计算的高级与低级中间表示。"}, {"topic": "图优化", "description": "实现图级转换，如算子融合、常量折叠和死代码消除。"}, {"topic": "循环转换", "description": "应用高级循环优化，如分块、展开和重排序，以最大化缓存局部性。"}, {"topic": "MLIR架构", "description": "运用多级中间表示框架构建自定义方言与下沉流水线。"}, {"topic": "自动调优", "description": "制定自动核函数搜索和成本建模策略，以确定最佳执行调度。"}]
hasProject: false
---

机器学习模型需要大量计算资源，因此需要将高级框架高效地转换为机器码。本课程介绍Apache TVM、XLA和MLIR等深度学习编译器的架构与工程原理。本课程讲述从计算图到硬件专用二进制文件的转换流程。

学员将分析中间表示（IR）、图级优化（包含算子融合和布局转换）以及循环优化（如分块和向量化）。课程内容包含多面体模型、自动调度策略以及GPU和专用加速器的内存层次管理。重点在于软件逻辑与硬件限制如何相互配合，以最大化吞吐量并减少延迟。

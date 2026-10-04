---
course: "advanced-tensorflow"
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-1-tensorflow-execution-graphs"
sourceId: 641
chapter: "tensorflow-execution-graphs"
title: "TensorFlow 执行与图"
order: 1
description: "了解 TensorFlow 的即时执行与图执行、tf.function、AutoGraph、梯度计算和调试技术。"
hasQuiz: false
---

了解 TensorFlow 如何执行操作是编写高效和可扩展代码的根本。尽管 TensorFlow 2 默认使用即时执行（像 Python 一样命令式地运行操作），但创建和优化静态计算图对于获得高性能和支持部署仍然非常重要。本章将讲解这种相互关系。

你将学习到：
*   即时执行与图执行之间的区别和权衡。
*   `tf.function` 如何使用 AutoGraph 将 Python 代码转换为优化过的 TensorFlow 图。
*   函数追踪的机制以及 TensorFlow 如何在内部表示计算。
*   在图中实现控制流（例如，条件语句、循环）。
*   使用 `tf.GradientTape` 进行自动微分，它是梯度下降训练（如 $L = \sum (y_{pred} - y_{true})^2$）的驱动力。
*   TensorFlow 程序的基本资源管理考量和调试技术。

掌握这些执行细节将为后续性能调优、分布式训练和自定义组件开发提供所需的支持。

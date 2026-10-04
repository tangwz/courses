---
course: "compiler-runtime-optimization-ml"
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-7-jit-compilation-ml"
sourceId: 882
chapter: "jit-compilation-ml"
title: "面向机器学习的即时 (JIT) 编译技术"
order: 7
description: "分析 JIT 编译方法 (XLA, TorchScript)，包括追踪、脚本化、特化和自适应方法。"
hasQuiz: false
---

机器学习框架中常见的执行模式，如即时执行或预先 (AOT) 编译，在灵活性和优化潜力之间有各自的取舍。即时 (JIT) 编译的工作方式则不同，它将最终的代码生成推迟到运行时进行。这种策略能够根据动态运行时环境的信息进行优化，例如实际观测到的张量形状 (例如，$B \times C \times H \times W$) 或数值。

本章将介绍与机器学习工作负载相关的 JIT 编译方法。我们将分析图获取技术，例如追踪 (tracing) 和脚本化 (scripting)，审视 JIT 专用中间表示的需求，并研究运行时特化和自适应编译策略。TensorFlow XLA 和 PyTorch JIT (TorchScript) 等重要 JIT 系统的架构和功能将作为具体实例进行说明。

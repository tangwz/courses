---
course: "compiler-runtime-optimization-ml"
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-1-ml-execution-stack-challenges"
sourceId: 865
chapter: "ml-execution-stack-challenges"
title: "开篇：机器学习执行栈与挑战"
order: 1
description: "了解机器学习模型执行流程、性能瓶颈以及编译器和运行时环境的功能。"
hasQuiz: false
---

部署训练好的机器学习模型，相比训练阶段，在效率上存在一些特殊问题。一个常见现象是，模型开发与在生产环境中执行时，性能表现存在差异。本章将阐明造成这一性能差距的原因，并阐述专用编译器和运行时环境在解决此问题上的作用。

我们将分析典型机器学习执行栈的组成部分，并指出与计算、内存和延迟相关的常见性能瓶颈。您将对用于机器学习加速的各种硬件目标有个大致了解，包括CPU、GPU和定制芯片，同时认识到它们的特性如何影响优化策略。最后，我们将说明为何通用编译技术通常无法满足复杂的机器学习工作负载需求，这正是本课程所讲解的先进、专门的优化方法的重要原因。

---
course: "deploying-diffusion-models-scale"
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-1-scaling-challenges-architectures"
sourceId: 989
chapter: "scaling-challenges-architectures"
title: "扩展难题与系统架构"
order: 1
description: "理解扩散模型扩展的独特难题，以及部署时常用的架构模式。"
hasQuiz: false
---

生成式扩散模型由于其计算密集性，带来了独特的运行难题。将这些模型从研究环境迁移到为真实用户提供可靠、高效服务的生产系统，会带来显著的工程障碍。本章重点讲解如何理解这些具体的难题。

我们将分析扩散过程中固有的计算和内存需求，特别是在推理阶段。我们会考量生成延迟（输出生成的速度）与吞吐量（能同时处理的请求数量）之间的权衡。此外，我们还将回顾用于服务大型生成模型的常见系统架构，对比处理用户请求的同步与异步处理方法，并调整核心 MLOps 原则，以便在生产环境中持续管理扩散模型。在本章结束时，您将对有效部署扩散模型的主要障碍和基本方法有清晰的认识。

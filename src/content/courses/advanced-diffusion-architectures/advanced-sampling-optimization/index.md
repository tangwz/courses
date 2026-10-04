---
course: "advanced-diffusion-architectures"
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-6-advanced-sampling-optimization"
sourceId: 1003
chapter: "advanced-sampling-optimization"
title: "进阶采样与优化"
order: 6
description: "掌握进阶采样算法，解决收敛问题，并优化扩散模型以提升推理速度和降低资源使用。"
hasQuiz: false
---

尽管DDPM和DDIM等基本采样方法是扩散模型生成数据所依赖的主要方式，但其迭代特性常常导致推理时间缓慢。本章着重介绍大幅加速采样过程和优化模型以用于实践的技术。

你将考察进阶采样算法，包括像DPM-Solver和UniPC这样的高阶ODE求解器，它们旨在以更少步骤生成高质量结果。我们还将介绍随机采样变体和对引导采样的改进。此外，本章还提供关于解决常见采样问题（例如伪影或模糊）的实用指导。最后，我们将讨论用于部署的模型优化，涵盖模型蒸馏、量化等方法，以及硬件加速的考量，以提高速度并减少资源消耗。

---
course: "mixture-of-experts-advanced-implementation"
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference"
sourceId: 1235
chapter: "efficient-moe-inference"
title: "高效的MoE模型推理"
order: 4
description: "掌握MoE模型推理的优化，掌握专家卸载、量化和推测解码等方式的使用。"
hasQuiz: false
---

训练一个专家混合模型是整个流程的一部分。将这些模型投入生产环境进行推理，会带来一系列特有的难题，这些难题主要与它们巨大的内存占用以及稀疏激活的计算方式有关。尽管MoE模型的稀疏性使其训练计算量（$FLOPs$）相对于参数数量而言容易应对，但其这一特点却也给低延迟服务带来了特殊的挑战。

本章将介绍优化MoE模型推理的实用方法。我们将介绍处理高内存占用的方法，包括将不活跃的专家模型卸载到CPU内存或NVMe中。你将学会如何应用适合稀疏计算的专门批处理策略，通过量化和蒸馏进行模型压缩，并使用推测解码来加快token生成。最终，你将能够为大型稀疏模型搭建一个高效的推理流程。

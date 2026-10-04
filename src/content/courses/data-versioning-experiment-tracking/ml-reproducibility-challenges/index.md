---
course: "data-versioning-experiment-tracking"
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-1-ml-reproducibility-challenges"
sourceId: 807
chapter: "ml-reproducibility-challenges"
title: "机器学习可复现性的必要性"
order: 1
description: "了解重现机器学习结果时的难题，以及为什么版本控制和追踪是必要的。"
hasQuiz: true
---

机器学习项目常涉及代码、数据集、模型配置和环境的频繁变动。重现特定结果，无论是为了调试、协作还是部署，都可能成为一个不小的难题。简单的Git代码版本控制有所帮助，但无法解决追踪大型数据集或记录特定模型训练运行所用精确参数等问题。

本章将让您明白，为什么系统化管理这些要素是必要的。我们会研究重现机器学习实验时常遇到的常见困难。您将了解为什么仅靠标准版本控制工具不足以应对机器学习工作流程的独特要求，尤其是在数据方面。我们将明确可复现性在此处的意思，并指出需要追踪的基本组成部分。最后，我们会介绍数据版本控制和实验追踪的初步设想，这些设想我们将在整个课程中，使用DVC和MLflow等工具进行后续讲解。

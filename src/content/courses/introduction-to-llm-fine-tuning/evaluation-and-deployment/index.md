---
course: "introduction-to-llm-fine-tuning"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-5-evaluation-and-deployment"
sourceId: 1267
chapter: "evaluation-and-deployment"
title: "评估与部署"
order: 5
description: "学习如何使用量化和定性指标来评估你微调后的LLM性能，并为其部署做准备。"
hasQuiz: false
---

完成微调过程后，下一步自然是评估模型的性能并为其推理做好准备。模型的价值取决于其在特定任务上的有效性，本章将介绍衡量这些有效性并使其投入实际使用的方法。

我们将从建立一个评估框架开始。这包括应用ROUGE、BLEU和困惑度等量化指标，这些指标可以来源于交叉熵损失 $H(p,q)$，通常表示为 $2^{H(p,q)}$。除了这些自动化分数外，你还将学习进行定性评估，即通过人工判断来检查输出的连贯性和相关性。然后，你将了解如何构建自动化流程来系统地应用这些技术。

本章最后将连接评估与部署。这包括将训练好的PEFT适配器与基础模型合并以创建独立成品，以及为最终模型的高效推理做准备等实际操作步骤。

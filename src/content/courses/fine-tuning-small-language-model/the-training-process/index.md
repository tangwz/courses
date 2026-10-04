---
course: "fine-tuning-small-language-model"
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-5-the-training-process"
sourceId: 1505
chapter: "the-training-process"
title: "训练过程"
order: 5
description: "设置训练超参数、管理状态并执行语言模型的微调循环。"
hasQuiz: false
---

数据格式已校对，环境已配置，参数高效适配器也已初始化，微调的各项准备工作已经就绪。接下来要运行训练循环，让模型从指令数据集中学习。这一过程需要在硬件算力限制与确保模型高效收敛之间取得平衡。

在这一部分，你将定义决定优化运行方式的训练参数。你将配置批次大小和累积步数，以控制内存使用。接着，你将设置学习率和调度器。设置合适的学习率调度方案可以确保权重更新随训练推进而逐渐减小。一种标准方法是应用衰减公式来使学习过程随时间趋于稳定：

$$ \alpha_{t} = \alpha_{initial} \cdot \beta^{t} $$

其中，$\alpha_{t}$ 是特定步数的学习率，$t$ 代表当前训练步数，$\beta$ 是衰减因子。

你还将实现检查点机制，用以保存中间模型状态。定期保存进度可以防止因硬件故障或进程中断导致的数据丢失。除了状态管理，你还将追踪验证指标和训练损失。通过监控这些数值，你可以察觉模型何时停止学习，或者何时开始死记硬背训练数据而失去了泛化能力。最后，你将整合这些模块，在部分数据上执行完整的训练循环，生成一套可供评估的微调权重。

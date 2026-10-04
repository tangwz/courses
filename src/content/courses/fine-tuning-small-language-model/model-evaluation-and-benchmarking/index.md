---
course: "fine-tuning-small-language-model"
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-6-model-evaluation-and-benchmarking"
sourceId: 1506
chapter: "model-evaluation-and-benchmarking"
title: "模型评估与基准测试"
order: 6
description: "评测生成文本的质量，应用定量指标，并检测过拟合现象。"
hasQuiz: false
---

你已经顺利完成了训练循环并生成了更新后的模型权重。接下来的任务是确定你的轻量级语言模型是否能正确执行预定任务。训练过程中损失值的下降表明模型正在拟合数据，但这并不保证生成的文本在实际应用中是准确、连贯或有用的。

在本章中，你将通过定性观察和定量测量来评估微调后的模型。首先，你将检查文本生成质量，观察模型对标准指令提示词的响应情况。随后，你将计算标准的自然语言处理指标，为模型表现进行评分。例如，你将计算困惑度（Perplexity）。如果 $N$ 代表序列中的标记数量，困惑度将通过以下公式衡量模型预测该样本的能力：

$$PP(W) = \exp \left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{<i}) \right)$$

你还将追踪 ROUGE 等指标得分，将生成的输出与参考文本进行对比。我们将测试提示词的泛化能力，确保模型在面对不熟悉的表述时不会失效。通过直接对比微调输出与基础模型，你将学习如何识别过拟合和灾难性遗忘的迹象。最后，你将编写一个自动化评估脚本来处理预留数据集。这为你提供了一套可重复的方法，在进入最终部署阶段前为模型建立性能基准。

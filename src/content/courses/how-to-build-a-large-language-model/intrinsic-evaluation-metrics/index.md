---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-21-intrinsic-evaluation-metrics"
sourceId: 1099
chapter: "intrinsic-evaluation-metrics"
title: "内在评估指标"
order: 21
description: "理解并计算困惑度等内在评估指标，以直接衡量语言建模性能。"
hasQuiz: false
---

在投入大量计算和数据资源训练大型语言模型后，下一步是衡量它在基本任务——建模语言——上的实际表现。评估这些复杂系统需要特定的方法。有一类评估方法直接关注模型预测文本序列的能力，而无需在下游应用中对其进行测试。

本章着重介绍这些*内在*评估方法。我们将研究直接从模型为文本序列分配的概率中得出的指标。最常见的内在指标是**困惑度**，它量化了概率模型预测样本的效果。它与训练期间使用的交叉熵损失有紧密联系。较低的困惑度分数通常表明模型在预测测试数据方面表现更好，这意味着它为观测到的序列分配了更高的概率。对于序列 $W = w_1, w_2, ..., w_N$，困惑度可以根据模型分配的概率 $p(w_i | w_{<i}; \theta)$ 表示为：

$$ PPL(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log p(w_i | w_{<i}; \theta)\right) $$

理解困惑度为模型的语言建模质量提供了一个基础评估。

在本章中，你将学到：

*   困惑度的定义及其从模型输出中的计算方法。
*   如何解释困惑度值，并理解其相对含义和局限性。
*   其他相关指标，例如每字符比特数或每词比特数。
*   不同分词策略对计算出的困惑度分数可能产生的影响。

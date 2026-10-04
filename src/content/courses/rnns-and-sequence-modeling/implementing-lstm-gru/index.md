---
course: "rnns-and-sequence-modeling"
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-7-implementing-lstm-gru"
sourceId: 615
chapter: "implementing-lstm-gru"
title: "实现 LSTM 和 GRU"
order: 7
description: "使用 TensorFlow 或 PyTorch 实现 LSTM 和 GRU 网络的实践指南，包括堆叠和双向变体。"
hasQuiz: true
---

在审视了长短期记忆网络（LSTM）和门控循环单元（GRU）的理论原理，包括其旨在管理信息流并缓解简单RNN中梯度问题的内部门控结构（LSTM中的$f_t, i_t, o_t$，GRU中的$z_t, r_t$）之后，我们现在开始进行它们的实际操作。

本章着重于使用当代深度学习框架将理论转化为可运行的代码。您将学习实例化由库API提供的LSTM和GRU层，配置它们的主要参数（如隐藏单元数量和激活函数选择），并处理输入和输出所需的预期三维张量形状（批次大小、时间步长、特征）。我们还将通过堆叠循环层来构建更复杂的模型，以提升表示能力，并实施双向处理，让网络能从序列的前向和后向处理中考虑上下文。一个实际的编码示例将通过将LSTM或GRU模型应用于情感分析任务来巩固这些思想。

# 第 7 章：实现 LSTM 和 GRU

来源：[原章节](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-7-implementing-lstm-gru)

[返回课程目录](../README.md)

在审视了长短期记忆网络（LSTM）和门控循环单元（GRU）的理论原理，包括其旨在管理信息流并缓解简单RNN中梯度问题的内部门控结构（LSTM中的$f_t, i_t, o_t$，GRU中的$z_t, r_t$）之后，我们现在开始进行它们的实际操作。

本章着重于使用当代深度学习框架将理论转化为可运行的代码。您将学习实例化由库API提供的LSTM和GRU层，配置它们的主要参数（如隐藏单元数量和激活函数选择），并处理输入和输出所需的预期三维张量形状（批次大小、时间步长、特征）。我们还将通过堆叠循环层来构建更复杂的模型，以提升表示能力，并实施双向处理，让网络能从序列的前向和后向处理中考虑上下文。一个实际的编码示例将通过将LSTM或GRU模型应用于情感分析任务来巩固这些思想。

## 小节

- 1. [在深度学习框架中使用LSTM层](01-%E5%9C%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E4%B8%AD%E4%BD%BF%E7%94%A8LSTM%E5%B1%82.md)
- 2. [在深度学习框架中使用GRU层](02-%E5%9C%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E4%B8%AD%E4%BD%BF%E7%94%A8GRU%E5%B1%82.md)
- 3. [配置 LSTM/GRU 层参数](03-%E9%85%8D%E7%BD%AE%20LSTM-GRU%20%E5%B1%82%E5%8F%82%E6%95%B0.md)
- 4. [堆叠循环层](04-%E5%A0%86%E5%8F%A0%E5%BE%AA%E7%8E%AF%E5%B1%82.md)
- 5. [理解双向循环神经网络](05-%E7%90%86%E8%A7%A3%E5%8F%8C%E5%90%91%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
- 6. [实现双向层](06-%E5%AE%9E%E7%8E%B0%E5%8F%8C%E5%90%91%E5%B1%82.md)
- 7. [动手实践：情感分析](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90.md)

章节测验：[在线测验](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-7-implementing-lstm-gru/quiz)

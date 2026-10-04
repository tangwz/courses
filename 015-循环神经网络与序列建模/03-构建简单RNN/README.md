# 第 3 章：构建简单RNN

来源：[原章节](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-3-building-simple-rnns)

[返回课程目录](../README.md)

在了解了简单循环神经网络的主要思想和数学原理，包括时间反向传播（BPTT）的工作方式之后，我们现在将重点转向实现。本章将指导您完成使用常用工具构建第一个RNN模型的实际步骤。

您将学习如何：

*   配置您的Python环境，安装所需的深度学习库（TensorFlow或PyTorch）。
*   使用框架API来定义 `SimpleRNN` 层。
*   正确管理循环层所需的输入和输出张量形状，通常表示为 `(batch_size, time_steps, features)`。
*   将这些层组合成一个完整的序列模型。
*   构建基本的训练循环，根据损失函数来输入数据并更新模型权重。
*   将这些步骤应用于一个涉及简单序列预测的实践案例。

在本章结束时，您将把RNN的理论知识转化为可运行的代码，为您处理更复杂的架构和应用做好准备。

## 小节

- 1. [搭建开发环境](01-%E6%90%AD%E5%BB%BA%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83.md)
- 2. [RNN 单元实现](02-RNN%20%E5%8D%95%E5%85%83%E5%AE%9E%E7%8E%B0.md)
- 3. [使用框架API构建简单RNN层](03-%E4%BD%BF%E7%94%A8%E6%A1%86%E6%9E%B6API%E6%9E%84%E5%BB%BA%E7%AE%80%E5%8D%95RNN%E5%B1%82.md)
- 4. [处理输入与输出形状](04-%E5%A4%84%E7%90%86%E8%BE%93%E5%85%A5%E4%B8%8E%E8%BE%93%E5%87%BA%E5%BD%A2%E7%8A%B6.md)
- 5. [构建一个简单RNN模型](05-%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95RNN%E6%A8%A1%E5%9E%8B.md)
- 6. [RNN 的训练循环](06-RNN%20%E7%9A%84%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)
- 7. [动手实践：简单序列预测](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%AE%80%E5%8D%95%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-3-building-simple-rnns/quiz)

# 第 3 章：训练深度神经网络

来源：[原章节](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-3-training-deep-neural-networks)

[返回课程目录](../README.md)

在使用Keras层定义了神经网络的架构后，下一步是让它从数据中学习。本章主要讲解训练过程的机制。

你将学习如何通过*编译*模型来准备训练，这包括选择一个损失函数来衡量误差，一个优化算法来更新模型的权重，以及用于监控性能的指标。我们将介绍梯度下降及其变体（例如Adam、SGD）等重要思想，反向传播的原理（网络如何学习调整），以及使用Keras的`fit()`方法进行实际操作。

我们还将解释$epochs$和$batch$ $size$等必要的训练参数，使用验证数据来监控进度的重要性，最后，如何使用`evaluate()`方法评估模型在未见过数据上的表现。到本章结束时，你将理解如何有效地定义Keras模型并对其进行训练的完整流程。

## 小节

- 1. [编译步骤](01-%E7%BC%96%E8%AF%91%E6%AD%A5%E9%AA%A4.md)
- 2. [理解损失函数](02-%E7%90%86%E8%A7%A3%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 3. [优化算法](03-%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95.md)
- 4. [反向传播](04-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD.md)
- 5. [训练循环：\`fit()\` 方法](05-%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF%EF%BC%9A%60fit%28%29%60%20%E6%96%B9%E6%B3%95.md)
- 6. [批次与周期](06-%E6%89%B9%E6%AC%A1%E4%B8%8E%E5%91%A8%E6%9C%9F.md)
- 7. [验证数据与性能监控](07-%E9%AA%8C%E8%AF%81%E6%95%B0%E6%8D%AE%E4%B8%8E%E6%80%A7%E8%83%BD%E7%9B%91%E6%8E%A7.md)
- 8. [模型评估：evaluate() 方法](08-%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%EF%BC%9Aevaluate%28%29%20%E6%96%B9%E6%B3%95.md)
- 9. [动手实践：训练一个简单分类器](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E5%88%86%E7%B1%BB%E5%99%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-3-training-deep-neural-networks/quiz)

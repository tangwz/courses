# 第 5 章：搭建与训练深度神经网络

来源：[原章节](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-5-building-training-dnns)

[返回课程目录](../README.md)

前面的章节介绍了神经网络的主要思想，包括其结构、激活函数、损失指标以及梯度下降和反向传播等优化方法。本章从理论转向实践，侧重于搭建和训练深度神经网络模型的工作流程。

你将使用 TensorFlow/Keras 或 PyTorch 等常用深度学习框架来逐层定义模型架构。涵盖的主要步骤包括准备和预处理输入数据、有效权重初始化的方法、通过指定损失函数和优化器来编译模型、执行训练循环、通过损失和准确度指标监控进展，最后，在独立的测试数据上评估训练好的模型。一个涉及图像分类的实践练习将为你提供这些步骤的动手实践。

## 小节

- 1. [深度学习框架简介 (TensorFlow/Keras, PyTorch)](01-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E7%AE%80%E4%BB%8B%20%28TensorFlow-Keras%2C%20PyTorch%29.md)
- 2. [搭建开发环境](02-%E6%90%AD%E5%BB%BA%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83.md)
- 3. [神经网络的数据准备](03-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87.md)
- 4. [定义一个前馈网络模型](04-%E5%AE%9A%E4%B9%89%E4%B8%80%E4%B8%AA%E5%89%8D%E9%A6%88%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B.md)
- 5. [权重初始化方法](05-%E6%9D%83%E9%87%8D%E5%88%9D%E5%A7%8B%E5%8C%96%E6%96%B9%E6%B3%95.md)
- 6. [配置模型：损失函数与优化器选择](06-%E9%85%8D%E7%BD%AE%E6%A8%A1%E5%9E%8B%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E4%B8%8E%E4%BC%98%E5%8C%96%E5%99%A8%E9%80%89%E6%8B%A9.md)
- 7. [模型训练：fit 方法](07-%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%EF%BC%9Afit%20%E6%96%B9%E6%B3%95.md)
- 8. [监控训练进展（损失与指标）](08-%E7%9B%91%E6%8E%A7%E8%AE%AD%E7%BB%83%E8%BF%9B%E5%B1%95%EF%BC%88%E6%8D%9F%E5%A4%B1%E4%B8%8E%E6%8C%87%E6%A0%87%EF%BC%89.md)
- 9. [模型性能评估](09-%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E8%AF%84%E4%BC%B0.md)
- 10. [动手实践：在MNIST上训练分类器](10-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%9C%A8MNIST%E4%B8%8A%E8%AE%AD%E7%BB%83%E5%88%86%E7%B1%BB%E5%99%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-5-building-training-dnns/quiz)

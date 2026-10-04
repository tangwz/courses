# 第 4 章：高级 API 使用与自定义组件

来源：[原章节](https://apxml.com/zh/courses/advanced-tensorflow/chapter-4-advanced-api-custom-components)

[返回课程目录](../README.md)

尽管标准 Keras API 提供了构建多种模型的有效方法，但某些研究目标或复杂应用需要更高的灵活性。本章介绍一些方法，用于扩展 TensorFlow 的核心功能以适应特定需求。

您将学习如何通过以下方式更精细地控制模型架构和训练过程：

*   继承 `tf.keras.Model` 类以创建完全自定义的模型结构。
*   通过继承 `tf.keras.layers.Layer` 类开发独特的处理模块。
*   实现根据特定目标定制的自定义损失函数和评估指标。
*   使用 `tf.GradientTape` 编写自定义训练循环，以精确控制梯度计算和权重更新。
*   处理特殊数据表示形式，例如用于变长输入的 `tf.RaggedTensor` 和用于包含大量零值的数据的 `tf.SparseTensor`。
*   使用 TensorFlow Addons 以获取社区贡献的额外组件。

掌握这些技术将使您能够摆脱预定义结构，并实现新颖的构想或高度专业的机器学习系统。

## 小节

- 1. [通过继承tf.keras.Model获得灵活性](01-%E9%80%9A%E8%BF%87%E7%BB%A7%E6%89%BFtf.keras.Model%E8%8E%B7%E5%BE%97%E7%81%B5%E6%B4%BB%E6%80%A7.md)
- 2. [构建定制tf.keras层](02-%E6%9E%84%E5%BB%BA%E5%AE%9A%E5%88%B6tf.keras%E5%B1%82.md)
- 3. [实现自定义损失函数](03-%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%AE%9A%E4%B9%89%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 4. [开发自定义指标](04-%E5%BC%80%E5%8F%91%E8%87%AA%E5%AE%9A%E4%B9%89%E6%8C%87%E6%A0%87.md)
- 5. [编写自定义训练循环](05-%E7%BC%96%E5%86%99%E8%87%AA%E5%AE%9A%E4%B9%89%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)
- 6. [处理参差张量和稀疏张量](06-%E5%A4%84%E7%90%86%E5%8F%82%E5%B7%AE%E5%BC%A0%E9%87%8F%E5%92%8C%E7%A8%80%E7%96%8F%E5%BC%A0%E9%87%8F.md)
- 7. [使用TensorFlow Addons处理特定功能](07-%E4%BD%BF%E7%94%A8TensorFlow%20Addons%E5%A4%84%E7%90%86%E7%89%B9%E5%AE%9A%E5%8A%9F%E8%83%BD.md)
- 8. [动手实践：构建自定义模型管线](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9E%8B%E7%AE%A1%E7%BA%BF.md)

# 第 6 章：保存和加载模型

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-6-saving-loading-models)

[返回课程目录](../README.md)

在投入时间和计算资源训练机器学习模型后，您会希望保存其学习到的状态。保存模型可以让您稍后停止并恢复训练，无需重新训练即可用于对新数据进行预测，或者与他人分享。本章侧重于TensorFlow和Keras中模型保存的实用方法。

您将学习如何：
*   在训练期间使用检查点定期保存模型权重。
*   仅保存模型学习到的参数（权重）。
*   保存完整的模型，包括其架构、权重和优化器配置。
*   了解标准的TensorFlow `SavedModel` 格式，该格式适用于部署。
*   加载以前保存的权重或整个模型，用于推断或继续训练。
*   了解TensorFlow Hub以获取预训练模型组件。

我们将介绍可用的不同方法，并讨论何时使用每种方法，以确保您能有效管理训练好的模型。

## 小节

- 1. [为什么要保存和加载模型？](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B%EF%BC%9F.md)
- 2. [训练期间保存检查点](02-%E8%AE%AD%E7%BB%83%E6%9C%9F%E9%97%B4%E4%BF%9D%E5%AD%98%E6%A3%80%E6%9F%A5%E7%82%B9.md)
- 3. [仅保存权重](03-%E4%BB%85%E4%BF%9D%E5%AD%98%E6%9D%83%E9%87%8D.md)
- 4. [保存整个模型（架构 + 权重 + 优化器状态）](04-%E4%BF%9D%E5%AD%98%E6%95%B4%E4%B8%AA%E6%A8%A1%E5%9E%8B%EF%BC%88%E6%9E%B6%E6%9E%84%20%2B%20%E6%9D%83%E9%87%8D%20%2B%20%E4%BC%98%E5%8C%96%E5%99%A8%E7%8A%B6%E6%80%81%EF%BC%89.md)
- 5. [TensorFlow SavedModel 形式](05-TensorFlow%20SavedModel%20%E5%BD%A2%E5%BC%8F.md)
- 6. [加载预训练模型](06-%E5%8A%A0%E8%BD%BD%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B.md)
- 7. [TensorFlow Hub 简介](07-TensorFlow%20Hub%20%E7%AE%80%E4%BB%8B.md)
- 8. [练习：模型训练的保存与恢复](08-%E7%BB%83%E4%B9%A0%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BF%9D%E5%AD%98%E4%B8%8E%E6%81%A2%E5%A4%8D.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-6-saving-loading-models/quiz)

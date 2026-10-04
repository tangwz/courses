# 第 4 章：模型的训练与评估

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models)

[返回课程目录](../README.md)

既然您已经使用Keras API构建了模型，本章将讲解训练模型并评估其性能的重要步骤。首先，我们将通过`model.compile()`配置学习过程，您将选择合适的损失函数（如回归任务的均方误差 $MSE$ 或分类任务的交叉熵），选择指导学习的优化器（如Adam或$SGD$），并定义用于观察进度的指标（如准确率）。

接着，您将学习如何使用`model.fit()`将数据喂入模型并执行训练循环，理解批次（epochs）和批处理大小（batch size）等参数。随后，我们将介绍使用`model.evaluate()`评估训练好的模型在测试数据上的效果，并使用`model.predict()`对新输入生成预测。最后，我们将介绍Keras回调函数，包括用于保存训练进度的`ModelCheckpoint`，用于防止过拟合的`EarlyStopping`，以及整合TensorBoard以展示训练指标和模型图。本章最后将通过结合这些技术的实践练习进行总结。

## 小节

- 1. [编译模型：损失函数](01-%E7%BC%96%E8%AF%91%E6%A8%A1%E5%9E%8B%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 2. [编译模型：优化器](02-%E7%BC%96%E8%AF%91%E6%A8%A1%E5%9E%8B%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8.md)
- 3. [编译模型：指标](03-%E7%BC%96%E8%AF%91%E6%A8%A1%E5%9E%8B%EF%BC%9A%E6%8C%87%E6%A0%87.md)
- 4. [使用 model.fit() 进行训练](04-%E4%BD%BF%E7%94%A8%20model.fit%28%29%20%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83.md)
- 5. [使用 model.evaluate() 评估模型表现](05-%E4%BD%BF%E7%94%A8%20model.evaluate%28%29%20%E8%AF%84%E4%BC%B0%E6%A8%A1%E5%9E%8B%E8%A1%A8%E7%8E%B0.md)
- 6. [使用 model.predict() 进行预测](06-%E4%BD%BF%E7%94%A8%20model.predict%28%29%20%E8%BF%9B%E8%A1%8C%E9%A2%84%E6%B5%8B.md)
- 7. [训练时使用回调](07-%E8%AE%AD%E7%BB%83%E6%97%B6%E4%BD%BF%E7%94%A8%E5%9B%9E%E8%B0%83.md)
- 8. [使用 TensorBoard 可视化训练](08-%E4%BD%BF%E7%94%A8%20TensorBoard%20%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83.md)
- 9. [实践：训练与监控](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83%E4%B8%8E%E7%9B%91%E6%8E%A7.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models/quiz)

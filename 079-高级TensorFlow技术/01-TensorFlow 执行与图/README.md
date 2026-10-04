# 第 1 章：TensorFlow 执行与图

来源：[原章节](https://apxml.com/zh/courses/advanced-tensorflow/chapter-1-tensorflow-execution-graphs)

[返回课程目录](../README.md)

了解 TensorFlow 如何执行操作是编写高效和可扩展代码的根本。尽管 TensorFlow 2 默认使用即时执行（像 Python 一样命令式地运行操作），但创建和优化静态计算图对于获得高性能和支持部署仍然非常重要。本章将讲解这种相互关系。

你将学习到：
*   即时执行与图执行之间的区别和权衡。
*   `tf.function` 如何使用 AutoGraph 将 Python 代码转换为优化过的 TensorFlow 图。
*   函数追踪的机制以及 TensorFlow 如何在内部表示计算。
*   在图中实现控制流（例如，条件语句、循环）。
*   使用 `tf.GradientTape` 进行自动微分，它是梯度下降训练（如 $L = \sum (y_{pred} - y_{true})^2$）的驱动力。
*   TensorFlow 程序的基本资源管理考量和调试技术。

掌握这些执行细节将为后续性能调优、分布式训练和自定义组件开发提供所需的支持。

## 小节

- 1. [TensorFlow 的运行模式：即时执行与图执行](01-TensorFlow%20%E7%9A%84%E8%BF%90%E8%A1%8C%E6%A8%A1%E5%BC%8F%EF%BC%9A%E5%8D%B3%E6%97%B6%E6%89%A7%E8%A1%8C%E4%B8%8E%E5%9B%BE%E6%89%A7%E8%A1%8C.md)
- 2. [理解 tf.function 和 AutoGraph](02-%E7%90%86%E8%A7%A3%20tf.function%20%E5%92%8C%20AutoGraph.md)
- 3. [追踪机制与图表示](03-%E8%BF%BD%E8%B8%AA%E6%9C%BA%E5%88%B6%E4%B8%8E%E5%9B%BE%E8%A1%A8%E7%A4%BA.md)
- 4. [在图结构中实现控制流](04-%E5%9C%A8%E5%9B%BE%E7%BB%93%E6%9E%84%E4%B8%AD%E5%AE%9E%E7%8E%B0%E6%8E%A7%E5%88%B6%E6%B5%81.md)
- 5. [使用 tf.GradientTape 进行自动微分](05-%E4%BD%BF%E7%94%A8%20tf.GradientTape%20%E8%BF%9B%E8%A1%8C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86.md)
- 6. [管理资源和内存](06-%E7%AE%A1%E7%90%86%E8%B5%84%E6%BA%90%E5%92%8C%E5%86%85%E5%AD%98.md)
- 7. [调试 TensorFlow 程序](07-%E8%B0%83%E8%AF%95%20TensorFlow%20%E7%A8%8B%E5%BA%8F.md)
- 8. [实践：优化函数追踪](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BC%98%E5%8C%96%E5%87%BD%E6%95%B0%E8%BF%BD%E8%B8%AA.md)

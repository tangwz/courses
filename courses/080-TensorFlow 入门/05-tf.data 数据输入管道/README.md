# 第 5 章：tf.data 数据输入管道

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata)

[返回课程目录](../README.md)

高效的数据处理对于训练机器学习模型很重要，尤其是在处理大型数据集时，I/O可能成为主要的瓶颈。TensorFlow 的 `tf.data` API 提供了一种强大而灵活的方式，用于构建高性能的输入管道，将数据提取和转换与模型训练分离。

在本章中，你将学会高效地使用 `tf.data` API。我们将学习以下内容：
*   从多种来源创建 `tf.data.Dataset` 对象，包括内存中的数组（NumPy, Tensors）、Python 生成器，以及 TFRecord 等优化过的文件格式。
*   使用 `map()` 进行元素级预处理、`batch()` 进行数据分组、`shuffle()` 进行随机化以及 `prefetch()` 进行性能优化等方法，应用常见的数据转换。
*   将你的自定义 `tf.data` 管道方便地与 Keras `model.fit()` API 集成，用于训练和评估。
*   在管道中直接实现图像数据增强等技术。

完成本章后，你将能够为你的 TensorFlow 模型构建可扩展且高效的数据加载机制。

## 小节

- 1. [为什么选择 tf.data？](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%80%89%E6%8B%A9%20tf.data%EF%BC%9F.md)
- 2. [从张量、NumPy 和生成器创建数据集](02-%E4%BB%8E%E5%BC%A0%E9%87%8F%E3%80%81NumPy%20%E5%92%8C%E7%94%9F%E6%88%90%E5%99%A8%E5%88%9B%E5%BB%BA%E6%95%B0%E6%8D%AE%E9%9B%86.md)
- 3. [使用 TFRecord 文件](03-%E4%BD%BF%E7%94%A8%20TFRecord%20%E6%96%87%E4%BB%B6.md)
- 4. [应用转换：map()](04-%E5%BA%94%E7%94%A8%E8%BD%AC%E6%8D%A2%EF%BC%9Amap%28%29.md)
- 5. [批处理与混洗](05-%E6%89%B9%E5%A4%84%E7%90%86%E4%B8%8E%E6%B7%B7%E6%B4%97.md)
- 6. [为提高性能而预取](06-%E4%B8%BA%E6%8F%90%E9%AB%98%E6%80%A7%E8%83%BD%E8%80%8C%E9%A2%84%E5%8F%96.md)
- 7. [将 tf.data 与 model.fit() 结合使用](07-%E5%B0%86%20tf.data%20%E4%B8%8E%20model.fit%28%29%20%E7%BB%93%E5%90%88%E4%BD%BF%E7%94%A8.md)
- 8. [使用 tf.data 进行图像数据增强](08-%E4%BD%BF%E7%94%A8%20tf.data%20%E8%BF%9B%E8%A1%8C%E5%9B%BE%E5%83%8F%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA.md)
- 9. [动手实践：构建图像数据管道](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%9B%BE%E5%83%8F%E6%95%B0%E6%8D%AE%E7%AE%A1%E9%81%93.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/quiz)

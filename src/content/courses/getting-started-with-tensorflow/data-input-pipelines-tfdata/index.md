---
course: "getting-started-with-tensorflow"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata"
sourceId: 504
chapter: "data-input-pipelines-tfdata"
title: "tf.data 数据输入管道"
order: 5
description: "学习使用 tf.data API 为 TensorFlow 模型构建可扩展且高性能的数据输入管道。"
hasQuiz: true
---

高效的数据处理对于训练机器学习模型很重要，尤其是在处理大型数据集时，I/O可能成为主要的瓶颈。TensorFlow 的 `tf.data` API 提供了一种强大而灵活的方式，用于构建高性能的输入管道，将数据提取和转换与模型训练分离。

在本章中，你将学会高效地使用 `tf.data` API。我们将学习以下内容：
*   从多种来源创建 `tf.data.Dataset` 对象，包括内存中的数组（NumPy, Tensors）、Python 生成器，以及 TFRecord 等优化过的文件格式。
*   使用 `map()` 进行元素级预处理、`batch()` 进行数据分组、`shuffle()` 进行随机化以及 `prefetch()` 进行性能优化等方法，应用常见的数据转换。
*   将你的自定义 `tf.data` 管道方便地与 Keras `model.fit()` API 集成，用于训练和评估。
*   在管道中直接实现图像数据增强等技术。

完成本章后，你将能够为你的 TensorFlow 模型构建可扩展且高效的数据加载机制。

# 使用 model.predict() 进行预测

来源：[原文](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models/making-predictions-predict)

[返回章节目录](README.md) · [返回课程目录](../README.md)

机器学习 (machine learning)模型在经过训练（例如，使用 `model.fit()` 方法）并评估了其在单独测试集上的泛化性能（例如，使用 `model.evaluate()` 方法）之后，其主要目的是对新的、未见过的数据点进行预测。`model.predict()` 方法便用于实现此功能。它接收输入数据并返回模型的预测输出。

### 使用 `model.predict()`

`predict()` 方法使用简单。你将需要预测的输入数据传递给它，它会返回模型的输出。输入数据通常应为 NumPy 数组或 `tf.data.Dataset` 对象的形式，类似于你用于 `fit()` 或 `evaluate()` 的数据。

```python
# 假设 'model' 是一个已训练的 Keras 模型
# 假设 'new_data' 是包含新样本的 NumPy 数组或 tf.data.Dataset

predictions = model.predict(new_data)

print(predictions)
```

### 输入数据形状

一个常见的错误点是确保输入数据具有正确的形状。`model.predict()` 期望接收一批样本，即使你只预测一个实例。输入形状通常应与模型构建时使用的 `input_shape` 匹配，但要在开头增加一个批次维度。

例如，如果你的模型期望的输入图片形状为 (28, 28, 1)（比如 MNIST 数字），并且你想要预测存储在形状为 (28, 28, 1) 的 NumPy 数组 `single_image` 中的单张图片，你需要增加一个批次维度：

```python
import numpy as np
import tensorflow as tf

# 假设 'model' 期望的输入形状是 (None, 28, 28, 1)
# 假设 'single_image' 是形状为 (28, 28, 1) 的 NumPy 数组

# 使用 np.expand_dims 或切片增加批次维度
input_for_predict = np.expand_dims(single_image, axis=0) # 形状变为 (1, 28, 28, 1)
# 或者：input_for_predict = single_image[tf.newaxis, ...]

prediction_for_single = model.predict(input_for_predict)
print(f"传递给 predict 的输入形状：{input_for_predict.shape}")
print(f"预测输出：{prediction_for_single}")
```

如果你传递多个样本，它们应该沿第一个维度堆叠：

```python
# 假设 'multiple_images' 是形状为 (10, 28, 28, 1) 的 NumPy 数组
# （10 个样本，每个都是 28x28 灰度图）

predictions_for_multiple = model.predict(multiple_images)
print(f"多个图片预测的形状：{predictions_for_multiple.shape}")
# 输出形状将取决于模型的最后一层，例如，分类任务为 (10, 类别数)
```

你也可以直接将 `tf.data.Dataset` 传递给 `predict()`。这对于大型数据集通常很高效，因为它利用了管道的批处理和预取功能。

```python
# 假设 'new_dataset' 是一个生成新数据批次的 tf.data.Dataset
predictions_from_dataset = model.predict(new_dataset)
# 预测结果将被连接成一个 NumPy 数组
```

### 理解输出

`model.predict()` 返回的数组的结构和含义完全取决于你模型的架构，特别是其最后一层和激活函数 (activation function)。

- **回归：** 如果你的模型执行回归任务（例如，预测房价），输出通常是一个 NumPy 数组，其中每个元素（或每行，如果每个样本预测多个值）对应于相应输入样本的预测连续值。如果最后一层只有一个单元且没有激活函数（或线性激活），则 $N$ 个输入样本的输出形状将是 $(N, 1)$。

  ```python
  # 例子：预测每个输入的单个值
  # 输入形状：(N, 特征数)
  # 输出形状：(N, 1)
  print(predictions[0]) # 输出：[样本0的预测值]
  ```
- **二元分类：** 对于二元分类，最后一层通常有一个单元，带有 sigmoid 激活函数。`model.predict()` 将返回一个概率数组（0 到 1 之间的值），表示每个样本属于正类的概率。你通常会应用一个阈值（通常是 0.5）将这些概率转换为类别标签（0 或 1）。

  
  $$
  \text{预测类别} = \begin{cases} 1 & \text{如果 } \text{预测值} > 0.5 \\ 0 & \text{否则} \end{cases}
  $$
  

  N 个输入样本的输出形状将是 $(N, 1)$。

  ```python
  probabilities = model.predict(new_data) # 形状 (N, 1)
  predicted_classes = (probabilities > 0.5).astype("int32")
  print(f"概率：{probabilities[0]}") # 输出：例如，[0.92]
  print(f"预测类别：{predicted_classes[0]}") # 输出：[1]
  ```
- **多类别分类：** 对于具有 $C$ 个类别的多类别分类，最后一层通常有 $C$ 个单元，带有 softmax 激活函数。`model.predict()` 返回一个数组，其中每行包含对应输入样本在 $C$ 个类别上的概率分布。每行概率之和将近似为 1。为了获取预测的类别标签，你通常使用 `np.argmax` 找到最高概率的索引。

  
  $$
  \text{预测类别索引} = \underset{i}{\operatorname{argmax}}(\text{预测值}_i)
  $$
  

  N 个输入样本的输出形状将是 $(N, C)$。

  ```python
  probabilities_multi = model.predict(new_data) # 形状 (N, C)
  predicted_class_indices = np.argmax(probabilities_multi, axis=1)
  print(f"第一个样本的概率：{probabilities_multi[0]}") # 输出：例如，[0.1, 0.7, 0.2]
  print(f"第一个样本的预测类别索引：{predicted_class_indices[0]}") # 输出：1
  ```

### `predict()` 与直接调用 `model()`

你可能会注意到，你也可以通过直接将模型实例作为函数调用来获取预测结果：`predictions = model(new_data, training=False)`。虽然这种方法可行，但对于大型数据集的推理 (inference)，`model.predict()` 通常更受青睐。

- `model.predict()`：为推理而优化。它可以处理 NumPy 数组和 `tf.data.Dataset` 对象，以批次形式处理数据（对于 NumPy 数组可以通过 `batch_size` 参数 (parameter)指定，尽管它通常会推断或在内存允许的情况下处理所有数据），并且始终在推理模式下运行（例如，dropout 层不活跃，批标准化使用已学习的统计数据）。
- `model(data, training=False)`：更灵活的 TensorFlow 原生方式。它返回张量而非 NumPy 数组。它在自定义训练循环中或当你需要直接张量输出时很有用。显式设置 `training=False` 很重要，以确保 Dropout 和 BatchNormalization 等层在推理期间行为正确。

对于处理潜在大型输入的标准预测任务，`model.predict()` 通常是更方便且性能可能更好的选项。

进行预测是构建监督学习 (supervised learning)模型的最终目标。`model.predict()` 在 Keras 中提供了一个简单高效的接口，用于将你训练好的模型应用于新数据，从而利用它在训练期间学习到的模式。请记住，在将新数据输入 `model.predict()` 之前，始终以与训练数据完全相同的方式对其进行预处理。

## 参考资料

- [tf.keras.Model.predict](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict) — Keras team (2023)
  Publisher: TensorFlow
  提供Keras `predict` 方法的API规范和使用示例。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras和TensorFlow的综合指南，涵盖模型构建、训练、评估和推断，以及数据准备和输出解析。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  使用TensorFlow和Keras实现机器学习模型的实践指南，提供关于进行预测的详细示例。
- [Build TensorFlow input pipelines with tf.data](https://www.tensorflow.org/guide/data) — TensorFlow Authors (2023)
  详细说明使用 `tf.data.Dataset` 构建高效数据管道的方法，这对于将大型数据集送入 `model.predict()` 进行预测有所帮助。

---

[上一节](05-%E4%BD%BF%E7%94%A8%20model.evaluate%28%29%20%E8%AF%84%E4%BC%B0%E6%A8%A1%E5%9E%8B%E8%A1%A8%E7%8E%B0.md) · [下一节](07-%E8%AE%AD%E7%BB%83%E6%97%B6%E4%BD%BF%E7%94%A8%E5%9B%9E%E8%B0%83.md)

# 复杂模型的函数式API

来源：[原文](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-3-building-models-keras/functional-api-complex-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`Sequential` API 能够构建数据通过层堆栈线性流动的模型，这对于许多任务来说是直接的方法。然而，许多应用程序需要更复杂的架构。您可能需要以下模型：

- 接受多个输入（例如，图像和元数据）。
- 生成多个输出（例如，同时进行分类和回归）。
- 具有共享权重 (weight)的层。
- 包含非线性连接模式，例如残差连接（跳跃连接）。

对于这些情况，Keras 提供函数式API。它是一种更灵活的模型定义方式，您可以将层视为对张量进行操作的函数，并直接连接它们以构建图。

### 核心理念：层即函数

将 Keras 层实例看作一个可调用对象。您将输入张量（或多个张量）传递给它，它会返回一个输出张量（或多个张量）。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 示例：一个全连接层实例
dense_layer = layers.Dense(units=64, activation='relu')

# 我们首先需要一个输入张量（符号张量）
# 形状：(batch_size, input_features) - 这里 batch_size 通常为 None
input_tensor = keras.Input(shape=(784,))

# 在输入张量上调用该层
output_tensor = dense_layer(input_tensor)

print(f"Input Tensor Shape: {input_tensor.shape}")
print(f"Output Tensor Shape: {output_tensor.shape}")
```

`keras.Input` 对象创建一个符号张量状对象，表示模型的入口点。它定义了输入数据预期的形状和数据类型 (`dtype`)。然后，您通过依次调用层来连接它们，将一个层的输出张量作为输入张量传递给下一个层。

最后，通过指定模型的输入和输出，您可以定义一个 `keras.Model`。

```python
# 定义完整模型
model = keras.Model(inputs=input_tensor, outputs=output_tensor)

# 显示模型结构
model.summary()
```

这个使用函数式API的简单示例创建了与 `keras.Sequential([layers.Dense(64, activation='relu', input_shape=(784,))])` 完全相同的单层架构。其优点在超越简单线性堆栈时显现。

### 处理多个输入

设想您想构建一个模型，它根据工单的文本描述（可能通过嵌入 (embedding)和LSTM处理）和一些分类元数据（如工单类型、来源）来预测其优先级。函数式API让这易于实现。

1. 为每种输入定义一个 `Input` 层。
2. 为每个输入创建单独的处理分支。
3. 组合这些分支的输出（例如，使用 `layers.Concatenate`）。
4. 定义最终的输出层。
5. 使用输入列表和最终输出实例化 `Model`。

```python
# 定义输入
text_input = keras.Input(shape=(None,), dtype='int32', name='text') # 可变长度的整数序列
metadata_input = keras.Input(shape=(5,), dtype='float32', name='metadata') # 5个元数据特征

# 文本处理分支
text_features = layers.Embedding(input_dim=10000, output_dim=64)(text_input)
text_features = layers.LSTM(32)(text_features)

# 元数据处理分支（可选，可直接拼接）
metadata_features = layers.Dense(16, activation='relu')(metadata_input)

# 组合分支
combined_features = layers.Concatenate()([text_features, metadata_features])

# 输出层
priority_output = layers.Dense(1, activation='sigmoid', name='priority')(combined_features)

# 创建模型
multi_input_model = keras.Model(
    inputs=[text_input, metadata_input],
    outputs=priority_output
)

# 可视化模型结构（需要 pydot 和 graphviz）
# keras.utils.plot_model(multi_input_model, "multi_input_model.png", show_shapes=True)
```

> 一个具有两个独立输入分支并随后合并的模型架构。

当使用 `model.fit()` 训练此模型时，您将提供与定义输入相匹配的列表或字典形式的输入数据：

```python
# 虚拟数据生成（请替换为实际数据加载）
import numpy as np
num_samples = 100
dummy_text = np.random.randint(1, 10000, size=(num_samples, 50)) # 最大序列长度 50
dummy_metadata = np.random.rand(num_samples, 5)
dummy_priority = np.random.randint(0, 2, size=(num_samples, 1))

# 训练调用结构（示意性）
# multi_input_model.compile(optimizer='adam', loss='binary_crossentropy')
# multi_input_model.fit(
#     {'text': dummy_text, 'metadata': dummy_metadata}, # 输入字典
#     {'priority': dummy_priority}, # 输出字典
#     epochs=5,
#     batch_size=32
# )
# 或者，如果顺序一致，可以使用列表作为输入：
# multi_input_model.fit([dummy_text, dummy_metadata], dummy_priority, ...)
```

### 处理多个输出

同样，一个模型可能需要从相同的输入中预测多项内容。例如，一个图像分析模型可以对主要对象进行分类并预测其边界框坐标。

1. 定义输入和常用处理层。
2. 从中间层创建单独的输出分支层。
3. 使用输入和输出列表实例化 `Model`。

```python
# 输入
image_input = keras.Input(shape=(128, 128, 3), name='image')

# 共享卷积基
x = layers.Conv2D(32, 3, activation='relu')(image_input)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
base_output = layers.Flatten()(x) # 常见特征

# 分支1：分类头
class_output = layers.Dense(10, activation='softmax', name='class_label')(base_output)

# 分支2：边界框回归头
bbox_output = layers.Dense(4, name='bounding_box')(base_output) # 4个坐标：x, y, 宽度, 高度

# 创建模型
multi_output_model = keras.Model(
    inputs=image_input,
    outputs=[class_output, bbox_output]
)

# 可视化
# keras.utils.plot_model(multi_output_model, "multi_output_model.png", show_shapes=True)
```

> 一个模型，其共享卷积基分为两个输出头：分类和回归。

编译此模型时，通常为每个输出提供单独的损失函数 (loss function)，并可能指定损失权重 (weight)。

```python
# 虚拟数据
num_samples = 100
dummy_images = np.random.rand(num_samples, 128, 128, 3)
dummy_classes = np.random.randint(0, 10, size=(num_samples, 1))
dummy_classes_one_hot = tf.keras.utils.to_categorical(dummy_classes, num_classes=10)
dummy_bboxes = np.random.rand(num_samples, 4)

# 使用多个损失函数和可能的权重进行编译
# multi_output_model.compile(
#     optimizer='adam',
#     loss={
#         'class_label': 'categorical_crossentropy',
#         'bounding_box': 'mse' # 用于回归的均方误差
#     },
#     loss_weights={'class_label': 1.0, 'bounding_box': 0.5} # 示例权重
# )

# 训练调用结构（示意性）
# multi_output_model.fit(
#     {'image': dummy_images},
#     {'class_label': dummy_classes_one_hot, 'bounding_box': dummy_bboxes},
#     epochs=5,
#     batch_size=16
# )
# 或者，如果顺序一致，可以使用列表作为输出：
# multi_output_model.fit(dummy_images, [dummy_classes_one_hot, dummy_bboxes], ...)
```

### 共享层

函数式API直接支持层共享。您只需实例化一个层一次，然后在不同的输入上多次调用它。该层会在每次调用时重复使用相同的权重 (weight)集。这在孪生网络等模型中，或将相同的处理应用于不同输入时很常见。

```python
# 两个文本序列的输入张量
input_a = keras.Input(shape=(None,), dtype='int32', name='text_a')
input_b = keras.Input(shape=(None,), dtype='int32', name='text_b')

# 共享嵌入层
shared_embedding = layers.Embedding(input_dim=10000, output_dim=128, name='shared_embed')

# 将共享层应用于两个输入
encoded_a = shared_embedding(input_a)
encoded_b = shared_embedding(input_b)

# 示例：在一些处理（例如 LSTM）之后计算余弦相似度
lstm_layer = layers.LSTM(64, name='lstm') # 如果需要，也可以共享
vector_a = lstm_layer(encoded_a)
# 要共享 LSTM 权重：vector_b = lstm_layer(encoded_b)
# 要使用独立的 LSTM 权重：
lstm_layer_b = layers.LSTM(64, name='lstm_b')
vector_b = lstm_layer_b(encoded_b)

# 示例输出：余弦相似度（需要自定义层或 TensorFlow 操作）
# 仅为示意，此处仅做拼接
concatenated_vectors = layers.Concatenate()([vector_a, vector_b])
output = layers.Dense(1, activation='sigmoid', name='similarity')(concatenated_vectors)

# 创建模型
shared_layer_model = keras.Model(inputs=[input_a, input_b], outputs=output)

# 可视化
# keras.utils.plot_model(shared_layer_model, "shared_layer_model.png", show_shapes=True)
```

> 一个模型，为两个不同的文本输入使用单个 `Embedding` 层实例 (`shared_embed`)。请注意，这里使用独立的 LSTM 仅为示意，但它们*也*可以共享。

### 非线性拓扑：残差连接

深度学习 (deep learning)中一种常见模式是残差连接（或跳跃连接），ResNet 架构广泛使用它。它涉及将一个层块的输入添加到其输出中，有助于在训练期间更轻松地流动梯度并实现更深的网络。

```python
# 输入
input_tensor = keras.Input(shape=(32, 32, 3))

# 初始卷积
x = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)

# 残差块
residual = x # 存储该块的输入

x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same')(x) # 在相加之前不使用激活函数

# 添加残差连接
x = layers.Add()([x, residual])
x = layers.Activation('relu')(x) # 在相加之后应用激活函数

# 最终层（示例）
x = layers.GlobalAveragePooling2D()(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

# 模型
resnet_like_model = keras.Model(inputs=input_tensor, outputs=output_tensor)

# 可视化
# keras.utils.plot_model(resnet_like_model, "resnet_like_model.png", show_shapes=True)
```

> 一个简化的模型结构，演示了残差连接，其中 `Conv2D_Initial` 的输出被添加到 `Conv2D_2` 的输出中。

`layers.Add()` 层对一个张量列表（它们必须具有兼容的形状）执行逐元素相加。

函数式API提供了构建这些复杂模型架构的灵活性。虽然对于简单的线性堆栈而言，它比 `Sequential` API 略显冗长，但其定义复杂层图的功能使其在高级深度学习任务中作用显著。一旦使用函数式API定义了 `Model`，使用损失/优化器/指标进行编译以及使用 `fit()` 进行训练的过程将与您在下一章中学到的相同。

## 参考资料

- [Guide to the Keras Functional API](https://keras.io/guides/functional_api/) — fchollet (2023)
  Keras 复杂模型定义的官方指南，详细介绍多输入/多输出和共享层。
- [Deep Residual Learning for Image Recognition](https://ieeexplore.ieee.org/document/7780459) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2016)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 770-778; DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
  介绍残差连接，一种模型架构，作为函数式 API 的主要应用，支持训练深度神经网络。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  使用 Keras 和 TensorFlow 构建和训练深度学习模型的实用参考书，包含函数式 API 在多种架构中的示例。
- [tf.keras.Model class](https://www.tensorflow.org/api_docs/python/tf/keras/Model) — TensorFlow Authors (2024)
  Publisher: TensorFlow
  `tf.keras.Model` 类的文档，该类通过组合输入和输出，利用函数式 API 定义模型。

---

[上一节](04-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md) · [下一节](06-%E8%87%AA%E5%AE%9A%E4%B9%89%E5%B1%82%E5%92%8C%E6%A8%A1%E5%9E%8B%20%28%E7%AE%80%E4%BB%8B%29.md)

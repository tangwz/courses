---
course: "advanced-tensorflow"
chapter: "advanced-api-custom-components"
lesson: "custom-keras-layers"
sourceId: 2872
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-4-advanced-api-custom-components/custom-keras-layers"
title: "构建定制tf.keras层"
description: "学习通过继承tf.keras.layers.Layer来制作您自己的神经网络层。"
order: 2
plots: []
sourceHash: "447f23e105a526ce766937f747496e23632f31d2c7c5bbdbb584eaedb620f212"
sourceCorrections: []
---

Keras提供了一套丰富的内置层（如`Dense`、`Conv2D`、`LSTM`等），但您总会遇到需要执行特定、非标准操作的层的情况。这可能涉及实现研究论文中的新颖变换，以独特顺序组合现有操作，或制作带有定制逻辑的有状态层。TensorFlow允许您通过继承`tf.keras.layers.Layer`基类来定义自己的层。

## `tf.keras.layers.Layer`基类

核心来说，Keras层封装了两个主要部分：

1. **状态：** 层的权重 (weight)（可训练或不可训练的变量）。这些通常创建一次，并在训练期间更新（如果可训练）。
2. **计算：** 定义输入张量如何转换为输出张量的逻辑，如有必要，使用层的状态。这在`call`方法中定义。

要制作一个定制层，您需要继承`tf.keras.layers.Layer`，并且通常实现三个重要方法：

- `__init__(self, ...)`：构造函数。用它来定义层特定的配置参数 (parameter)（比如全连接层中的单元数量），并执行不依赖于输入形状的初始设置。始终首先调用`super().__init__(**kwargs)`。
- `build(self, input_shape)`：这个方法是使用`self.add_weight()`制作层权重（变量）的标准位置。它在Keras层首次被输入调用时自动调用。`input_shape`参数（一个`tf.TensorShape`对象或它们的结构）允许您创建维度取决于输入维度的权重。在大多数情况下，您不需要显式调用`super().build(input_shape)`，但要确保在此处创建权重。
- `call(self, inputs, ...)`：这个方法定义了层的前向传播逻辑。它接收输入张量（以及可能其他参数，例如用于在训练与推理 (inference)期间行为不同的层的`training`参数），并返回输出张量。层的所有计算都在这里进行。

### 为什么要用`build`？

您可能会想为什么权重 (weight)通常不在`__init__`中创建。使用`build`提供了灵活性。通常，层权重的确切形状取决于其输入的形状。例如，一个`Dense`层的核矩阵需要`input_shape[-1]`行。将权重创建推迟到`build`方法，意味着该层在实例化时不需要知道输入形状，只需在首次使用时知道。这简化了模型构建，尤其是在函数式或序贯式API中，输入形状可能无法立即得知。

> 定制Keras层的简化生命周期，展示了`__init__`、`build`和`call`通常的执行顺序。

## 示例：一个简单的全连接层

让我们实现一个简化版的`Dense`层来说明这些原理。我们的层将执行核心操作 $输出 = 激活(点积(输入, 核) + 偏置)$。

```python
import tensorflow as tf

class SimpleDense(tf.keras.layers.Layer):
    """一个基本的全连接层实现。

    参数：
        units: 正整数，输出空间的维度。
        activation: 要使用的激活函数。如果不指定，则不应用激活（即“线性”激活：`a(x) = x`）。
        name: 字符串，层的名称。
    """
    def __init__(self, units, activation=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        # 获取激活函数；tf.keras.activations.get 将字符串标识符或函数
        # 转换为激活函数。
        self.activation = tf.keras.activations.get(activation)
        # 我们将权重创建推迟到build方法

    def build(self, input_shape):
        """创建层的权重。"""
        # input_shape 是一个 tf.TensorShape 对象。
        # 输入形状的最后一个维度是输入特征的数量。
        input_dim = input_shape[-1]

        # self.add_weight 创建层的变量（权重）。
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer='glorot_uniform',  # 常用初始化器
            trainable=True)               # 权重可训练

        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',          # 将偏置初始化为零
            trainable=True)
        
        # 'built'属性在build()成功完成后自动设置为True。
        # 您通常不需要手动设置 self.built = True。

    def call(self, inputs):
        """定义层执行的计算。"""
        # 执行矩阵乘法
        z = tf.matmul(inputs, self.kernel)
        # 添加偏置
        z = z + self.bias
        # 如果指定，则应用激活函数
        if self.activation is not None:
            return self.activation(z)
        return z

    # 可选：实现get_config用于序列化
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            # 序列化激活函数标识符（例如，'relu'）
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config

    # 可选：实现from_config用于反序列化
    @classmethod
    def from_config(cls, config):
        # 反序列化激活函数
        config['activation'] = tf.keras.activations.deserialize(config['activation'])
        return cls(**config)
```

让我们分解`build`方法对`self.add_weight`的使用：

- `name`：为变量提供一个有意义的名称，有助于调试和保存。
- `shape`：定义了权重 (weight)张量的维度。这里，核的形状取决于`input_dim`（派生自`input_shape`）和`self.units`。偏置 (bias)的形状仅取决于`self.units`。
- `initializer`：指定如何初始化权重值（例如，`'zeros'`、`'glorot_uniform'`）。Keras提供了许多标准初始化器。
- `trainable`：一个布尔值，指示变量的值是否应在训练期间由优化器更新。大多数权重都是可训练的，但有时您可能需要不可训练的状态变量。

`call`方法使用标准的TensorFlow操作（`tf.matmul`、`+`）实现了实际的数学运算。它接收`inputs`张量并返回转换后的输出。

您现在可以像使用任何内置Keras层一样使用这个定制层：

```python
# 使用示例
# 制作一些模拟数据
input_data = tf.random.normal(shape=(32, 64)) # 32个样本的批次，每个样本有64个特征

# 实例化定制层
custom_dense_layer = SimpleDense(units=128, activation='relu')

# 在数据上调用该层（这会隐式触发build方法）
output_data = custom_dense_layer(input_data)

# 检查输出形状
print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output_data.shape}") # 应该是 (32, 128)

# 检查已创建的权重
print(f"Kernel shape: {custom_dense_layer.kernel.shape}") # 应该是 (64, 128)
print(f"Bias shape: {custom_dense_layer.bias.shape}") # 应该是 (128,)
```

## 序列化：`get_config`和`from_config`

对于包含您的定制层的模型要能够保存（例如，使用`model.save()`）和加载（`tf.keras.models.load_model()`），该层需要可序列化。这通常涉及实现`get_config`方法。

- `get_config(self)`：这个方法应该返回一个可JSON序列化的字典，其中包含重新制作层实例所需的配置参数 (parameter)。通常，您会调用父类的`get_config`并用您层特有的参数（如我们`SimpleDense`示例中的`units`和`activation`）更新结果字典。请注意，像激活函数 (activation function)这样复杂的对象应使用Keras工具（例如`tf.keras.activations.serialize`）进行序列化。

有时，如果您的层具有复杂的初始化逻辑或需要定制反序列化（例如仅将配置字典传递给构造函数），您可能还需要实现类方法`from_config(cls, config)`。Keras在模型加载期间使用`get_config`返回的字典调用此方法来重构层对象。通常，如果`get_config`实现正确，默认实现（即将`config`传递给构造函数`cls(**config)`）就足够了，但为了完整性，在示例中有所展示，特别是为了处理像激活函数这类对象的反序列化。

## 具有不同训练/推理 (inference)行为的层

某些层，如Dropout或Batch Normalization，在训练期间（例如，应用dropout，更新移动平均）的行为与推理期间（例如，禁用dropout，使用冻结统计数据）不同。Keras通过`call`方法中的可选`training`参数 (parameter)处理此问题。

如果您的定制层需要不同的行为，请在`call`中接受一个`training`参数：

```python
import tensorflow as tf

class CustomLayerWithTrainingArg(tf.keras.layers.Layer):
    # ... __init__ 和 build ...

    def call(self, inputs, training=None): # 接受 training 参数
        if training:
            # 训练期间的行为
            # 例如，应用dropout，更新状态
            print("在训练模式下执行")
            # 示例：简化的dropout
            # return tf.nn.dropout(inputs, rate=0.5) 
        else:
            # 推理期间的行为
            print("在推理模式下执行")
            # 示例：不应用dropout
            # return inputs 
        
        # 占位符返回值
        return inputs + 0.0 # 确保返回某些东西

# 当您使用 model.fit()、model.evaluate() 或 model.predict() 时，
# Keras会自动为 'training' 传递正确的布尔值。
```

当您使用标准的Keras训练循环（`model.fit()`）、评估（`model.evaluate()`）或预测（`model.predict()`）时，Keras会自动为您的层的`call`方法中的`training`参数传递正确的布尔值（`True`或`False`）。如果您正在编写定制训练循环，则需要显式传递此参数。

制作定制层是扩展TensorFlow能力的一个强大技术。通过理解`__init__`、`build`和`call`的作用，以及如何管理权重 (weight)和序列化，您可以实现您的特定机器学习 (machine learning)任务所需的几乎任何层架构。

## 参考资料

- [Custom layers and models](https://www.tensorflow.org/guide/custom_layers) — TensorFlow Authors (2024)
  Journal: TensorFlow Documentation; Publisher: Google
  本指南直接解释了如何通过子类化`Layer`创建自定义Keras层，涵盖了`__init__`、`build`和`call`方法。
- [Layer class](https://keras.io/api/layers/base_layer/) — Keras Team (2024)
  Journal: Keras Documentation
  `tf.keras.layers.Layer`基类的官方API参考，详细说明了其方法和属性。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本涵盖高级Keras技术（包括创建自定义层和模型）的全面书籍。

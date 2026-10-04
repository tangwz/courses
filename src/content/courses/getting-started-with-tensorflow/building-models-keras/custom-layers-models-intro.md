---
course: "getting-started-with-tensorflow"
chapter: "building-models-keras"
lesson: "custom-layers-models-intro"
sourceId: 1721
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-3-building-models-keras/custom-layers-models-intro"
title: "自定义层和模型 (简介)"
description: "简要介绍通过继承 Keras 类创建自定义层和模型的方法。"
order: 6
plots: []
sourceHash: "c31b34083ad444b9b37178104d82d8ff0a43f203a2852e3b0c5350eda18bf45f"
sourceCorrections: []
---

Keras 的 Sequential 和 Functional API 提供了便捷的方法来构建各种常见的模型架构，但你可能会遇到需要更大灵活性的情形。例如，你可能正在实现一篇研究论文中描述的新颖层，或者你需要一个具有复杂内部状态的层，再或者你的模型的前向传播逻辑无法很好地适应 Functional API 的标准有向无环图结构。

对于这些情况，TensorFlow 允许你通过继承 Keras 提供的基本类来创建 **自定义层** 和 **自定义模型**。这种面向对象的方法使你能够完全控制组件的行为。

### 创建自定义层

自定义层本质上是一个继承自 `tf.keras.layers.Layer` 的 Python 类。通过继承 `Layer`，你的自定义组件可以与 Keras 生态系统的其余部分顺畅地集成。你可以像使用任何内置层（例如 `Dense`、`Conv2D`）一样，在 Sequential 或 Functional 模型中使用它。

要创建自定义层，你通常需要实现三个方法：

1. `__init__(self, **kwargs)`: 这是构造函数。使用它来定义不依赖于输入形状的层特定属性。这包括超参数 (parameter) (hyperparameter)，例如单元数量、激活函数 (activation function)或正则化 (regularization)强度。首先调用 `super().__init__(**kwargs)`。此处应接受配置所需的任何参数。
2. `build(self, input_shape)`: 这个方法是定义层权重 (weight)（可训练变量）的地方。它会在层第一次使用时自动调用，因为权重形状通常取决于输入数据的形状，而输入数据的形状在层实例化时是未知的。在 `build` 方法内部使用 `self.add_weight()` 方法来创建权重。`input_shape` 参数（一个 `tf.TensorShape` 对象）提供了必要的信息。最佳做法是在此方法结束时设置 `self.built = True`。
3. `call(self, inputs, **kwargs)`: 这个方法定义了层的前向传播逻辑。它接收输入张量作为参数（以及可选的关键字参数，如 `training=None`，用于在训练和推理 (inference)期间行为不同的层，例如 `Dropout`），并返回输出张量。所有涉及输入和权重的计算都在这里进行。

下面是自定义层结构的草图：

```python
import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, config_param1, config_param2=None, **kwargs):
        super().__init__(**kwargs)
        # 存储不依赖于输入形状的配置参数
        self.config_param1 = config_param1
        self.config_param2 = config_param2
        # ... 其他初始化

    def build(self, input_shape):
        # 首次使用层时调用一次。
        # 使用 input_shape 来确定权重维度。
        # 示例：创建一个可训练的权重矩阵 'w'
        self.w = self.add_weight(
            name='kernel', # 一个有意义的名称
            shape=(input_shape[-1], self.config_param1), # 形状取决于输入特征和一个配置参数
            initializer='random_normal', # 如何初始化权重
            trainable=True # 该权重应在训练期间更新
        )
        # 示例：创建一个不可训练的偏置 'b'（不太常见，仅用于说明）
        self.b = self.add_weight(
            name='bias',
            shape=(self.config_param1,),
            initializer='zeros',
            trainable=False # 该权重不会被更新
        )
        # 将层标记为已构建
        self.built = True

    def call(self, inputs, training=None):
        # 使用 self.w、self.b 和 inputs 定义前向传播逻辑
        # 示例：一个简单的线性变换
        output = tf.matmul(inputs, self.w) + self.b
        # 如果在 __init__ 中指定了激活函数，则应用
        # if self.activation: output = self.activation(output)

        # 如果需要在训练与推理期间处理行为差异
        # if training:
        #     # 应用 dropout 等
        # else:
        #     # 使用推理行为

        return output

    # 可选：为序列化实现 get_config 方法
    def get_config(self):
        config = super().get_config()
        config.update({
            "config_param1": self.config_param1,
            "config_param2": self.config_param2,
        })
        return config
```

### 创建自定义模型

正如你可以通过继承 `tf.keras.layers.Layer` 来创建自定义层一样，你也可以通过继承 `tf.keras.Model` 来创建自定义模型。当你需要比 `Sequential` 或 Functional API 所提供的对模型架构或训练循环有更多控制时，这会很有用。

过程与创建自定义层相似：

1. `__init__(self, **kwargs)`: 将模型将要使用的层定义为类实例的属性。这些可以是标准的 Keras 层，也可以是你自己的自定义层。
2. `call(self, inputs, **kwargs)`: 定义整个模型的前向传播。你将在此处调用在 `__init__` 中定义的层，指定数据如何流经它们。

继承 `tf.keras.Model` 提供了最大的灵活性。你本质上是将模型的前向传播定义为一个 Python 函数。虽然你仍然可以将标准的 `model.compile()` 和 `model.fit()` 方法用于自定义模型，但继承 `Model` 也为编写完全自定义的训练循环提供了可能（尽管这是一个更高级的话题）。

下面是草图：

```python
class CustomModel(tf.keras.Model):
    def __init__(self, num_units_l1, num_units_l2, num_classes, **kwargs):
        super().__init__(**kwargs)
        # 定义模型将使用的层
        self.layer1 = tf.keras.layers.Dense(num_units_l1, activation='relu')
        self.layer2 = CustomLayer(num_units_l2) # 使用我们的自定义层
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None):
        # 使用这些层定义前向传播
        x = self.layer1(inputs)
        x = self.layer2(x, training=training) # 如果层需要训练标志，则传递
        outputs = self.classifier(x)
        return outputs

# 实例化自定义模型
# model = CustomModel(num_units_l1=128, num_units_l2=64, num_classes=10)
# 然后你可以像往常一样编译和训练这个模型
```

### 何时使用自定义组件

虽然在可能的情况下，通常应优先选择 Sequential 和 Functional API 的直接性，但在以下情况可以考虑创建自定义层或模型：

- 你需要实现一个标准 Keras 层中不具备操作的层。
- 你正在根据一篇研究论文重现一个使用非标准组件的架构。
- 你的层需要复杂的内部状态管理。
- 你的模型架构涉及复杂的连接、前向传播中的条件逻辑，或难以用 Functional API 表达的递归结构。
- 你需要对权重 (weight)的创建或前向传播中操作的精确序列进行细粒度控制。

本简介展示了 Keras 继承所提供的灵活性。虽然在本入门课程中我们不会实现复杂的自定义组件，但理解存在这种能力在你遇到更高级的机器学习 (machine learning)架构时是有价值的。对于大多数常见任务，Sequential 和 Functional API 将足够使用且通常更易于操作。

## 参考资料

- [Writing Custom Layers and Models](https://www.tensorflow.org/guide/custom_layers) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  TensorFlow官方指南，介绍如何通过继承`tf.keras.layers.Layer`实现自定义Keras层，以及通过继承`tf.keras.Model`实现自定义模型，涵盖了`__init__`、`build`、`call`和`get_config`等核心方法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本综合性的书籍，涵盖了使用TensorFlow和Keras构建和训练深度学习模型，包括创建自定义组件和架构的实践示例。

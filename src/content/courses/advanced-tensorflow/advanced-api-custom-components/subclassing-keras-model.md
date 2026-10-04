---
course: "advanced-tensorflow"
chapter: "advanced-api-custom-components"
lesson: "subclassing-keras-model"
sourceId: 2869
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-4-advanced-api-custom-components/subclassing-keras-model"
title: "通过继承tf.keras.Model获得灵活性"
description: "通过继承tf.keras.Model类来构建高度自定义的模型架构。"
order: 1
plots: []
sourceHash: "d7e3fa387cb0d061e5921a18b1f0b9edf5736fac0e6cf0f63d8f16b01e555c55"
sourceCorrections: []
---

Keras函数式API和序贯式API虽然提供了构建许多标准神经网络 (neural network)结构的便捷方式，但在处理高度定制化或动态的模型行为时，它们可能会变得笨重或不够用。对于需要对模型前向传播逻辑进行最大化灵活和控制的场景，TensorFlow提供了通过继承`tf.keras.Model`类来定义模型的能力。

这种方法将您的模型定义视为一个普通的Python类。您继承`tf.keras.Model`并实现必要的方法来定义模型的组成部分及其计算逻辑。这种命令式风格让您能够完全自由地在Python代码中直接实现精细的架构、条件逻辑或递归结构。

### 主要结构：`__init__` 和 `call`

当您继承`tf.keras.Model`时，需要实现两个基本方法：

1. **`__init__(self, ...)`**: 构造函数。您在此处定义模型将使用的所有层和子模块。重要的是在`__init__`中将层定义为模型实例的属性（例如，`self.my_layer = tf.keras.layers.Dense(...)`）。这可确保Keras能够自动追踪层的变量（权重 (weight)和偏置 (bias)）。您也可以在此处定义模型逻辑所需的其他常量或属性。
2. **`call(self, inputs, training=None, mask=None)`**: 前向传播方法。此方法包含模型的核心逻辑，定义输入如何转换为输出。您在此处使用在`__init__`中定义的层，将它们应用于输入张量或中间张量。`inputs`参数 (parameter)接收输入数据。可选的`training`参数是一个布尔值，表示模型是在训练模式还是推理 (inference)模式下运行。这对于`Dropout`或`BatchNormalization`这类在两种模式下行为不同的层来说很重要。一个好的做法是在`call`方法签名中包含`training`参数，并将其传递给该方法中使用的任何此类层。

### 一个基本示例：自定义多层感知机

我们使用子类化API来实现一个简单的多层感知机（MLP）来演示这个思路。

```python
import tensorflow as tf

class SimpleMLP(tf.keras.Model):

    def __init__(self, num_units_l1, num_units_l2, num_classes, name="simple_mlp", **kwargs):
        super().__init__(name=name, **kwargs)
        # 在构造函数中定义层
        self.dense_layer_1 = tf.keras.layers.Dense(num_units_l1, activation='relu')
        self.dropout_layer = tf.keras.layers.Dropout(0.5) # 需要`training`参数的层示例
        self.dense_layer_2 = tf.keras.layers.Dense(num_units_l2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None):
        # 定义前向传播逻辑
        x = self.dense_layer_1(inputs)
        # 将`training`参数传递给需要的层
        x = self.dropout_layer(x, training=training)
        x = self.dense_layer_2(x)
        outputs = self.output_layer(x)
        return outputs

# 实例化模型
mlp_model = SimpleMLP(num_units_l1=128, num_units_l2=64, num_classes=10)

# 构建模型（可选，首次调用时自动发生）
# 此步骤根据输入形状初始化权重
mlp_model.build(input_shape=(None, 784))

# 您可以查看模型
mlp_model.summary()

# 示例调用（前向传播）
# 创建一些模拟数据（批量大小4，特征大小784）
dummy_input = tf.random.normal([4, 784])
output_tensor = mlp_model(dummy_input, training=False) # 在推理时调用
print("Output shape:", output_tensor.shape)
```

在此示例中：

- 层（`Dense`、`Dropout`）在`__init__`中定义并作为实例属性存储。
- `call`方法决定了操作序列：输入 -> dense1 -> dropout -> dense2 -> 输出。
- `training`参数 (parameter)在`call`方法中显式传递给`Dropout`层。当您使用`model.fit()`、`model.evaluate()`或`model.predict()`时，Keras会自动处理为`training`提供正确的布尔值。

### 为何选择子类化？优点与应用场合

使用子类化API的主要原因是**灵活性**。它使您能够：

- **实现复杂的拓扑**: 构建层可重用、连接具有条件性，或图结构并非严格线性或无环的模型。这可例如编程定义的循环连接，或根据输入特性改变结构的模型。
- **使用原生Python控制流**: 您可以在`call`方法中使用Python的`if/else`语句、`for`循环，甚至调用外部Python函数。尽管`tf.function`（Keras自动使用）在将Python代码转换为图时会施加某些限制，但它能有效处理常见的控制流结构（参见第一章）。
- **管理内部状态**: 轻松地在模型类中定义和管理超出典型层权重 (weight)的状态。
- **提升复用性**: 将复杂的逻辑块封装到可复用的模型类中，有助于复杂系统的代码更加整洁。
- **简化调试**: 调试前向传播有时会更直接，因为您可以在`call`方法中单步调试Python代码，尤其是在即时执行模式下。

当函数式API感觉受到限制时，例如在实现新颖的研究思路、具有动态行为的模型，或前向传播过程中计算路径通过编程决定的架构时，应选择子类化。

### 子类化与函数式API对比

对比子类化方法与Keras函数式API会很有帮助：

| 特性 | 函数式API | 子类化API |
| --- | --- | --- |
| **定义** | 声明式：定义图结构 | 命令式：定义前向传播逻辑 |
| **灵活性** | 适用于静态、有向无环图 | 对动态/复杂逻辑具有高灵活性 |
| **架构** | 由层连接显式定义 | 由`call`方法隐式定义 |
| **可视化** | 易于模型绘图 (`tf.keras.utils.plot_model`) | 静态结构可视化可能更难 |
| **序列化** | 通常直接序列化 | `SavedModel`效果好；某些复杂Python逻辑可能需额外处理 |
| **调试** | 调试图的构建/执行 | 调试`call`中的Python代码（通常更简单） |

函数式API会事先创建模型的静态图表示。这个图易于检查、绘制和分析。子类化API通过`call`方法中执行的Python代码命令式地定义前向传播。虽然`tf.function`会将其编译成图以提高性能，但定义本身更具动态性。

> 函数式API与子类化API定义模型的方式对比。函数式API显式定义层图，而子类化API则在`call`方法中定义前向传播逻辑，使用在`__init__`中定义的层。

### `training`参数 (parameter)的处理

请记住，`call(self, inputs, training=None)`中的`training`参数很重要。Keras在`model.fit()`期间会自动将其设置为`True`，而在`model.evaluate()`或`model.predict()`期间设置为`False`。您*必须*将此参数传递给`call`方法中那些在训练和推理 (inference)期间行为不同的任何层，例如`tf.keras.layers.Dropout`或`tf.keras.layers.BatchNormalization`。忘记这样做是常见的错误来源，会导致模型在推理时行为异常（例如，Dropout仍然活跃）。

```python
# 在子类化模型的call方法内部：
def call(self, inputs, training=None):
    x = self.conv_layer(inputs)
    x = self.batch_norm_layer(x, training=training) # 正确：传递训练标志
    x = tf.nn.relu(x)
    x = self.dropout_layer(x, training=training)     # 正确：传递训练标志
    # ... 前向传播的其余部分
    return outputs
```

### 保存和加载子类化模型

通过子类化创建的模型通常可以使用`model.save()`和`tf.keras.models.load_model()`像序贯式或函数式模型一样保存和加载。这会将模型架构（通过检查`tf.function`追踪的`call`方法）、权重 (weight)和优化器状态保存为TensorFlow `SavedModel`格式。

但是，请注意序列化依赖于TensorFlow将`call`方法追踪成图的能力。如果您的`call`方法包含难以轻易追踪或严重依赖外部Python状态的复杂Python逻辑，那么保存和加载可能需要更仔细的处理或自定义序列化逻辑。对于大多数标准的深度学习 (deep learning)操作和控制流，`tf.function`和`SavedModel`都运行良好。

通过掌握子类化API，您将能够实现TensorFlow中几乎任何模型架构或行为，从而能够构建真正自定义的机器学习 (machine learning)解决方案。

## 参考资料

- [Custom layers and models with Keras](https://www.tensorflow.org/guide/keras/custom_layers_and_models) — fchollet (2024)
  Keras扩展的官方指南，包含如何子类化 `tf.keras.Model` 以完全控制模型架构和前向传播。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本关于Keras自定义模型的综合书籍，提供实践指导，涵盖实现细节和最佳实践。
- [Better performance with tf.function](https://www.tensorflow.org/guide/function) — TensorFlow authors (2024)
  解释 `tf.function` 的官方指南，它自动将 Python 代码转换为 TensorFlow 图以提高性能，对理解子类化模型如何优化很重要。

---
course: "advanced-tensorflow"
chapter: "advanced-api-custom-components"
lesson: "practice-custom-model-pipeline"
sourceId: 2889
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-4-advanced-api-custom-components/practice-custom-model-pipeline"
title: "动手实践：构建自定义模型管线"
description: "构建使用自定义层的模型，使用自定义损失函数进行训练，并实现自定义训练循环。"
order: 8
plots: []
sourceHash: "cbf24c124d991c2e6e57ba587b575ccb25e9b32e38e421737853e8dd84abcc0b"
sourceCorrections: []
---

让我们将本章的概念付诸实践。我们将构建一个小型机器学习 (machine learning)管线，其中包含几个定制组件：一个自定义层、一个通过子类化定义的自定义模型结构、一个自定义损失函数 (loss function)，以及一个手动实现的训练循环。本练习展示了如何组合这些元素，以便对模型的架构和训练过程获得精细的控制。

### 场景：带定制化的二分类

设想您有一个二分类问题，其中您需要特定类型的层交互以及旨在处理潜在类别不平衡或特定错误成本的损失函数 (loss function)。我们将使用合成数据进行模拟。

首先，让我们生成一些数据：

```python
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 生成合成数据（非线性可分）
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将y重塑为TensorFlow的列向量
y = y.reshape(-1, 1).astype(np.float32)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 转换为TensorFlow数据集
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Data sample: {X_train[0]}, Label: {y_train[0]}")
```

### 1. 创建自定义Keras层

让我们创建一个简单的自定义全连接层。尽管 Keras 提供了 `tf.keras.layers.Dense`，但构建我们自己的层有助于阐明子类化 `tf.keras.layers.Layer` 的机制。我们将其命名为 `MySimpleDense`。

```python
class MySimpleDense(tf.keras.layers.Layer):
    """一个用于演示的基本全连接层实现。"""
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        print(f"Initializing MySimpleDense with {units} units.")

    def build(self, input_shape):
        """创建层的权重。在层首次被使用时调用。"""
        input_dim = input_shape[-1]
        # 添加权重变量
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform", # Xavier均匀初始化器
            trainable=True,
            name="kernel" # 标准名称
        )
        # 添加偏置变量
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias" # 标准名称
        )
        print(f"Building MySimpleDense: Input shape {input_shape}, Weight shape {self.w.shape}")
        super().build(input_shape) # 确保父类的build方法被调用

    def call(self, inputs):
        """定义层的前向传播逻辑。"""
        # 矩阵乘法：inputs @ w
        z = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            return self.activation(z)
        return z

    def get_config(self):
        """允许序列化。"""
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation)
        })
        return config
```

要点：

- `__init__`: 存储配置，例如单元数量和激活函数 (activation function)。不创建权重 (weight)。
- `build`: 使用 `add_weight` 创建可训练权重（`w` 和 `b`）。此方法在层首次处理输入时由 Keras 自动调用，用于推断输入维度。
- `call`: 使用输入张量和已创建的权重定义层的计算。
- `get_config`: 对于保存和加载包含此自定义层的模型非常重要。

### 2. 子类化 `tf.keras.Model`

现在，我们将通过子类化 `tf.keras.Model` 来构建模型。这在定义前向传播时提供了最大灵活性。我们的模型将使用 `MySimpleDense` 层。

```python
class CustomClassifier(tf.keras.Model):
    """一个使用我们自定义全连接层的简单分类器模型。"""
    def __init__(self, num_hidden_units, name="custom_classifier", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_hidden_units = num_hidden_units
        # 在__init__中实例化层
        self.hidden_layer = MySimpleDense(num_hidden_units, activation="relu")
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid") # 用于输出的标准全连接层
        print("Initializing CustomClassifier model.")

    def call(self, inputs, training=None):
        """定义模型的前向传播逻辑。"""
        x = self.hidden_layer(inputs)
        # 如果需要，您可以在此处添加更复杂的逻辑
        return self.output_layer(x)

    # 可选：如果需要处理复杂的输入形状逻辑，可以定义 build，
    # 但通常 __init__ 和首次调用就足够了。

    # 可选：如果不使用自定义循环，可以自定义 train_step, test_step, predict_step
    # （我们将在下面使用自定义循环，因此此处不覆盖这些方法）

    def get_config(self):
        """允许序列化。"""
        config = super().get_config()
        config.update({"num_hidden_units": self.num_hidden_units})
        return config

    @classmethod
    def from_config(cls, config):
        # 如果需要，需要处理自定义层的反序列化
        # 对于此类简单情况，Keras 可能会自动处理
        # 如果自定义层已注册或通过 custom_objects 传递
        return cls(**config)

# 实例化模型
model = CustomClassifier(num_hidden_units=10)

# 通过调用一次来构建模型（或使用 model.build）
# 这会触发内部层的 build 方法
_ = model(tf.keras.Input(shape=(X_train.shape[1],)))
model.summary()
```

在此，我们在 `__init__` 中定义层，并在 `call` 方法中指定数据如何流经它们。`model.summary()` 确认我们的自定义层是架构的一部分。

### 3. 实现自定义损失函数 (loss function)

让我们定义一个简单的自定义损失函数。我们将手动实现一个基本的二元交叉熵。尽管 `tf.keras.losses.BinaryCrossentropy` 存在，但这展示了该过程。

```python
def manual_binary_crossentropy(y_true, y_pred):
    """手动计算二元交叉熵损失。"""
    # 添加一个小的 epsilon 以防止 log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # 计算正例的损失项
    loss_pos = y_true * tf.math.log(y_pred)
    # 计算负例的损失项
    loss_neg = (1. - y_true) * tf.math.log(1. - y_pred)

    # 组合并计算批次的平均损失
    loss = -tf.reduce_mean(loss_pos + loss_neg)
    return loss

# 使用虚拟数据的示例用法：
y_true_ex = tf.constant([[1.], [0.], [1.], [0.]], dtype=tf.float32)
y_pred_ex = tf.constant([[0.9], [0.2], [0.8], [0.1]], dtype=tf.float32)
loss_value = manual_binary_crossentropy(y_true_ex, y_pred_ex)
print(f"\nCustom Loss Example: {loss_value.numpy()}")

# 与Keras实现比较（应该非常接近）
bce = tf.keras.losses.BinaryCrossentropy()
keras_loss_value = bce(y_true_ex, y_pred_ex)
print(f"Keras BCE Loss Example: {keras_loss_value.numpy()}")
```

此函数接收真实标签和预测值，逐项计算交叉熵，并对批次求平均。它遵循标准定义。对于涉及层权重 (weight)或内部模型状态的更复杂损失，您可以子类化 `tf.keras.losses.Loss`。

### 4. 编写自定义训练循环

现在，我们使用 `tf.GradientTape` 组织训练过程。这使我们能够对每个步骤进行明确控制。

```python
# 超参数
learning_rate = 0.01
epochs = 20

# 优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 要跟踪的指标
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
test_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

# 核心训练步骤，使用 tf.function 装饰以提高性能
@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        # 前向传播
        predictions = model(features, training=True)
        # 使用我们的自定义函数计算损失
        loss = manual_binary_crossentropy(labels, predictions)
        # 添加模型/层可能产生的正则化损失
        if model.losses: # 如果层添加了正则化损失，这一点很重要
          loss += tf.add_n(model.losses)

    # 计算梯度
    gradients = tape.gradient(loss, model.trainable_variables)
    # 应用梯度以更新权重
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 更新训练指标
    train_loss_metric.update_state(loss)
    train_accuracy_metric.update_state(labels, predictions)

# 测试/评估步骤
@tf.function
def test_step(features, labels):
    # 推理模式下的前向传播
    predictions = model(features, training=False)
    # 计算损失
    loss = manual_binary_crossentropy(labels, predictions)

    # 更新测试指标
    test_loss_metric.update_state(loss)
    test_accuracy_metric.update_state(labels, predictions)

# 用于存储每个 epoch 指标的历史字典
history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

# 主训练循环
print("\nStarting Custom Training Loop...")
for epoch in range(epochs):
    # 在每个 epoch 开始时重置指标
    train_loss_metric.reset_state()
    train_accuracy_metric.reset_state()
    test_loss_metric.reset_state()
    test_accuracy_metric.reset_state()

    # 遍历训练批次
    for batch_features, batch_labels in train_dataset:
        train_step(batch_features, batch_labels)

    # 遍历测试批次进行验证
    for batch_features, batch_labels in test_dataset:
        test_step(batch_features, batch_labels)

    # 获取指标结果
    epoch_loss = train_loss_metric.result()
    epoch_acc = train_accuracy_metric.result()
    epoch_val_loss = test_loss_metric.result()
    epoch_val_acc = test_accuracy_metric.result()

    # 存储历史记录
    history['loss'].append(epoch_loss.numpy())
    history['accuracy'].append(epoch_acc.numpy())
    history['val_loss'].append(epoch_val_loss.numpy())
    history['val_accuracy'].append(epoch_val_acc.numpy())

    # 打印进度
    print(f"Epoch {epoch + 1}/{epochs} - "
          f"Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f} - "
          f"Val Loss: {epoch_val_loss:.4f} - Val Accuracy: {epoch_val_acc:.4f}")

print("Custom Training Loop Finished.")
```

自定义循环的重要方面：

- **`tf.GradientTape`：** 记录在其上下文 (context)中执行的操作，以实现自动微分。
- **前向传播：** `model(features, training=True)` 执行模型的 `call` 方法。将 `training=True` 设置为重要，因为像 Dropout 或 BatchNormalization 这样的层在训练和推理 (inference)期间行为不同。
- **损失计算：** 使用我们的 `manual_binary_crossentropy` 函数。我们还会检查并添加模型或其层中定义的任何正则化 (regularization)损失（`model.losses`）。
- **梯度计算：** `tape.gradient(loss, model.trainable_variables)` 计算损失相对于模型可训练参数 (parameter)的梯度。
- **权重 (weight)更新：** `optimizer.apply_gradients()` 根据优化器算法（在此例中是 Adam）应用计算出的梯度来更新模型的权重。
- **指标：** `tf.keras.metrics` 对象用于在批次和周期之间累积统计信息（如平均损失或准确率）。请记住在每个周期开始时调用 `reset_state()`。
- **`@tf.function` 装饰器：** 将 Python 函数（`train_step`、`test_step`）编译为可调用的 TensorFlow 图。这通常通过减少 Python 开销和启用图优化来提供明显的性能提升。

### 可视化训练进度

我们可以使用 `history` 字典来绘制训练和验证指标。

```python
epochs_range = range(1, epochs + 1)

# 绘制损失
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['loss'], label='训练损失', color='#1c7ed6', marker='o')
plt.plot(epochs_range, history['val_loss'], label='验证损失', color='#f76707', marker='x')
plt.title('训练和验证损失')
plt.xlabel('周期')
plt.ylabel('损失')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 绘制准确率
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['accuracy'], label='训练准确率', color='#1c7ed6', marker='o')
plt.plot(epochs_range, history['val_accuracy'], label='验证准确率', color='#f76707', marker='x')
plt.title('训练和验证准确率')
plt.xlabel('周期')
plt.ylabel('准确率')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
```

> 训练和验证损失以及准确率随周期变化的曲线。

此可视化有助于评估模型收敛性并识别潜在的过拟合 (overfitting)（即训练性能持续提高，但验证性能停滞或恶化）。

### 总结

本次实践练习展示了如何整合几项高级 TensorFlow/Keras 功能：

1. 我们通过子类化 `tf.keras.layers.Layer` 定义了 `MySimpleDense` 层，管理其权重 (weight)并定义其前向传播。
2. 我们通过子类化 `tf.keras.Model` 创建了 `CustomClassifier` 模型，结合了我们的自定义层并定义了模型的结构。
3. 我们实现了一个 `manual_binary_crossentropy` 函数，展示了如何整合自定义损失计算。
4. 我们使用 `tf.GradientTape` 构建了一个自定义训练循环，明确控制梯度计算、权重更新和指标跟踪。

掌握这些技术为实现高度定制的架构、损失函数 (loss function)和训练过程提供了基础，这些过程是尖端研究或特殊应用需求所必需的，超出了标准 `model.fit()` 工作流程。

## 参考资料

- [Writing custom layers and models](https://www.tensorflow.org/guide/keras/custom_layers_and_models) — TensorFlow Authors (2024)
  官方指南，详细说明如何子类化 `tf.keras.layers.Layer` 和 `tf.keras.Model`，包括实现 `build`、`call` 和 `get_config` 方法以实现高级模型定制。
- [Custom training loops](https://www.tensorflow.org/guide/advanced_training) — TensorFlow Authors (2024)
  官方 TensorFlow 指南，说明如何实现自定义训练循环，涵盖用于自动微分的 `tf.GradientTape`、用于性能优化的 `tf.function` 以及用于跟踪训练进度的 `tf.keras.metrics`。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用书籍，为在 TensorFlow 2 和 Keras 中构建自定义组件和实现自定义训练循环提供了大量示例和清晰说明。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  由 Keras 创始人撰写，本书提供了使用 Keras 构建和自定义深度学习模型（包括自定义层和模型）的权威理论和实践说明。

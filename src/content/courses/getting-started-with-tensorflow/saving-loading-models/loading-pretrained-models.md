---
course: "getting-started-with-tensorflow"
chapter: "saving-loading-models"
lesson: "loading-pretrained-models"
sourceId: 1797
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-6-saving-loading-models/loading-pretrained-models"
title: "加载预训练模型"
description: "如何加载已提前保存了权重或完整结构的模型。"
order: 6
plots: []
sourceHash: "923f14f92603e19e9dba0dbcd8dd77449ec8768dff5e7a3f4af1541b4d581892"
sourceCorrections: []
---

为了高效地使用训练好的机器学习 (machine learning)模型，加载它们的能力是必不可少的。这个过程允许用户检索模型的权重 (weight)或其完整状态，使其可用于各种应用。常见的流程包括使用模型进行预测（推理 (inference)）、在新数据集上微调 (fine-tuning)模型，或者恢复中断的训练过程。TensorFlow 和 Keras 提供了简单易用的函数来处理这些情况。

### 将权重 (weight)加载到现有模型架构中

通常，您可能拥有定义模型架构的 Python 代码，但需要将之前训练好的权重加载到其中。当您使用 `model.save_weights()` 或配置为仅保存权重的 `ModelCheckpoint` 回调（例如 `save_weights_only=True`）时，这种情况很常见。

主要方法是 `tf.keras.Model.load_weights()`。它将保存的权重文件的路径作为主要参数 (parameter)。

**重要提示：** 要使 `load_weights()` 成功，您的代码中定义的模型架构*必须*与保存权重时所用模型的架构完全一致。这包括层数、层类型、层顺序以及每个层的配置（如单元数、激活函数 (activation function)等）。如果架构不匹配，TensorFlow 通常会引发错误，因为它不知道如何将保存的权重张量映射到当前模型中的层。

假设您定义了一个简单的 Sequential 模型，并且权重已保存到名为 `'my_model_weights.weights.h5'` 的文件中。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. 定义*完全相同*的模型架构
def build_simple_model():
  model = keras.Sequential([
      layers.Dense(64, activation='relu', input_shape=(784,)),
      layers.Dropout(0.2),
      layers.Dense(10, activation='softmax')
  ])
  # 注意：通常在加载权重*后*编译模型
  # 如果您只进行推理。如果恢复训练，则在加载前编译。
  return model

# 创建模型实例
model = build_simple_model()

# 打印摘要以查看架构（可选）
# model.summary()

# 保存的权重文件路径
weights_filepath = 'my_model_weights.weights.h5'

# 2. 加载权重
try:
  model.load_weights(weights_filepath)
  print(f"成功从 {weights_filepath} 加载了权重")

  # 现在模型实例已包含训练好的参数
  # 您可以继续进行评估或预测
  # model.compile(...) # 如果评估/训练需要，则编译
  # loss, acc = model.evaluate(x_test, y_test, verbose=0)
  # print(f"恢复的模型准确率: {acc:.2f}")

except Exception as e:
  print(f"加载权重出错: {e}")
  print("确保模型架构与保存的权重匹配。")
```

这种方法很灵活，因为它将模型定义（代码）与学习到的参数（权重文件）分开。

### 加载整个模型（架构、权重 (weight)和优化器状态）

如果您使用 `model.save(filepath)` 保存了整个模型，您不仅保存了权重，还保存了模型的架构以及可能的优化器状态。这通常以 TensorFlow 的 SavedModel 格式（一个目录）或单个 HDF5 文件（`.h5` 或 `.keras`）存储。

要加载这样一个完整的模型，您可以使用顶层函数 `tf.keras.models.load_model()`。

```python
import tensorflow as tf

# 保存的模型目录或文件路径
saved_model_path = 'my_full_model_savedmodel' # 或者 'my_full_model.h5' / 'my_full_model.keras'

try:
  # 加载整个模型
  loaded_model = tf.keras.models.load_model(saved_model_path)
  print(f"成功从 {saved_model_path} 加载了模型")

  # loaded_model 对象是一个已编译的 Keras 模型，可以直接使用。
  loaded_model.summary()

  # 您可以立即将其用于评估或预测
  # loss, acc = loaded_model.evaluate(x_test, y_test, verbose=0)
  # print(f"加载模型的准确率: {acc:.2f}")

  # 或者如果优化器状态已保存，甚至可以继续训练
  # history = loaded_model.fit(x_train, y_train, epochs=5, ...)

except Exception as e:
  print(f"加载模型出错: {e}")
```

这里的一个显著好处是您不需要定义模型架构的原始 Python 代码。`load_model` 根据保存的文件或目录中存储的信息重建模型。如果保存了训练配置（优化器、损失、指标），它还会一并恢复，使得恢复训练或使用模型创建者设定的精确设置变得方便。

**处理自定义对象：** 如果您的已保存模型包含不属于标准 TensorFlow/Keras 库的自定义层、自定义损失函数 (loss function)或自定义激活函数 (activation function)，`load_model` 可能会失败，因为它不知道如何解释这些自定义组件。为处理此问题，您可以向 `load_model` 传递一个 `custom_objects` 字典，将自定义对象的保存名称映射到其对应的 Python 类或函数。

```python
# 假设您定义了一个自定义层类：MyCustomLayer

# 加载时：
# loaded_model = tf.keras.models.load_model(
#     saved_model_path,
#     custom_objects={'MyCustomLayer': MyCustomLayer}
# )
```

或者，您可以使用 `tf.keras.utils.register_keras_serializable` 来全局注册您的自定义对象。

### 总结：`load_weights` 对比 `load_model`

- 当满足以下条件时，使用 `model.load_weights(filepath)`：
  - 您拥有定义模型架构的 Python 代码。
  - 您只保存了模型的权重 (weight)（参数 (parameter)）。
  - 您想将预训练 (pre-training)权重加载到可能已修改的架构中（例如，用于迁移学习 (transfer learning)，确保加载权重时层名称/结构对齐 (alignment)）。
- 当满足以下条件时，使用 `tf.keras.models.load_model(filepath)`：
  - 您保存了整个模型（架构、权重、优化器状态）。
  - 您需要一个即用型模型，而无需原始定义代码。
  - 您打算部署模型（特别是使用 SavedModel 格式时）。
  - 您想从上次停止的地方精确地恢复训练。

成功加载预训练模型可以帮助您在现有成果上继续，节省大量时间和计算资源。这是有效应用深度学习 (deep learning)的一项基本技能。正如我们将在下一节中简要看到的，TensorFlow Hub 等平台进一步简化了访问和重用由更广泛社区训练的模型的过程。

## 参考资料

- [Save and load Keras models](https://www.tensorflow.org/guide/keras/save_and_serialize) — Neel Kovelamudi, Francois Chollet (2024)
  官方指南，解释Keras模型的保存和加载，涵盖仅保存权重和完整模型序列化。
- [tf.keras.models.load_model](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model) — TensorFlow Authors (2024)
  Publisher: Google
  加载完整Keras模型的直接API文档，包括自定义对象的细节。
- [tf.keras.Model.load_weights](https://www.tensorflow.org/api_docs/python/tf/keras/Model#load_weights) — TensorFlow Authors (2024)
  将权重加载到现有Keras模型的直接API文档。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  一本全面的书籍，包含TensorFlow和Keras模型保存和加载的实践示例。第三版。

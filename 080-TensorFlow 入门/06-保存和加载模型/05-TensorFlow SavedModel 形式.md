# TensorFlow SavedModel 形式

来源：[原文](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-6-saving-loading-models/tensorflow-savedmodel-format)

[返回章节目录](README.md) · [返回课程目录](../README.md)

保存检查点或仅保存模型权重 (weight)在开发和训练时很有用，但部署模型通常需要更全面、标准化的格式。TensorFlow 为此提供了 `SavedModel` 这种格式。它是保存一个完整的 TensorFlow 程序（包括模型架构、训练好的权重以及计算图本身）的推荐方式，这种格式是语言无关且可恢复的。

可以将 `SavedModel` 看作是训练好的模型的自包含包。它存储权重并捕获执行推断所需的实际 TensorFlow 图操作。这使其非常适合于可能没有用于定义模型的原始 Python 代码的环境，例如：

- **TensorFlow Serving：** 一个为在生产环境中服务机器学习 (machine learning)模型而设计的高性能系统。
- **TensorFlow Lite：** 用于在移动设备、微控制器和其他边缘设备上部署模型。
- **TensorFlow.js：** 用于在网页浏览器或 Node.js 应用中直接运行模型。
- 与可能使用不同编程语言或环境的其他人共享模型。

### 以 SavedModel 形式保存

以 SavedModel 形式保存 Keras 模型很简单。如果你使用 `model.save()` 方法并提供一个没有 `.h5` 或 `.weights.h5` 扩展名的目录路径，TensorFlow 会默认采用 SavedModel 形式。

```python
import tensorflow as tf
# 假设 'model' 是你训练好的 Keras 模型
# model = tf.keras.Sequential([...])
# model.compile(...)
# model.fit(...)

# 以 SavedModel 形式保存模型
model.save("my_first_savedmodel")
```

此命令会创建一个名为 `my_first_savedmodel` 的目录（或你指定的任何路径）。与仅保存权重 (weight)（通常会创建带 `.weights.h5` 或 `.keras` 扩展名的一个或多个文件）不同，以 SavedModel 形式保存会创建特定的目录结构。

### SavedModel 目录的结构

让我们查看 `model.save()` 创建的目录内容：

- **`saved_model.pb`**：这是 SavedModel 的核心。它是一个协议缓冲区文件，包含序列化的 `MetaGraphDef`。每个 `MetaGraphDef` 都包含图结构（即 `GraphDef`）、变量信息、资产详情，以及重要的模型签名（我们接下来会介绍这些）。
- **`variables/`**：这个子目录保存了模型变量（权重 (weight)和偏差）的训练值。数据通常被分片到多个文件（`variables.data-xxxxx-of-yyyyy` 和 `variables.index`）中，以便高效加载。
- **`assets/`**：一个可选目录。如果你的模型依赖外部文件（如文本处理的词汇文件或查找表），它们会被复制到这里。这保证了 SavedModel 的自包含性。
- **`fingerprint.pb`**：包含用于标识 SavedModel 形式版本和创建者的元数据。
- **`keras_metadata.pb`**：（可选，但从 Keras 保存时通常会存在）此文件存储有关模型架构、损失函数 (loss function)、优化器状态和指标的 Keras 特有信息。这使得 `tf.keras.models.load_model` 能够完美重建原始 Keras 模型对象，从而可以使用 Keras API 进行进一步的训练或修改。

```text
my_first_savedmodel/
├── assets/
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
├── fingerprint.pb
├── keras_metadata.pb  # Keras 特有信息
└── saved_model.pb     # 计算图和签名
```

### 签名：定义模型端点

SavedModel 的一个重要特点是**签名**的理念。签名定义了模型导出的一个特定函数或计算。它指定了特定任务（通常是推断）的预期输入张量和结果输出张量。可以将它们视为模型计算图的定义入口点。

当你保存 Keras 模型时，它通常会自动导出一个名为 `serving_default` 的默认签名。这个签名通常对应于模型的正向传播（`call` 方法或 `predict` 行为），接收模型的输入并产生其输出。

对于像 TensorFlow Serving 这样的部署系统，这些签名是必不可少的。它们准确地告诉服务系统如何与加载的模型交互以获取预测。你还可以使用 `tf.function` 为不同的功能定义自定义签名，并在保存标准 Keras `Model` 类之外构建的模型时或在导出特定预处理步骤时指定输入签名。

### 加载 SavedModel

你可以使用 `tf.keras.models.load_model()` 加载以此形式保存的模型：

```python
# 从目录加载模型
loaded_model = tf.keras.models.load_model("my_first_savedmodel")

# 验证其是否为相同类型的对象
print(type(loaded_model))
# <class 'keras.src.engine.sequential.Sequential'> (or Functional, etc.)

# 你现在可以用它进行预测、评估，甚至继续训练
# predictions = loaded_model.predict(new_data)
```

由于包含了 `keras_metadata.pb` 文件，`load_model` 成功地重建了原始的 Keras `Sequential`（或 `Functional`）模型对象，其中包含其架构、权重 (weight)，如果保存了，甚至还包含优化器状态。

另外，你可以使用更低层次的 `tf.saved_model.load()` 函数：

```python
loaded_generic = tf.saved_model.load("my_first_savedmodel")

# 这会返回不同类型的对象
print(type(loaded_generic))
# <class 'tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject'>

# 访问默认的服务签名
inference_func = loaded_generic.signatures["serving_default"]

# 使用签名进行推断（需要 Tensor 格式的输入）
# output_tensor = inference_func(tf.constant(input_data_as_numpy))['output_layer_name']
```

使用 `tf.saved_model.load()` 会得到一个更通用的 TensorFlow 对象。如果你不需要完整的 Keras 模型结构，这会很有用，例如将其集成到非 Keras 的 TensorFlow 代码中，或者在 Keras 不可用的部署环境中。你主要通过其定义的签名来与之交互。

总之，SavedModel 形式是 TensorFlow 用于序列化模型以进行部署和共享的标准。它将图、权重和资产打包成一种自包含、语言无关的形式，使其与 TensorFlow Serving、Lite 和 JS 兼容。尽管在开发过程中保存权重或使用检查点很方便，但 SavedModel 是将模型投入实际应用的优选。

## 参考资料

- [TensorFlow SavedModel Guide](https://www.tensorflow.org/guide/saved_model) — TensorFlow Authors (2024)
  介绍了保存完整TensorFlow程序的标准格式，涵盖其结构、组件和主要应用。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的书籍，介绍模型保存和部署策略，包括TensorFlow SavedModel。

---

[上一节](04-%E4%BF%9D%E5%AD%98%E6%95%B4%E4%B8%AA%E6%A8%A1%E5%9E%8B%EF%BC%88%E6%9E%B6%E6%9E%84%20%2B%20%E6%9D%83%E9%87%8D%20%2B%20%E4%BC%98%E5%8C%96%E5%99%A8%E7%8A%B6%E6%80%81%EF%BC%89.md) · [下一节](06-%E5%8A%A0%E8%BD%BD%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B.md)

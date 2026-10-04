# 特征工程与 Transform

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-5-production-ml-pipelines-tfx/tfx-feature-engineering-transform)

[返回章节目录](README.md) · [返回课程目录](../README.md)

数据处理的一致性是可靠机器学习 (machine learning)系统的根基。为模型准备特征是数据摄取和验证之后的一个主要步骤，这些过程通常由 `ExampleGen`、`StatisticsGen` 和 `SchemaGen` 等组件处理。在训练和服务阶段对特征应用完全相同的转换非常必要，以避免模型遇到数据时性能下降。根据训练集计算的统计量（例如使用训练集的最小值和最大值进行缩放）应用转换时，即使一次处理一个样本，在转换用于推断的数据时也必须使用 *完全相同* 的统计量。任何差异都会导致训练/服务偏差，这是生产机器学习中常见问题的起因。

TFX `Transform` 组件专门设计用于处理此问题，它将特征预处理逻辑直接嵌入 (embedding)到导出的模型图中。它确保了在训练期间用于特征工程的相同 TensorFlow 代码在评估和服务期间得到一致应用。

### TFX Transform 的作用

TFX `Transform` 将 `ExampleGen` 输出的原始数据和 `SchemaGen` 推断的模式作为主要输入。其主要作用是执行特征工程操作，例如：

- **归一化 (normalization)和缩放：** 根据对整个训练数据集计算的统计量，应用 z-score 归一化 ($(x - \mu) / \sigma$) 或 Min-Max 缩放 ($(x - min) / (max - min)$) 等技术。
- **词汇表 (vocabulary)生成：** 根据对训练数据的频率分析，创建从类别特征值（字符串）到整数索引的映射。
- **嵌入 (embedding)：** 将稀疏类别特征（通常由词汇表中的索引表示）转换为密集向量 (vector) (dense vector)表示。
- **特征交叉：** 通过组合现有特征来创建新的交互特征（例如，交叉“纬度”和“经度”的桶）。
- **文本处理：** 应用分词 (tokenization)、TF-IDF 计算或生成 n-grams 等技术。
- **缺失值处理：** 实现填充策略（例如，用从训练数据计算的均值或中位数替换缺失的数值）。

`Transform` 的亮点在于它应用这些转换的方式。它不仅转换训练数据；它会生成一个可复用的 TensorFlow 图，通常称为 `transform_graph` 或 `transform_fn`，该图封装了预处理逻辑 *以及任何必要的计算统计量*（如均值、方差、最小值、最大值或词汇表）。

> 流程图示 TFX Transform 组件的输入和输出。原始数据和模式根据用户定义的逻辑进行处理，生成用于训练的转换特征和用于服务阶段一致应用的图。

### 定义转换：`preprocessing_fn`

你在 Python 函数中定义特征工程逻辑，该函数通常命名为 `preprocessing_fn`。此函数接收一个原始输入张量字典（代表数据集中的特征），并且必须返回一个转换后的输出张量字典。

在 `preprocessing_fn` 内部，你使用标准 TensorFlow 操作以及 `tensorflow_transform` 库中的专门函数（通常导入为 `tft`）。`tensorflow_transform` 提供分析器和映射器：

- **分析器：** 这些是诸如 `tft.min`、`tft.max`、`tft.mean`、`tft.vocabulary` 或 `tft.compute_and_apply_vocabulary` 等函数。它们需要对数据集进行完整遍历以计算必要的统计量（例如，最小值/最大值、词汇表 (vocabulary)列表）。`Transform` 在其“分析”阶段使用 Apache Beam 协调此计算。
- **映射器：** 这些是标准 TensorFlow 操作或 `tft` 函数，如 `tft.scale_to_0_1`、`tft.string_to_int` 或逐元素算术运算。它们根据输入张量以及分析器计算的常数应用转换。

以下是 `preprocessing_fn` 的一个示例：

```python
import tensorflow as tf
import tensorflow_transform as tft

# 假设输入数据中存在的特征键
_NUMERICAL_FEATURE_KEYS = ['trip_miles', 'trip_seconds']
_CATEGORICAL_FEATURE_KEYS = ['payment_type', 'company']
_LABEL_KEY = 'trip_total'

def preprocessing_fn(inputs):
    """使用 tf.Transform 定义特征工程逻辑。

    参数：
        inputs: 原始张量的字典。

    返回：
        转换后张量的字典。
    """
    outputs = {}

    # 将数值特征缩放到 z-score
    for key in _NUMERICAL_FEATURE_KEYS:
        # tft.scale_to_z_score 计算数据集的均值和方差
        # 并应用转换：(输入 - 均值) / 标准差
        outputs[key + '_scaled'] = tft.scale_to_z_score(inputs[key])

    # 生成词汇表并将类别特征映射为整数
    for key in _CATEGORICAL_FEATURE_KEYS:
        # tft.compute_and_apply_vocabulary 根据频率生成词汇表，
        # 并将字符串映射为整数索引。
        # num_oov_buckets=1 用于处理服务期间的未见值。
        outputs[key + '_index'] = tft.compute_and_apply_vocabulary(
            inputs[key],
            num_oov_buckets=1
        )

    # 不修改地传递标签
    outputs[_LABEL_KEY] = inputs[_LABEL_KEY]

    return outputs
```

### Transform 的执行方式

1. **分析阶段：** TFX `Transform` 通常在 Apache Beam 上运行，对 *整个* 训练数据集执行 `preprocessing_fn`。在此过程中，它会专门计算 `tft` 分析器所需的值（例如，均值、方差、词汇表 (vocabulary)）。这些计算出的值会被存储。
2. **转换阶段：** `Transform` 构建最终的 TensorFlow 图 (`transform_graph`)。此图包含在 `preprocessing_fn` 中定义的 TensorFlow 操作，并将分析阶段计算出的统计量作为常量嵌入 (embedding)图中。然后，它将此图应用于训练和评估数据集，生成 `Trainer` 组件所需的转换样本。

### 优势和输出

使用 `Transform` 带来了显著的优势：

- **消除训练/服务偏差：** 通过将精确的转换逻辑和统计量嵌入 (embedding)到可导出图中，它保证了一致性。
- **简化服务：** 服务基础设施只需在原始输入数据上执行已保存的 `transform_graph`；它无需重新实现预处理逻辑或管理统计量。
- **效率高：** 统计量在分析阶段对完整数据集计算一次。转换阶段和后续的服务推断通常是高效的图执行。
- **流水线集成：** `Transform` 自然地融入 TFX 工作流程。其输出直接被 `Trainer` 和 `Evaluator` 使用，并且 `transform_graph` 会由 `Trainer` 自动与训练好的模型打包，以便通过 `Pusher` 进行部署。

`Transform` 组件的主要输出包括：

1. **`transformed_examples`：** 应用了特征工程的训练和评估数据集，准备用于模型训练。
2. **`transform_output`（或 `transform_graph`）：** 一个包含 TensorFlow 图定义和元数据的文件，表示特征处理逻辑。此文件对确保下游的一致性很重要。

通过使用 TFX `Transform`，你为 ML 流水线增添了稳健性，确保对模型性能重要的特征工程从开发到生产部署都保持一致。该组件是使用 TFX 构建可靠、可用于生产的 ML 系统的基本组成部分。

## 参考资料

- [TFX Transform Component](https://www.tensorflow.org/tfx/guide/transform) — TensorFlow (2024)
  TFX Transform组件的官方指南，详细介绍了其功能以及在TFX管道中的集成。
- [TensorFlow Transform (tf.Transform)](https://www.tensorflow.org/tfx/transform/api_docs/python/tft) — TensorFlow (2024)
  TensorFlow Transform库的官方API文档，详细介绍了在`preprocessing_fn`中使用的分析器和映射器。
- [TFX: A TensorFlow-Based Production-Ready ML Platform](https://dl.acm.org/doi/10.1145/3097983.3098006) — Denis Baylor, Jian Li, Jean-François Puget, Andy Honig, Ethan Snider, Kathy Lu, Clemens Mewald, Tushar Chandra, H. Verena Nguyen, Vivek Kumar, Steven Whittaker (2017)
  Journal: KDD '17: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: Association for Computing Machinery; Pages: 2209–2217; DOI: [10.1145/3097983.3098006](https://doi.org/10.1145/3097983.3098006)
  介绍了TensorFlow Extended (TFX) 平台，概述了其架构以及Transform等组件在生产机器学习中的作用。
- [Machine Learning Engineering](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFM39EjWe841tH-Tv4g-3dLM4WBMzXwuygsw0_xXwUonBsM3X5mc-HrPJS4wuBrSDqUfTAZ1mQW3eEfdmAwQ0u2xfYtI3nosnntrN3nn9QRrIE9s_OIp2NKf2hEpnmCZ7-EDxoGp3oCTIuJNxqtYeMFQebc5fLTEnvnKMnCw-Wm832UA4Tc2UdpC6z-4_V-BubKE8P29xs=) — Andriy Burkov (2020)
  Publisher: True Positive Inc.; Pages: 310
  构建和部署机器学习系统的综合指南，涵盖了特征工程一致性以及如何在生产中避免训练/服务偏差等主题。

---

[上一节](03-%E6%95%B0%E6%8D%AE%E6%91%84%E5%8F%96%E4%B8%8E%E9%AA%8C%E8%AF%81.md) · [下一节](05-%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%B0%83%E4%BC%98.md)

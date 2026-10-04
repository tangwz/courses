---
course: "advanced-tensorflow"
chapter: "production-ml-pipelines-tfx"
lesson: "tfx-data-ingestion-validation"
sourceId: 2898
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-5-production-ml-pipelines-tfx/tfx-data-ingestion-validation"
title: "数据摄取与验证"
description: "在 TFX 管道中使用 ExampleGen、StatisticsGen 和 SchemaGen 实现可靠的数据摄取和验证。"
order: 3
plots: ["plots/2898-0.json", "plots/2898-1.json"]
sourceHash: "7ee82ecfb00753303243f97554d200d765e52fee1aa503d96448013281ba1e40"
sourceCorrections: []
---

任何可靠的机器学习 (machine learning)系统的根本是可信赖的数据。在生产环境中，数据持续流入，可能来自不同来源，并且其特性会随时间变化。手动检查每个批次是不可行的。TFX 提供自动化组件来处理最初的重要步骤：将数据引入到管道中并彻底验证其完整性。`ExampleGen`、`StatisticsGen` 和 `SchemaGen` 是负责数据摄取并为其预期结构和属性建立基准的组件。

### ExampleGen: 数据摄取标准化

大多数 TFX 管道中的第一个活动组件是 `ExampleGen`。其主要作用是从外部源摄取数据，并将其转换为适用于下游 TFX 组件的格式，通常是包含序列化 `tf.train.Example` 协议缓冲区的 `TFRecord` 文件。`tf.train.Example` 是 TensorFlow 中表示特征数据的标准格式，能够处理多种数据类型。

`ExampleGen` 开箱即用地支持多种输入格式，包括 CSV、TFRecord、Avro 和 Parquet。您可以通过指定输入源位置和所需的输出拆分（例如，'train' 和 'eval'）来配置它。例如，要从 CSV 文件目录中摄取数据，您可以使用 `CsvExampleGen`：

```python
# TFX 组件配置示例（在管道定义中）
from tfx.components import CsvExampleGen
from tfx.utils.dsl_utils import external_input

# 指向包含 CSV 文件的目录
data_root = external_input("/path/to/your/csv/data")

# 配置 CsvExampleGen
example_gen = CsvExampleGen(input_base=data_root)

# 下游组件将访问 example_gen.outputs['examples']
```

`ExampleGen` 通常将数据划分为至少两个拆分：`train` 用于模型训练，`eval` 用于评估和验证。这种划分可以根据文件模式或比例进行配置。输出是 `TFRecord` 文件的集合，可以通过组件的 `examples` 输出通道访问，为管道的后续阶段做好准备。

### StatisticsGen: 了解您的数据

数据摄取后，下一步是了解其特性。`StatisticsGen` 计算由 `ExampleGen` 生成的数据集的描述性统计数据。它独立处理每个拆分（'train'、'eval'）。

`StatisticsGen` 的输出是一个 `DatasetFeatureStatisticsList` 协议缓冲区，其中包含数据集中每个特征的详细统计数据。这些包括：

- **计数：** 每个特征的样本数和缺失值。
- **数据类型：** 推断的数据类型（例如，INT、FLOAT、STRING）。
- **数值特征：** 最小值、最大值、平均值、标准差、用于分布可视化的直方图。
- **类别特征：** 唯一值的数量，前 k 个常见值及其频率。

这些统计数据提供了数据的量化 (quantization)摘要。它们对于以下方面很重要：

1. **数据审查：** 得到特征分布以及潜在问题（如稀疏性或偏斜值）的认识。
2. **模式推断：** 为定义预期数据结构提供经验依据。
3. **异常检测：** 建立一个基准，未来数据批次可以与其比较，以检测漂移或偏斜。

可视化这些统计数据通常很有帮助。例如，我们可以查看'年龄'等数值特征的分布，或者'产品类别'等特征不同类别的频率。



!['年龄'特征的分布](plots/2898-0.json)



> 直方图显示了示例“年龄”特征的频率分布。



!['产品类别'的频率](plots/2898-1.json)



> 条形图展示了“产品类别”特征中不同值的频率计数。

这些统计数据会被后续组件（如 `SchemaGen` 和 `ExampleValidator`）使用。

### SchemaGen: 定义数据契约

统计数据计算完成后，`SchemaGen` 推断出数据模式。模式作为正式契约，定义了管道应处理的数据的预期属性。它规范了对特征名称、类型、存在性以及值范围或域的预期。

`SchemaGen` 分析 `StatisticsGen` 的输出统计数据，以生成初始的 `Schema` 协议缓冲区。此模式通常定义：

- **特征名称：** 每个特征的标识符。
- **存在性：** 特征是否在所有样本中都是必需的 (`required`)、可选的 (`optional`)，或者可以缺失 (`min_count`、`min_fraction`)。
- **类型：** 预期的数据类型 (INT, FLOAT, BYTES/STRING)。
- **价态：** 特征是应表示单个值 (`single`) 还是值的列表/向量 (vector) (`multi`)。
- **域：** 对可接受值的约束。对于类别特征，这通常被推断为数据中出现的唯一字符串值的集合。对于数值特征，它可能指定一个范围。

```protobuf
# 简化的 tf.metadata.proto.v0.Schema 结构示例

feature {
  name: "age"
  type: FLOAT
  presence {
    min_fraction: 1.0 # 在所有样本中都是必需的
  }
  # domain: "age_range" # 可选域名称
}

feature {
  name: "product_category"
  type: BYTES # 字符串通常表示为 BYTES
  domain: "product_category" # 引用 string_domain 定义
  presence {
    min_fraction: 1.0
  }
}

string_domain {
  name: "product_category"
  value: "Electronics"
  value: "Clothing"
  value: "Home Goods"
  value: "Books"
  value: "Toys"
}
```

虽然 `SchemaGen` 提供了一个好的起始点，但推断出的模式通常需要手动审查和管理。例如：

- 您可能会根据业务逻辑收紧数值特征的域约束。
- 您可能会明确定义类别特征允许的值，而不是仅仅依赖于初始数据集中观察到的值。
- 如果某些特征预期会偶尔缺失，您可能会调整其存在性要求。

这个经过管理的模式成为与管道代码一起管理的重要产物。它确保后续组件，特别是 `Transform` 和 `Trainer`，接收到符合预期的数据。

### ExampleValidator: 强制执行契约

`ExampleValidator` 组件（通常在 `SchemaGen` 或 `StatisticsGen` 之后立即使用）使用模式和统计数据来检测数据中的异常。它将给定数据拆分的统计数据与模式中定义的预期进行比较。如果发现不一致，它会生成一个 `Anomalies` 协议缓冲区，详细说明问题。

常见检测到的异常包括：

- **模式一致性：** 数据中存在但模式中不存在的特征，或反之。具有意外数据类型或价态的特征。
- **值约束：** 超出指定范围的数值。在定义域中未找到的类别值。
- **分布偏斜/漂移：** 当前数据拆分（例如，'eval'）与用于生成模式的基准数据（例如，'train'）之间，或后续运行之间，统计属性（例如，平均值、中位数、频率分布）的显著差异。

以下图表展示了这些初始组件的典型流程：

> 流程图展示了 TFX 管道中 `ExampleGen`、`StatisticsGen`、`SchemaGen` 和 `ExampleValidator` 之间的相互作用。

及早检测异常可以防止有问题的数据在管道中传播，从而提高训练过程和生成模型的可靠性。如果检测到严重异常，甚至可以配置管道执行停止。

`ExampleGen`、`StatisticsGen`、`SchemaGen` 和 `ExampleValidator` 共同构成了一个系统，用于摄取数据、了解其属性、通过模式定义预期，并根据这些预期验证传入数据。这个自动化过程对于使用 TFX 构建稳定且可维护的生产级机器学习 (machine learning)管道非常重要。

## 参考资料

- [TensorFlow Extended (TFX) documentation](https://www.tensorflow.org/tfx/guide) — Google (2024)
  TFX的全面指南，解释其架构和核心组件，如ExampleGen、StatisticsGen、SchemaGen和ExampleValidator。
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf26747205edb427b2-Paper.pdf) — D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison (2015)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems; Volume: 28; Pages: 2503
  一篇基础性论文，讨论构建和维护生产级ML系统所面临的挑战，强调TFX组件旨在解决的数据依赖性和系统层面的问题。
- [TensorFlow Data Validation (TFDV) documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEpnD8JNB4EjQry7F0cxcUlfOelUxiebvoX_fLISscLFQrKDxFEBH4xRakdxXtM8_qw6ah94LcPlBNIcz_iMawQUpAOCPXi-1vCuhrKKuu0jka7ZKQ2IX1I4qWI_sRFUwo5NvfWhxIGZ6vidgb6PeVA==) — Google (2024)
  Publisher: Google
  提供使用TFDV进行数据统计计算、模式生成和异常检测的详细解释，TFDV是TFX组件用于数据验证的底层库。
- [Building Machine Learning Pipelines](https://www.oreilly.com/library/view/building-machine-learning/9781492053187/) — Hannes Hapke, Catherine Nelson, Rahul Raizada (2020)
  Publisher: O'Reilly Media
  一本实践性书籍，涵盖使用TFX设计和实现生产级ML管道，包括对数据摄取、验证和模式管理的全面讨论。

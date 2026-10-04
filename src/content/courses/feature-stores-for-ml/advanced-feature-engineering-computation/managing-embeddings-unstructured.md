---
course: "feature-stores-for-ml"
chapter: "advanced-feature-engineering-computation"
lesson: "managing-embeddings-unstructured"
sourceId: 3366
sourceUrl: "https://apxml.com/zh/courses/feature-stores-for-ml/chapter-2-advanced-feature-engineering-computation/managing-embeddings-unstructured"
title: "管理嵌入和非结构化数据"
description: "关于存储、检索和管理嵌入以及从非结构化数据中派生特征的方法。"
order: 3
plots: []
sourceHash: "99573ab9fc1ad81f5568dca3cbdce2a7fdc12992e20083b59dd6943fc28ee904"
sourceCorrections: []
---

现代机器学习 (machine learning)系统经常处理文本、图像、音频和图等复杂的非结构化数据源。将这类数据表示为数值通常需要生成*嵌入 (embedding)*：一种密集、低维的向量 (vector)表示，能够捕获语义信息。例如，自然语言处理模型可能将句子转换为512维向量，或者计算机视觉模型可能将图像表示为2048维向量。尽管这些嵌入特征功能强大，但与传统的标量特征（如计数或平均值）相比，它们在特征存储中带来了特殊的管理难题。本节讨论了有效存储、检索和管理嵌入以及其他源自非结构化数据特征的方法。

## 嵌入 (embedding)在特征存储中的挑战

嵌入与典型的特征类型有很大不同：

1. **高维度:** 嵌入可以有数百或数千个维度，这与简单的数值或分类特征不同。
2. **数据类型:** 它们通常表示为浮点数的数组或列表。
3. **大小:** 单个嵌入向量 (vector)比标量特征占用显著更多的存储空间。对于具有多个嵌入的实体（例如，用户画像嵌入、用户活动嵌入），存储空间会迅速增加。
4. **生成源:** 嵌入通常由独立的、复杂的深度学习 (deep learning)模型生成，有时需要大量的计算资源（如GPU）。它们经常在主要的特征转换管道之外进行预计算。
5. **上游模型依赖:** 嵌入的质量和含义完全取决于生成它的模型。上游模型的变化要求对存储在特征存储中的嵌入进行版本控制或重新计算。

这些特性影响了特征存储架构中的存储选择、检索延迟、计算工作流程和版本控制方法。

## 管理嵌入 (embedding)的方法

将嵌入成功集成到特征存储中，需要仔细考虑存储、访问模式和元数据管理。

### 存储预计算嵌入 (embedding)

嵌入通常在专门的上游流程中计算，例如使用Spark结合TensorFlow/PyTorch的批处理管道，或通过专门的模型服务端点。然后，这些预计算的向量 (vector)会被摄入到特征存储中。

> 预计算嵌入摄入特征存储的管道。

在特征存储注册表中为嵌入定义特征时，需要指定适当的数据类型。许多特征存储支持数组类型（例如，`List[float]`、`numpy.ndarray`）。

```python
# 特征定义示例 (SDK)
from my_feature_store_sdk import FeatureGroup, Feature, FloatList

product_embeddings_fg = FeatureGroup(
    name="product_embeddings",
    entities=["product_id"],
    features=[
        Feature(name="description_embedding_bert_v1", dtype=FloatList(size=768)),
        Feature(name="image_embedding_resnet_v2", dtype=FloatList(size=2048)),
        Feature(name="embedding_model_versions", dtype=String) # 元数据
    ],
    online=True,
    offline=True,
    source=batch_pipeline_output, # 对上游数据源的引用
    ttl=timedelta(days=30)
)

# 在特征存储中注册特征组
feature_store.register_feature_group(product_embeddings_fg)

# 摄入（通常在批处理/流式作业中完成）
embedding_data = [
    {"product_id": 101,
     "description_embedding_bert_v1": [0.1, 0.5, ..., -0.2], # 768个浮点数
     "image_embedding_resnet_v2": [0.9, -0.1, ..., 0.3],  # 2048个浮点数
     "embedding_model_versions": "bert_v1;resnet_v2",
     "event_timestamp": "2023-10-27T10:00:00Z"},
    # ... 更多产品
]
feature_store.ingest("product_embeddings", embedding_data)
```

### 存储考量

- **离线存储:** 在离线存储（例如，使用Parquet、Delta Lake、Iceberg的数据湖仓）中高效存储高维向量 (vector)相对简单。列式存储格式可以很好地处理数组。请确保所选格式能够有效存储大型数组，以最小化存储占用并优化训练数据生成期间的扫描性能。
- **在线存储:** 由于低延迟要求，这更具挑战性。
  - **标准键值存储（Redis、DynamoDB）：** 存储嵌入 (embedding)通常需要将浮点数组序列化为二进制大对象（例如，使用MessagePack或Protocol Buffers）或将其存储为JSON字符串。这在读写过程中会增加序列化/反序列化开销。对于维度非常高的嵌入，一些键值存储的最大项目大小限制也可能成为问题。
  - **专用向量数据库（Pinecone、Weaviate、Milvus）：** 这些数据库针对向量嵌入的存储和查询进行了优化，通常提供高效的近似最近邻（ANN）搜索功能。一些先进的特征存储架构会集成或使用向量数据库作为其在线存储的后端，特别是在服务时向量相似性搜索是主要要求的情况下。然而，对于按实体ID进行的标准特征查找，它们的专用索引可能无法提供比简单键值存储更多的优势，并且可能增加操作复杂性。
  - **数据库选择：** 最佳的在线存储取决于所需的延迟、查询模式（按ID查找与相似性搜索）、向量维度和整体系统架构。

### 检索模式

- **离线检索：** 用于训练时，时间点正确的连接会根据实体ID和时间戳获取嵌入 (embedding)及其他特征。这通常涉及从离线存储中读取大批量数据，其中列式格式是高效的。
- **在线检索：** 对于推理 (inference)，应用程序提供实体ID（例如，`product_id`），特征存储服务API从低延迟在线存储中检索相应的嵌入向量 (vector)。在此环节，最小化反序列化开销和网络传输大小很重要。

### 直接处理非结构化数据

虽然对于高分辨率图像或长文档等原始、大型非结构化数据来说不太常见，但一些特征存储可能通过转换函数管理直接源自非结构化输入的数据特征。例如，特征存储转换可能会调用外部情感分析服务处理文本，或者运行简单的图像哈希函数。

然而，通常不建议在特征存储中存储*原始*非结构化数据本身。更好的做法是，如果需要，存储指向原始数据在其主要存储位置的标识符（例如，S3 URL、数据库ID），并仅将派生特征或嵌入 (embedding)摄入到特征存储中。

### 版本控制和元数据

由于嵌入 (embedding)与创建它们的模型相关联，因此跟踪其出处很重要。

- **将嵌入作为特征：** 将每个特定的嵌入（例如，`product_description_embedding_bert_v1`）视为一个独立的特征。当引入新的模型版本（例如，`bert_v2`）时，注册一个*新*的特征（例如，`product_description_embedding_bert_v2`）。这使得使用这些特征的模型可以显式选择所需的版本。
- **元数据：** 将元数据与嵌入向量 (vector)一同存储。这可以包括所使用的嵌入模型的版本、处理的原始数据版本或置信度分数。这些元数据可以作为独立特征存储在同一特征组中。

## 权衡与替代方案

在通用特征存储中管理嵌入 (embedding)，为模型统一特征访问提供了便利。然而，需要考虑以下权衡：

- **复杂性：** 处理大型向量 (vector)会增加特征存储在线服务层和存储管理的复杂性。
- **成本：** 存储潜在的大型嵌入向量会增加存储成本，特别是在低延迟在线存储中。
- **专门需求：** 如果主要用例涉及向量相似性搜索（例如，根据图像嵌入查找相似产品），则专用向量数据库可能更适合，或者与特征存储结合使用。特征存储可以保存标识符，而向量数据库负责处理ANN搜索，返回ID供特征存储用其他特征进行补充。

请根据您对延迟、查询模式（ID查找与相似性搜索）以及您愿意管理的运营开销的具体要求来选择方法。对于许多需要嵌入特征以及其他特征类型通过实体ID查找进行模型推理 (inference)的应用，将预计算的嵌入集成到特征存储中提供了一个统一且易于管理的方案。

## 参考资料

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (2013)
  Journal: arXiv preprint arXiv:1301.3781; DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  介绍了Word2Vec，一种生成词的稠密向量表示（嵌入）的基础模型，展示了捕获语义的概念。
- [Feast Documentation](https://docs.feast.dev/) — The Feast Authors (2024)
  提供了在特征存储中实现和管理特征的详细指导，包括对嵌入等复杂数据类型的考虑。
- [Pinecone Documentation: What is a vector database?](https://docs.pinecone.io/home/what-is-a-vector-database) — The Pinecone Team (2024)
  Publisher: Pinecone
  解释了向量数据库的架构和目的，它们是专门用于有效管理和查询高维嵌入向量的存储。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供构建和部署机器学习系统的广阔视角，其中包含特征工程、数据管理以及模型投入生产的挑战。

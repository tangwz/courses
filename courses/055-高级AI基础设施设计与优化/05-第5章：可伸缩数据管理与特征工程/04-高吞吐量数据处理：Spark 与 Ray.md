# 高吞吐量数据处理：Spark 与 Ray

来源：[原文](https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-5-scalable-data-feature-engineering/data-processing-spark-ray)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当训练数据集从千兆字节增长到太字节时，它们会超出任何单台机器的内存和处理能力。在这种规模下进行特征工程、数据转换和验证需要分布式计算。为了解决这个挑战，Apache Spark 和 Ray 作为两种功能强大但架构不同的框架而出现，用于并行处理数据。两者间的选择将显著影响您平台的效率、成本和灵活性。

### Apache Spark：批处理 ETL 的标准

Apache Spark 是长期以来的行业标准，用于大规模、容错的批处理数据处理。它擅长抽取、转换、加载（ETL）工作负载，使其非常适合生成模型训练所需的干净、结构化数据集。Spark 的核心抽象——弹性分布式数据集 (RDD)——及其更高级别的 DataFrame API 提供了一种强大的、声明性的方式来表达复杂的数据转换。

Spark DataFrame API 与 pandas 非常相似，允许您构建一个转换的逻辑计划。Spark 的 Catalyst 优化器随后将此计划转换为一个高效的物理执行计划，并分发到工作节点集群上执行。逻辑计划和物理计划的这种分离是 Spark 最重要的优势之一，因为它可以在无需用户手动调整的情况下，优化数据混洗、谓词下推和查询执行。

对于典型的机器学习 (machine learning)数据准备流程，您可以使用 Spark 来：

1. 读取数据湖中的原始数据（例如：Amazon S3, Google Cloud Storage）。
2. 执行一系列转换：清理空值、对类别变量进行独热编码、缩放数值特征。
3. 将处理后的特征集以 Parquet 等优化格式写回存储。

以下是一个简化的 PySpark 示例，演示了用户活动数据集的特征工程。

```python
from pyspark.sql import SparkSession
from pysspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# Initialize Spark Session
# 初始化 Spark 会话
spark = SparkSession.builder \
    .appName("FeatureEngineering") \
    .getOrCreate()

# 1. Read raw data from object storage
# 1. 从对象存储读取原始数据
raw_data = spark.read.parquet("s3a://my-ml-datalake/raw/user_activity/")

# 2. Define transformation stages
# 2. 定义转换阶段
# Convert categorical 'device_type' to a numerical index
# 将类别型 'device_type' 转换为数值索引
indexer = StringIndexer(inputCol="device_type", outputCol="device_index")

# Assemble feature columns into a single vector
# 将特征列组合成一个向量
assembler = VectorAssembler(
    inputCols=["session_duration", "clicks", "device_index"],
    outputCol="features_raw"
)

# Scale the feature vector to have zero mean and unit variance
# 将特征向量进行缩放，使其均值为零，方差为一
scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled")

# 3. Create a pipeline to chain the stages together
# 3. 创建一个管道，将各阶段连接起来
pipeline = Pipeline(stages=[indexer, assembler, scaler])
model = pipeline.fit(raw_data)

# 4. Transform the data and select the final columns
# 4. 转换数据并选择最终列
processed_data = model.transform(raw_data) \
                      .select("user_id", "features_scaled", "has_purchased")

# 5. Write the processed data back to the data lake for training
# 5. 将处理后的数据写回数据湖以供训练
processed_data.write.mode("overwrite").parquet("s3a://my-ml-datalake/processed/training_set/")

spark.stop()
```

虽然 Spark 对批处理作业来说功能非常强大，但其架构也存在权衡。它被设计用于大型、粗粒度的任务。启动 Spark 作业的开销可能很大，使其不太适合低延迟或高度迭代的工作负载。其 Java 虚拟机 (JVM) 基础也可能在以 Python 为主的机器学习环境中造成不便。

### Ray：一个用于分布式 Python 的灵活框架

Ray 提供了一种不同的方法。它不是一个专门的数据处理引擎，而是一个通用框架，用于扩展任何 Python 工作负载。其设计理念是提供一个简单、通用的并行 API，使其感觉像是 Python 的自然扩展。这使得它特别适合异构的机器学习 (machine learning)工作负载，这些工作负载在一个应用程序中结合了数据处理、模型训练和超参数 (parameter) (hyperparameter)调优。

Ray 的基本构建块很简单：

- **任务 (Tasks):** 使用 `ray.remote()` 远程执行的无状态函数。
- **Actor (Actors):** 使用 `ray.remote()` 在集群中作为微服务执行的有状态 Python 类。

此模型提供了对分布式资源的细粒度控制。对于数据处理，`Ray Data` 库提供了一个类似 DataFrame 的 API，可以在 Ray 集群上执行。Ray Data 被设计为一个“分布式数据交换”层，能够与其他库（如 Dask、Spark 和 Mars）进行集成，并充当数据摄取和模型训练之间的高性能连接。

设想一个场景，您需要并行预处理数据分片以供分布式训练作业使用。借助 Ray，数据工作器和训练工作器可以位于同一个集群中，在内存中传递对象引用，并避免了将中间结果写入外部存储的开销。

```python
import ray
import pandas as pd
from typing import Dict

# Initialize Ray cluster
# 初始化 Ray 集群
ray.init()

# Define a remote function (a Ray Task) for preprocessing a data shard
# 定义一个用于预处理数据分片的远程函数（Ray 任务）
@ray.remote
def preprocess_shard(shard: Dict[str, list]) -> pd.DataFrame:
    df = pd.DataFrame(shard)
    # Perform some transformations
    # 执行一些转换
    df['feature_c'] = df['feature_a'] * df['feature_b']
    return df

# Use Ray Data to load a dataset
# 使用 Ray Data 加载数据集
# Ray automatically parallelizes the read and represents it as a distributed dataset
# Ray 自动并行化读取操作，并将其表示为分布式数据集
ds = ray.data.read_parquet("s3a://my-ml-datalake/raw/some_data/")

# Apply the preprocessing function to each shard of the dataset in parallel
# 并行地将预处理函数应用于数据集的每个分片
processed_ds = ds.map_batches(preprocess_shard, batch_format="pyarrow")

# The processed dataset can now be directly consumed by a Ray Train job
# without being written back to S3.
# 现在，处理后的数据集可以直接被 Ray Train 作业使用，
# 而无需写回 S3。
# for epoch in range(5):
#     for batch in processed_ds.iter_batches():
#         train_model(batch)

print(processed_ds.show(limit=1))
ray.shutdown()
```

### 比较 Spark 和 Ray 在机器学习 (machine learning)数据工作负载中的应用

Spark 和 Ray 之间的选择并非哪个更好，而是哪个更适合特定的工作。它们不同的架构导致在常见的机器学习任务中表现出不同的性能特点。

> 批处理预处理的数据流。Spark 使用独立的、协调的作业并伴随中间存储，而 Ray 可以在单个集群内集成数据处理和训练，从而实现高效的内存数据传输。

以下是它们适用性的细分：

- **大规模批处理 ETL：** Spark 通常在此任务中表现更优。其成熟的 Catalyst 优化器和容错机制专为可靠地转换太字节结构化数据而构建。如果您的数据准备是由数据工程团队执行的独立离线步骤，那么 Spark 是一个很棒的选择。
- **集成式机器学习工作负载：** Ray 在数据处理与其他机器学习生命周期部分（如训练或调优）紧密结合的场景中表现出色。通过避免将完整数据集物化到中间存储的需求，Ray 可以显著减少 I/O 瓶颈和总体执行时间。这对于迭代开发和复杂流程而言是一个显著优势。
- **生态系统与 API：** Spark 拥有丰富的生态系统，包含 Spark SQL、Structured Streaming 和 MLlib。其声明式、类 SQL 的 API 受到许多数据分析师和工程师的熟悉。Ray 是 Python 原生的，这吸引了那些偏好命令式、灵活编程模型且无需离开 Python 生态系统的机器学习工程师和研究人员。

### 架构考量

将这些工具集成到您基于 Kubernetes 的平台时，您将使用 Spark on Kubernetes Operator 或 KubeRay 等操作符来管理分布式作业的生命周期。一个主要考量是数据本地性。为了最大程度地减少昂贵的跨区域数据传输成本并降低延迟，您应始终将 Spark 或 Ray 集群配置为与您的对象存储桶在同一云区域运行。

最终，决定取决于您团队的技能和您平台的架构理念。您是更倾向于使用一组专业化的、同类最佳的工具（即“多石”方法），让 Spark 处理数据而另一个系统处理训练？还是更喜欢一个像 Ray 这样能够管理机器学习 (machine learning)应用程序端到端计算需求的统一框架（即“整体”方法）？两种都是有效策略，并且理解这些权衡对于构建高性能 AI 平台非常重要。

## 参考资料

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/) — The Apache Software Foundation (2024)
  提供关于 Spark 架构、DataFrame API、PySpark 和分布式处理能力的全面指导。
- [Ray Documentation](https://docs.ray.io/en/latest/) — Anyscale (2024)
  提供关于 Ray Core、Ray Data、任务和 Actor 原语以及构建分布式 Python 应用的详细信息。
- [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf) — Matei Zaharia, Mosharaf Chowdhury, Tathagata Das, Ankur Dave, Blake Goldin, Joseph Hellerstein, Mohammad Isard, S. Krishnan, Haoyuan Li, Scott McVeety, Andy Konwinski, Patrick Wendell, Adam Wilde, Michael Franklin, Ion Stoica (2012)
  Journal: Proceedings of the 9th USENIX Conference on Networked Systems Design and Implementation (NSDI '12); Publisher: USENIX; Pages: 15-28; DOI: [10.1145/2228298.2228301](https://doi.org/10.1145/2228298.2228301)
  介绍了 RDD 抽象，它是 Spark 设计容错和高效分布式数据处理的基础。
- [Ray: A Distributed System for AI Applications](https://www.usenix.org/system/files/osdi18-moritz.pdf) — Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Gungor, Eric Choi, Joseph E. Gonzalez, Ion Stoica (2018)
  Journal: Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI '18); Pages: 569-586
  描述了 Ray 灵活的架构，用于通用分布式 Python 计算，旨在支持多样化的 AI 工作负载。

---

[上一节](03-%E7%BB%93%E5%90%88%20DVC%20%E5%92%8C%20Pachyderm%20%E7%9A%84%E6%95%B0%E6%8D%AE%E7%89%88%E6%9C%AC%E6%8E%A7%E5%88%B6%E4%B8%8E%E8%A1%80%E7%BC%98%E8%BF%BD%E6%BA%AF.md) · [下一节](05-AI%E6%95%B0%E6%8D%AE%E6%B9%96%E5%92%8C%E6%95%B0%E6%8D%AE%E4%BB%93%E5%82%A8%E7%AE%A1%E7%90%86.md)

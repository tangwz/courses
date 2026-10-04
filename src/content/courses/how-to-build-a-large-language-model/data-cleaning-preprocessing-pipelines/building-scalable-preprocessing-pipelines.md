---
course: "how-to-build-a-large-language-model"
chapter: "data-cleaning-preprocessing-pipelines"
lesson: "building-scalable-preprocessing-pipelines"
sourceId: 5984
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-7-data-cleaning-preprocessing-pipelines/building-scalable-preprocessing-pipelines"
title: "构建可扩展的预处理流水线"
description: "使用Apache Spark或Dask等工具设计数据处理流程。"
order: 6
plots: []
sourceHash: "f74889f2c94b8ca2700fa81020a8690996eb79df0d0df98ae64a994cdef2e3e8"
sourceCorrections: []
---

处理PB级文本数据，不仅仅需要巧妙的过滤算法；它还需要一套能够应对信息庞大体量和高速率的基础设施。对于Common Crawl或大型内部语料库这类数据集，在单台机器上顺序执行标准的数据预处理步骤，包括清洗、过滤、规范化和去重，是不可行的。因此，处理流水线必须明确为扩展性设计，采用分布式计算方法。

这些可扩展的流水线通常将复杂的预处理任务分解为一系列独立的阶段，在机器集群上执行。Apache Spark和Dask等工具常用于此目的，它们为分布式数据结构和计算提供抽象功能，并提供长时间运行任务不可或缺的容错机制。

## 选择合适的框架

Apache Spark凭借其弹性分布式数据集（RDD）和更高级的DataFrame API（PySpark），擅长大规模批处理。它管理数据分布、在工作节点间调度任务以及处理节点故障。Dask提供类似功能，侧重于与Python生态系统（NumPy、pandas、scikit-learn）原生集成，并提供更灵活的调度选项。

让我们考虑一个简单的过滤任务：删除短于特定长度的文档。在PySpark中，这可能看起来像这样：

```python
# 假设 'spark' 是一个活跃的 SparkSession，'raw_documents_df' 是一个 DataFrame
# 包含 'text' 列。

from pyspark.sql.functions import length

MIN_LENGTH = 100 # 示例最小文档字符长度

# 根据文本长度过滤文档
filtered_df = raw_documents_df.filter(length(raw_documents_df.text) >= MIN_LENGTH)

# 进一步处理或保存过滤后的 DataFrame
# filtered_df.write.parquet("path/to/filtered_data")
```

Dask使用其DataFrame API提供类似的使用感受：

```python
# 假设 'ddf' 是一个从文件加载的Dask DataFrame
# 包含 'text' 列。

import dask.dataframe as dd

MIN_LENGTH = 100 # 示例最小文档长度

# 使用 map_partitions 根据文本长度过滤文档以提高效率
# 注意：这假设 'text' 列包含字符串
filtered_ddf = ddf[ddf['text'].str.len() >= MIN_LENGTH]

# 计算结果并保存（触发执行）
# filtered_ddf.to_parquet("path/to/filtered_data")
```

这些示例说明了如何在分布式数据结构上表达操作，从而使得框架能够在集群中并行执行。

## 设计流水线阶段

典型的预处理流水线不仅仅是单个脚本，而是操作的有向无环图（DAG）。每个阶段执行特定的转换，通常读取前一阶段处理过的数据，并为下一阶段写入结果。

考虑一个处理网络爬取数据的流水线：

1. **加载原始数据：** 读取原始文件（例如，来自Common Crawl的WARC文件，存储在HDFS或S3中）。
2. **初始提取与清理：** 提取纯文本，移除样板HTML/JavaScript，执行基本的Unicode规范化。
3. **语言识别：** 识别每份文档的主要语言。
4. **语言过滤：** 仅保留被识别为目标语言的文档。
5. **质量过滤：** 应用启发式或基于模型的过滤器，以删除低质量内容（例如，基于长度、停用词比例、小型模型的困惑度分数）。
6. **去重：** 使用MinHash LSH等技术识别并删除近似重复的文档。
7. **最终规范化：** 应用所有剩余的规范化步骤（例如，特定字符替换）。
8. **分词 (tokenization)（可选）：** 有时分词作为最后的预处理步骤执行，存储分词后的数据以便在训练期间更快地加载。否则，它由训练数据加载器即时完成。
9. **保存处理后的数据：** 将清理、过滤后的数据存储为Parquet等高效格式。

这可以如下图所示：

> 一个典型的数据预处理流水线，说明了从原始数据加载到保存处理后的文本的顺序阶段。

## 可扩展地实现阶段

让我们看看一些主要阶段如何在分布式环境中实现：

- **过滤和规范化：** 这些任务通常“易于并行化”。每个文档通常可以独立处理。分布式框架使用map操作高效处理此任务。您可以定义Python函数（Spark中的用户定义函数或UDF，或Dask中通过`map_partitions`应用的函数），封装对单个文档或一批文档进行清洗、规范化或应用质量启发式方法的逻辑。

  ```python
  # PySpark 示例：使用 UDF 进行规范化
  from pyspark.sql.functions import udf
  from pyspark.sql.types import StringType
  import unicodedata

  def normalize_text(text):
      if text is None:
          return None
      # 示例：应用 NFKC 规范化
      return unicodedata.normalize('NFKC', text)

  normalize_udf = udf(normalize_text, StringType())

  # 将 UDF 应用于 'text' 列
  normalized_df = filtered_df.withColumn("normalized_text", normalize_udf(filtered_df.text))
  ```
- **去重：** 使用文档内容的哈希比较来查找完全重复项相对简单，然后进行分布式分组或去重操作即可。然而，检测*近似重复项*需要更精巧的技术，例如MinHash结合局部敏感哈希（LSH）。

  分布式去重的MinHash LSH核心思路是：

  1. **分片：** 将每份文档表示为一组短字符序列（分片，例如5-gram）。
  2. **MinHashing：** 为每份文档的分片集计算固定数量的哈希值（MinHash签名）。具有相似分片集的文档很可能具有相似的MinHash签名。
  3. **LSH分带：** 将签名组件分组到带中。如果文档在至少一个完整带上匹配，则认为它们是潜在的重复项。这显着减少了所需的比较次数。
  4. **候选配对：** 生成在至少一个LSH带中匹配的文档对。
  5. **验证：** 计算候选对的精确相似度（例如，Jaccard相似度），并根据阈值进行过滤。

  高效地实现这一点通常涉及Spark中的多个MapReduce风格阶段或Dask中的等效操作，同时仔细管理数据混洗。存在诸如`spark-deduplication`之类的库，或者您可以利用Spark的MLlib LSH功能来实现，或使用Dask Bags/DataFrames构建自定义逻辑。
- **语言识别：** `pycld2`、`langdetect`等库或`fastText`的模型可以集成。由于这些通常按文档操作，因此它们非常适合map操作。然而，在每个工作任务上加载语言识别模型可能效率不高。策略包括广播模型（如果足够小）或使用`mapPartitions`为每个分区/工作器加载一次模型。

  ```python
  # Dask 示例：使用 map_partitions 进行语言识别
  import dask.dataframe as dd
  # 假设 'fasttext_model' 已加载（例如，使用 fasttext.load_model）

  def identify_language_partition(partition_df):
      # 如果未广播，则为每个分区加载一次模型
      # model = fasttext.load_model('path/to/lid.176.bin') # 示例
      langs = []
      for text in partition_df['text']:
          if text and isinstance(text, str) and len(text.strip()) > 20: # 基本检查
               # 预测返回类似 (('__label__en',), array([0.99], dtype=float32)) 的元组
              pred = fasttext_model.predict(text.replace('\n', ' '))
              lang = pred[0][0].replace('__label__', '') if pred[0] else 'und'
          else:
              lang = 'und' # 未确定
          langs.append(lang)
      partition_df['language'] = langs
      return partition_df

  # 将函数应用于每个分区
  lang_identified_ddf = filtered_ddf.map_partitions(
      identify_language_partition,
      meta=filtered_ddf._meta.assign(language=str)
  )

  # 过滤英语
  # english_ddf = lang_identified_ddf[lang_identified_ddf['language'] == 'en']
  ```

## 优化和效率考量

构建真正可扩展的流水线不仅仅是使用分布式框架。

- **中间存储：** 避免将中间结果写入纯文本文件或CSV。使用Apache Parquet或Apache Arrow等列式格式。这些格式提供高效的压缩、编码，并支持谓词下推（允许阶段只读取必要的数据）。
- **分区：** 数据在节点和任务间如何拆分（分区）至关重要。确保数据合理地良好分区，通常通过哈希相关键或使用文件块边界。分区不佳可能导致“拖延”任务，从而拖慢整个阶段（数据倾斜）。重新分区可能是必要的，但这会引入网络混洗成本。
- **缓存：** 对于多次重用的中间DataFrame或RDD（例如，用于质量过滤和去重的已清理数据集），将其缓存到集群内存或磁盘中（Spark中的`.cache()`或`.persist()`，Dask中的`.persist()`）可以显着加快后续阶段的速度。
- **资源管理：** 调整执行器/工作器数量、每个工作器的核心数和内存分配对性能很重要。这通常需要根据特定的集群硬件和处理任务的性质（CPU密集型与I/O密集型）进行试验。
- **批处理：** 在应用复杂函数（例如运行模型进行质量过滤或语言识别）时，在每个任务内部批量处理数据，而不是逐条处理，以分摊函数调用开销或模型加载时间。

## 整合（流水线）

这是一个显示链式操作的简化PySpark结构：

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, length, col
from pyspark.sql.types import StringType, BooleanType

# 假设 SparkSession 'spark' 已初始化

# --- UDFs（根据需要定义） ---
def complex_quality_filter(text):
    # 更复杂的启发式或基于模型的过滤器占位符
    if text is None: return False
    has_enough_words = len(text.split()) > 10
    # ... 添加更多检查 ...
    return has_enough_words

quality_filter_udf = udf(complex_quality_filter, BooleanType())

def normalize_text(text):
    # 规范化逻辑占位符
    if text is None: return None
    return text.lower().strip() # 简单示例

normalize_udf = udf(normalize_text, StringType())

# --- 流水线阶段 ---

# 1. 加载数据（假设初始提取后是Parquet源）
input_path = "s3://my-bucket/data/extracted_text/"
df = spark.read.parquet(input_path)

# 2. 基本过滤（例如，长度）
df_filtered_basic = df.filter(length(col("text")) > 50)

# 3. 质量过滤（使用UDF）
df_filtered_quality = df_filtered_basic.filter(quality_filter_udf(col("text")))

# 4. 规范化（使用UDF）
df_normalized = df_filtered_quality.withColumn("processed_text", normalize_udf(col("text"))) \
                                   .select("doc_id", "processed_text") # 选择相关列

# （去重阶段通常会在这里进行，通常更复杂）

# 5. 保存处理后的数据
output_path = "s3://my-bucket/data/processed_text/"
df_normalized.write.mode("overwrite").parquet(output_path)
```

构建这些流水线需要仔细规划和反复迭代。从少量数据样本开始开发和调试各个阶段，然后再扩展到集群上的完整数据集。Spark或Dask提供的监控工具对于找出瓶颈和优化大型数据集上的性能非常有帮助。目标是创建一个可重复且高效的过程，将原始文本转换为大型语言模型的高质量数据源。

## 参考资料

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/) — Apache Spark Community (2024)
  Publisher: Apache Software Foundation
  Apache Spark的官方文档，提供了关于其架构、API（包括PySpark DataFrames）、分布式数据处理和优化技术的全面指南，对于构建可扩展的管道非常有用。
- [Mining of Massive Datasets](http://www.mmds.org/) — Jure Leskovec, Anand Rajaraman, and Jeffrey D. Ullman (2020)
  Publisher: Cambridge University Press; Pages: Chapter 3
  一本被广泛引用的教科书章节，解释了局部敏感哈希（LSH）和MinHash的理论基础和实际应用，用于在大型数据集中高效检测近似重复项，这是数据去重的一项核心技术。
- [Dask Documentation](https://docs.dask.org/en/latest/) — Dask Developers (2024)
  Publisher: Coiled
  Dask的官方文档，涵盖了其DataFrame API、分布式计算功能以及与Python生态系统的集成，对于设计可扩展的数据预处理工作流程很有价值。

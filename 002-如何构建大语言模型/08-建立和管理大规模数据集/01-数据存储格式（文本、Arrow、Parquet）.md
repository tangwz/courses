# 数据存储格式（文本、Arrow、Parquet）

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-8-building-managing-large-scale-datasets/data-storage-formats)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型需要海量文本语料库，这些语料库在训练期间必须高效存储并快速访问。此类语料通常会经过抓取、清洗和预处理。选择合适的存储格式是一个根本性决定，它会影响存储成本、I/O性能以及数据加载流程的整体效率。虽然简单格式容易开始使用，但随着数据集扩展到TB或PB级别，专门格式的益处将越来越明显。

我们来考察常见选择：纯文本文件、Apache Arrow和Apache Parquet。

### 纯文本格式（例如，`.txt`、`.jsonl`）

最直接的方法通常是将文本数据存储在纯文本文件中。每行可能代表一个文档，或者您可以使用JSON Lines（`.jsonl`）等格式，其中每行都是一个表示数据记录的有效JSON对象（例如，包含文本和元数据）。

**优点：**

- **简单性：** 易于手动创建、检查和调试。标准文本处理工具（`grep`、`sed`、`awk`）可直接使用。
- **普适性：** 几乎任何编程语言或工具无需特殊库即可读取。

**缺点：**

- **低效：** 文本解析（特别是复杂的JSON）在数据加载期间可能占用大量CPU资源。
- **压缩：** 可以应用Gzip或Zstandard等标准压缩，但由于列内数据相似性，列式格式通常能实现更好的压缩比。
- **无模式/类型：** 格式本身不强制执行数据类型或结构，如果解析假设被违反，可能导致后续错误。
- **随机访问慢：** 访问特定记录需要从头开始扫描和解析，或维护单独的索引。

一个典型的`.jsonl`文件可能如下所示：

```json
{
    "text": "The quick brown fox jumps over the lazy dog.",
    "source": "wikipedia",
    "id": "doc_001"
}
{
    "text": "Large language models require immense amounts of text data.",
    "source": "common_crawl",
    "id": "cc_abc"
}
{
    "text": "...",
    "source": "...",
    "id": "..."
}
```

在Python中读取这个文件很简单，但对于非常大的文件来说可能很慢，特别是当反序列化复杂的JSON时：

```python
import json
import gzip

def read_jsonl_gz(filepath):
    """从一个gzipped JSON Lines文件中读取记录。"""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 处理或记录格式错误行
                print(f"正在跳过格式错误的行：{line.strip()}")

# 使用示例
# 数据加载器会迭代这个生成器
# data_generator = read_jsonl_gz("my_large_dataset.jsonl.gz")
# for record in data_generator:
#    process(record['text'])
```

对于较小的数据集或初步原型来说尚可接受，但逐行解析文本的CPU开销会成为瓶颈，当向多个GPU提供高速消耗数据的LLM训练时。

### Apache Arrow

Apache Arrow是一个内存列式数据格式标准。它旨在提高分析查询性能，并以最小的序列化/反序列化开销（通常是零拷贝读取）在系统和语言之间高效交换数据。

**理念：**

- **列式布局：** 数据按列而非按行组织。特定字段（例如“text”字段）的所有值在内存中连续存储。
- **零拷贝读取：** 进程通常可以直接访问内存中的Arrow数据结构而无需复制，这显著加快了数据访问速度，尤其是在进程间通信中（例如数据加载进程和主训练进程之间）。
- **语言无关：** 许多语言（Python、Java、C++、Rust等）都有官方库，使其非常适合涉及不同技术的流程。
- **丰富的数据类型：** 支持全面的数据类型集，包括嵌套结构、时间戳和数值类型。

**优点：**

- **快速读取/扫描：** 列式布局对缓存友好，并支持向量 (vector)化操作（使用SIMD指令），使顺序读取速度非常快。
- **高效互操作性：** 非常适合在进程之间（例如使用共享内存）或库之间传递数据（例如Pandas DataFrames可以几乎即时地转换为Arrow表或从Arrow表转换）。
- **内存效率：** 对于相同数据，通常比Python对象表示更节省内存。

**缺点：**

- **内存侧重：** 尽管Arrow有文件格式（`.arrow`或`.feather`），但它主要针对内存表示进行优化。Parquet通常更适合磁盘上的持久性、压缩存储。
- **写入开销：** 构建Arrow数组有时比写入纯文本的开销更高。
- **可读性差：** 二进制格式，需要特定工具或库来检查。

Arrow在Hugging Face `datasets`等库的*底层*被大量使用。当您使用`datasets`加载或映射数据集时，它通常在内部使用Arrow表进行缓存和快速访问。

```python
import pyarrow as pa
import pyarrow.feather as feather
import time

# 示例：创建和写入一个Arrow表
# 假设'data'是一个字典列表，例如[{'text': '...', 'id': '...'}, ...]
# 将Python对象转换为Arrow数组
texts = pa.array([d['text'] for d in data], type=pa.string())
ids = pa.array([d['id'] for d in data], type=pa.string())

# 创建一个Arrow表
table = pa.Table.from_arrays([texts, ids], names=['text', 'id'])

# 写入到Feather文件（Arrow文件格式）
feather.write_feather(table, 'my_dataset.arrow')

# 读取通常非常快
start_time = time.time()
read_table = feather.read_table('my_dataset.arrow')
end_time = time.time()
print(f"Arrow读取时间：{end_time - start_time:.4f} 秒")

# 访问列是高效的
text_column = read_table['text']
# print(text_column[0].as_py()) # 访问第一个文本条目
```

当您流程的多个阶段都使用Arrow时，才能体现其真正的效用，从而避免昂贵的序列化步骤。在Spark上运行的数据预处理任务可以直接将Arrow文件输出到云存储，然后PyTorch `DataLoader`可以使用`pyarrow`高效读取这些文件。

### Apache Parquet

Apache Parquet是一个广泛采用的列式*存储*格式，针对大规模数据仓库和分析进行了优化，对于在磁盘上存储LLM数据集也非常有效。

**理念：**

- **磁盘列式存储：** 和Arrow一样，它逐列存储数据，但专为磁盘持久化设计（例如HDFS、S3、本地文件系统）。
- **压缩与编码：** 通过应用适合每列数据类型和特点的编码（如字典编码、行程长度编码），然后进行通用压缩（Snappy、Gzip、Zstd、Brotli），实现了出色的压缩比。
- **模式演进：** 支持模式演进，允许您稍后添加列而无需重写旧数据。
- **谓词下推：** 存储系统通常可以在将数据加载到内存*之前*，直接在Parquet文件内部根据查询谓词过滤行（尽管这对于分析查询比顺序训练读取更相关）。

**优点：**

- **存储效率：** 与文本格式相比，显著节省磁盘空间，降低存储成本和网络传输时间。
- **快速顺序扫描：** 高效读取训练所需的列，跳过不相关的列。如果存在其他元数据，只读取“text”列比读取整个行快得多。
- **生态系统整合：** 在大数据框架（Spark、Dask、Presto、Hive）和Python库（`pandas`、`pyarrow`）中都有出色的支持。

**缺点：**

- **写入复杂性：** 由于编码和压缩，写入Parquet文件比纯文本涉及更多的计算开销。
- **不可读性：** 二进制格式，需要特定工具。
- **行级操作：** 与基于行的格式相比，对于需要一次性访问整行的操作效率较低（但由于I/O节省，通常总体上仍然更快）。

Parquet通常是最终处理数据集的首选格式，该数据集将在训练期间重复读取。

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd # 通常用作中间件

# 直接使用PyArrow的示例（类似于Arrow Feather示例）
# 假设“table”是Arrow示例中创建的pyarrow.Table

# 写入一个Parquet文件（默认使用Snappy压缩）
pq.write_table(table, 'my_dataset.parquet', compression='snappy')

# 读取Parquet文件
start_time = time.time()
# 可以只读取特定列，减少I/O
read_parquet_table = pq.read_table('my_dataset.parquet', columns=['text'])
end_time = time.time()
print(
    f"Parquet读取时间（仅文本列）：" # Parquet读取时间（仅文本列）：
    f"{end_time - start_time:.4f} 秒" # {end_time - start_time:.4f} 秒
)

# 使用Pandas的示例（常见工作流）
# df = pd.DataFrame(data) # data是一个字典列表
# df.to_parquet('my_dataset_pd.parquet', engine='pyarrow', compression='snappy')
# read_df = pd.read_parquet('my_dataset_pd.parquet', engine='pyarrow', columns=['text'])
```

### 对比与建议

以下是权衡的汇总：

| 特点 | 文本 (.txt, .jsonl) | Apache Arrow | Apache Parquet |
| --- | --- | --- | --- |
| **类型** | 基于行 | 列式（内存中） | 列式（磁盘上） |
| **主要用途** | 简单性，调试 | 快速内存分析，进程间通信 | 高效存储，分析 |
| **可读性** | 是 | 否 | 否 |
| **压缩** | 外部（Gzip, Zstd） | 有限（进程间通信通常是原始数据） | 优秀（内置） |
| **读取速度** | 慢（解析开销） | 非常快（零拷贝） | 快（列式扫描） |
| **写入速度** | 快 | 中等 | 中等-慢（编码） |
| **CPU使用（读取）** | 高（解析） | 低 | 低-中等（解码） |
| **随机访问** | 差 | 中等（内存中） | 中等（行组） |
| **生态系统** | 通用 | 增长中（Pandas, Spark） | 优秀（大数据） |

> 大型文本数据集常见数据存储格式的比较。

**对LLM数据集的建议：**

1. **中间处理：** 在复杂预处理流程的各阶段之间，使用Apache Arrow进行高效数据传输，尤其是在使用Spark或Dask等工具，或在Python进程间传递数据时。Hugging Face `datasets`等库大量使用Arrow进行缓存和内存映射。
2. **训练的最终存储：** 将最终的、大型的、已清洗的数据集以Apache Parquet文件格式存储在分布式文件系统（如S3或HDFS）上。其压缩功能显著降低存储成本，其列式特性允许在训练期间高效读取仅必要的列（通常仅为文本标记 (token)），从而最大限度地提高I/O吞吐量 (throughput)。使用Snappy或Zstandard等高效压缩方式。
3. **小型数据集/调试：** 对于小型实验或当人工可读性对于调试特定数据点非常重要时，纯文本或JSON Lines是可接受的。

为存储在云存储或HDFS上的大规模训练数据选择Parquet，并通过使用Arrow进行内存表示的库（如`datasets`或使用`pyarrow.parquet`的自定义`DataLoader`）进行读取，为将数据馈送给您的分布式训练任务提供了高效的根本。这最大限度地减少了I/O瓶颈和存储占用，使您能够将计算资源集中在模型训练本身上。

## 参考资料

- [Apache Arrow Documentation](https://arrow.apache.org/docs/) — Apache Arrow Project (2024)
  Apache Arrow的官方文档，详细介绍了其内存列式格式、数据类型和多语言互操作性，对于高效数据交换和零拷贝读取至关重要。
- [Apache Parquet Documentation](https://parquet.apache.org/docs/) — Apache Parquet Project (2022)
  Publisher: Apache Software Foundation
  Apache Parquet的官方文档，描述了其磁盘列式存储、压缩技术以及针对大规模持久化数据集的编码策略。
- [C-Store: A Column-Oriented DBMS](http://www.vldb.org/archives/website/2005/program/paper/thu/p553-stonebraker.pdf) — Michael Stonebraker, Daniel J. Abadi, Adam Batkin, Xuedong Chen, Mitch Cherniack, Miguel Ferreira, Edmond Lau, Amerson Lin, Samuel Madden, Elizabeth J. O'Neil, Patrick E. O'Neil, Alex Rasin, Nga Tran, Stanley B. Zdonik (2005)
  Journal: Proceedings of the 31st International Conference on Very Large Data Bases; Publisher: ACM; Pages: 553-564; DOI: [10.1109/VLDB.2005.1509709](https://doi.org/10.1109/VLDB.2005.1509709)
  一篇介绍列式数据库管理系统原理和优势的基础学术论文，直接影响了Apache Arrow和Parquet等现代列式数据格式的设计。
- [Hugging Face Datasets Library Documentation](https://huggingface.co/docs/datasets/) — Hugging Face (2024)
  Hugging Face `datasets`库的官方文档，提供了用于管理和加载大型文本数据集以进行LLM训练的工具，内部常使用Apache Arrow来提高性能。

---

[上一节](../07-%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86%E6%B5%81%E6%B0%B4%E7%BA%BF/06-%E6%9E%84%E5%BB%BA%E5%8F%AF%E6%89%A9%E5%B1%95%E7%9A%84%E9%A2%84%E5%A4%84%E7%90%86%E6%B5%81%E6%B0%B4%E7%BA%BF.md) · [下一节](02-%E5%88%86%E5%B8%83%E5%BC%8F%E6%96%87%E4%BB%B6%E7%B3%BB%E7%BB%9F%20%28HDFS%2C%20S3%29.md)

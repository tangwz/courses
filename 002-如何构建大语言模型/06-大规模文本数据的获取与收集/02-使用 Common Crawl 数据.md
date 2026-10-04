# 使用 Common Crawl 数据

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-6-sourcing-acquiring-massive-text-datasets/utilizing-common-crawl-data)

[返回章节目录](README.md) · [返回课程目录](../README.md)

获取足够大的数据集用于大型语言模型（LLM）预训练 (pre-training)是一项重要的工程难题。为此，Common Crawl (CC) 语料库是最大且常用的资源之一。它通过大规模网络爬取操作收集，代表了公共网络很大一部分的快照。Common Crawl 公开提供数PB的原始网页数据、元数据和提取的文本，通常每隔一两个月发布新的爬取数据集。其规模之大使其成为训练先进语言模型不可或缺的资源，尽管其利用存在一定困难。

### 理解 Common Crawl 档案

Common Crawl 是一个开放的网络爬取数据仓库。可以将其视为一个庞大的图书馆，其中包含多年来收集的数十亿网页副本。这些数据存储在亚马逊云服务（AWS）的简单存储服务（S3）上，所有人均可访问，尽管在“请求者付费”模式下通常会产生数据传输费用。

每次爬取的数据主要分为三种格式：

1. **WARC (Web ARChive)：** 这些文件包含每个被爬取 URL 的原始响应数据，包括完整的 HTTP 头和未经处理的 HTML 内容。WARC 文件内容全面但非常大，需要大量的解析工作才能提取有用文本。
2. **WAT (WARC Archive Transformation)：** 这些文件存储从 WARC 记录中提取的元数据。这包括 HTTP 头、页面上检测到的链接以及在爬取处理过程中获取的其他元数据。WAT 文件对于分析网络结构或特定的元数据点很有用，但它们不直接包含页面的主要文本内容。
3. **WET (WARC Encapsulated Text)：** 这些文件包含从 WARC 文件中存储的 HTML 内容中提取的纯文本。Common Crawl 应用启发式方法去除 HTML 标签、导航栏、广告和其他非内容元素。WET 文件明显小于 WARC 文件，通常是为 LLM 预训练 (pre-training)收集文本的最实用起点。但是，请注意文本提取的质量可能有所不同，有时重要的上下文 (context)或格式可能会丢失。

### 访问 AWS S3 上的 Common Crawl 数据

访问批量数据最直接的方式是通过 AWS S3。每次爬取都有一个唯一标识符（例如，`CC-MAIN-2023-50`），其数据组织在相应的 S3 存储桶中（例如，`s3://commoncrawl/`）。您可以使用标准 AWS 工具，如 AWS 命令行界面 (CLI) 或 SDK（例如 Python 的 `boto3`），来列出和下载文件。

```bash
# 示例：列出特定爬取分段的 WET 文件路径
aws s3 ls s3://commoncrawl/crawl-data/CC-MAIN-2023-50/segments/1700495631538.95/wet/ --request-payer requester
```

请记住，`--request-payer requester` 标志是必需的，因为 Common Crawl 存储桶被配置为由*请求*数据的人支付下载带宽费用。下载数 TB 或 PB 数据可能会产生可观的费用。

### 处理 WET 文件以提取文本

由于 WET 文件包含提取的纯文本，它们通常是最方便的处理格式。一个典型的 WET 文件（`.wet.gz`）是一个 gzip 压缩档案，其中包含多个文本记录，每个记录对应一个网页。每个记录都以提供元数据（如目标 URI 和内容长度）的 WARC 头开始，后跟提取的纯文本内容。

以下是一个使用 `warcio` 库（它简化了 WARC 格式文件，包括 WET 的读取）在 WET 文件中遍历记录的 Python 代码片段：

```python
import gzip
from warcio.archiveiterator import ArchiveIterator
import io

# 假设 wet_file_path 指向一个已下载的 .wet.gz 文件
# 实际上，您很可能会从 S3 或分布式文件系统流式读取此文件

try:
    with open(wet_file_path, 'rb') as stream:
        # 如果文件是 gzipped，使用 gzip.open 进行透明解压缩
        # 注意：warcio 通常可以直接处理 gzip 解压缩
        # 如果从流中读取
        # 对于本地文件，明确使用 gzip 是有效的。
        with gzip.open(stream, 'rb') as compressed_stream:
            # 为 warcio 创建一个内存字节流
            # 这可以避免底层流不可查找的问题
            # 对于非常大的文件，可能需要更复杂的流式处理
            bytes_stream = io.BytesIO(compressed_stream.read())

            for record in ArchiveIterator(bytes_stream):
                # 检查记录是否是转换（文本提取）记录
                if record.rec_type == 'conversion':
                    # record.rec_headers 包含元数据，如 WARC-Target-URI
                    uri = record.rec_headers.get_header('WARC-Target-URI')

                    # 读取有效负载（提取的文本）并解码
                    content_bytes = record.content_stream().read()
                    try:
                        content_text = content_bytes.decode('utf-8')

                        # --- 您的文本处理逻辑在此处 ---
                        # 示例：基本过滤和打印
                        if len(content_text.strip()) > 100: # 过滤非常短的文本
                            print(f"URI: {uri}")
                            print(f"Content Length: {len(content_text)}")
                            # print(content_text[:500] + "...") # 打印片段
                            # 添加到您的数据集，执行清理等

                    except UnicodeDecodeError:
                        # 如果内容不是有效的 UTF-8，处理潜在的解码错误
                        print(f"Skipping record for URI {uri} "
                              f"because of decoding error.")
                        continue

except FileNotFoundError:
    print(f"Error: File not found at {wet_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
```

> 使用 Python 和 `warcio` 库处理 WET 文件的简要概述。

处理整个 Common Crawl 数据集需要分布式计算框架，如 Apache Spark 或 Dask，并在可访问 S3 的集群上运行。这些框架允许您在多台机器上并行化 WET 文件的读取、解析和过滤，考虑到 PB 级别的数据规模，这是必不可少的。我们将在第 7 章讨论如何构建此类可扩展的数据处理流程。

### 使用 CC-Index 服务器

仅仅为了查找来自特定网站或页面类型的数据而下载整个数 TB 的数据段是效率低下的。Common Crawl 提供了一个索引服务（CC-Index），允许您查询爬取的元数据，而无需首先下载大型 WARC/WET 文件。

该索引包含关于每个被爬取 URL 的信息，包括其 MIME 类型、时间戳、在相应 WARC 文件中的位置以及状态码。您可以使用 `cdx-toolkit` 等工具或通过直接向 CDX API 端点发出 HTTP 请求来查询此索引。

例如，您可以查询索引以查找特定域名（`example.com`）下爬取的所有页面，或被识别为特定语言的所有页面。这使得数据获取更具针对性。一旦您从索引查询中获得了 WARC 文件位置（文件名和偏移量），您就可以执行范围 HTTP GET 请求，仅获取您需要的特定 WARC 记录，如果您只需要部分数据，这将显著减少下载量。

```bash
# 示例：使用 cdx-toolkit（需要安装）
# 在特定爬取中查找 'example.com' 下的英文页面
# cdx-toolkit "http://index.commoncrawl.org/CC-MAIN-2023-50-index" \
#   --url "example.com/*" \
#   --output-fields url,warc_filename,warc_record_offset,warc_record_length,content_language \
#   --filter "content_language == eng" \
#   --limit 10
```

### 数据质量和过滤考量

尽管 Common Crawl 提供了庞大的规模，但原始数据，即使是 WET 格式，本质上也是嘈杂的，需要大量清理。常见问题包括：

- **样板内容：** 导航菜单、页脚、侧边栏、Cookie 同意表单。
- **广告：** 与主要内容混合的广告文本。
- **低质量内容：** 垃圾邮件、机器生成文本、语法差的论坛帖子。
- **标记 (token)残余：** HTML/JavaScript 代码未完全去除。
- **语言污染：** 非目标语言的文本。
- **重复：** 多次爬取到的近似相同或完全相同的页面。

> 典型的过滤流程应用于 Common Crawl WET 数据，然后才用于 LLM 训练。

解决这些问题通常需要构建一个多阶段流程，包括语言识别、基于启发式的质量过滤（例如，根据文本长度、符号比例、停用词频率）和近似重复检测技术（如 MinHash）。这些重要的预处理步骤是第 7 章的重点。

总之，Common Crawl 是获取 LLM 预训练 (pre-training)所需规模的网络数据的基础资源。访问和处理其 WET 文件提供了获取大量文本的直接途径，但这需要操作 AWS S3、管理成本，并实施分布式处理和清理流程，以处理网络内容的噪声和多样性。理解如何有效使用 Common Crawl 是 LLM 开发数据获取阶段的重要一步。

## 参考资料

- [Common Crawl](https://commoncrawl.org/) — Common Crawl Foundation (2025)
  提供关于Common Crawl项目的全面信息，包括数据格式（WARC, WAT, WET）、访问方法和CC-Index。
- [warcio Documentation](https://warcio.readthedocs.io/en/latest/) — Webrecorder Project (2024)
  `warcio` Python库的官方文档，提供了分析WARC、WAT和WET文件的详细信息和示例，与本节的代码片段中引用的内容一致。

---

[上一节](01-%E7%A1%AE%E5%AE%9A%E6%BD%9C%E5%9C%A8%E6%95%B0%E6%8D%AE%E6%9D%A5%E6%BA%90.md) · [下一节](03-%E8%A7%84%E6%A8%A1%E5%8C%96%E7%BD%91%E9%A1%B5%E6%8A%93%E5%8F%96%E6%8A%80%E6%9C%AF.md)

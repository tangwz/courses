# 区分ETL与ELT

来源：[原文](https://apxml.com/zh/courses/intro-etl-pipelines/chapter-1-understanding-etl-fundamentals/etl-vs-elt)

[返回章节目录](README.md) · [返回课程目录](../README.md)

ETL代表提取（Extract）、转换（Transform）和加载（Load）。这是一种常用的数据传输方式，将数据从源系统移出，进行清洗和重塑，然后加载到目标系统，通常是数据仓库。其顺序是严格的：先将数据取出（提取），然后修改（转换），最后将其放入最终位置（加载）。

现在，我们来介绍一种相关但有区别的模式：**ELT**，它代表**提取（Extract）、加载（Load）、转换（Transform）**。

注意到变化了吗？在ELT模式中，转换步骤发生在数据加载到目标系统*之后*。

### 为何不同？

传统的ETL方法出现时，数据仓库不如现在强大。转换通常需要专门的ETL服务器或暂存区，配备专用处理能力来处理复杂的清洗和重塑操作，*在*数据到达资源相对受限的目标仓库*之前*。数据是先仔细准备，然后才加载。

ELT模式随着强大、可扩展的云数据仓库（如Amazon Redshift、Google BigQuery、Snowflake）和数据湖的兴起而流行。这些现代系统通常拥有强大的计算能力。首先将原始或经过少量处理的数据直接加载到目标系统变得可行，有时也更高效。然后，您可以使用目标系统自身的处理能力来*原地*执行转换。

### ETL与ELT比较

以下是主要区别的分类：

1. **操作顺序：**

   - **ETL：** 提取 -> 转换 -> 加载
   - **ELT：** 提取 -> 加载 -> 转换
2. **转换位置：**

   - **ETL：** 通常发生在独立的处理器或暂存区，*在*到达目标数据仓库*之前*。
   - **ELT：** 发生在数据加载后，*在*目标数据仓库或数据湖*内部*。
3. **目标系统中的数据：**

   - **ETL：** 只将最终的、已转换的、可供分析的数据加载到目标系统。
   - **ELT：** 首先加载原始或接近原始的数据。然后应用转换，通常在目标系统中与原始数据一同创建新表或视图。如果您以后需要使用不同逻辑重新处理原始数据，这可能很有用。
4. **灵活性和速度：**

   - **ETL：** 转换是预先定义的。加载步骤可能较慢，因为它需要等待转换完成。它确保数据在进入目标系统*之前*的质量。
   - **ELT：** 加载通常更快，因为它处理的是原始数据。它提供灵活性，允许快速存储原始数据，并在之后应用转换，可能直接在仓库中使用不同的工具或技术（通常使用SQL）。
5. **使用场景：**

   - **ETL：** 仍被广泛使用，特别适用于结构化数据、明确定义的转换、要求在加载*之前*进行数据掩码/清洗的合规性需求，以及与处理能力较低的目标系统集成时。
   - **ELT：** 随着云数据平台、大量数据（大数据）、半结构化或非结构化数据以及需要对原始数据进行分析或转换逻辑可能演变的情况，变得越来越常见。

### 流程图示

以下图表展示了ETL和ELT流程之间的数据流区别。

> ETL在加载前处理数据；ELT在目标系统内处理前加载数据。

ETL和ELT都是有效且有用的数据集成模式。如何选择它们取决于您的具体需求、可用工具、数据源的特性、目标系统的能力以及您的数据处理目标。理解顺序上的根本区别——即转换*何时*发生——是您开始使用数据管道时最重要的收获。

## 参考资料

- [Fundamentals of Data Engineering: Planning and Building Robust Data Systems](https://www.amazon.com/Fundamentals-Data-Engineering-Planning-Building/dp/1098108182) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  提供对数据工程原则的最新和全面分析，包括ETL和ELT模式的详细比较。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  本书提供了对数据系统架构的基础理解，有助于掌握ELT为何在现代数据仓库中变得可行。

---

[上一节](03-ETL%E6%B5%81%E7%A8%8B%E7%9A%84%E7%9B%AE%E7%9A%84.md) · [下一节](05-%E5%B8%B8%E8%A7%81%E6%95%B0%E6%8D%AE%E6%BA%90%E5%92%8C%E6%95%B0%E6%8D%AE%E7%9B%AE%E6%A0%87.md)

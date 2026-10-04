---
course: "intro-etl-pipelines"
chapter: "the-extraction-stage"
lesson: "extracting-structured-data"
sourceId: 5507
sourceUrl: "https://apxml.com/zh/courses/intro-etl-pipelines/chapter-2-the-extraction-stage/extracting-structured-data"
title: "处理结构化数据（例如：数据库、CSV）"
description: "学习从关系型数据库和 CSV 文件等结构化源提取数据的技术。"
order: 3
plots: []
sourceHash: "f5175b6b8dfdae52f6e49960929dae45485deb14c439603f84e0ec0a74d198c4"
sourceCorrections: []
---

结构化数据可能是构建 ETL 管道时最常遇到的类型。可以将其视为整齐地组织成行和列的信息，很像电子表格或数据库中的表。这种可预测的格式，通常由明确的模式（描述数据结构的蓝图）定义，使得数据提取相对简单。

我们将查看结构化数据的两种主要来源：关系型数据库和逗号分隔值 (CSV) 文件。

### 从关系型数据库提取数据

关系型数据库（如 PostgreSQL、MySQL、SQL Server）将数据存储在具有预定义列和数据类型的表中。与这些数据库交互的标准语言是 SQL（结构化查询语言）。数据提取通常涉及编写 SQL 查询来获取所需数据。

最基本的 SQL 提取命令是 `SELECT`。以其最简单的形式，您可以从表中获取所有数据：

```sql
SELECT * FROM customers;
```

星号 (`*`) 是一个通配符，表示“所有列”。然而，获取所有列通常效率不高，特别是对于大型表。更好的做法是只指定 ETL 过程中所需的列：

```sql
SELECT customer_id, first_name, last_name, email, registration_date
FROM customers;
```

通常，您也不需要 *所有* 行。您可能只需要在特定日期之后注册的客户，或位于特定区域的客户。SQL 中的 `WHERE` 子句允许您根据特定条件筛选行：

```sql
SELECT customer_id, first_name, email
FROM customers
WHERE country = 'USA' AND registration_date >= '2023-01-01';
```

此查询选择 2023 年 1 月 1 日或之后注册的美国客户的 ID、名字和电子邮件。

执行这些查询需要连接到数据库（如“连接数据源”中所述），并使用适当的凭据和连接详细信息。查询结果通常是一个表格数据集，您的 ETL 工具或脚本随后可以对其进行处理。

### 从 CSV 文件提取数据

CSV 文件是表示表格数据的简单文本文件。文件中的每一行通常对应一个数据行，行中的值由分隔符分隔，最常见的是逗号。通常，第一行包含列的标题名称。

以下是一个 `products.csv` 小文件示例：

```csv
ProductID,ProductName,Category,Price
101,Laptop,Electronics,1200.00
102,Coffee Maker,Home Goods,85.50
103,Notebook,Stationery,2.99
104,Desk Chair,Furniture,150.00
```

从 CSV 文件提取数据涉及逐行读取文件并根据分隔符解析每行。虽然您可以手动编写代码来完成此操作，但大多数编程语言和 ETL 工具都有专门为高效读取 CSV 文件而设计的内置库或组件。

这些工具处理常见的复杂情况：

- **分隔符：** 尽管逗号是标准分隔符，但文件可能使用制表符 (`\t`)、分号 (`;`) 或竖线 (`|`) 作为分隔符。提取过程需要知道使用哪个分隔符。
- **标题行：** 文件是否包含标题行？如果包含，应使用它来识别列；否则，列可能通过位置（例如，第 1 列、第 2 列）识别。
- **引用：** 包含分隔符（例如，“椅子，可倾斜”）或换行符的值通常必须用引号（通常是双引号 `"`）括起来。解析器需要正确处理这一点。
- **编码：** 文本文件可以有不同的字符编码（如 UTF-8 或 ASCII）。使用错误的编码可能导致乱码。

设置从 CSV 文件提取数据时，您通常需要配置这些参数 (parameter)（文件路径、分隔符、是否存在标题行、编码），以便读取器可以正确解析文件。

> 示意图展示了使用 SQL 从关系型数据库和使用解析器从 CSV 文件进行数据提取的流程。

了解如何查询数据库和解析像 CSV 这样的结构化文件是许多数据提取任务的基础。尽管工具和具体命令可能有所不同，但在表格格式中识别、选择和获取数据的基本原则保持不变。

## 参考资料

- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本广泛使用的教材，全面介绍了数据库系统原理，包括关系模型和SQL。
- [Learning SQL](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057611/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media
  一本实用且易懂的SQL学习指南，涵盖数据检索和操作的基本命令。
- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball, Margy Ross (2013)
  Publisher: John Wiley & Sons; Pages: 608
  数据仓库的基础著作，详细介绍了设计数据结构和实现ETL方法，包括提取策略。
- [csv - CSV File Reading and Writing](https://docs.python.org/3/library/csv.html) — Python Software Foundation (2023)
  Python内置模块的官方文档，展示了如何以编程方式读写CSV文件，并关注各种解析参数。

---
course: "intro-etl-pipelines"
chapter: "the-extraction-stage"
lesson: "connecting-to-sources"
sourceId: 5502
sourceUrl: "https://apxml.com/zh/courses/intro-etl-pipelines/chapter-2-the-extraction-stage/connecting-to-sources"
title: "连接数据源"
description: "查看建立与数据库、文件等多种数据源连接的方法。"
order: 1
plots: []
sourceHash: "e2312bd56a3d516e45a6a7e59bb13cc5bef5ffb623180da79e85f971f0c75b56"
sourceCorrections: []
---

提取是任何提取、转换、加载（ETL）过程中的基础性第一步。在转换或加载任何数据之前，您首先需要从其来源获取数据。这需要您的ETL系统（无论是工具还是脚本）与各种数据源之间建立连接。这就像将数据线插入设备一样；您需要正确的连接器和正确的端口才能使其正常运行。

数据存在于许多不同的位置和格式中。您的ETL管道可能需要从传统数据库、服务器上存储的简单文件，甚至是互联网上提供数据的网络服务中提取信息。我们来看看一些常见数据源类型如何建立连接。

### 连接到数据库

数据库是高度组织化的数据存储库，通常是关系型（如PostgreSQL、MySQL、SQL Server），有时是非关系型（如MongoDB、Cassandra）。连接它们通常需要将特定信息打包成所谓的**连接字符串**。

连接字符串就像是数据库的地址和一套密钥。它通常包含：

1. **驱动程序/提供程序信息：** 指定您要连接的数据库软件类型（例如，MySQL、PostgreSQL）。
2. **服务器地址：** 数据库服务器运行的网络主机名或IP地址。
3. **端口号：** 数据库监听的特定网络端口（例如，PostgreSQL为5432，MySQL为3306）。
4. **数据库名称：** 您要访问的服务器上特定数据库的名称。
5. **凭据：** 用户名和密码（或其他身份验证令牌），用于证明您有权访问数据。

例如，连接字符串可能看起来像这样（具体格式因数据库类型和驱动程序而异）：

```
postgresql://username:password@database.server.com:5432/mydatabase
```

为了建立实际连接，ETL工具常使用标准接口，如**JDBC**（Java数据库连接）或**ODBC**（开放数据库连接）。它们充当中间层，将ETL工具的请求翻译成特定数据库能够理解的命令。您需要安装要连接的数据库的相应驱动程序。

> ETL过程与数据库、文件系统和Web API等各种数据源之间建立的连接，重点说明了所涉及的典型组成部分（连接字符串、驱动程序、路径、权限、端点、身份验证）。

### 连接到文件

数据常以文件形式出现，例如逗号分隔值 (CSV)、JavaScript对象表示法 (JSON) 或可扩展标记 (token)语言 (XML)。这些文件可能位于运行ETL进程的同一机器上、共享网络驱动器中，或者越来越多地，在Amazon S3、Azure Blob Storage或Google Cloud Storage等云存储服务中。

连接文件主要涉及指定指向文件位置的**文件路径**或**统一资源标识符 (URI)**。

- **本地文件：** 路径如`/data/source/customers.csv`或`C:\Input\orders.json`。
- **网络共享：** 路径如`\\fileserver\share\logs.xml`。
- **云存储：** URI如`s3://my-data-bucket/raw/sales/data.csv`或`wasbs://container@account.blob.core.windows.net/input/data.json`。

除了位置，ETL过程还需要必要的**读取权限**才能访问文件或其所在的目录。对于云存储，这通常涉及配置访问密钥或角色，以授予从特定存储桶或容器读取的权限。

### 连接到API

许多现代应用程序和服务通过**应用程序编程接口 (API)** 公开数据，特别是使用HTTP/HTTPS上的**REST**（表述性状态传输）等协议的Web API。可以将API视为计算机系统通过网络相互请求信息的结构化方式。

连接到API通常涉及：

1. **端点URL：** 您发送请求的特定网址（URL）。例如，`https://api.example.com/v1/products`。
2. **HTTP 方法：** 通常使用`GET`方法来获取数据。
3. **身份验证：** API几乎总是需要身份验证来识别请求应用程序并确保其拥有权限。常见方法包括：
   - **API 密钥：** 随请求发送的唯一字符串（常在请求头或查询参数 (parameter)中）。
   - **OAuth 令牌：** 一种更复杂但安全的标准，应用程序可获取临时访问令牌。
4. **参数（可选）：** 您可以在URL（例如，`?status=active`）或请求体中包含参数，以筛选或指定所需数据。

ETL过程向API端点发出HTTP请求，并包含身份验证详细信息。如果成功，API会返回请求的数据，通常格式为JSON或XML。

### 安全考量

无论数据源类型如何，安全性都是基本。连接凭据，如数据库密码、API密钥或云存储访问密钥，必须安全处理。避免将它们直接硬编码到脚本或配置文件中。使用安全的凭据管理系统或环境变量。

此外，确保连接已加密，尤其是在数据通过网络传输时。对数据库连接使用SSL/TLS（通常是连接字符串中的一个选项），对API调用使用HTTPS，以保护传输中的数据免受窃听。

建立这些可靠且安全的连接是任何提取过程中的重要第一步。连接建立后，您可以继续实际获取数据，我们将在下文中讨论这一点。

## 参考资料

- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  本书详细介绍了各种数据系统，包括如何连接数据库以及如何集成来自不同来源的数据。
- [RESTful Web Services](https://www.oreilly.com/library/view/restful-web-services/9780596529260/) — Leonard Richardson and Sam Ruby (2007)
  Publisher: O'Reilly Media
  这本基础性书籍描述了RESTful网络服务的架构风格和原则，这对于与现代API交互至关重要。
- [Fundamentals of Data Engineering: Plan and Build Robust Data Systems](https://books.google.com/books/about/Fundamentals_of_Data_Engineering_Plan_an.html?id=lq_2EAAAQBAJ) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  一本关于现代数据工程实践的指南，包括连接各种数据源、处理基于云的数据以及数据管道中的安全问题。

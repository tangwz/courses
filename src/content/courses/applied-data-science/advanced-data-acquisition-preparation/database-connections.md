---
course: "applied-data-science"
chapter: "advanced-data-acquisition-preparation"
lesson: "database-connections"
sourceId: 1825
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-1-advanced-data-acquisition-preparation/database-connections"
title: "连接数据库和数据仓库"
description: "学习将Python数据科学流程连接到SQL数据库和数据仓库的方法。"
order: 1
plots: []
sourceHash: "46a0adc0d6e71455713e40cc4a72b5e14a0561411f9dc5ad8063565417553d71"
sourceCorrections: []
---

您在专业环境中遇到的许多数据都位于结构化数据库或数据仓库中。与获取CSV等平面文件不同，访问这类数据需要建立与数据库服务器的连接、进行认证、执行查询（通常使用SQL），并将结果导入您的数据科学环境，通常是直接导入Pandas DataFrame。

关系型数据库（如PostgreSQL、MySQL、SQL Server、Oracle）将数据组织成具有预定义模式和关系的表，并使用结构化查询语言（SQL）进行数据操作和检索。数据仓库（如Amazon Redshift、Google BigQuery、Snowflake）是专门的数据库，优化用于对大量历史数据进行分析查询。虽然连接方法可能存在些许不同，但基本思路是类似的。

### 从Python连接

要从Python与数据库进行交互，您通常需要两个组成部分：

1. **数据库驱动：** 针对目标数据库的底层库（例如，PostgreSQL的`psycopg2`，MySQL的`mysql.connector`）。这些库处理直接的通信协议。
2. **高级工具包/ORM（可选但建议）：** 像SQLAlchemy这样的库为不同的数据库和驱动程序提供了一致的API。SQLAlchemy作为一个抽象层，简化了连接管理、查询执行，甚至允许使用Python对象而非原始SQL进行交互（尽管我们这里将侧重于执行SQL查询）。使用SQLAlchemy通常更受青睐，因为它使您的代码在不同数据库后端之间更具可移植性。

### 使用SQLAlchemy和Pandas

SQLAlchemy结合Pandas，提供了一种有效的方式来查询数据库并将结果直接加载到DataFrames。SQLAlchemy中用于连接的核心组件是`Engine`。您可以通过提供一个连接字符串（URL）来创建引擎，该字符串指定了数据库类型、驱动、用户名、密码、主机、端口和数据库名称。

连接字符串的典型格式是：
`dialect+driver://username:password@host:port/database`

以下是常用数据库的示例：

- **PostgreSQL（使用psycopg2驱动）：**
  `postgresql+psycopg2://user:password@hostname:5432/mydatabase`
- **MySQL（使用mysql-connector-python驱动）：**
  `mysql+mysqlconnector://user:password@hostname:3306/mydatabase`
- **SQLite（基于文件）：**
  `sqlite:///path/to/my_database.db`

让我们看看如何创建引擎并获取数据。首先，请确保您已安装所需库：

```bash
pip install sqlalchemy pandas psycopg2-binary # 或 mysql-connector-python等
```

现在，您可以连接并查询：

```python
import sqlalchemy as sa
import pandas as pd
import os # 用于安全管理凭据

# --- 最佳实践：从环境变量加载凭据 ---
DB_USER = os.getenv("DB_USER", "default_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "default_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432") # PostgreSQL示例
DB_NAME = os.getenv("DB_NAME", "mydatabase")

# 构建连接字符串
# PostgreSQL示例
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 对于SQLite数据库文件
# DATABASE_URL = "sqlite:///path/to/your_data.db"

# --- 创建SQLAlchemy引擎 ---
try:
    engine = sa.create_engine(DATABASE_URL)
    print("成功创建数据库引擎。")
except Exception as e:
    print(f"创建数据库引擎出错：{e}")
    # 妥善处理错误，例如退出或使用备用数据
    engine = None

# --- 使用引擎和Pandas查询数据 ---
if engine:
    try:
        # 定义您的SQL查询
        sql_query = "SELECT customer_id, purchase_date, amount FROM sales WHERE amount > 100 ORDER BY purchase_date DESC LIMIT 50;"

        # 使用pandas.read_sql执行查询并将结果加载到DataFrame
        with engine.connect() as connection:
             df_sales = pd.read_sql(sql_query, connection)

        print(f"成功加载{len(df_sales)}条记录到DataFrame。")
        print(df_sales.head())

    except sa.exc.SQLAlchemyError as e:
        print(f"执行SQL查询或读取数据出错：{e}")
        # 处理潜在的SQL错误（表不存在、语法错误等）
    except Exception as e:
        print(f"发生了一个意外错误：{e}")

else:
    print("没有数据库引擎无法继续。")
```

> 连接过程的简化视图：您的Python脚本使用SQLAlchemy，后者反过来使用特定的数据库驱动与目标数据库通信。

### 重要事项

- **凭据管理：** 切勿将密码等敏感信息直接硬编码在脚本中。请使用环境变量（如示例所示）、配置文件或专用密钥管理工具。Git等版本控制系统应忽略包含凭据的文件。
- **查询优化：** 对于超大型数据集，使用`pd.read_sql`将整个表检索到内存中可能效率低下或无法实现。请考虑：
  - 编写更具体的SQL查询（`WHERE`子句、`LIMIT`）以仅获取所需数据。
  - 仅选择所需列（`SELECT col1, col2 FROM ...`）。
  - 使用`pd.read_sql`中的`chunksize`参数 (parameter)分块处理数据。
  - 直接在数据库内部使用SQL执行聚合或初始过滤，这通常快得多。
- **错误处理：** 数据库操作可能因网络问题、凭据不正确、SQL语法无效或表缺失而失败。请实现`try...except`块来捕获潜在错误（例如，SQLAlchemy特定问题的`sa.exc.SQLAlchemyError`），并优雅地处理它们。
- **连接关闭：** 使用`pd.read_sql`与引擎通常会隐式处理连接的打开和关闭。如果您手动管理连接（例如，使用`engine.connect()`），请确保它们正确关闭，最好使用`with`语句，即使发生错误也能保证关闭。
- **数据仓库：** 连接到BigQuery或Snowflake等数据仓库通常涉及特定的SQLAlchemy方言（例如，`sqlalchemy-bigquery`）或由云厂商提供的专用客户端库（例如，`google-cloud-bigquery`、`snowflake-connector-python`）。虽然具体的连接字符串或认证方法可能有所不同（通常涉及API密钥或服务账户），但执行SQL并将数据加载到Pandas的模式仍然是常见做法。

"熟练掌握数据库连接是访问数据科学项目中常用的大量结构化数据的基本步骤。通过使用SQLAlchemy和Pandas等库，您可以创建一个可重复且高效的流程，将数据库源整合到您的分析工作流程中。"

## 参考资料

- [SQLAlchemy 2.0 Documentation - Unified Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/index.html) — The SQLAlchemy Project (2025)
  提供 SQLAlchemy 的全面文档，包括其核心组件、引擎创建和连接管理，这些是 Python 数据库交互的基础。
- [pandas.read_sql](https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html) — The pandas development team (2024)
  pandas.read_sql 函数的官方文档，展示了如何执行 SQL 查询并将结果直接加载到 Pandas DataFrame 中。
- [PEP 249 -- Python Database API Specification v2.0](https://www.python.org/dev/peps/pep-0249/) — Marc-André Lemburg (1999)
  定义了 Python 数据库驱动程序的标准接口，解释了 psycopg2 和 mysql.connector 等库的底层运行机制。

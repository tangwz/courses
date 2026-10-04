---
course: "building-advanced-llm-agent-tools"
chapter: "developing-custom-python-tools"
lesson: "practice-database-query-tool"
sourceId: 6567
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-2-developing-custom-python-tools/practice-database-query-tool"
title: "实践：构建数据库查询工具"
description: "一个动手实践环节，用于构建一个Python工具，该工具查询数据库并为LLM代理返回结构化结果。"
order: 7
plots: []
sourceHash: "143b65c14a713223ea905835294c7818f6dd94a7aad0120fb8e50e6330525505"
sourceCorrections: []
---

本次动手练习着重构建一个实用的Python工具：一个数据库查询工具。该工具使LLM代理能够从关系型数据库中获取信息，这是代理访问结构化数据的常见需求。主要关注点是创建一个Python函数，该函数连接到SQLite数据库，执行查询，并以LLM友好的格式返回结果。本次练习涵盖定义工具接口、处理输入和输出，以及与外部数据源的交互。

### 场景：产品信息代理

设想一个旨在回答客户产品问题的LLM代理。产品信息（名称、类别、价格）存储在数据库中。我们的目标是创建一个代理可以用来查询该数据库的工具。

### 前提条件：设置数据库

首先，我们需要一个简单的数据库来使用。我们将使用SQLite，它内置于Python中，使设置变得简单。让我们创建一个名为 `product_catalog.db` 的小型数据库，其中包含一个 `products` 表。

创建一个Python脚本（例如 `setup_db.py`），包含以下内容，并运行一次以创建和填充你的数据库：

```python
import sqlite3

def initialize_database():
    """如果product_catalog.db不存在，则创建并填充它。"""
    db_path = 'product_catalog.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建products表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock_quantity INTEGER
        )
    ''')

    # 在插入示例数据之前，检查表是否为空
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        sample_data = [
            ('Laptop Pro', 'Electronics', 1299.99, 50),
            ('Wireless Mouse', 'Electronics', 25.50, 200),
            ('Coffee Maker', 'Appliances', 79.00, 75),
            ('Office Chair', 'Furniture', 150.75, 30),
            ('Python Programming Book', 'Books', 45.00, 120)
        ]
        cursor.executemany('''
            INSERT INTO products (name, category, price, stock_quantity) 
            VALUES (?, ?, ?, ?)
        ''', sample_data)
        print("数据库已初始化并填充了示例数据。")
    else:
        print("数据库已存在并包含数据。")

    conn.commit()
    conn.close()

if __name__ == '__main__':
    initialize_database()
```

运行此脚本后，你将在同一目录下获得一个 `product_catalog.db` 文件。

### 设计数据库查询工具

我们的工具需要：

1. 接受来自LLM的SQL查询和可选参数 (parameter)。为了本练习的安全性和简单性，我们将限制工具只执行 `SELECT` 语句。
2. 连接到 `product_catalog.db`。
3. 使用参数化输入执行查询，以防止SQL注入。
4. 获取结果。
5. 将结果格式化为JSON字符串，LLM可以轻松解析这种格式。
6. 妥善处理潜在错误。

### 实现工具

让我们将此工具实现为一个Python函数。

```python
import sqlite3
import json

DB_PATH = 'product_catalog.db' # 定义数据库路径为常量

def query_product_database(sql_query: str, parameters: tuple = ()) -> str:
    """
    对product_catalog.db数据库执行只读SQL查询（SELECT）并返回结果。

    Args:
        sql_query (str): 要执行的SELECT SQL查询。
                         只允许SELECT语句。
        parameters (tuple): 一个参数元组，用于安全地替换到查询中。
                            在WHERE子句中使用此参数进行动态值替换。
                            示例：对于"WHERE category = ?"，参数为("Electronics",)

    Returns:
        str: 一个JSON字符串，表示查询结果（字典列表），
             如果查询失败或不允许，则返回错误消息。
    """
    # 输入验证：确保是SELECT查询
    if not sql_query.strip().upper().startswith("SELECT"):
        return json.dumps({
            "error": "无效操作。此工具只允许SELECT查询。"
        })

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(sql_query, parameters)

        rows = cursor.fetchall()

        if not rows:
            return json.dumps({
                "message": "查询执行成功。没有找到符合你条件的记录。"
            })

        # 从cursor.description获取列名
        column_names = [description[0] for description in cursor.description]

        # 将元组列表转换为字典列表
        results = [dict(zip(column_names, row)) for row in rows]

        return json.dumps(results, indent=2) # 缩进是为了可读性，LLM可选

    except sqlite3.IntegrityError as e: # 特定数据库错误的示例
        return json.dumps({
            "error": f"数据库完整性错误：{str(e)}。检查你的查询参数和表约束。"
        })
    except sqlite3.OperationalError as e: # 表未找到、语法错误的示例
        return json.dumps({
            "error": f"数据库操作错误：{str(e)}。检查你的SQL语法或表是否存在。"
        })
    except sqlite3.Error as e: # 捕获其他sqlite3特定错误
        return json.dumps({
            "error": f"发生数据库错误：{str(e)}"
        })
    except Exception as e: # 捕获其他任何意外错误
        return json.dumps({
            "error": f"查询执行期间发生意外错误：{str(e)}"
        })
    finally:
        if 'conn' in locals() and conn:
            conn.close()
```

**此实现的方面：**

- **Docstring：** Docstring很详细。这很重要，因为LLM代理框架通常使用docstring来了解工具的功能以及如何使用其参数 (parameter)。
- **类型提示：** `sql_query: str`、`parameters: tuple`、`-> str` 提高了代码可读性，并且可以被代理框架使用。
- **输入验证：** 我们明确检查查询是否以 `SELECT` 开头。对于执行任意SQL的工具来说，这是一个简单但重要的安全措施。在更高级的场景中，你可能需要更详细地解析SQL或使用查询构建器。
- **参数化查询：** 使用了 `cursor.execute(sql_query, parameters)`。这是防止SQL注入漏洞的标准方法。LLM将被指示分别提供查询结构和值。
- **结果格式化：** 结果被转换为字典列表，其中每个字典代表一行，键是列名。然后，此结构被序列化为JSON字符串，这是一种LLM可以轻松处理的格式。
- **错误处理：** `try...except` 块用于捕获各种数据库相关错误及其他异常，返回带有错误消息的JSON对象。在出现问题时，这会向LLM提供有用的反馈。
- **连接管理：** 数据库连接在 `finally` 块中关闭，以确保即使发生错误，连接也始终会被释放。

### 直接测试工具

在将此工具与LLM代理集成之前，直接测试它是一个很好的做法。你可以将以下内容添加到你的Python脚本中，以查看其运行效果：

```python
if __name__ == '__main__':
    print("请确保'product_catalog.db'存在。如果不存在，请运行'setup_db.py'。
")

    # 测试用例1：查询“电子产品”类别中的产品，成功
    print("测试用例1：电子产品类别中的产品")
    query1 = "SELECT name, price FROM products WHERE category = ? ORDER BY price DESC"
    params1 = ('Electronics',)
    result1 = query_product_database(query1, params1)
    print(json.dumps(json.loads(result1), indent=2)) # 格式化输出JSON
    print("-" * 40)

    # 测试用例2：查询特定产品，成功
    print("测试用例2：特定产品“办公椅”")
    query2 = "SELECT name, stock_quantity FROM products WHERE name = ?"
    params2 = ('Office Chair',)
    result2 = query_product_database(query2, params2)
    print(json.dumps(json.loads(result2), indent=2))
    print("-" * 40)

    # 测试用例3：查询无结果
    print("测试用例3：不存在的类别“玩具”")
    query3 = "SELECT name, price FROM products WHERE category = ?"
    params3 = ('Toys',)
    result3 = query_product_database(query3, params3)
    print(json.dumps(json.loads(result3), indent=2))
    print("-" * 40)

    # 测试用例4：尝试非SELECT查询（应被阻止）
    print("测试用例4：尝试UPDATE查询（安全检查）")
    query4 = "UPDATE products SET price = 10.00 WHERE name = ?"
    params4 = ('Wireless Mouse',)
    result4 = query_product_database(query4, params4)
    print(json.dumps(json.loads(result4), indent=2))
    print("-" * 40)

    # 测试用例5：语法错误的查询
    print("测试用例5：SQL语法错误查询")
    query5 = "SELEC name, price FROM products WHERE category = ?" # SELECT中故意打错字
    params5 = ('Books',)
    result5 = query_product_database(query5, params5)
    print(json.dumps(json.loads(result5), indent=2))
    print("-" * 40)

    # 测试用例6：查询不存在的列
    print("测试用例6：查询不存在的列")
    query6 = "SELECT non_existent_column FROM products WHERE category = ?"
    params6 = ('Books',)
    result6 = query_product_database(query6, params6)
    print(json.dumps(json.loads(result6), indent=2))
    print("-" * 40)
```

运行这些测试将帮助你验证该工具在各种场景下的行为是否符合预期，包括成功的查询、无结果的查询和错误情况。

### 与LLM代理集成

工具开发并测试完成后，就可以将其注册到LLM代理框架（如LangChain、LlamaIndex或自定义代理）中。代理在遇到需要产品信息的任务时，会：

1. **确定对工具的需求**：根据用户的查询（例如，“你们有什么50美元以下的电子产品？”）以及可用工具的描述，LLM将判断 `query_product_database` 是合适的。
2. **组织参数 (parameter)**：LLM将生成 `sql_query` 字符串和 `parameters` 元组。对于“你们有什么50美元以下的电子产品？”这个例子，它可能会生成：
   - `sql_query`： `"SELECT name, price FROM products WHERE category = ? AND price < ?"`
   - `parameters`： `('Electronics', 50.0)`
3. **调用工具**：代理框架使用这些参数调用我们的 `query_product_database` 函数。
4. **接收并处理结果**：工具返回的JSON字符串由LLM解析，然后LLM使用这些信息来为用户组织答案。

以下图表阐明了这种交互流程：

> LLM代理使用 `query_product_database` 工具回答用户问题的流程图。

本次动手练习展示了如何构建一种基本类型的工具，使LLM代理能够访问结构化数据。通过定义清晰的接口、仔细处理输入和输出以及确保错误管理，你可以创建功能强大且可靠的工具，大大扩展代理的能力。随着你构建更复杂的代理，你经常会发现自己创建了此类数据库交互工具的变体，以适应不同的数据源和需求。

## 参考资料

- [\`sqlite3\` - DB-API 2.0 interface for SQLite databases](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF2W4c5I_vjTsXXcP8kZ8hIbv6ZScZ0chlsza2DfzrVre0dTADK8Srgy30bP7FNpGxwy8-Izyt1DpWd9OFXFV8MKGOQCprRg40BVHzIbOzVqBOQQ8D-5GQu4_hBlyD8f3Mz2ltElN8w) — Python Software Foundation (2022)
  Python内置SQLite数据库接口的官方文档，内容涵盖连接、游标、查询执行和参数绑定。
- [SQL Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html) — OWASP Foundation (2023)
  提供预防SQL注入漏洞的最佳实践和策略，特别强调使用参数化查询来实现安全的数据库交互。
- [Python for Data Analysis](https://github.com/wesm/pydata-book) — Wes McKinney (2021)
  Publisher: O'Reilly Media, Inc.
  一本涵盖Python数据加载、清洗、转换和数据库交互的综合性书籍，为处理从数据库检索的数据提供了背景。
- [Tools | LangChain](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEbWeOYpTN0oB0TzJU7y0iLJG-yLtgBFhpOIjfhIxlAqVAHMt0MDoDg98eU1EwVDI8PaokOyB5lOsLYxpGvl0jbp1IjsoNZSW7ClC2y5xqxKwj4WRae0i9NkfwmCA7ou6H1fkLSrtxVrr9xs4BKMWc=) — LangChain Developers (2024)
  LangChain框架中创建和集成自定义工具的官方文档，对于将LLM代理连接到外部功能很有用。

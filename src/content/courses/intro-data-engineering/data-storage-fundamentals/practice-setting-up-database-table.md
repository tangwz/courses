---
course: "intro-data-engineering"
chapter: "data-storage-fundamentals"
lesson: "practice-setting-up-database-table"
sourceId: 5570
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-4-data-storage-fundamentals/practice-setting-up-database-table"
title: "实践：设置一个简单的数据库表"
description: "执行动手任务，在关系数据库中创建一个基本表。"
order: 7
plots: []
sourceHash: "05ad05c447037449ef31f8c77f685e869b0e3ea3fc427aa18c139df1fed835be"
sourceCorrections: []
---

关系数据库将数据组织成行和列的表，提供了一种结构化的数据存储方法。尽管存在多种数据存储方式，但关系数据库被广泛使用。本内容将指导你使用SQL（与关系数据库交互的标准语言）创建一个简单的表。

本练习将引导你定义表的结构（模式）并在数据库中创建它。我们将使用对处理结构化数据很重要的基本SQL命令。

### 前提条件

要跟随操作，你需要一种方法来针对关系数据库执行SQL命令。为简单起见和方便设置，我们推荐使用**SQLite**。它是一个轻量级、基于文件的数据库系统，不需要独立的服务器进程。

- **使用命令行：** 如果你安装了`sqlite3`（macOS和许多Linux发行版预装，Windows上易于安装），打开终端或命令提示符，输入`sqlite3 my_first_database.db`。此命令会创建（如果不存在）并打开一个名为`my_first_database.db`的数据库文件。
- **使用GUI工具：** 另外，你可以使用一个免费的图形工具，如[DB Browser for SQLite](https://sqlitebrowser.org/)。下载并安装它，然后通过其界面创建一个新的数据库文件。这些工具提供了一种可视化方式来执行SQL并查看结果。

选择你喜欢的方法。SQL命令本身是相同的。

### 定义表结构

假设我们想为应用程序创建一个表来存储用户的基本信息。我们将这个表命名为`users`。一个表需要列，每列都需要一个名称和数据类型。

我们将定义以下列：

1. `user_id`：每个用户的唯一标识符。这将是我们的**主键**，表示它唯一标识表中的每一行。我们将使用自动递增的整数类型。
2. `username`：用户选择的名称。这将是文本类型。
3. `email`：用户的电子邮件地址。这也应该是每个用户的唯一标识，并存储为文本类型。
4. `signup_date`：用户注册日期。我们将它存储为日期类型。

### `CREATE TABLE`语句

在SQL中，创建新表的命令是`CREATE TABLE`。其基本结构如下：

```sql
CREATE TABLE table_name (
    column1_name data_type constraints,
    column2_name data_type constraints,
    ...
    columnN_name data_type constraints
);
```

- `CREATE TABLE table_name`：指定要创建的表的名称。
- `(...)`：括号内包含列定义的列表。
- `column_name`：你为每列指定的名称。
- `data_type`：指定列将存储何种数据（例如，`INTEGER`、`TEXT`、`REAL`、`DATE`）。常见类型包括：
  - `INTEGER`：整数。
  - `TEXT`：字符串。
  - `REAL`：浮点数。
  - `DATE`：日历日期。
  - `BOOLEAN`：真或假值。
    *(注意：不同数据库系统（如PostgreSQL、MySQL和SQLite）之间的数据类型名称可能略有不同，但基本思想相似)。*
- `constraints`：应用于列的可选规则（例如，`PRIMARY KEY`、`NOT NULL`、`UNIQUE`）。
  - `PRIMARY KEY`：唯一标识每一行。一个表只能有一个主键。常用于ID列。在SQLite中，`INTEGER PRIMARY KEY`通常意味着自动递增行为。
  - `NOT NULL`：确保列不能有NULL（空）值。
  - `UNIQUE`：确保列中的所有值都是唯一的。

### 创建`users`表

现在，让我们将`users`表的期望结构转换为SQL `CREATE TABLE`语句。

```sql
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    signup_date DATE
);
```

让我们来分解一下：

- `CREATE TABLE users`：我们正在创建一个名为`users`的表。
- `user_id INTEGER PRIMARY KEY`：将`user_id`列定义为整数并使其成为主键。在SQLite中，这通常将其设置为自动递增。
- `username TEXT NOT NULL`：将`username`列定义为存储文本，并要求它必须有一个值（不能为空）。
- `email TEXT UNIQUE NOT NULL`：将`email`列定义为存储文本，确保每封电子邮件在所有行中都是唯一的，并要求它必须有一个值。
- `signup_date DATE`：将`signup_date`列定义为存储日期。在这个例子中，我们允许它为空值（NULL）。

### 执行命令

1. **如果使用`sqlite3`命令行工具：** 在提示符下输入或粘贴整个`CREATE TABLE`语句（从`CREATE`到最后的 `;`），然后按Enter键。如果没有错误，你将只获得一个新的提示行。
2. **如果使用DB Browser for SQLite：** 转到“执行SQL”选项卡，在编辑器窗口中输入或粘贴命令，然后点击“执行”按钮（通常看起来像一个“播放”图标）。

### 验证表创建

你怎么知道它成功了？

- **在`sqlite3`中：** 你可以输入`.tables`并按Enter键。你应该会看到`users`被列出。你也可以输入`.schema users`来查看你刚刚执行的`CREATE TABLE`语句。
- **在DB Browser for SQLite中：** `users`表应该出现在“数据库结构”选项卡或面板中，通常在左侧。你通常可以点击它来查看其列和数据类型。

### （可选）使用`INSERT INTO`添加数据

创建表是第一步。让我们使用`INSERT INTO`语句添加一行数据。

```sql
INSERT INTO users (username, email, signup_date) 
VALUES ('data_explorer', 'explorer@example.com', '2024-01-15');
```

*注意，我们不需要指定`user_id`，因为它被设置为自动递增。*

执行此命令，就像你执行`CREATE TABLE`命令一样。

### （可选）使用`SELECT`查看数据

要查看你刚刚插入的数据，请使用`SELECT`语句。星号`*`表示“选择所有列”。

```sql
SELECT * FROM users;
```

执行此命令。你应该会看到你插入的行：

```
1|data_explorer|explorer@example.com|2024-01-15
```

*(确切的输出格式可能因你使用的工具而略有不同)。*

恭喜！你已成功创建了你的第一个数据库表，使用适当的数据类型和约束定义了其结构，甚至还使用基本SQL添加和查看了一行数据。定义和创建表的这个过程是关系数据库中数据管理的一项基本活动，也是数据工程师的一项常见任务。

## 参考资料

- [CREATE TABLE](https://www.sqlite.org/lang_createtable.html) — D. Richard Hipp (2024)
  SQLite CREATE TABLE 语句的官方文档，详细说明了语法、数据类型和约束。
- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本全面的学术教科书（第7版），深入涵盖了关系数据库理论、SQL 基础知识和数据库设计原则。

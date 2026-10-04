---
course: "introduction-to-databases"
chapter: "working-with-databases"
lesson: "using-simple-database-environment"
sourceId: 1633
sourceUrl: "https://apxml.com/zh/courses/introduction-to-databases/chapter-5-working-with-databases/using-simple-database-environment"
title: "使用一个简单的数据库环境"
description: "指导学习者如何访问或设置一个简单的练习数据库环境（例如SQLite、在线SQL演练场）。"
order: 4
plots: []
sourceHash: "8686615d576fe7180465d519322977bad01d293ba3f5f2ce60b01a338fe6c957"
sourceCorrections: []
---

不同类型的数据库，包括它们的结构（特别是关系型数据库），以及用于交互的基本SQL命令，是核心概念。选择数据库时应考虑的因素以及用于数据库操作的工具种类，也是重要的知识。访问或设置一个直接明了的环境，能让你安全地练习编写和运行SQL查询，同时避免设置复杂生产级数据库服务器的麻烦。

这里的目标是练习，而非构建大型应用。因此，我们希望有一个设置迅速、易于使用且能让你专注于学习SQL和数据库知识的环境。在此阶段，复杂的配置、服务器管理和网络设置都是不必要的。这就像在驶上高速公路之前，在一个安全、空旷的停车场学习驾驶。

### 简单的练习环境选择

有几种很好的方法可以运行一个简单的数据库环境进行练习。以下是两种广受欢迎且非常适合初学者的做法：

1. **使用SQLite和GUI工具：**

   - **它是什么：** SQLite是一个优秀的数据库引擎，因为它*无服务器*且*自包含*。它将整个数据库（定义、表、数据）存储在您计算机上的一个文件中，通常以`.db`或`.sqlite`结尾。无需单独的服务器进程来安装、配置或管理。
   - **为何简单：** 因为它只是一个文件，入门极其容易。许多编程语言都内置支持它，并且有优秀的免费工具可用于直观地与其交互。
   - **推荐工具：DB Browser for SQLite：** 这是一个免费、开源的图形用户界面 (GUI) 工具，适用于Windows、macOS和Linux。它能让你：
     - 创建新的数据库文件。
     - 可视化地或使用SQL定义表结构。
     - 浏览表中的数据。
     - 在专用面板中编写并执行SQL查询，并立即查看结果。
   - **开始使用：** 从其官方网站下载并安装DB Browser for SQLite。安装完成后，您只需点击“新建数据库”，选择保存数据库文件的位置（例如，`practice.db`），然后使用“执行SQL”选项卡开始创建表和运行SQL命令。
2. **使用在线SQL演练场：**

   - **它们是什么：** 许多网站提供在线环境，常被称为“SQL Fiddle”或“演练场”，您可以在您的网络浏览器中直接编写和执行SQL查询，无需安装任何东西。例如DB Fiddle、SQL Fiddle等。
   - **为何简单：** 无需任何设置！您只需访问网站。它们通常提供预构建的示例数据库模式（如客户/订单数据库），或者让您使用SQL `CREATE TABLE`语句轻松定义自己的表。如果需要，您通常可以从不同的数据库系统（如PostgreSQL、MySQL、SQLite）中选择，以练习SQL语法的不同之处。
   - **开始使用：** 访问像DB Fiddle这样的网站。您通常会看到用于您的模式（SQL `CREATE`语句）、查询（SQL `SELECT`、`INSERT`等）和结果的面板。选择一种数据库类型（SQLite通常是一个好的起点），可以加载示例模式或在模式面板中编写自己的`CREATE TABLE`语句，然后在查询面板中编写查询并运行它们。

> 设置练习数据库环境的常见方式：本地使用SQLite和像DB Browser这样的工具，或使用基于网页的在线SQL演练场。

### 实际连接

您在上一节中了解了连接数据库的一般方法（通常涉及连接字符串、主机、凭证）。对于这些简单的环境：

- **SQLite：** “连接”通常指使用像DB Browser for SQLite这样的工具打开数据库文件。该工具处理与文件的直接交互。如果从代码中使用它，连接字符串通常只是`.db`文件的路径。
- **在线演练场：** 您不需要以传统方式明确地“连接”。您访问网站，平台会为您处理与后端数据库实例的连接。您的交互是通过网页界面进行的。

选择SQLite与GUI工具或在线演练场，提供了一种低障碍的方式，开始运用您在第3章中学到的SQL命令。下一节将指导您使用这样的环境进行实际练习。准备好执行您的第一个`SELECT`、`INSERT`、`UPDATE`和`DELETE`语句吧！

## 参考资料

- [SQLite Home Page](https://www.sqlite.org/) — D. Richard Hipp (2025)
  Publisher: SQLite Consortium
  SQLite数据库引擎的官方网站，提供全面的文档、功能介绍和下载链接。
- [DB Browser for SQLite](https://sqlitebrowser.org/) — DB Browser for SQLite Development Team (2024)
  Publisher: DB Browser for SQLite
  推荐用于管理和操作SQLite数据库的图形用户界面工具的官方来源。
- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本基础教材，深入介绍了数据库系统，包括关系模型、SQL和数据库设计原则。
- [Learning SQL: Master SQL Fundamentals](https://www.oreilly.com/library/view/learning-sql-3rd/9781492057611/) — Alan Beaulieu (2020)
  Publisher: O'Reilly Media; Pages: 377
  一本面向初学者的实用指南，学习SQL基础知识，涵盖适用于不同数据库系统的各种SQL命令和概念。

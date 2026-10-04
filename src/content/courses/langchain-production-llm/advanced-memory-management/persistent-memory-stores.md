---
course: "langchain-production-llm"
chapter: "advanced-memory-management"
lesson: "persistent-memory-stores"
sourceId: 3875
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-3-advanced-memory-management/persistent-memory-stores"
title: "实现持久化记忆存储"
description: "配置和管理用于对话记忆的持久化存储（数据库、文件系统）。"
order: 2
plots: []
sourceHash: "2ae07d8398bd8237f89dfcc7a0166f22075e27678fc6c1be5cc55b930ce36a4d"
sourceCorrections: []
---

尽管基本内存缓冲区将对话历史存储在易失性内存中，但生产应用通常要求数据持久性。交互可能跨越多个会话，需要长期保存用户特定上下文 (context)，或者需要在应用重启和部署后依然存在。实现持久化记忆存储可以满足这些要求，确保对话上下文不会丢失。

本节将介绍如何配置和使用LangChain记忆模块的持久化存储后端。我们将介绍常见的数据库选择，并演示LangChain的抽象层如何让集成变得简单。

## 持久性的必要性

内存存储对于短时交互或开发来说简单快速。然而，它在生产环境中存在明显局限：

1. **易失性：** 应用实例停止或崩溃时，数据会丢失。
2. **可扩展性：** 在无状态应用的多个实例之间，内存不容易共享。每个实例将拥有自己独立的内存。
3. **长期上下文 (context)：** 将大量历史记录完全存储在RAM中会变得低效且消耗大量资源。

持久化存储通过将对话历史保存到数据库或文件系统等持久性存储中来解决这些问题。这使得：

- **连续性：** 用户可以在不同会话或设备之间恢复对话。
- **无状态环境中的有状态性：** 部署为无状态服务的应用（例如，无服务器函数、容器编排器）可以从共享的持久化存储中获取对话上下文。
- **分析和审计：** 持久化后的历史记录可以用于分析或合规性目的。

## 选择持久化后端

LangChain通过其`BaseChatMessageHistory`接口，提供对各种存储解决方案的内置支持。后端的选择依赖于您的应用规模、现有基础设施和具体要求。

### 关系型数据库 (SQL)

PostgreSQL、MySQL、SQLite或SQL Server等数据库提供结构化存储、ACID一致性以及强大的查询功能。

- **优点：** 成熟的技术、事务完整性，适用于与消息一起存储结构化元数据。
- **缺点：** 可能需要模式管理；对于简单的消息列表而言，可能感觉基础设施过于繁重。
- **LangChain集成：** `SQLChatMessageHistory`（使用SQLAlchemy）。需要一个表来存储消息，通常包含会话ID、消息类型（人类/AI）和内容。

### NoSQL数据库

这些数据库提供更大的灵活性，对于大型数据集通常具有更好的横向扩展性。

- **Redis：** 一种内存数据结构存储，常用于缓存，但具备持久化能力。非常适合低延迟访问。
  - **优点：** 读写速度非常快，简单的键值或列表存储。
  - **缺点：** 主要基于内存（尽管存在持久化选项），查询功能不如SQL。
  - **LangChain集成：** `RedisChatMessageHistory`。通常将消息存储在与会话ID关联的Redis列表中。
- **MongoDB：** 一种流行的文档数据库，以类似JSON的BSON格式存储数据。
  - **优点：** 灵活的模式，适合半结构化数据，易于扩展。
  - **缺点：** 在分布式设置中可能需要考虑最终一致性模型。
  - **LangChain集成：** `MongoDBChatMessageHistory`。将消息作为文档存储在集合中，通常通过会话ID进行索引。
- **Cassandra：** 一种分布式宽列存储，专为高可用性和可扩展性设计。
  - **优点：** 处理海量数据集和高写入吞吐量 (throughput)，容错性强。
  - **缺点：** 管理起来更复杂，需要特定的查询模式才能获得良好性能。
  - **LangChain集成：** `CassandraChatMessageHistory`。

### 文件系统

直接在文件中存储历史记录是适用于特定情况的简单选择。

- **优点：** 设置极其简单，无需外部数据库。适用于单用户应用或本地开发/测试。
- **缺点：** 不适合并发用户扩展，管理文件锁可能很复杂，不适合分布式应用。
- **LangChain集成：** `FileChatMessageHistory`。通常将消息以JSON或文本形式保存在以会话ID命名的文件中。

## LangChain的`ChatMessageHistory`抽象

核心思想是将消息存储与对话逻辑分离。LangChain使用`BaseChatMessageHistory`的实现来处理从不同后端（如Redis或SQL）保存和获取消息的细节。

在现代LangChain表达式语言（LCEL）中，您无需手动将历史对象传递给记忆类。相反，您可以使用`RunnableWithMessageHistory`。这个包装器接受一个工厂函数，该函数接受`session_id`并为该会话返回相应的`BaseChatMessageHistory`实例。这确保在链运行前自动加载正确的历史记录，并在完成后进行更新。

## 实现范例

让我们看看如何使用LCEL配置不同后端的持久性。

### 使用Redis

首先，确保您已安装必要的库：`pip install redis langchain-community`。

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# 配置
REDIS_URL = "redis://localhost:6379/0" # 替换为您的Redis实例URL

# 1. 定义历史工厂
# 此函数为给定会话ID返回历史对象
def get_redis_history(session_id: str):
    return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)

# 2. 设置LLM和提示
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 3. 创建链
# 注意这里我们不绑定记忆；链是无状态逻辑
chain = prompt | llm

# 4. 用消息历史包装
# 这增加了有状态的持久性层
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_redis_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 5. 使用会话配置调用
config = {"configurable": {"session_id": "user123_conversation456"}}

# 第一次交互
response1 = chain_with_history.invoke({"input": "Hi! My name is Bob."}, config=config)
print(response1.content)

# 第二次交互（记忆从Redis加载）
response2 = chain_with_history.invoke({"input": "What is my name?"}, config=config)
print(response2.content)

# 验证消息是否在Redis中（使用redis-cli）：
# > KEYS *
# > LRANGE message_store:user123_conversation456 0 -1
```

在此范例中，`RunnableWithMessageHistory`使用在`config`中找到的`session_id`调用`get_redis_history`。它将加载的消息注入到提示的`chat_history`占位符中，并自动将新的交流保存回Redis。

### 使用SQL数据库（SQLite范例）

首先，安装所需的库：`pip install sqlalchemy langchain-community`。

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# 配置
DB_CONNECTION = "sqlite:///langchain_memory.db"

# 1. 定义历史工厂
def get_sql_history(session_id: str):
    # 如果表不存在，它会自动创建
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=DB_CONNECTION,
        table_name="message_store"
    )

# 2. 设置LLM和提示（与Redis范例相同）
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的旅行代理。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 3. 创建链
chain = prompt | llm

# 4. 用消息历史包装
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_sql_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 与特定会话ID的交互
config = {"configurable": {"session_id": "user789_chat001"}}

chain_with_history.invoke({"input": "I want to plan a trip to Italy."}, config=config)
response = chain_with_history.invoke({"input": "What was the destination I mentioned?"}, config=config)
print(response.content)

# 验证SQLite数据库中的消息：
# > sqlite3 langchain_memory.db
# > SELECT * FROM message_store WHERE session_id = 'user789_chat001';
```

这种设置允许您通过更改传递给`RunnableWithMessageHistory`的工厂函数，轻松切换后端。

## 生产环境方法

在生产环境中实现持久化记忆时，请考虑以下事项：

- **会话ID管理：** 这很基础。您需要一种方式为每个独立的对话或用户会话生成和管理唯一的会话ID。此ID将请求关联到持久化存储中正确的聊天历史记录。常用方法包括使用用户ID、浏览器会话令牌或专用的对话标识符。
- **数据库性能：** 选择一个能处理预期负载的数据库。对`session_id`列（或等效列）进行索引对于快速查询来说是必要的。监控查询性能，并在需要时考虑数据库扩展方法（读副本、分片）。
- **连接池：** 对于高吞吐量 (throughput)应用，特别是SQL数据库，使用连接池来有效管理数据库连接，并避免为每个请求建立新连接的开销。SQLAlchemy等库提供连接池机制。
- **数据模式和演变：** 虽然NoSQL提供灵活性，但拥有一定的结构是有益的。决定消息将如何存储（例如，每条消息一个文档，每个会话一个消息列表）。规划未来可能对与消息一起存储的数据进行的更改。
- **安全：** 安全地管理数据库凭据。使用环境变量或密钥管理系统（如AWS Secrets Manager、HashiCorp Vault）而不是硬编码凭据。确保您的数据库有适当的网络安全（防火墙）。请参阅第8章以了解更多安全信息。
- **备份和恢复：** 按照标准的数据库管理实践，为您的持久化存储实施定期备份。
- **成本：** 考虑托管数据库服务的成本或托管所选后端所需的基础设施成本。

通过仔细选择和配置使用LangChain的`BaseChatMessageHistory`实现的持久化后端，您可以构建能够有效维护长期对话上下文 (context)的有状态LLM应用。这个持久化层是迈向提供连续且连贯用户体验的生产就绪系统的一大步。

## 参考资料

- [Chat Message History](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGZPOWGSPKx7WCyTBvVqSYTlaHzX6jHWGLc22HbLwItO0UtgnjJ-ELn_MmmXhMWsUcSDVYTujZiO0hgvXZhnR6CzigslXhzzPb_2uWb3OQPguVV-2yp5QKnlTkzzjUoEPb40e-1Lo6XRLrlqGK1W-IUEynWj_PwGoDQhEnCro2NfC9RWw==) — LangChain (2024)
  Publisher: LangChain
  了解LangChain消息历史抽象及其各种持久化存储实现的指南。
- [Database System Concepts](https://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  全面介绍关系型和NoSQL数据库原理、事务管理、索引和模式设计的书籍，对后端选择和性能有帮助。
- [Redis Persistence](https://redis.io/topics/persistence) — Redis (2024)
  Publisher: Redis
  解释Redis持久化机制（RDB和AOF），对于将Redis用于内存存储时确保数据持久性至关重要。
- [System Design Interview - An Insider's Guide](https://openlibrary.org/works/OL22244243W/System_Design_Interview_-_an_Insider%27s_Guide) — Alex Xu, Sahn Lam (2020)
  Publisher: Independently published
  提供实用的系统设计模式和考量，涵盖数据库选择、可扩展性和分布式系统，适用于设计持久化LLM应用。

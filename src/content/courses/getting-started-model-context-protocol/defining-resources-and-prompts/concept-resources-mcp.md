---
course: "getting-started-model-context-protocol"
chapter: "defining-resources-and-prompts"
lesson: "concept-resources-mcp"
sourceId: 7843
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-2-defining-resources-and-prompts/concept-resources-mcp"
title: "MCP中的资源理念"
description: "领会资源作为向客户端公开数据上下文的主要方式。"
order: 1
plots: []
sourceHash: "d516f6c098cf65c398844ed516b6d9b3ef46d6b6635808cdd5f83d30c07adb4f"
sourceCorrections: []
---

模型上下文 (context)协议的中心是，将大型语言模型植根于具体的外部数据中的方法。架构规定了组件如何连接，而资源则界定了这些组件了解什么。资源代表MCP服务器向客户端公开的某个数据项。这些数据作为上下文窗口内容，支持大型语言模型回答关于您的具体代码库、数据库或内部文档的问题。

在常规软件工程术语中，您可以将资源视为REST API中的GET端点。它是一个只读接口，旨在供客户端应用程序使用。然而，与旨在供程序使用的普通API不同，MCP资源经过特殊结构化，供大型语言模型使用，它们提供元数据和内容格式，模型可以轻松获取并进行处理。

### 资源构成

MCP中的资源不仅仅是原始文本字符串。它是一个结构化对象，含有三个主要部分，支持客户端识别、获取和解释数据。

1. **URI（统一资源标识符）：** 这是资源的独有地址。它遵循常规URI语法，充当访问数据的主要标识符。URI的方案通常表明数据源的性质，例如 `file:///logs/error.txt` 或 `postgres://users/schema`。
2. **名称：** 一个人类可读的标识符，说明资源。这对于客户端的用户界面来说十分重要，它让用户可以选择希望将哪个上下文 (context)附加到对话中。
3. **MIME类型：** 这表明内容的格式，例如 `text/plain`、`application/json` 或 `image/png`。MIME类型告知客户端如何在将数据提供给模型之前处理或渲染数据。

规范地说，我们可以将资源 $R$ 界定为一个元组，它包含这些元素以及读取时的实际数据载荷：

$R = (u, n, m, c)$

其中 $u$ 是URI，$n$ 是名称，$m$ 是MIME类型，$c$ 是内容块。服务器负责维护URI $u$ 和内容 $c$ 之间的映射。

### 资源生命周期

客户端（如Claude Desktop或IDE）与MCP服务器之间关于资源的互动，遵循特定的发现和获取模式。客户端不会自动获取服务器上所有可用数据，因为这会使上下文 (context)窗口不堪重负。相反，协议要求采用“先列表后读取”的工作流程。

首先，客户端请求一份可用资源列表。服务器回复一个它公开的目录，包含URI和名称，但不含大量内容。然后，用户或大型语言模型挑选与当前任务相关的具体资源。最后，客户端发出针对这些具体URI的读取请求，以获取完整内容。

> 此序列图展现了发现（列出）和获取（读取）之间的区分。这种惰性加载方法改善了带宽和上下文窗口的使用。

### 静态资源与动态资源

MCP中的资源用途广泛。它们可以表示磁盘上的静态文件，也可以表示按需产生的动态数据。

**静态资源**直接对应不变数据。例如，服务器公开一个日志文件或配置文档。当客户端读取 `file:///project/config.json` 时，服务器打开该文件，读取字节，并将其发送回去。内容仅在磁盘上的文件发生变化时才变化。

**动态资源**在请求时被计算或获取。URI为 `weather://current/sf` 的资源并非指向一个文件。相反，当客户端请求此URI时，服务器的内部逻辑会启动对天气提供商的API调用，格式化结果，并将其作为资源内容返回。对于客户端和大型语言模型而言，读取文件与读取动态系统状态没有作用上的区别；两者都只是从URI返回的文本。

### 区分资源与工具

对于初次接触MCP的开发者来说，一个普遍的困惑点在于资源与工具的差异。两者都涉及数据传送，但其意图根本不同。

资源是被动的。它们表示一种“读取”操作。读取资源应该是幂等的，这意味着多次读取它不会改变系统的状态（除了可能记录访问之外）。资源提供上下文 (context)，协助大型语言模型理解情境。

工具是主动的。它们表示可执行函数，可以完成“写入”操作或需要参数 (parameter)的繁复逻辑。一个工具可能是 `execute_sql_query(query)` 或 `send_slack_message(channel, text)`。这些是大型语言模型根据资源提供的上下文判定的行动。

您应在以下情况将数据界定为资源：

1. 数据是只读的。
2. 数据充当参考资料（代码、日志、文档）。
3. 数据获取不需要繁复参数（它可通过单个URI定位）。

您应在以下情况将功能界定为工具：

1. 运作有副作用（更改数据库，发送电子邮件）。
2. 运作需要数个参数（例如，使用具体筛选器搜索数据库）。
3. 产出是行动的成果，而非对象的状态。

### 资源模板

在许多场景下，服务器可能含有数千个类似项，例如数据库表中的行或支持系统中的工单。在 `resources/list` 响应中单独列出每个项目效率低下且常常办不到。

MCP通过资源模板处理此问题。服务器不列出静态URI，而是公开一个包含变量的URI模式，例如 `postgres://users/{id}/profile`。这告知客户端，它可以通过用具体用户ID替换 `{id}` 来构建一个合规URI。尽管大型语言模型不能以列表形式“浏览”这些，但如果它从先前的上下文 (context)中知道ID，它就可以基于该模式生成（构建）一个合规URI，支持直接存取细粒度数据，而不会造成目录膨胀。

通过规范化数据的识别和取回方式，资源支持开发者建立一个通用上下文层。无论数据是源自本地文件系统、云数据库还是实时API，MCP资源原语都会将其规范化为一种格式，从而提升关联AI模型的智能和实用性。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  描述了利用外部知识来为大型语言模型提供基础的架构和方法，这是MCP资源概念的核心。
- [Architectural Styles and the Design of Network-based Software Architectures](https://www.ics.uci.edu/~fielding/pubs/dissertation/top.htm) — Roy T. Fielding (2000)
  Journal: PhD Dissertation, University of California, Irvine
  定义了表征状态传输（REST）架构风格，介绍了资源及其通过URI识别的理念，这为MCP的方法提供了依据。
- [Function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2023)
  Publisher: OpenAI
  提供关于大型语言模型如何与外部函数交互以执行操作的指南，阐明了被动资源与主动工具之间的区别。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  一本全面的教材，涵盖了信息检索的基本原则，包括索引、查询和高效数据访问模式，与“先列表后读取”工作流相关。

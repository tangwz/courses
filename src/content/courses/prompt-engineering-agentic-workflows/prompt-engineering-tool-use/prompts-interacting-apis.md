---
course: "prompt-engineering-agentic-workflows"
chapter: "prompt-engineering-tool-use"
lesson: "prompts-interacting-apis"
sourceId: 6729
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-3-prompt-engineering-tool-use/prompts-interacting-apis"
title: "与API交互的提示"
description: "编写指导代理进行API调用并处理其响应的提示。"
order: 5
plots: []
sourceHash: "97e7cb8dc7ac76b1f9ecf25a2c7742eedce143a9027be0551cb8ade9abe1cfbb"
sourceCorrections: []
---

应用程序编程接口（API）是连接AI代理与互联网及私有网络中各种服务和数据源的核心工具。当代理需要获取实时股票价格、翻译文本、查询客户数据库或与几乎任何外部系统通信时，它很可能会通过API进行。在此，我们将研究如何编写提示，以引导代理正确地进行这些API调用，并理解和使用返回的信息，主要关注针对API交互编写提示的具体要求。

主要目的是在提示中向代理提供所有必要的信息和指示，使其能够准确地构建和执行API请求，然后处理后续响应。

### API交互提示中的要素

为了成功引导代理使用API，您的提示通常需要传达以下几部分信息：

1. **API端点URL：** 代理需要发送请求的特定地址。这必须精确。
   - 示例：`https://api.weather.com/v3/weather/current?location=NewYork`
2. **HTTP方法：** 要发出的请求类型。常见方法包括：
   - `GET`：用于获取数据。
   - `POST`：用于提交数据以创建新资源。
   - `PUT`：用于更新现有资源。
   - `DELETE`：用于删除资源。
     您的提示应明确指示代理针对预期操作使用哪种方法。
3. **请求头：** API请求通常需要请求头，用于认证、指定内容类型或其他元数据。
   - `Authorization`：用于API密钥或令牌（例如，`Authorization: Bearer YOUR_API_KEY`）。
   - `Content-Type`：对于`POST`或`PUT`请求，指示请求体的格式（例如，`Content-Type: application/json`）。
     提示应指示代理包含必要的请求头。
4. **请求体（负载）：** 对于`POST`和`PUT`请求，这是发送到API的数据。它通常是JSON格式。提示必须指导代理如何组织此负载，可能使用它之前收集的信息。
5. **查询参数 (parameter)：** 对于`GET`请求，参数通常附加到URL上，以筛选或指定请求。提示需要帮助代理识别并格式化这些参数。

### 提示API调用执行

您的提示不仅应描述API，还应指示代理*何时*以及*如何*进行调用。这包括告知代理：

- 根据提供的规范（端点、方法、请求头、请求体、参数 (parameter)）组装请求。
- 调用负责发起实际HTTP请求的工具或函数。

设计提示时，鼓励代理首先阐明它打算进行的API调用是一个好的做法。例如，代理可能会在执行请求之前输出一个表示该请求的JSON对象。这为调试提供了一个检查点，并确保代理正确理解了指示。

```text
你需要获取巴黎的当前天气。
使用天气API，详情如下：
- API端点: https://api.exampleweather.com/current
- 方法: GET
- 查询参数:
  - city: [城市名称]
  - units: metric
- 必需的请求头:
  - X-API-Key: [你的API密钥，工具将从安全存储中获取]

组织API请求。然后，使用'call_api'工具执行请求。
```

### 处理API响应

API调用完成后，代理会收到响应。提示必须指导代理如何处理此响应：

1. **解析状态码：** HTTP状态码（例如，200表示成功，401表示未经授权，404表示未找到）非常重要。提示可以指示代理预期的成功码以及如何对不同的错误码作出反应。
2. **提取数据：** API响应（通常为JSON或XML格式）包含请求的信息。您的提示需要告知代理从响应中提取哪些字段。
   - 示例：“从JSON响应中，提取位于'main'对象下的'temperature'字段的值。”
3. **使用数据：** 提取的数据应随后用于告知代理下一步行动，更新其内部状态或记忆，或呈现给用户。
4. **错误处理：** 如果API返回错误，提示应指导代理如何报告错误或尝试纠正措施（例如，如果是暂时性错误则重试）。这与处理工具执行错误这个更广泛的议题密切相关。

以下图表描绘了代理在提示指引下与API交互的典型流程。

> 代理接收提示，解析提示以构建API请求，通过工具执行调用，处理响应，然后决定其下一步行动。

### API交互提示示例

我们来看几个例子。

#### 示例1：获取用户数据的GET请求

假设代理需要使用`user_id`获取用户信息。

**提示：**

```text
目标：获取user_id 'u101'的用户详情。

API交互详情：
使用工具: 'api_caller'
端点: 'https://api.example.com/users/{user_id}' (将{user_id}替换为实际ID)
方法: 'GET'
必需的请求头:
  'Authorization': 'Bearer YOUR_SECURE_TOKEN_HERE' (假设工具安全地处理令牌注入)

指示：
1. 构建完整的API请求URL。
2. 使用方法和URL调用'api_caller'工具。
3. 如果请求成功（状态码200），从JSON响应中提取'email'和'full_name'。
4. 将提取的电子邮件存储为'user_email'，全名存储为'user_name'，以备后续任务使用。
5. 如果状态码是404，报告用户未找到。对于其他错误，报告状态码和错误信息。
```

**预期代理行为（简化）：**

1. 代理识别`user_id`为'u101'。
2. 构建URL：`https://api.example.com/users/u101`。
3. 准备使用`GET`方法、URL和必要的请求头调用`api_caller`。
4. 调用后，检查状态码。
5. 如果是200，则解析JSON（例如，`{"id": "u101", "email": "test@example.com", "full_name": "Test User"}`），并提取`test@example.com`和`Test User`。
6. 存储这些值。

#### 示例2：创建任务的POST请求

假设代理需要在项目管理系统中创建新任务。

**提示：**

```text
目标：创建新任务，标题为'最终确定Q3报告'，描述为'起草、审查并提交Q3财务报告。'

API交互详情：
使用工具: 'api_caller'
端点: 'https://api.projectmanager.com/tasks'
方法: 'POST'
必需的请求头:
  'Authorization': 'Bearer YOUR_SECURE_TOKEN_HERE'
  'Content-Type': 'application/json'
请求体（JSON结构）：
{
  "title": "[任务标题]",
  "description": "[任务描述]",
  "status": "pending"
}

指示：
1. 根据目标，准备请求体的JSON负载。新任务的状态应始终为'pending'。
2. 使用方法、URL、请求头和JSON负载调用'api_caller'工具。
3. 如果请求成功（状态码201 Created），从响应中提取新创建任务的'id'。
4. 报告：“任务成功创建，ID为：[task_id]”。
5. 如果是任何其他状态码，报告错误。
```

**预期代理行为（简化）：**

1. 代理组织JSON负载：

   ```json
   {
     "title": "Finalize Q3 Report",
     "description": "Draft, review, and submit the Q3 financial report.",
     "status": "pending"
   }
   ```
2. 准备使用`POST`方法、URL、请求头和此JSON体调用`api_caller`。
3. 调用后，检查状态码。
4. 如果是201，则解析响应（例如，`{"id": "task_789", "title": "...", ...}`），并提取`task_789`。
5. 报告成功。

### 提示API交互的最佳做法

- **具体明确：** 清楚说明端点、方法、预期参数 (parameter)和数据格式。模糊性会导致不正确的API调用。
- **提供示例（少样本）：** 如果API的请求或响应结构复杂，直接在提示中包含一个小示例可以极大帮助大型语言模型。

  ```text
  ...
  请求体（JSON结构）：
  {
    "item_name": "[项目名称]",
    "quantity": [项目数量],
    "details": {
      "color": "[颜色]",
      "size": "[尺寸]"
    }
  }
  例如，要订购2件M码红色衬衫：
  {
    "item_name": "shirt",
    "quantity": 2,
    "details": {
      "color": "red",
      "size": "M"
    }
  }
  现在，创建一个请求以订购3顶L码蓝色帽子。
  ...
  ```
- **引用API文档片段：** 对于更复杂的API，您可以在提示中包含API文档中的相关片段（如参数描述、可接受的值或示例响应）。这为代理提供了直接、权威的信息。
- **凭证安全：** 避免将API密钥或令牌等敏感信息直接硬编码到可能被记录或广泛可见的提示中。相反，指示代理使用命名凭证或占位符，由底层工具执行环境安全地解析。例如，“使用名为'WEATHER\_API\_KEY'的API密钥”。运行代理的系统随后负责在调用工具时安全地注入此密钥。
- **迭代开发和测试：** API交互可能很精细。请预期需要测试和改进您的提示。记录代理组织好的请求和API的响应，以有效调试问题。提示措辞的微小变化都可能显著改变代理对API要求的理解方式。

通过掌握这些提示API交互的技巧，您能显著扩展AI代理的能力，使其能够获取外部数据和功能。这是构建能够执行有用任务的代理的基本技能。

## 参考资料

- [Function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2023)
  Publisher: OpenAI
  官方文档，详细介绍了如何通过在提示中提供结构化函数描述，使大型语言模型 (LLM) 能够调用外部工具或 API。
- [Toolformer: Language Models That Can Use Tools](https://arxiv.org/abs/2302.04761) — Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2302.04761](https://doi.org/10.48550/arXiv.2302.04761)
  一篇研究论文，介绍了语言模型通过自监督训练学习使用外部工具（包括 API）的方法，为代理工具使用奠定了基础。
- [An overview of HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview) — Mozilla Contributors (2024)
  Publisher: Mozilla
  提供了超文本传输协议 (HTTP) 的基础知识，包括请求方法、状态码和头部信息，这些对于与 Web API 交互至关重要。
- [LangChain Agents](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHupi0VHrkphW5-oFAnYdjG7TgvbbhTlbK0y1Nytvk4r9Qw6qwbEvGT3mtiYgXgs4bKrGchgIB2EIQX8VAT8jTF1dKIXJg1bMCG5eLbTwE5Mor3NXq96BQVTMSLThye9oMPUa3-unEDDrMS1WHJroHeOPKqFTQw) — LangChain Inc. (2025)
  Publisher: LangChain Inc.
  LangChain 代理的官方文档，展示了构建能够与包括 API 在内的各种工具进行交互的基于 LLM 的代理的实用模式和实现方法。

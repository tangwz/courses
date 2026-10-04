---
course: "prompt-engineering-agentic-workflows"
chapter: "prompt-engineering-tool-use"
lesson: "addressing-tool-errors-via-prompts"
sourceId: 6730
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-3-prompt-engineering-tool-use/addressing-tool-errors-via-prompts"
title: "通过提示处理工具执行中的错误"
description: "设计提示，帮助代理处理工具执行中遇到的错误并尝试恢复。"
order: 6
plots: []
sourceHash: "9c126542144ed2bc2b484a08ae307176f9de7965e6dddf529061e3c190c9dabb"
sourceCorrections: []
---

AI 代理在工具执行过程中会遇到错误，即使为工具选择和操作设计了良好的提示。API 可能会暂时无法使用，输入可能以意外方式格式不正确，工具可能返回错误，或者网络问题可能中断通信。构建工作流程需要预见这些问题，并向代理提供如何处理它们的指示。提示工程 (prompt engineering)是指导代理应对此类故障的 主要 方法，使其能够恢复、重试，或者至少以恰当方式结束。

### 工具执行错误的常见来源

当代理尝试使用外部工具时，可能出现多种错误。了解这些类别有助于编写能有效处理它们的提示：

1. **连接与可用性错误**：

   - **网络超时**：工具或 API 服务器无响应。
   - **服务不可用 (HTTP 503)**：服务器暂时过载或因维护而关闭。
   - **DNS 解析失败**：找不到工具的地址。
2. **API 使用错误（客户端）**：

   - **认证/授权错误 (HTTP 401/403)**：代理缺少有效的凭据或权限。
   - **错误请求 (HTTP 400)**：提供给工具的输入格式不正确、缺少必要参数 (parameter)或使用了错误的数据类型。
   - **未找到 (HTTP 404)**：特定的 API 端点或资源不存在。
   - **频率限制 (HTTP 429)**：代理在给定时间内发出了过多请求。
3. **工具特定操作错误（服务器端或工具逻辑）**：

   - **内部服务器错误 (HTTP 500)**：服务器端的通用错误，表示工具本身遇到了问题。
   - **工具特定错误码/消息**：许多工具在其响应体中返回自定义错误消息或代码（例如，`{"status": "error", "message": "Invalid search query"}`）。
   - **意外输出格式**：工具返回的数据格式，代理未准备好处理，即使从工具的角度来看这并非严格意义上的“错误”。
4. **资源错误**：

   - **文件未找到**：如果工具尝试访问不存在的本地文件。
   - **权限不足**：代理或工具缺少操作系统级别的权限。

### 错误检测和初步处理的提示策略

处理错误的第一步是让代理识别出错误已经发生。您的提示应指导代理检查工具输出，寻找问题的迹象。

- **检查状态码**：对于通过 HTTP 交互的工具（如大多数 API），指示代理检查 HTTP 状态码。

  ```text
  在调用 `weather_api` 后，检查 HTTP 状态码。状态码 200 表示成功。任何其他状态码，特别是属于 4xx 或 5xx 范围的，都应视为错误。
  ```
- **解析错误字段**：许多 API 返回结构化错误响应（例如，带有 `error` 键的 JSON）。

  ```text
  使用 `user_database_tool` 时，检查 JSON 响应。如果响应包含名为 "error_message" 或 "detail" 的顶级键，则表示发生了错误。提取此键的值以进行进一步处理。
  ```
- **关键词识别**：对于结构化程度较低的输出或工具日志，您可以指示代理寻找特定的错误关键词。

  ```text
  如果 `image_processing_script` 的输出包含 "failed"、"exception"、"Traceback" 或 "Error:" 等术语，则假定发生了错误。
  ```

### 通过提示指导代理应对错误

一旦检测到错误，代理需要关于如何继续的指示。您的提示可以指定一系列行动：

1. **简单重试**：对于网络故障或临时服务不可用等瞬时问题，简单的重试（可能带延迟）通常是有效的。

   ```text
   如果 `stock_quote_api` 返回 503 服务不可用错误或发生超时，等待 3 秒，然后再次尝试 API 调用。如果第二次仍然失败，报告此次失败。
   ```
2. **修改输入后重试**：如果错误表明输入有问题（例如，“错误请求”错误），代理可能会被提示在可行的情况下使用更正或简化的输入再次尝试。

   ```text
   如果 `product_search_tool` 对复杂查询返回“InvalidParameterValue”错误，尝试通过删除可选过滤器来简化查询，并再次尝试搜索。例如，如果搜索“blue striped cotton shirt size M”失败，则尝试“blue shirt”。
   ```
3. **使用备用工具或策略**：如果主要工具持续失败，您可以指示代理尝试替代方案。

   ```text
   如果 `primary_translation_service` 两次尝试后仍无法提供翻译，则使用 `secondary_translation_service` 处理相同的输入文本。如果两者都失败，通知用户翻译目前不可用。
   ```
4. **请求澄清或更正**：如果错误源于代理无法自行解决的模糊或不正确的用户输入，提示代理向用户寻求帮助。

   ```text
   如果 `calendar_tool` 报告“日期格式模糊”错误，请要求用户以“YYYY-MM-DD”格式提供日期。
   ```
5. **服务降级**：有时，非关键工具的故障意味着任务的一部分无法完成，但总体目标可能仍然部分实现。

   ```text
   生成研究报告时，如果 `fetch_latest_news_tool` 失败，请仅使用 `company_database_tool` 中的信息继续生成报告，并在报告中包含一条注释：“由于新闻服务暂时出现问题，无法获取最新新闻。”
   ```
6. **结构化错误报告**：为了调试和监控，让代理以一致的格式报告错误很有用。

   ```text
   如果任何工具执行导致无法恢复的错误，输出以下信息：
   工具名称：[失败工具的名称]
   提供输入：[提供给工具的精确输入]
   错误类型：[例如，HTTP_500、API_Auth_Error、Timeout、Malformed_Output]
   错误详情：[从工具收到的具体错误消息或代码]
   恢复尝试：[重试次数、尝试过的替代工具]
   ```

### 示例：API 错误处理的提示

考虑一个负责从 API 获取用户数据的代理。

**初始提示片段（无错误处理）：**

```text
目标：获取用户 ID 123 的用户详情。
可用工具：`get_user_data(user_id)` - 调用 `https://api.example.com/users/{user_id}`。
```

如果 `api.example.com` 关闭，代理可能会停止或返回原始网络错误。

**改进的提示片段（带错误处理）：**

```text
目标：获取用户 ID 123 的用户详情。
可用工具：`get_user_data(user_id)` - 调用 `https://api.example.com/users/{user_id}`。

`get_user_data` 的指令：
1. 调用 API。
2. 如果 HTTP 状态为 200，返回 JSON 响应。
3. 如果 HTTP 状态为 401 或 403（认证/授权错误）：报告“API 访问被拒绝。请检查凭据。”不要重试。
4. 如果 HTTP 状态为 404（未找到）：报告“用户 ID 未找到。”不要重试。
5. 如果 HTTP 状态为 500 或 503（服务器错误/不可用），或者发生网络超时：
    a. 等待 5 秒。
    b. 重试调用一次。
    c. 如果再次失败，报告“用户数据 API 暂时不可用。已尝试 2 次。”
6. 对于任何其他 4xx 错误：报告“API 请求错误：[状态码] - [如果可用，则为响应文本]。”
```

### 错误处理简单流程的可视化

处理工具错误的决策过程通常可以表示为流程。即使简单的指令也意味着代理的一个决策树。

> 代理在工具调用遇到问题时的决策路径，由基于提示的错误处理规则引导。

### 提示错误处理的最佳实践

- **具体但不冗长**：为常见且重要的错误提供清晰的指示。您不需要列出每一个 HTTP 错误代码，但要涵盖认证、服务器错误和错误请求等类别。
- **优先处理非常重要的故障**：将错误处理工作集中在那些其失败会导致整个工作流程脱轨的工具上。
- **迭代和优化**：错误处理提示最初很少是完美的。在测试和生产中观察代理在故障期间的行为，然后优化您的提示以涵盖未处理的情况或改进现有响应。这是代理发展过程中的一个持续部分。
- **考虑代理的推理 (inference)能力**：非常复杂、嵌套的错误处理逻辑可能难以仅通过提示让大型语言模型可靠地遵循。如有必要，将复杂的错误恢复分解为更小、更易于管理的步骤。
- **广泛记录日志**：确保您的代理或其周围的系统记录工具调用、响应、检测到的错误和恢复尝试。这些数据对于调试与错误处理相关的提示非常有价值。

通过周密地设计提示，引导代理应对错误情境，您可以显著提高代理系统的可靠性和稳定性。能够智能地响应工具故障的代理更具自主性，并提供更好的用户体验。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  本文介绍了大型语言模型在使用工具时结合推理和行动的关键框架，强调了代理与外部环境交互的架构考量，其中包含处理各种结果和潜在故障。
- [Designing Data-Intensive Applications: The Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  虽然与大型语言模型无直接关联，但本书提供了构建弹性系统的基本原理，涵盖了容错、分布式系统挑战以及各种错误处理模式（例如重试、超时、优雅降级），这些对于设计稳健的代理工作流具有直接应用价值。
- [Prompt engineering](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api) — OpenAI (2024)
  Publisher: OpenAI
  官方文档提供了为大型语言模型设计有效提示的最新策略和实践，包括在复杂场景中引导其行为以及管理意外输出的方法，这与指导代理进行错误检测和恢复相关。
- [Challenges and Applications of Large Language Model Agents](https://arxiv.org/abs/2403.17725) — Andrii Kompanets, Gautam Pai, Remco Duits, Davide Leonetti, Bert Snijder (2024)
  Journal: arXiv preprint arXiv:2403.17725; DOI: [10.48550/arXiv.2403.17725](https://doi.org/10.48550/arXiv.2403.17725)
  这项调查讨论了大型语言模型代理当前面临的挑战和限制，包括其在现实场景中的可靠性和稳健性方面，为理解为何有效的错误处理对这些系统至关重要提供了更广阔的视角。

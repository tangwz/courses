---
course: "building-advanced-llm-agent-tools"
chapter: "integrating-external-apis-tools"
lesson: "api-tool-authentication-authorization"
sourceId: 6585
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-4-integrating-external-apis-tools/api-tool-authentication-authorization"
title: "工具API访问的认证与授权"
description: "为与外部API交互的工具实现安全认证和授权机制。"
order: 1
plots: []
sourceHash: "2336fd2b8087b4d438e19f11a6fd0974f6cd788cf2598dd75a19b9064e770107"
sourceCorrections: []
---

当你的LLM代理需要通过外部API与外部交流时，这不仅仅是发送请求和获取响应那么简单。大多数提供有价值数据或支持操作的API，都需要一种方式来知晓*谁*在发起请求以及他们被*允许*执行哪些操作。这时，认证和授权便显得重要，它们构成了工具安全受控地访问API的基础。忽视这些方面可能导致未经授权的访问、数据泄露或服务滥用，从而损害代理的可靠性和可信度。

本节将详细阐述在将外部API封装为工具时，如何实现认证和授权机制。我们将讲解常用的模式和最佳实践，以确保你的工具能够安全、负责地与API进行交互，充当LLM代理的可信中介。

### 认证：验证“谁”在访问API

认证是API服务器验证发起请求的客户端身份的过程，在这里指的是你的LLM代理工具。如果没有正确的认证，API无法知晓请求是否合法。API使用了几种常见的认证方法：

#### API密钥

API密钥是最简单也最常用的认证形式之一。API密钥通常是一个唯一的字符串，你的工具在请求中包含它，以便向API提供方表明身份。

- **如何运作**：当你注册API服务时，通常会获得一个API密钥。你的工具随后会在每个请求中发送此密钥，通常放在HTTP请求头中（例如，`X-API-Key: YOUR_API_KEY` 或 `Authorization: ApiKey YOUR_API_KEY`），或作为URL查询参数 (parameter)（例如，`?apiKey=YOUR_API_KEY`）。
- **安全考虑**：关于API密钥最重要的规则是：**切勿将其直接硬编码到你的工具源代码中**。如果你的代码被公开分享或进行版本控制，硬编码的密钥很容易暴露。相反，应将API密钥存储在环境变量中，或使用专用的密钥管理服务（如HashiCorp Vault、AWS Secrets Manager或Google Cloud Secret Manager）。你的Python工具可以在运行时从环境中读取密钥。

  ```python
  import os
  import openai

  # 最佳实践：从环境变量加载API密钥
  openai.api_key = os.getenv("OPENAI_API_KEY")

  if not openai.api_key:
     print("Error: OPENAI_API_KEY environment variable not set.")
     # 妥善处理此错误，例如抛出异常
     # 或向LLM返回错误信息。
  else:
     try:
         # 示例：调用OpenAI GPT-3.5-turbo聊天补全API
         chat_completion = openai.chat.completions.create(
             model="gpt-3.5-turbo",
             messages=[{"role": "user", "content": "Hello world"}]
         )
         print(chat_completion.choices[0].message.content)
     except openai.APIStatusError as e:
         print(f"OpenAI API request failed with status {e.status_code}: {e.response}")
         # 处理特定的OpenAI API错误（例如，401、403、429）
     except openai.APIConnectionError as e:
         print(f"OpenAI API connection error: {e}")
         # 处理网络错误、超时等
     except Exception as e:
         print(f"An unexpected error occurred: {e}")
  ```
- **管理**：像对待密码一样对待API密钥。如果API提供方支持，请定期轮换它们，并确保每个密钥只拥有其所需权限。

#### OAuth 2.0

OAuth 2.0是一个行业标准的授权框架（也常用于认证），它使第三方应用程序（如你的工具）能够代表用户访问网络资源，而无需暴露用户的主要凭证（如他们的用户名和密码）。

- **运作方式（简化）**：
  1. **授权请求**：你的工具会将用户（如果适用，用于用户委托访问）或自身（用于机器对机器访问）重定向到授权服务器。
  2. **用户授予权限/客户端认证**：用户批准请求，或客户端应用程序直接进行认证。
  3. **授权许可**：授权服务器返回一个授权许可（例如，一个授权码）。
  4. **访问令牌**：你的工具将此许可换取一个访问令牌。
  5. **API访问**：你的工具使用此访问令牌（通常是Bearer令牌）向API发起认证请求。
- **授权类型**：OAuth 2.0定义了几种用于获取访问令牌的“授权类型”（流程）。对于工具来说，常见的包括：
  - **客户端凭证授权**：当工具代表自身（机器对机器）而非代表用户访问资源时使用。工具通过其客户端ID和客户端密钥进行认证，以获取访问令牌。这通常适用于后端工具。
  - **授权码授权**：更为复杂，通常涉及通过浏览器进行用户交互。当你的工具需要访问用户特定数据时使用此类型。
- **令牌处理**：访问令牌通常是短期有效的。你的工具必须安全地存储访问令牌，通常还有“刷新令牌”。当当前访问令牌过期时，刷新令牌可用于获取新的访问令牌，而无需用户或客户端从头重新认证。

#### 持有者令牌

持有者令牌，通常以JSON Web Token (JWT)的形式存在，是实现基于令牌认证的常用方式。“持有者”一词表示令牌的持有者被授权访问相关资源。

- **用法**：工具在`Authorization` HTTP请求头中包含令牌，使用`Bearer`方案：`Authorization: Bearer YOUR_ACCESS_TOKEN`。
- **获取方式**：持有者令牌通常通过OAuth 2.0流程或其他登录机制获取。
- **有效期**：与OAuth访问令牌类似，持有者令牌通常有时间限制，并且可能需要刷新机制。

#### 基本认证

基本认证是一种内置于HTTP协议的简单认证方案。它涉及在`Authorization`请求头中，将用户名和密码进行Base64编码后，随每个请求发送。

- **用法**：`Authorization: Basic BASE64_ENCODED_USERNAME_PASSWORD`
- **安全风险**：基本认证在**未加密的HTTP连接上不安全**，因为凭证尽管经过Base64编码，但很容易被解码。它只能在HTTPS上使用。许多现代API已弃用基本认证，转而支持API密钥或OAuth 2.0等更安全的方法。如果你必须使用它，请确保你的工具严格使用HTTPS。

下面的图表说明了API工具封装器通常如何处理凭证：

> 工具封装器作为安全中介，从安全位置获取凭证，并代表LLM代理使用它们与外部API进行通信。

### 授权：规定“可以做什么”

一旦API服务器知晓*谁*在发起请求（认证），它需要确定该身份被*允许*执行哪些操作。这就是授权。例如，一个经过认证的用户可能被授权读取数据，但无权删除数据。

#### 范围（OAuth 2.0常用）

使用OAuth 2.0时，访问令牌通常与“范围”相关联。范围定义了授予访问令牌的特定权限。例如，API可能定义`read_profile`（读取资料）、`write_files`（写入文件）或`send_notifications`（发送通知）等范围。

- **最小权限原则**：当你的工具请求访问令牌时，它应该只请求其预期功能所需的最少范围集。如果一个工具只需读取日历条目，它就不应请求删除它们的权限。这限制了如果工具的凭证或令牌被泄露可能造成的损害。
- **API文档**：API的文档将详细说明可用的范围以及它们授予的权限。你的工具配置应体现对这些范围的仔细考量。

#### 基于角色的访问控制（RBAC）

有些API，特别是那些用于企业服务的API，可能会使用更细粒度的基于角色的访问控制（RBAC）系统。在这种模式中，API密钥或服务账户被分配角色（例如，“查看者”、“编辑者”、“管理员”），每个角色都有一组预定义的权限。

- **配置**：在设置你的工具将使用的API密钥或服务账户时，请确保为其分配所需的最少权限角色。避免使用高权限的管理员账户进行常规工具操作。

### 在工具中实现安全凭证管理

你的工具处理凭证的方式与认证方法本身同样重要。

- **配置**：设计你的工具以安全地接收凭证。
  - **初始化**：在工具设置时，将凭证（如API密钥或预获取的令牌）传递给工具的类构造函数或函数参数 (parameter)。这些凭证应从安全来源加载，例如环境变量或*不*提交到版本控制的配置文件。
  - **动态获取**：对于需要刷新的OAuth令牌，工具可以封装获取/刷新令牌的逻辑，并安全地存储客户端ID和密钥。
- **运行时处理**：
  - **避免日志记录**：绝不记录原始API密钥、令牌或其他敏感凭证。如果你需要记录请求详情用于调试，请务必对`Authorization`或`X-API-Key`等请求头进行清理或省略。
  - **内存存储**：在需要时将凭证保留在内存中。除非绝对必要且已妥善保护，否则避免将其写入临时文件。
  - **HTTP客户端使用**：在Python中使用`requests`等HTTP客户端库时，正确传递凭证。对于请求头中的API密钥或持有者令牌：

    ```python
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    ```

    对于使用`requests`的基本认证：

    ```python
    from requests.auth import HTTPBasicAuth
    response = requests.get(url, auth=HTTPBasicAuth('username', 'password'))
    ```

    （请记住从安全来源加载“username”和“password”）。
- **令牌刷新逻辑**：如果你的工具使用会过期的OAuth 2.0访问令牌，它需要处理令牌刷新。这通常包括：
  1. 使用当前访问令牌发起API请求。
  2. 如果API返回`401 Unauthorized`错误（或指示令牌过期的特定错误），使用刷新令牌向授权服务器请求新的访问令牌。
  3. 安全地存储新的访问令牌（如果提供，也存储新的刷新令牌）。
  4. 使用新的访问令牌重试原始API请求。
     特定于API提供方的框架或库通常能简化此过程。

### 处理认证和授权错误

你的工具必须准备好处理与认证和授权相关的错误。API通常使用标准HTTP状态码：

- **`401 Unauthorized`（未经授权）**：这通常意味着请求缺少有效的认证凭证。API密钥可能缺失或无效，访问令牌可能已过期或格式错误。你的工具在未解决问题（例如，刷新令牌）之前，不应使用相同凭证重试请求。
- **`403 Forbidden`（禁止访问）**：这意味着服务器理解了请求并认证了身份，但该已认证的身份没有权限访问所请求的资源或执行所请求的操作。这可能是由于权限范围不足或RBAC角色限制造成的。重试相同的请求很可能导致同样的错误。

当你的工具遇到这些错误时，它应该：

1. **避免暴露敏感错误细节**：如果错误信息包含不应透露的内容，则不应直接暴露给LLM。
2. **提供清晰反馈**：给LLM或调用系统，例如“服务[服务名称]认证失败”或“服务[服务名称]上对[资源]的访问被拒绝。可能缺少所需权限。”
3. **实现适当的重试逻辑**：如果错误是瞬时的（例如，可以刷新的过期令牌）。对于永久性权限问题（`403`），重试通常是徒劳的。

通过认真落实这些认证和授权策略，你可以构建出不仅能扩展LLM代理能力，还能在其交互的数字生态系统中安全、合规运行的工具。请记住，LLM代理依赖工具作为其通往外部服务的可信安全门户。

## 参考资料

- [The OAuth 2.0 Authorization Framework](https://www.rfc-editor.org/rfc/rfc6749) — D. Hardt, Ed. (2012)
  Journal: RFC; Volume: 6749; Pages: 1-76; DOI: [10.17487/RFC6749](https://doi.org/10.17487/RFC6749)
  阐述OAuth 2.0协议（包括授权类型）以实现安全委托访问的基础规范。
- [OWASP API Security Top 10 (2023)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEG9d5z8Ys-bhnroU1br_eegPbB74S-CrEeNquTHLvU7TQXs99IBv_kjLt6wYzM24wJFY0YKX64fvRfqecvfgQcBoeM4jaTv2qbxbAdmmzYC5qjnpHjl5fUhBKHlhtBZ4cXCtfRlfpycN-XRJkJdf1bf6UM) — OWASP Foundation (2023)
  关于常见API安全漏洞和最佳实践的指南，涵盖认证、授权和安全凭证处理。
- [What is Vault?](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGnyconT3A8FwSDUqbDYfK9tGDJamjqQMuFA5WTvexc3FLZ1YVvIQwzwc399pbsl0u6QYQt6a3-w4pFKTF823L4ROdTdYi9Xf4mtF2jbaLELjXLAuUnDDPuFofqhWZPKkEM-rjMqEfvvgqhUQiiApsvRu_TsQSMOjvTkTG5ZRAn) — HashiCorp (2024)
  HashiCorp Vault的官方文档，详细说明了其在章节中提及的安全密钥管理方法。
- [Authentication](https://readthedocs.io/en/latest/user/authentication/) — Kenneth Reitz (2024)
  使用流行的Python `requests` 库实现各种认证方法的实践指南，与代码示例相关。

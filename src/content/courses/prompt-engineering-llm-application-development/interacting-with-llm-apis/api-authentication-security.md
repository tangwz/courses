---
course: "prompt-engineering-llm-application-development"
chapter: "interacting-with-llm-apis"
lesson: "api-authentication-security"
sourceId: 3022
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-4-interacting-with-llm-apis/api-authentication-security"
title: "API 认证与安全"
description: "安全处理 API 密钥和认证方法。"
order: 2
plots: []
sourceHash: "4cba39d9d4094c0d4e1214a06b615eee2a1e1f3954f2d17e26ea8ae864fa5aca"
sourceCorrections: []
---

通过编程方式与大型语言模型交互时，您的应用程序需要向 API 提供方进行认证。此过程确认您的应用程序有权限使用该服务，通常会将使用情况与您的账户关联，用于计费和监控。安全地管理认证凭据是构建可靠应用程序的一个主要方面。

### API 密钥：常用方法

大多数大型语言模型提供商使用 API 密钥进行认证。API 密钥通常是一个长而唯一的字符字符串，您会将其包含在 API 请求中。当提供商收到请求时，会在处理前检查 API 密钥的有效性。可以将其视为专门用于您的应用程序访问大型语言模型服务的密码。

您通常会通过提供商的网页控制台或管理界面生成 API 密钥。例如，OpenAI、Anthropic、Cohere 和 Google AI Studio 等服务都提供界面来创建和管理与您的账户关联的 API 密钥。

### 安全处理 API 密钥

请像对待敏感密码一样对待您的 API 密钥。将其公开，例如直接提交到您的源代码仓库（如 Git），可能导致未经授权的访问，如果他人使用您的密钥，还可能产生可观的费用。

以下是安全管理 API 密钥的基本做法：

1. **避免硬编码：** 切勿将 API 密钥直接嵌入 (embedding)到应用程序的源代码中。这是一个重要的安全风险。任何能访问您代码的人，包括公共仓库的用户，都可以找到并滥用您的密钥。

   ```python
   # 不好的做法：不要硬编码 API 密钥！
   # api_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
   # headers = {"Authorization": f"Bearer {api_key}"}
   # response = requests.post(api_url, headers=headers, json=payload)
   ```
2. **使用环境变量：** 一种常用且有效的方法是将 API 密钥存储为运行您应用程序的系统上的环境变量。您的代码可以在运行时从环境中读取密钥。

   - **在 Linux/macOS 上：**

     ```bash
     export OPENAI_API_KEY="your_actual_api_key_here"
     ```
   - **在 Windows 上（命令提示符）：**

     ```bash
     set OPENAI_API_KEY=your_actual_api_key_here
     ```
   - **在 Windows 上（PowerShell）：**

     ```powershell
     $Env:OPENAI_API_KEY="your_actual_api_key_here"
     ```

   然后您可以在 Python 代码中使用 `os` 模块访问它：

   ```python
   import os
   import requests

   # 好的做法：从环境变量加载 API 密钥
   api_key = os.getenv("OPENAI_API_KEY")
   if not api_key:
       raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")

   api_url = "https://api.openai.com/v1/chat/completions" # 示例 URL
   headers = {"Authorization": f"Bearer {api_key}"}
   # ... 构建有效载荷并发送请求 ...
   # response = requests.post(api_url, headers=headers, json=payload)
   ```
3. **使用配置文件（请谨慎）：** 您可以将密钥存储在配置文件中（例如，`.env` 文件、JSON、YAML）。确保这些文件：

   - **不**提交到版本控制（将其添加到您的 `.gitignore` 文件中）。
   - 受适当的文件系统权限保护。
   - 像 `python-dotenv` 这样的库可以将 `.env` 文件中的变量加载到环境中。

   `.env` 文件示例：

   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

   使用 `python-dotenv` 的 Python 示例：

   ```python
   import os
   import requests
   from dotenv import load_dotenv

   load_dotenv() # 将变量从 .env 文件加载到环境中

   api_key = os.getenv("OPENAI_API_KEY")
   if not api_key:
       # ... 处理错误 ...
       pass
   # ... 代码的其余部分 ...
   ```
4. **密钥管理系统：** 对于生产环境或团队设置，专用的密钥管理工具（如 HashiCorp Vault、AWS Secrets Manager、Google Secret Manager、Azure Key Vault）提供更安全、更易管理的方式来处理 API 密钥及其他敏感凭据。这些系统提供集中存储、访问控制、审计和自动轮换等功能。集成这些系统通常需要使用由密钥管理器提供的特定 SDK。

### API 请求中的认证

您如何在请求中包含 API 密钥取决于特定大型语言模型提供商的要求。两种常见模式是：

1. **授权头（Bearer 令牌）：** 这非常常见。API 密钥在 `Authorization` HTTP 头中发送，通常以 `Bearer`  作为前缀。

   ```python
   import os
   import requests

   api_key = os.getenv("PROVIDER_API_KEY")
   api_url = "PROVIDER_API_ENDPOINT" # 替换为实际端点
   headers = {
       "Authorization": f"Bearer {api_key}",
       "Content-Type": "application/json"
   }
   payload = { # 您的请求数据
       "model": "some-model-name",
       "prompt": "Explain API authentication.",
       # ... 其他参数
   }

   # response = requests.post(api_url, headers=headers, json=payload)
   # print(response.json())
   ```
2. **自定义头：** 有些 API 可能要求密钥位于自定义头中，例如 `X-API-Key`。

   ```python
   import os
   import requests

   api_key = os.getenv("PROVIDER_API_KEY")
   api_url = "PROVIDER_API_ENDPOINT"
   headers = {
       "X-API-Key": api_key, # 自定义头示例
       "Content-Type": "application/json"
   }
   payload = { # 您的请求数据
       "model": "some-model-name",
       "prompt": "Explain API authentication.",
       # ... 其他参数
   }

   # response = requests.post(api_url, headers=headers, json=payload)
   # print(response.json())
   ```
3. **请求体或查询参数 (parameter)（不常见/安全性较低）：** 偶尔，API 可能期望密钥位于 JSON 有效载荷内或作为 URL 查询参数。这通常安全性较低，尤其是通过查询参数发送时，因为它们可能被记录在服务器历史或浏览器缓存中。在可用时，始终优先选择基于头的认证。

务必查阅您使用的大型语言模型提供商的 API 文档，以了解其所需的认证方法。

> 这是一个简化流程，展示了存储在环境变量中的 API 密钥如何用于您的应用程序向大型语言模型提供商服务发起 API 请求的认证过程。

### 安全考量

- **密钥轮换：** 定期重新生成（轮换）您的 API 密钥。如果密钥遭到泄露，轮换它能限制未经授权访问的时间范围。一些提供商可能会提供自动轮换工具或策略。
- **最小权限原则：** 如果您的提供商允许创建具有受限权限的密钥（例如，只读访问、仅访问特定模型），请使用这些功能来限制密钥暴露时可能造成的损害。
- **监控：** 通过提供商的控制面板留意您的 API 使用情况。异常活动高峰可能表明密钥已被泄露。
- **撤销：** 如果您怀疑 API 密钥已被泄露，请知道如何立即撤销它。

安全管理认证是构建与外部 API 交互的可信应用程序的基本步骤。通过使用环境变量或密钥管理器并遵循最佳实践，您可以显著降低处理敏感 API 密钥相关的风险。

## 参考资料

- [OWASP API Security Top 10](https://owasp.org/API-Security/) — Erez Yalon, Inon Shkedy, Paulo Silva (2023)
  Publisher: OWASP Foundation
  提供API安全风险及建议缓解策略的列表，包含安全认证和授权。
- [Authentication](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFSCnSTle1QyxRDfj9RSISxJv75kZW9beiK3l6S6RvRexQOYDc0xAyjKaoRFOjecon2NEAxA95s2L4NzU4f8mlSBvZoQDGDhM7YiTJgp4vF74Js6puQIpR19sI5qE4=) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI LLM API请求的官方认证指南，详述API密钥的使用和HTTP授权头。
- [What is Vault?](https://developer.hashicorp.com/vault/docs/what-is-vault) — HashiCorp (2024)
  Publisher: HashiCorp
  介绍一个领先的密钥管理系统，阐释其安全存储、访问和审计API密钥等敏感凭证的原理。

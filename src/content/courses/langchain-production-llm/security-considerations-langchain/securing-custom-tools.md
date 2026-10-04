---
course: "langchain-production-llm"
chapter: "security-considerations-langchain"
lesson: "securing-custom-tools"
sourceId: 3936
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-8-security-considerations-langchain/securing-custom-tools"
title: "保护自定义工具和API交互"
description: "为LangChain代理使用的工具实施身份验证、授权和速率限制。"
order: 5
plots: []
sourceHash: "951c7dde19951c98ad246a06616ed4c10309511fe2939ccc843e707e25f08c64"
sourceCorrections: []
---

当LangChain代理使用自定义工具时，它们连接了语言模型的概率性特点与外部API、数据库和系统的确定性（且通常敏感）系统。这项能力使得自动化和数据访问得以大范围实现，但同时也带来了重大的安全难题，需要细致处理。一个受损或安全防护不佳的工具可能成为未经授权操作、数据泄露或针对后端系统的拒绝服务攻击的通道。

保护这些交互需要多层面地应用安全措施，将每个工具都视为一个潜在的攻击面。目标是确保工具只执行授权操作，使用经过验证的输入，并在其整个生命周期中安全地处理数据。

### 工具的身份验证和授权

在工具执行任何操作之前，必须回答两个优先问题：

1. 调用代理（进而调用工具）的实体是否有权请求此操作？
2. 工具本身是否已正确进行身份验证并获得授权，以与目标系统（例如，外部API、数据库）交互？

**代理层面授权：** 代理本身可能需要权限。通常，这与启动LangChain应用交互的用户有关。代理周围的应用层应强制执行用户身份验证，并安全地传递相关的用户上下文 (context)或权限。然后，代理或编排工具使用的逻辑可以检查当前用户上下文是否允许执行特定工具或操作。避免将大范围权限直接嵌入 (embedding)到代理的核心逻辑中。

**工具到系统身份验证：** 工具需要凭据才能与外部服务交互。将API密钥或密码直接硬编码到工具代码中是不安全的。应使用安全的秘密管理解决方案：

- **环境变量：** 适用于较简单的部署，但要确保环境本身是安全的。
- **秘密管理系统：** HashiCorp Vault、AWS Secrets Manager或Google Secret Manager等系统提供秘密的安全存储、访问控制和审计。工具应在需要时从这些系统动态获取凭据。
- **OAuth或服务账户：** 对于与支持这些协议的云服务或API的交互，请使用OAuth流程或具有窄范围权限的专用服务账户。这可避免直接管理静态密钥。

确保工具使用的凭据遵循最小权限原则，仅授予工具特定功能所需的权限。

### 严格输入验证

代理工具的优先风险之一是，控制代理的大型语言模型 (LLM) 可能会为工具生成意想不到的、恶意的或仅仅是错误的输入。**在没有明确验证的情况下，绝不要信任LLM生成的输入。**

LangChain建议使用Pydantic模型定义工具输入的预期架构，这为验证提供了坚实的基础。

```python
from pydantic import BaseModel, Field, field_validator
from langchain.tools import BaseTool
import re
import asyncio

# 使用Pydantic定义预期的输入架构
class GetUserDetailsInput(BaseModel):
    user_id: str = Field(description="用户的唯一标识符。")
    include_history: bool = Field(default=False, description="是否包含用户的订单历史记录。")

    # Pydantic验证器，用于user_id格式
    @field_validator('user_id')
    @classmethod
    def user_id_must_be_valid_format(cls, v: str) -> str:
        if not re.match(r'^user_[a-zA-Z0-9]+$', v):
            raise ValueError('无效的user_id格式。必须以"user_"开头。')
        return v

# 使用架构的自定义工具示例
class GetUserDetailsTool(BaseTool):
    name = "get_user_details"
    description = "获取特定用户ID的详细信息。可选择包含订单历史记录。"
    args_schema: type[BaseModel] = GetUserDetailsInput # 关联架构

    def _run(self, user_id: str, include_history: bool = False) -> str:
        # 在调用_run之前，user_id和include_history输入已
        # 由Pydantic根据GetUserDetailsInput架构进行验证。

        try:
            # 安全地与后端系统交互
            # 示例：user_data = call_secure_user_api(user_id, include_history)
            user_data = self._fetch_user_data_from_backend(user_id, include_history)

            # 处理并返回数据
            return f"用户详情: {user_data}" # 必要时过滤敏感数据

        except Exception as e:
            # 安全地记录错误
            # logger.error(f"获取用户数据出错 {user_id}: {e}")
            # 向代理返回一个安全的错误消息
            return f"错误: 无法获取用户 {user_id} 的详情。请检查ID或稍后重试。"

    async def _arun(self, user_id: str, include_history: bool = False) -> str:
        # 工具执行的异步版本
        # 实现与_run类似的逻辑，但使用异步调用
        try:
            user_data = await self._async_fetch_user_data_from_backend(user_id, include_history)
            return f"用户详情: {user_data}"
        except Exception as e:
            # logger.error(f"异步获取用户数据出错 {user_id}: {e}")
            return f"错误: 无法异步获取用户 {user_id} 的详情。"

    def _fetch_user_data_from_backend(self, user_id: str, include_history: bool) -> dict:
        # 占位符：替换为实际的安全API调用
        # 确保此函数处理身份验证、HTTPS等
        print(f"模拟后端调用，user_id: {user_id}, include_history: {include_history}")
        # 示例响应结构
        data = {"user_id": user_id, "name": "张三", "email": f"{user_id}@example.com"}
        if include_history:
            data["history"] = ["订单_123", "订单_456"]
        return data

    async def _async_fetch_user_data_from_backend(self, user_id: str, include_history: bool) -> dict:
        # 占位符：后端调用的异步版本
        print(f"模拟异步后端调用，user_id: {user_id}, include_history: {include_history}")
        # 示例响应结构
        data = {"user_id": user_id, "name": "张三", "email": f"{user_id}@example.com"}
        if include_history:
            data["history"] = ["订单_123", "订单_456"]
        await asyncio.sleep(0.1) # 模拟异步延迟
        return data

# 实例化工具
user_details_tool = GetUserDetailsTool()

# LangChain代理执行器将处理此工具的调用
# 并将输入根据GetUserDetailsInput进行解析和验证
```

在工具逻辑中实施特定领域的验证：

- 在尝试操作之前，检查标识符（如 `user_id`）是否确实存在于目标系统中。
- 根据后端系统预期或允许的范围，验证数值范围、字符串长度或允许值。
- 如果输入用于构建查询或命令，则清理输入以去除潜在的有害字符或序列（尽管参数 (parameter)化查询/API调用比字符串构建更受推荐）。

下图显示了工具交互流程中的优先验证步骤：

> 数据流显示LLM生成的输入在与后端系统交互之前由工具进行严格验证，并可选地在返回之前对输出进行过滤。

### 应用最小权限原则

设计工具时应使其范围尽可能窄。避免创建执行许多不同操作的单一工具，特别是混合读写操作。例如，不应使用接受原始SQL查询的单个 `database_tool`（一种非常危险的模式），而应创建如下特定工具：

- `get_order_details_tool(order_id: str)` （获取订单详情工具）
- `update_customer_address_tool(customer_id: str, address: dict)` （更新客户地址工具）
- `list_recent_products_tool(category: str, limit: int)` （列出最新产品工具）

每个工具都应使用仅授予其特定任务所需权限的凭据或角色（例如，检索工具的只读访问权限，更新工具的特定写入权限）。

### 速率限制和资源管理

代理，特别是在自主循环中或处理并发请求时，可能会无意中或恶意地触发大量工具调用。这可能会使后端系统超载或产生高昂费用（例如，API调用费用）。

- **实施速率限制：** 可以在工具逻辑中（使用 `ratelimit` 等库）或最好在基础设施层面（例如，使用API网关）对后端服务应用速率限制。限制可以基于用户、API密钥或整体代理使用情况。
- **资源限制：** 设计工具以处理合理量的数据。例如，一个检索数据的工具应内置限制或分页功能，而不是试图一次性获取数百万条记录。
- **熔断器：** 如果后端服务无响应或频繁失败，请考虑实施熔断器模式，以防止代理反复攻击一个故障系统。

### 安全输出处理

正如工具输入需要细致审查一样，工具返回给代理/LLM的数据也可能需要过滤或清理。

- **防止敏感信息泄露：** 避免向LLM返回原始数据库记录、带有堆栈跟踪的完整API错误消息或过多的内部系统细节。这些可能在日志或后续LLM响应中无意间泄露敏感信息。
- **过滤相关性：** 仅返回代理继续操作所需的信息。这减少了传回给LLM的上下文 (context)大小，并尽量减小了潜在数据泄露的风险面。
- **妥善处理错误：** 当工具遇到错误（例如，API不可用，输入无效）时，在工具内部捕获异常，为开发人员安全地记录详细错误，并向代理返回简洁、安全的错误消息（例如，“未能获取用户数据。”）。不要直接传回原始异常消息。

### 安全日志记录和监控

为工具调用实施结构化日志记录：

- 记录被调用的工具名称。
- 记录*已验证*的输入参数 (parameter)（对于密码或个人身份信息等敏感数据的日志记录要极其谨慎；应进行屏蔽或省略）。
- 记录结果（成功或失败）。
- 如果可用，记录来源/用户上下文 (context)。

监控这些日志以查找异常模式：

- 对特定工具的过度调用。
- 工具反复失败。
- 来自非预期上下文的工具调用。
- 尝试使用格式错误输入调用工具（已被验证捕获）。

将这些日志与安全信息和事件管理 (SIEM) 系统集成，有助于检测针对代理工具的潜在滥用或攻击。

通过将自定义工具视为核心安全组件，并应用身份验证、授权、输入验证、最小权限、速率限制和安全输出处理这些原则，可以显著降低LLM代理与外部系统交互相关的风险。请记住，每个工具都扩展了应用程式的信任边界，需要认真进行安全考量。

## 参考资料

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/) — OWASP Foundation (2023)
  Publisher: OWASP Foundation
  提供API最关键安全风险的全面列表，包括认证失败、授权失败、注入和安全配置错误，这些与保护工具和外部系统交互直接相关。
- [OWASP Top 10 for Large Language Model Applications](https://genai.owasp.org/llm-top-10/) — OWASP Foundation (2024)
  专门关注使用大型语言模型（LLM）的应用程序中的安全漏洞，涵盖提示注入、不安全的输出处理和过度代理等问题，这些在LLM代理使用自定义工具时至关重要。
- [Pydantic Documentation: Validation](https://docs.pydantic.dev/latest/) — Pydantic Team (2024)
  Pydantic官方数据验证功能文档，在正文内容中明确推荐并用于严格验证LLM生成工具输入。
- [HashiCorp Vault Documentation](https://developer.hashicorp.com/docs/vault) — HashiCorp (2024)
  Publisher: HashiCorp
  HashiCorp Vault的官方文档，这是一种被广泛采用的秘密管理系统，正文中提及它用于安全管理工具与外部系统交互时使用的凭据。

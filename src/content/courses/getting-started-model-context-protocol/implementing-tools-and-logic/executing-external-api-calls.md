---
course: "getting-started-model-context-protocol"
chapter: "implementing-tools-and-logic"
lesson: "executing-external-api-calls"
sourceId: 7851
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-3-implementing-tools-and-logic/executing-external-api-calls"
title: "执行外部API调用"
description: "在MCP工具处理程序中封装REST API请求的模式。"
order: 3
plots: ["plots/7851-0.json"]
sourceHash: "2716e039aa0e0a703c6b65bbcc93a25e1ca1a50905e5a9d604ad72e8f0f62b53"
sourceCorrections: []
---

验证输入参数 (parameter)仅仅是工具执行生命周期的第一步。一旦参数符合Pydantic模型，服务器就会调用注册的处理函数。此函数充当JSON-RPC协议与外部系统之间的桥梁。在模型上下文 (context)协议（MCP）的背景下，这座桥梁通常涉及对第三方REST或GraphQL API的异步网络请求。

实现这些处理程序需要仔细管理异步上下文、数据转换和网络延迟。与本地文件操作不同，外部API调用会带来不可预测性。服务器必须高效处理这些交互，以便为与大型语言模型（LLM）交互的用户维持良好的响应体验。

### 异步工具处理程序

大多数用Python实现的MCP服务器使用`asyncio`来处理并发请求。定义工具处理程序时，应使用`async def`语法。这允许服务器在等待外部API响应时让出控制权，从而避免整个服务器在网络请求期间阻塞。

使用Python MCP SDK时，工具执行逻辑被封装在一个装饰函数中，用于处理`CallToolRequestSchema`。在此函数内部，您通常会采用异步HTTP客户端，例如`httpx`或`aiohttp`。

下图展示了LLM调用封装外部API的工具时的执行流程。

> 工具请求从客户端发出，经过校验和外部执行，最后以文本形式返回给客户端的流程。

### 实现HTTP封装

实现的核心在于使用工具调用提供的参数 (parameter)来构建HTTP请求。最佳实践是在服务器生命周期内实例化一个HTTP客户端实例，或者为单个请求使用上下文 (context)管理器，以确保连接正确关闭。

设想一个场景，我们希望提供一个名为`get_crypto_price`的工具，用于获取加密货币的当前价格。该工具接受一个`ticker`符号（例如“BTC”）和一个`currency`（例如“USD”）。

以下是使用`httpx`的实现方式：

```python
import httpx
import mcp.types as types
from mcp.server import Server

app = Server("crypto-server") # 实例化服务器

async def fetch_crypto_price(ticker: str, currency: str) -> str:
    url = f"https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ticker.lower(),
        "vs_currencies": currency.lower()
    }
    
    async with httpx.AsyncClient() as client:
        # 设置超时，避免LLM生成停滞
        response = await client.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        
        # 提取具体数值
        price = data.get(ticker.lower(), {}).get(currency.lower())
        
        if price:
            return f"The current price of {ticker} is {price} {currency.upper()}."
        else:
            return f"Could not retrieve price for {ticker} in {currency}."

@app.call_tool()
async def handle_tool_call(
    name: str, 
    arguments: dict
) -> list[types.TextContent]:
    if name == "get_crypto_price":
        # 此时参数已由schema校验
        ticker = arguments.get("ticker")
        currency = arguments.get("currency", "usd")
        
        result = await fetch_crypto_price(ticker, currency)
        
        return [
            types.TextContent(
                type="text",
                text=result
            )
        ]
    
    raise ValueError(f"Unknown tool: {name}")
```

在此实现中，`fetch_crypto_price`函数封装了外部逻辑。它处理URL构建、查询参数和网络调用。`handle_tool_call`函数充当路由器，解包字典参数并委托给特定的逻辑函数。

### 数据转换和上下文 (context)窗口

外部API通常会返回冗长的JSON对象，其中包含元数据、时间戳和大量细节，这些可能与用户的查询无关。大型语言模型具有有限的上下文窗口。将50KB的原始JSON响应直接放入对话历史中会降低模型性能并增加成本（如果按token付费）。

工具实现的一个重要职责是将这些原始数据过滤并转换成简洁、易读的格式或最小化的JSON结构。

例如，一个天气API可能返回：

```json
{
  "coord": { "lon": -0.13, "lat": 51.51 },
  "weather": [{ "id": 300, "main": "Drizzle", "description": "light intensity drizzle", "icon": "09d" }],
  "base": "stations",
  "main": { "temp": 280.32, "pressure": 1012, "humidity": 81, "temp_min": 279.15, "temp_max": 281.15 },
  "visibility": 10000,
  "wind": { "speed": 4.1, "deg": 80 },
  "clouds": { "all": 90 },
  "dt": 1485789600,
  "sys": { "type": 1, "id": 5091, "message": 0.0103, "country": "GB", "sunrise": 1485762037, "sunset": 1485794875 },
  "id": 2643743,
  "name": "London",
  "cod": 200
}
```

传递整个对象效率不高。工具应解析此内容并返回：“伦敦当前天气：小雨，温度：280.32K，湿度：81%。”

下图比较了原始API响应与转换后响应的token使用影响。最小化负载大小对于维持长时间对话非常重要。



![Token使用量：原始JSON与转换后输出对比](plots/7851-0.json)



> 原始API响应与精选文本响应之间的预估token消耗量比较。

### 安全措施：SSRF和输入净化

当工具接受影响网络请求（特指URL或主机名）的输入时，会带来服务器端请求伪造（SSRF）的风险。在SSRF攻击中，攻击者（或出现幻觉 (hallucination)的模型）诱导服务器向不应从外部访问的内部资源发出请求。

例如，如果您实现了一个工具`fetch_url_content(url: str)`，用户可能会要求LLM获取`http://localhost:8080/admin`。如果您的MCP服务器运行在开发者的机器上，或者一个带有内部服务的云容器中，该工具可能会泄露敏感的内部配置数据。

为减轻此风险，在实现外部调用时：

1. **允许列表：** 如果可行，将工具限制在一组特定域。如果工具旨在查询维基百科，请确保URL以`https://wikipedia.org`开头。
2. **避免通用代理：** 避免构建充当开放代理的工具（例如，“获取[URL]内容”）。而应构建特定工具，例如“获取Jira工单”或“搜索GitHub仓库”。
3. **校验协议：** 确保请求只使用`http`或`https`，阻止诸如`file://`或`gopher://`等可能访问本地文件系统的方案。

### 超时管理

LLM为用户实时运行。如果一个工具执行需要30秒，用户可能会认为生成失败。Python的`httpx`库默认为超时，但通常比较宽松。您应该为外部调用明确设置严格的超时。

5到10秒的超时通常适用于MCP工具。如果外部API慢于此速度，工具应返回消息说明操作耗时超出预期或服务不可用，而不是无限期保持连接打开。

在前面提供的`httpx`示例中，`timeout=10.0`确保如果CoinGecko API挂起，工具处理程序会抛出异常（必须捕获），而不是使MCP服务器卡住。

```python
# 推荐的超时配置
timeout_config = httpx.Timeout(10.0, connect=5.0)
async with httpx.AsyncClient(timeout=timeout_config) as client:
    # ... 执行请求
```

此配置允许5秒用于建立连接，操作总共10秒，确保服务器能够迅速从网络问题中恢复。

## 参考资料

- [asyncio - Asynchronous I/O](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2025)
  Publisher: Python Software Foundation
  Python异步I/O框架的官方文档，对构建非阻塞工具处理器至关重要。
- [httpx documentation](https://www.python-httpx.org/) — encode/httpx contributors (2024)
  httpx库的官方文档，说明了异步HTTP客户端的用法和超时配置。
- [Server-Side Request Forgery Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_For_Forgery_Prevention_Cheat_Sheet.html) — OWASP Foundation (2023)
  Publisher: OWASP Foundation
  关于预防Web应用程序和API交互中服务器端请求伪造(SSRF)漏洞的指南。
- [Scaling Up Models with Function Calling](https://openai.com/blog/function-calling) — OpenAI (2023)
  Publisher: OpenAI Blog
  讨论了使语言模型通过函数调用与外部工具交互的机制和优势。

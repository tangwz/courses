# 处理API响应和错误

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-3-interacting-llm-apis-python/handling-api-responses-errors)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在与LLM API交互时，妥善处理响应和潜在错误是必不可少的。应用程序经常会遇到意料之外的行为或错误，简单地认为请求成功且数据已就绪常常会导致严重的程序错误。因此，应用程序需要可靠的机制来优雅地管理成功的API响应和各种错误情况。

### 检查响应状态

首先要检查的是API服务器返回的HTTP状态码。网页标准为这些代码定义了范围：

- **2xx (例如，200 OK):** 表示成功。请求已接收、已理解并已接受。
- **4xx (例如，400 Bad Request，401 Unauthorized，404 Not Found，429 Too Many Requests):** 客户端错误。您的应用程序发送的请求有问题（数据格式错误、缺少认证、请求的资源不存在、超出速率限制）。
- **5xx (例如，500 Internal Server Error，503 Service Unavailable):** 服务器错误。API提供方那边出错了。

`requests`库使得检查成功变得简单。`Response`对象有一个`status_code`属性和一个布尔型`ok`属性，如果状态码小于400（即成功或重定向），则`ok`为`True`，否则为`False`。

```python
import requests
import os
import json

# 假设API_ENDPOINT和API_KEY已正确设置
# API_ENDPOINT = "YOUR_LLM_API_ENDPOINT"
# API_KEY = os.getenv("LLM_API_KEY") # 安全获取密钥的示例

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "prompt": "Explain the difference between HTTP status codes 200 and 404.",
    "max_tokens": 50
}

try:
    # response = requests.post(API_ENDPOINT, headers=headers, json=data, timeout=10) # 示例调用
    # 为了演示，我们模拟一个响应对象
    class MockResponse:
        def __init__(self, status_code, text_content, is_json=True):
            self.status_code = status_code
            self.text = text_content
            self._is_json = is_json

        @property
        def ok(self):
            return self.status_code < 400

        def json(self):
            if not self._is_json:
                raise requests.exceptions.JSONDecodeError("Expecting value", "dummy", 0)
            try:
                return json.loads(self.text)
            except json.JSONDecodeError as e:
                 raise requests.exceptions.JSONDecodeError(e.msg, e.doc, e.pos)

    # 模拟一个成功的响应
    response = MockResponse(200, '{"completion": "200 OK表示成功，404 Not Found表示资源不存在。"}', is_json=True)

    # 总是首先检查状态
    if response.ok:
        print(f"请求成功 (状态码: {response.status_code})")
        # 继续解析响应体
    else:
        print(f"请求失败 (状态码: {response.status_code})")
        print(f"响应体: {response.text}")
        # 在此处处理特定的错误码（4xx, 5xx）

except requests.exceptions.RequestException as e:
    print(f"请求过程中发生错误: {e}")
```

在尝试处理响应体之前，务必检查`response.ok`或`response.status_code`。尝试从错误响应（例如可能是HTML的404页面）中解析JSON很可能会引发异常。

### 解析响应体

如果状态码表示成功（通常是200 OK），LLM的输出通常包含在响应体中，并且常以JSON格式呈现。`requests`库提供了`response.json()`方法来方便地解析它。

```python
# 延续上述成功响应的模拟...

if response.ok:
    try:
        # 解析JSON响应
        response_data = response.json()
        print("成功解析JSON响应:")
        # print(json.dumps(response_data, indent=2)) # 美观打印

        # 提取相关信息（此结构因API而异）
        if "completion" in response_data:
             llm_output = response_data["completion"]
             print(f"LLM 输出: {llm_output}")
        elif "choices" in response_data and len(response_data["choices"]) > 0:
             # 处理类似OpenAI聊天补全的结构
             first_choice = response_data["choices"][0]
             if "message" in first_choice and "content" in first_choice["message"]:
                 llm_output = first_choice["message"]["content"]
                 print(f"LLM 输出: {llm_output}")
             else:
                 print("在第一个选项中找不到预期的消息内容。")
        else:
            print("响应JSON不包含预期的'completion'或'choices'字段。")
            print("完整响应:", response_data)

    except requests.exceptions.JSONDecodeError:
        # 处理响应状态为OK但响应体不是有效JSON的情况
        print(f"JSON解码失败，即使状态码为{response.status_code}。")
        print(f"原始响应文本: {response.text}")
    except KeyError as e:
        # 处理JSON有效但缺少预期键的情况
        print(f"JSON响应中缺少预期键 {e}。")
        print("完整响应:", response_data)

else:
     # 像之前一样处理非OK状态码
     print(f"请求失败 (状态码: {response.status_code})")
     print(f"响应体: {response.text}")
```

请注意这个嵌套的`try...except`代码块。即使状态码是200 OK，您也可能收到一个不是有效JSON的响应体，或者是一个有效JSON但缺少您预期键（例如`completion`、`choices`、`message`）的响应体。您的代码需要预料到这些可能性。JSON响应的具体结构在不同的LLM提供商和API端点之间差异很大，因此请查阅您正在使用的特定API文档。

### 处理API特有错误

有时，API可能会返回200 OK状态码，但仍在JSON负载本身中指示错误。例如，API可能从结构上接受您的请求，但报告一个问题，如参数 (parameter)值无效、提示被内容过滤器标记 (token)或超出特定配额。

```json
{
  "error": {
    "code": "invalid_prompt",
    "message": "提供的提示被内容过滤器拒绝。",
    "type": "validation_error"
  }
}
```

在成功解析JSON后，检查其中的这类错误结构很重要，即使HTTP状态是`200 OK`。

```python
# 假设response.ok为True且response_data = response.json()成功

if "error" in response_data:
    error_info = response_data["error"]
    print(f"API在JSON负载中返回了一个错误:")
    print(f"  代码: {error_info.get('code', 'N/A')}")
    print(f"  消息: {error_info.get('message', '未提供消息。')}")
    # 根据错误码或类型实施具体处理
el:
    # 像之前一样处理成功的响应数据
    print("负载中未发现API特有错误。正在处理数据...")
    # ... 提取补全、选项等 ...
```

### 常见错误和基本处理策略

我们来总结一下常见错误并提出基本处理建议：

1. **网络/连接错误 (`requests.exceptions.RequestException`):** 这些错误甚至在您收到响应之前就发生了（超时、DNS解析失败、连接被拒绝）。

   - **处理：** 在您的`requests.post`或`requests.get`调用周围使用`try...except requests.exceptions.RequestException`代码块。记录错误。考虑在短时间延迟后重试请求，可能带有指数退避机制（每次失败尝试后等待更长时间）。
2. **认证错误 (401 Unauthorized, 403 Forbidden):** 您的API密钥可能无效、已过期，或者对所请求的操作缺少权限。

   - **处理：** 停止此请求的执行。清晰地记录错误。验证您的API密钥和权限。不要使用相同的凭据自动重试。可能需要通知开发人员。
3. **速率限制错误 (429 Too Many Requests):** 您在给定时间段内发送了太多请求。

   - **处理：** 暂停发送请求一段时间。API响应通常包含一个`Retry-After`头，指示需要等待多长时间（以秒为单位）。如果不存在，则实施指数退避（例如，等待1秒，然后2秒，4秒，8秒等）后重试。
4. **客户端错误 (其他4xx，例如，400 Bad Request):** 您的请求可能格式不正确（无效JSON，缺少必需参数 (parameter)）。

   - **处理：** 停止此请求的执行。记录错误和响应体，响应体通常包含关于错误内容的详细信息。在您的代码中修正请求结构。不要自动重试。
5. **服务器错误 (5xx，例如，500 Internal Server Error，503 Service Unavailable):** API提供方出现临时问题。

   - **处理：** 这些错误通常是暂时的。延迟后重试请求，使用指数退避。如果多次重试后错误仍然存在，请记录并可能需要通知运维人员。
6. **JSON解码错误 (`requests.exceptions.JSONDecodeError`):** 响应体无法解析为JSON。

   - **处理：** 记录错误和原始`response.text`。这可能表明响应格式不符合预期（例如，来自错误页面的HTML），或者是由真正的服务器问题导致JSON格式错误。
7. **API特有错误 (在JSON负载中):** 在200 OK响应中报告的错误。

   - **处理：** 从JSON中解析错误详情。记录具体的错误码和消息。根据错误类型进行处理（例如，如果提示违反内容策略，则修改提示；如果超出配额，则停止）。

> 这是处理API响应和潜在错误的一个典型流程。

有效处理API响应和错误对于构建可靠的LLM应用程序很重要。通过检查状态码、仔细解析预期的数据格式以及预料到各种故障模式，您可以编写更具弹性的Python代码，使其可预测地与LLM服务交互。请记住查阅具体的API文档以获取响应结构和错误码的详情。

## 参考资料

- [HTTP response status codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status) — MDN Web Docs Contributors (2024)
  对HTTP状态码的权威解释，详细说明其在网络通信中的作用。
- [Requests: HTTP for Humans™ Documentation](https://requests.readthedocs.io/en/latest/) — The Requests Community (2024)
  Python `requests` 库的官方文档，包含其API、响应对象和异常处理。
- [Error Retries and Exponential Backoff in AWS](https://docs.aws.amazon.com/general/latest/gr/api-retries.html) — Amazon Web Services (2023)
  提供API交互中实现带指数退避的重试机制的说明，有助于构建可靠的客户端应用程序。
- [json - JSON encoder and decoder](https://docs.python.org/3/library/json.html) — Python Software Foundation (2024)
  Python 内置 `json` 模块的文档，该模块用于解析和序列化 JSON 数据。

---

[上一节](02-%E4%BD%BF%E7%94%A8requests%E5%BA%93%E5%8F%91%E9%80%81API%E8%AF%B7%E6%B1%82.md) · [下一节](04-%E4%BD%BF%E7%94%A8%E5%AE%98%E6%96%B9Python%E5%AE%A2%E6%88%B7%E7%AB%AF%E5%BA%93.md)

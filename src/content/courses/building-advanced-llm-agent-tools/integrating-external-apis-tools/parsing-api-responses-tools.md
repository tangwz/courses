---
course: "building-advanced-llm-agent-tools"
chapter: "integrating-external-apis-tools"
lesson: "parsing-api-responses-tools"
sourceId: 6587
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-4-integrating-external-apis-tools/parsing-api-responses-tools"
title: "解析与转换API响应"
description: "介绍解析多种API响应格式（JSON、XML）并将其转换为LLM可用的方法。"
order: 2
plots: []
sourceHash: "1671dc7c346763467bd868bd40bc7f546cdace74a131fb3cdccb123d472425d7"
sourceCorrections: []
---

API返回的数据格式多样，且这些原始数据通常不处于LLM可以直接使用的理想状态。它们可能过于冗长、包含不相关信息，或其结构模型难以有效理解。因此，解析和转换API响应是构建有效LLM代理API工具的主要一步。您的目标是将API的输出转换为一种整洁、简洁且结构化的格式，以便LLM能够方便地理解并据此操作。

## 了解常见API响应格式

在转换数据之前，您首先需要解析它，这意味着解读其结构。大多数现代Web API使用几种常用格式进行数据交换。API的HTTP响应中的`Content-Type`头部通常会告知您预期的数据格式。

### JSON (JavaScript Object Notation)

JSON因其轻量级特性和易于被各种编程语言解析，成为许多Web API的事实上的标准。JSON数据以键值对的形式表示，类似于Python字典，并且可以包含嵌套的对象和数组（列表）。

例如，天气API可能会返回：

```json
{
  "location": "San Francisco, CA",
  "timestamp": "2023-10-27T10:00:00Z",
  "data": {
    "temperature": 18,
    "unit": "Celsius",
    "condition": "Partly Cloudy",
    "humidity": 65
  },
  "source": "WeatherProviderX"
}
```

在Python中，内置的`json`模块是您的主要工具。如果您使用像`requests`这样的库进行HTTP调用，响应对象通常会有一个便捷的`.json()`方法，直接将JSON响应解析为Python字典。否则，`json.loads(response_text)`会将JSON字符串转换为Python字典或列表。

### XML (可扩展标记 (token)语言)

XML是您会遇到的另一种格式，尤其是在较旧或企业级的API中。它使用标签来定义元素并分层结构化数据。

天气数据的XML等效形式可能如下所示：

```xml
<weatherReport>
  <location>San Francisco, CA</location>
  <timestamp>2023-10-27T10:00:00Z</timestamp>
  <data>
    <temperature units="Celsius">18</temperature>
    <condition>Partly Cloudy</condition>
    <humidity>65</humidity>
  </data>
  <source>WeatherProviderX</source>
</weatherReport>
```

Python的标准库包含`xml.etree.ElementTree`用于解析XML。您可以使用`ET.fromstring(xml_string)`解析XML字符串以获取根元素，然后使用`find()`（查找第一个匹配的子元素）、`findall()`（查找所有匹配的子元素）等方法遍历树结构，并通过`.attrib`访问元素属性或通过`.text`访问文本内容。

### 其他格式

虽然JSON和XML覆盖了大多数情况，但您偶尔可能会遇到API返回CSV（逗号分隔值）、纯文本甚至YAML格式的数据。Python有用于处理CSV数据的`csv`模块，而纯文本通常需要自定义字符串处理或正则表达式。对于YAML，可以使用`PyYAML`等库。

## 解析和转换策略

一旦您识别出格式，下一步就是将原始响应解析成Python数据结构。然后，您可以应用转换以使其适用于LLM。

### 解析API数据

对于JSON响应，解析通常简单直接：

```python
import json

api_response_text = '{"city": "London", "details": {"temp": 15, "desc": "Cloudy"}}'
try:
    data = json.loads(api_response_text)
    city = data.get("city")
    temperature = data.get("details", {}).get("temp") # Using .get() for safety
except json.JSONDecodeError:
    print("Error: Could not decode JSON response.")
    data = None
# Now 'data' is a Python dictionary
```

使用`data.get("key", default_value)`通常比直接键访问（`data["key"]`）更安全，因为它允许您在键缺失时提供一个默认值，从而避免`KeyError`异常。

对于XML，您将遍历树结构：

```python
import xml.etree.ElementTree as ET

xml_response_text = '<product><id>123</id><name>Wireless Mouse</name><price>25.99</price></product>'
try:
    root = ET.fromstring(xml_response_text)
    product_id = root.findtext("id")
    product_name = root.findtext("name")
    price_str = root.findtext("price")
    price = float(price_str) if price_str else None
except ET.ParseError:
    print("Error: Could not parse XML response.")
    root = None
# product_id is '123', product_name is 'Wireless Mouse', price is 25.99
```

`findtext()`是一种方便的方法，可以直接返回子元素的文本内容。

### 转换数据以供LLM使用

原始解析数据，即使是Python对象，对于LLM来说仍可能不够理想。转换的目标是：

1. **减少冗余**：LLM有上下文 (context)窗口限制（token限制）。只发送必要信息是有效的。
2. **提高清晰度**：集中、结构良好的数据更容易让LLM正确理解和使用。
3. **与任务要求对齐 (alignment)**：转换后的数据应直接服务于LLM当前的业务目标。
4. **标准化格式**：如果您的代理使用多个工具，标准化它们的输出格式可以简化代理的逻辑。

常见的转换技术包括：

- **字段选择**：只提取LLM所需的具体信息。如果API返回50个字段但LLM只需要3个，您的工具就应该过滤掉其余的。

  ```python
  # Assuming 'api_data' is a parsed JSON dictionary from a complex user profile
  transformed_user_info = {
      "username": api_data.get("username"),
      "email": api_data.get("user_contact", {}).get("primary_email"),
      "last_login": api_data.get("activity", {}).get("last_seen_at")
  }
  # transformed_user_info is now a smaller dictionary with only relevant fields
  ```
- **数据简化与扁平化**：API响应可能具有深度嵌套的结构。如果嵌套不相关，就将其扁平化。如果将具有单个键值对的对象列表转换为简单的值列表更有用，则进行此转换。

  ```python
  # api_data = {"items": [{"name": "Apple"}, {"name": "Banana"}, {"name": "Cherry"}]}
  # Simplified:
  # item_names = [item.get("name") for item in api_data.get("items", []) if item.get("name")]
  # item_names will be: ["Apple", "Banana", "Cherry"]
  ```
- **摘要或截断**：对于API返回的长文本 (long context)字段（例如，文章内容、产品描述），您可能需要截断它们或提取重要句子。这里要小心，因为复杂的摘要生成通常是LLM自身的任务。简单的截断或提供前N个字符/单词可以由工具完成。
- **为LLM提高可读性**：

  - **简洁的字符串**：有时，总结API数据的自然语言字符串是最佳选择。例如，与其返回`{"temperature": 22, "unit": "C", "condition": "Sunny"}`，不如返回：`"当前天气：22°C，晴朗。"`。如果LLM的角色是将此信息传达给用户，这会非常有效。
  - **整洁的JSON/字典**：如果LLM需要对数据执行进一步的逻辑操作或将其用作另一个工具的输入，通常首选整洁、最小的JSON结构或Python字典。
  - **项目列表**：清晰地呈现结果列表，也许是字符串中的项目符号点或简单的JSON数组。

### 解析和转换过程中的错误处理

事情可能会出错。API可能返回意料之外的结构，字段可能缺失，或者数据类型不正确。您的工具必须妥善处理这些情况。

```python
def process_product_api_response(response_text):
    try:
        data = json.loads(response_text)

        # Ensure 'product_info' and 'name' keys exist
        product_info = data.get("product_info")
        if not product_info or "name" not in product_info:
            return "Error: Essential product information (name) is missing in API response."

        name = product_info["name"]

        # Safely get price, convert to float, handle potential missing or invalid price
        price_str = product_info.get("price_details", {}).get("amount")
        price = None
        if price_str:
            try:
                price = float(price_str)
            except ValueError:
                return f"Warning: Product '{name}' has an invalid price format: {price_str}. Price not available."

        return {"name": name, "price": price if price is not None else "Not available"}

    except json.JSONDecodeError:
        return "Error: API response was not valid JSON."
    except Exception as e: # Catch any other unexpected errors
        return f"An unexpected error occurred during processing: {str(e)}"

# Example usage:
# result = process_product_api_response(api_response_string)
# if isinstance(result, str) and result.startswith("Error:"):
#     # Handle error message for LLM
# else:
#     # Use the processed dictionary
```

返回提供信息的错误消息或合理的默认值，使LLM代理能够理解问题并可能尝试其他方法或告知用户，而不是简单地失败。

## 流程可视化

从原始API响应到LLM可以有效使用的数据，涉及多个步骤。

> 该图显示了API响应数据的流程。它从原始响应开始，经过验证和解析，然后进行转换，最终生成可供LLM使用的数据，或在出现问题时生成错误消息。

通过仔细思考您的工具如何解析和转换API响应，您可以显著提升其可靠性以及LLM代理的整体效率。目标始终是以最直接、明确和有用的形式为LLM提供信息，以完成手头的任务。这通常涉及一个迭代过程：设计您的解析和转换逻辑，测试LLM如何使用输出，并根据需要调整。

## 参考资料

- [json - JSON encoder and decoder](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGJmiMvWXTqgEu_CqL7oaklkQd9xzxnVGCtLpeylpJiQ36X_QzuGIVp3NoalI974KJ-b6L6t4fNiaNs2mXElI13HlWC5gp32WnkBBOhXnpYN6arRlA3Mw3bp8rmCqU3n3CeqVOq) — Python Software Foundation (2023)
  介绍了Python标准库中用于解析和生成JSON数据的模块。
- [xml.etree.ElementTree - The ElementTree XML API](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE3NiKGOkSoY6st08VdHvF6jH_0GU8Ab4QrzsVJ_BQe3VSy8EaLECL1Burrhl0UcLVdvBTLexlBwRVEAqnDEVNaf7v8XjCIy94jTs2J0juajUemW2MBgNmH9M5ch17hyp75SO0xZ-j4Oy9P1iidlY51qv20gU9J) — Python Software Foundation (2023)
  详细说明了Python标准库中用于解析和构建XML结构的模块。
- [RESTful Web APIs](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEAurJ3DLpW6Xe2KYdUR7yRpiDvyk5120xElaIpNbbOMyTZbbEUYJKNtYrCCuUQpApvud6gLsvWJBl1privgyUBZwdof5l8XE2odd5Xk8XcCdr5S11W6hC2Bt7LHWUb-t-i2xV0LsnHWs39KPNVbRLaKiYgTurDMV0j35Rm3nAev4FHdQ==) — Leonard Richardson, Mike Amundsen, Sam Ruby (2013)
  Publisher: O'Reilly Media
  提供了RESTful API设计原则指南，包括常用数据格式及其客户端使用方法。

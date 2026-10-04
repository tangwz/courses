---
course: "building-advanced-llm-agent-tools"
chapter: "integrating-external-apis-tools"
lesson: "practice-wrapping-public-api"
sourceId: 6598
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-4-integrating-external-apis-tools/practice-wrapping-public-api"
title: "实践：封装公共API作为LLM工具"
description: "一个动手实践环节，用于封装公共API，使其可作为LLM智能体的工具使用。"
order: 7
plots: []
sourceHash: "efdcc11a1afb533799a94bde0b325fb30401efc36df6b5477403097703709c63"
sourceCorrections: []
---

这个动手练习将引导你封装一个公共API，将其变为LLM智能体可以使用的实用工具。我们将使用Open-Meteo API，这是一个免费的天气预报服务，基础使用无需API密钥，非常适合本次学习实践。我们的目标是创建一个工具，让LLM智能体能获取给定地理坐标的当前天气。

在此过程中，我们将涉及本章前面讨论的几个主题，例如为LLM定义工具接口、进行API调用、解析响应，以及以对智能体有用的方式构造输出。

### 选择目标：Open-Meteo API

我们选择Open-Meteo API是因为它简单易用。具体来说，我们将使用其获取当前天气数据的端点。获取柏林（纬度：52.52，经度：13.41）当前天气的典型请求如下：
`https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current_weather=true`

此请求返回一个包含各种天气细节的JSON响应。对于我们的工具，我们将提取并展示温度和风速。

### 步骤1：为LLM定义工具

在编写工具的Python代码之前，我们必须首先定义LLM如何理解和使用此工具。这包括确定工具的名称、清晰的描述以及预期的输入和输出模式。

- **工具名称：** `get_current_weather`
- **工具描述：** “获取指定经纬度的当前天气（温度和风速）。当你需要查询特定地理位置的当前天气状况时，请使用此工具。”
- **输入模式：**
  - `latitude` (浮点数, 必填)：地点的纬度。
  - `longitude` (浮点数, 必填)：地点的经度。
- **输出模式（字典或JSON对象）：**
  - `temperature_celsius` (浮点数)：当前温度，单位摄氏度。
  - `wind_speed_kmh` (浮点数)：当前风速，单位公里/小时。
  - `summary` (字符串)：天气状况的简要文字概括。

LLM智能体将依据此定义来理解何时以及如何调用我们的工具。精确的描述有助于LLM做出更好的判断。

### 步骤2：实现Python工具封装

现在，我们来实现作为工具的Python函数。我们将使用`requests`库进行HTTP调用。如果尚未安装，可以使用pip安装：`pip install requests`。

```python
import requests
import json

def get_current_weather_from_api(latitude: float, longitude: float) -> dict:
    """
    从Open-Meteo API获取给定经纬度的当前天气数据。

    参数：
        latitude: 地点的纬度。
        longitude: 地点的经度。

    返回：
        包含温度、风速和摘要的字典，
        或API调用失败时的错误消息。
    """
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # 对于错误的响应（4XX或5XX）抛出HTTPError

        data = response.json()

        # 提取相关信息
        current_weather_data = data.get("current_weather", {})
        temperature = current_weather_data.get("temperature")
        wind_speed = current_weather_data.get("windspeed")
        weather_code = current_weather_data.get("weathercode")

        if temperature is None or wind_speed is None:
            return {"error": "Could not retrieve complete weather data from API response."}

        # 对天气代码进行非常简单的解释以生成摘要
        # 对于生产工具，这会更全面
        weather_summary = f"Temperature: {temperature}°C, Wind Speed: {wind_speed} km/h."
        if weather_code is not None:
            if weather_code == 0:
                weather_summary += " Condition: Clear sky."
            elif weather_code in [1, 2, 3]:
                weather_summary += " Condition: Mainly clear to partly cloudy."
            elif weather_code > 40 and weather_code < 70 : # various rain codes
                 weather_summary += " Condition: Rainy."
            # 根据需要添加更多条件
            else:
                weather_summary += " Condition: Unspecified (code: " + str(weather_code) + ")."

        # 这是我们之前定义的结构化输出
        return {
            "temperature_celsius": float(temperature),
            "wind_speed_kmh": float(wind_speed),
            "summary": weather_summary
        }

    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error occurred: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"Request error occurred: {req_err}"}
    except json.JSONDecodeError:
        return {"error": "Failed to decode API response."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# 示例用法（用于直接测试）：
if __name__ == "__main__":
    # 柏林坐标
    berlin_lat = 52.52
    berlin_lon = 13.41
    weather_info = get_current_weather_from_api(berlin_lat, berlin_lon)
    print(json.dumps(weather_info, indent=2))

    # 无效请求示例（例如，超出范围的坐标）
    # Open-Meteo API可能仍然返回一些东西，但测试边界情况是好的
    invalid_lat = 200.0 
    invalid_lon = 200.0
    error_info = get_current_weather_from_api(invalid_lat, invalid_lon)
    print(f"\n使用无效坐标进行测试（预期API错误或已处理的错误）：")
    print(json.dumps(error_info, indent=2))
```

**该实现的一些方面：**

1. **API交互：** 我们通过参数 (parameter)构造URL，并使用`requests.get()`获取数据。
2. **响应解析：** `response.json()`将JSON响应解析为Python字典。
3. **数据提取与转换：** 我们操作字典以查找`temperature`和`windspeed`。注意我们如何选择特定信息并将其转换为我们所需的输出模式。这是为LLM总结和呈现API数据的简单形式。`weather_summary`字符串也是一种处理后的输出形式。
4. **错误处理：** `try-except`块处理潜在问题，如网络问题（`requests.exceptions.RequestException`）、来自API的HTTP错误（例如，通过`response.raise_for_status()`产生的404未找到、500服务器错误），以及解析响应的问题（`json.JSONDecodeError`）。返回错误字典允许智能体或编排器妥善处理失败情况。
5. **结构化输出：** 函数返回一个与我们定义的`输出模式`匹配的字典。这种一致性对于LLM很重要。

### 步骤3：智能体集成考量

虽然将此工具注册到LLM智能体的具体方法取决于特定的框架（如LangChain、LlamaIndex或自定义智能体循环），但核心组成部分是：

1. **Python函数：** `get_current_weather_from_api`是可执行代码。
2. **工具定义：** 名称（`get_current_weather`）、描述以及输入/输出模式会提供给LLM智能体。智能体使用这些元数据来决定何时调用函数以及传递哪些参数 (parameter)。

例如，在类似LangChain的设置中，你可能会将`get_current_weather_from_api`封装在一个`Tool`对象中，提供名称、描述，以及可能基于Pydantic模型用于输入验证的`args_schema`。

当LLM遇到“柏林（纬度52.52，经度13.41）天气如何？”这样的查询时，它会使用`get_current_weather`的描述来识别其相关性，从查询中提取`latitude`和`longitude`，并调用我们的Python函数。然后，我们的工具生成的结构化JSON输出会返回给LLM，LLM可以使用它来形成自然语言的回答。

### 步骤4：测试工具

**直接测试：**
如`if __name__ == "__main__":`块中所示，你可以直接测试Python函数：

```python
# 柏林坐标
berlin_lat = 52.52
berlin_lon = 13.41
weather_info = get_current_weather_from_api(berlin_lat, berlin_lon)
print(json.dumps(weather_info, indent=2))

# 无效位置示例，例如海洋中部或无效坐标
# Open-Meteo API可能会对此返回错误或特定值
pacific_lat = 0.0
pacific_lon = -150.0 # 太平洋中部
pacific_weather = get_current_weather_from_api(pacific_lat, pacific_lon)
print(f"\n太平洋中部天气：")
print(json.dumps(pacific_weather, indent=2))
```

这有助于确保核心逻辑、API交互和数据解析正常工作。

**通过LLM智能体测试：**
一旦集成到智能体框架中，你可以通过向LLM提出应触发工具的问题来测试它。例如：

- “你能告诉我纬度34.05，经度-118.24的当前温度和风速吗？”
- “我需要知道我当前位置的天气：纬度40.71，经度-74.00。”

观察LLM是否正确识别出需要使用该工具，准确提取参数 (parameter)，以及工具是否成功执行并将结构化数据返回给LLM以生成最终响应。

### 回顾本章要点

这个实践练习体现了前面讨论的几个原则：

- **设计工具接口：** 我们为LLM定义了清晰的名称、描述以及输入/输出模式。
- **解析API响应：** 我们使用了`response.json()`和字典导航。
- **构造和概括API数据：** 我们将原始API数据转换为针对LLM的简洁、结构化格式，包括文本摘要。
- **错误处理：** 我们包含了API调用的基本错误处理。
- **安全性（简要说明）：** 我们选择的API是公共的且无需密钥。对于需要密钥的API，你需要安全地管理这些密钥（例如，使用环境变量或秘密管理器），并在`requests.get()`调用的`headers`或`params`中传递它们。在生产系统中，切勿将敏感密钥直接硬编码到工具的源代码中。例如：

  ```python
  # API key usage
  # 导入 os 模块
  # api_key = os.environ.get("MY_WEATHER_API_KEY")
  # headers = {"Authorization": f"Bearer {api_key}"}
  # response = requests.get(url, params=params, headers=headers)
  ```
- **速率限制和重试：** 尽管在此简单示例中未实现，但对于访问外部API的生产工具，你需要注意API的速率限制。实现重试逻辑（例如，对于`503 Service Unavailable`等瞬时错误采用指数退避）也是常见的需求。`tenacity`等库可以简化重试机制的实现。

这个练习提供了一个基础模板。你可以调整此方法来封装各种其他公共或私有API，显著增强你的LLM智能体的功能。核心在于始终为LLM明确定义工具的用途和接口，并可靠地处理与外部服务的交互。

## 参考资料

- [LangChain: Tools](https://python.langchain.com/docs/modules/agents/tools/) — LangChain Developers (2024)
  介绍如何使用LangChain框架为LLM代理定义和集成自定义工具，包括输入验证的模式定义。
- [requests: HTTP for Humans](https://readthedocs.io/en/latest/) — Kenneth Reitz (2024)
  提供了使用requests库进行HTTP请求、处理响应和管理错误的全面指南，这是API封装的基础。
- [Open-Meteo API Documentation](https://open-meteo.com/en/docs) — Open-Meteo (2024)
  Open-Meteo天气API的官方文档，详细说明了端点、参数和响应格式，对于正确集成该服务至关重要。
- [tenacity documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHXNDF53joZMz0I0G3uQwpxUBJoMlCobhrP6QoJ2PPrp_WF4w4XQ3BwkUCAJXDusf3O_iYVVxUJOhjrnYf3XB-a7Kishcic_c5lt06kRcQUKnuw9v3F0fJAToTF0DJAxaeeFfCZ) — tenacity Developers (2024)
  tenacity库的文档，提供了实现指数退避和抖动重试逻辑的模式，有助于处理API调用中的瞬时错误。

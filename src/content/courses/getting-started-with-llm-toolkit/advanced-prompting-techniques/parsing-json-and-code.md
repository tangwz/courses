---
course: "getting-started-with-llm-toolkit"
chapter: "advanced-prompting-techniques"
lesson: "parsing-json-and-code"
sourceId: 7758
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-2-advanced-prompting-techniques/parsing-json-and-code"
title: "解析JSON与代码片段"
description: "使用 Kerb 提供的专用解析器，可靠地从文本中提取JSON对象和代码块。"
order: 6
plots: []
sourceHash: "10578abd9215122c18df40fcbb1e1bc9776a8f4089095059e03075e18af01acc"
sourceCorrections: []
---

从模型生成结构化输出是一个常见需求，但原始响应通常无法直接使用。大型语言模型本质上是文本生成器，其输出可能被对话文本包裹、包含在Markdown块中，甚至含有小的格式错误。直接使用Python的`json`模块等标准库进行解析可能不稳定，并造成应用程序出错。

这时，专用解析器就显得不可或缺。它们被设计用于从大型语言模型有时产生的嘈杂、不可预测的文本中查找并提取结构化数据，为您提供一种可靠的方法，将原始字符串转换为可用数据结构。

### 从周围文本中提取JSON

您会常遇到的一种模式是大型语言模型提供一个被解释性文本或Markdown代码块包裹的JSON对象。例如，您可能会请求用户数据，并收到类似这样的响应：

````
当然！这是您请求的用户信息，格式为JSON对象：

```json
{
  "name": "Alice Johnson",
  "email": "alice@example.com",
  "age": 28,
  "roles": ["developer", "team_lead"]
}
```

如果您需要其他详细信息，请告诉我。
````

标准的JSON解析器无法处理这段文本。然而，`extract_json`函数就是为了处理这种特定情况而设计的。它会扫描文本，识别出JSON块，并将其提取出来进行解析。

````python
from kerb.parsing import extract_json

llm_output = """这是您请求的用户数据：

```json
{
  "name": "Alice Johnson",
  "email": "alice@example.com",
  "age": 28,
  "roles": ["developer", "team_lead"]
}
```

此数据是从用户数据库中提取的。"""

result = extract_json(llm_output)

if result.success:
    user_data = result.data
    print(f"成功提取用户：{user_data.get('name')}")
else:
    print(f"提取JSON失败：{result.error}")
````

该函数返回一个包含结果的`ParseResult`对象。`success`属性告知您操作是否成功，如果成功，`data`属性则包含已解析的Python字典。这使得构建能够应对大型语言模型输出多样性（冗长程度）的灵活工作流变得简单。

### 处理格式错误和嵌入 (embedding)式JSON

有时，大型语言模型的输出*几乎*正确，但包含小的语法错误，这会使严格的解析器无法运行。常见问题包括尾随逗号、键缺少引号或使用单引号而非双引号。

此外，模型可能会将JSON对象直接嵌入句子中，而没有任何特殊格式。例如：`配置是：{"api_key": "sk-xxx", "timeout": 30}，这应该可以运行。`

为了处理这些情况，您可以使用`parse_json`或`extract_json`，并采用更宽松的解析模式。

- `ParseMode.STRICT`：要求输入为格式完全正确的JSON字符串。
- `ParseMode.LENIENT`：在解析前尝试修复常见错误，例如尾随逗号或缺少引号。它仍只会在整个字符串旨在为JSON时进行解析。
- `ParseMode.BEST_EFFORT`：扫描字符串以找到第一个看起来有效的JSON对象或数组，即使它嵌入在其他文本中。

我们来看看`ParseMode.LENIENT`如何自动修复格式错误的JSON字符串。

```python
from kerb.parsing import parse_json, ParseMode

# 这个JSON有未加引号的键和尾随逗号
malformed_json = """{
    name: "Bob",
    age: 35,
    active: true,
}"""

result = parse_json(malformed_json, mode=ParseMode.LENIENT)

if result.success:
    print(f"数据：{result.data}")
    print(f"JSON是否已修复？{result.fixed}")
```

处理嵌入在文本中的JSON时，`ParseMode.BEST_EFFORT`是您的最佳选择。它会找到并提取结构化部分，忽略其余内容。

```python
from kerb.parsing import extract_json, ParseMode

llm_output = '配置是：{"api_key": "sk-xxx", "timeout": 30, "retries": 3}，这应该可以运行。'

# BEST_EFFORT 会在句子中找到JSON
result = extract_json(llm_output, mode=ParseMode.BEST_EFFORT)

if result.success:
    print(f"提取的配置：{result.data}")
```

使用这些模式提供了安全保障，使您的应用程序在面对大型语言模型输出质量中虽小但常见的多样性时更加可靠。

### 解析代码片段

提取代码也遵循类似的模式。尽管大型语言模型擅长生成代码，但它们通常会将其包裹在Markdown块中并添加解释。一种可靠的代码提取方法是指导模型将其置于JSON对象内部。这把解析代码的问题变成了更易于处理的解析JSON问题。

例如，您可以提示模型以特定JSON结构返回其输出。

**提示：**

````
分析以下Python函数，并将其转换为包含这些字段的JSON表示形式：“name”、“docstring”、“parameters”和“return_type”。

```python
def calculate_bmi(weight: float, height: float) -float:
    """计算身体质量指数。"""
    return weight / (height ** 2)
```

仅返回一个单一的、有效的JSON对象。
````

模型理想情况下会返回一个干净的JSON对象，您可以使用我们已讨论过的技术进行解析。

````python
from kerb.parsing import extract_json

# 对上述提示的模拟大型语言模型响应
llm_response = """
```json
{
  "name": "calculate_bmi",
  "docstring": "计算身体质量指数。",
  "parameters": [
    {"name": "weight", "type": "float"},
    {"name": "height", "type": "float"}
  ],
  "return_type": "float"
}
```
"""

result = extract_json(llm_response)

if result.success:
    code_info = result.data
    print(f"函数名称：{code_info.get('name')}")
    print(f"文档字符串：{code_info.get('docstring')}")
````

这种方法使用了模型遵循格式指令的能力，使输出变得可预测且易于使用现有工具进行解析。通过将代码视为大型JSON结构中的字符串值，您可以可靠地提取它，用于代码分析、生成或执行工作流。

## 参考资料

- [JSON encoder and decoder](https://docs.python.org/3/library/json.html) — Python Software Foundation (2024)
  Python内置JSON解析模块的官方文档，说明其标准的、严格的行为，这在处理LLM输出的变化时常遇到困难。
- [JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode) — OpenAI (2024)
  关于如何提示大型语言模型可靠生成结构化JSON输出的实用指南，这是高级提示中的一项关键技术，需要强大的下游解析能力。
- [JSON](https://www.json.org/json-en.html) — Douglas Crockford (2017)
  定义JSON数据交换格式的官方网站，阐述其严格的语法规则，这使得处理LLM生成的文本中常见偏差时需要专门的解析器。

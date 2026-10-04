---
course: "getting-started-with-llm-toolkit"
chapter: "advanced-prompting-techniques"
lesson: "dynamic-prompt-templates"
sourceId: 7754
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-2-advanced-prompting-techniques/dynamic-prompt-templates"
title: "使用模板引擎创建动态提示词"
description: "使用 Kerb 的模板引擎创建可变且可重复使用的提示词。"
order: 2
plots: []
sourceHash: "a744822bc789229821c959e904ddc291cc6b6f25a17409f056422505e67cde4c"
sourceCorrections: []
---

提示词 (prompt)在实际应用中很少是固定不变的。将用户输入、获取到的文档或对话历史插入到预设结构中是一种普遍需求。通过手动拼接字符串来硬编码提示词既困难又容易出错。这会将应用程序的逻辑与提示词内容混在一起，使得代码难以阅读、维护和更新。

一个更好的方法是使用模板引擎。这会将提示词的固定结构与填充其中的动态数据分开。您可以定义一个带有变量占位符的模板，在运行时，引擎会将这些占位符替换为实际值。这使得您的提示词可重复使用，更易于管理，也更便于版本控制。

### 基本变量替换

模板引擎使用双大括号，例如 `{{variable_name}}`，来标记 (token)占位符。要生成完整的提示词 (prompt)，您需要提供模板字符串以及一个包含变量值的字典。

`render_template` 函数处理这种替换。让我们创建一个简单的系统提示词，它可以适应不同的人工智能助手角色。

```python
from kerb.prompt import render_template

template = """You are a {{role}} assistant specialized in {{domain}}.
Your task is to help users with {{task}}."""

variables = {
    "role": "helpful",
    "domain": "Python programming",
    "task": "code review and debugging"
}

final_prompt = render_template(template, variables)

print(final_prompt)
```

此代码会产生以下输出，所有变量都被正确替换：

```text
You are a helpful assistant specialized in Python programming.
Your task is to help users with code review and debugging.
```

通过更改 `variables` 字典中的值，您可以重复使用相同的模板来为许多其他角色生成提示词，而无需改变提示词的结构。

### 使用嵌套变量处理结构化数据

您的动态数据不限于简单的键值对。您也可以传递嵌套对象，比如字典，并使用点表示法访问它们的值。这在用结构化资料填充提示词 (prompt)时非常有用，例如用户档案或配置设置。

设想一个场景，您希望根据用户的档案资料来个性化回复。

```python
from kerb.prompt import render_template

template = """为以下用户生成个性化回复。

用户档案：
- 姓名: {{user.name}}
- 角色: {{user.role}}
- 兴趣: {{user.interests}}

在回复关于 {{context}} 的查询时，请提及他们对 {{user.interests}} 的兴趣。"""

variables = {
    "user": {
        "name": "Alice Johnson",
        "role": "Senior Developer",
        "interests": "机器学习, 云架构"
    },
    "context": "推荐学习资源"
}

personalized_prompt = render_template(template, variables)

print(personalized_prompt)
```

生成的提示词将正确填充嵌套的用户数据：

```text
为以下用户生成个性化回复。

用户档案：
- 姓名：Alice Johnson
- 角色：高级开发人员
- 兴趣：机器学习, 云架构

在回复关于推荐学习资源的查询时，请提及他们对机器学习、云架构的兴趣。
```

### 适应不同模板格式

尽管 `{{...}}` 是一种常见语法，但一些大型语言模型（LLM）提供商或框架可能会使用不同的占位符分隔符，例如单大括号 (`{...}`) 或百分号 (`%...%`)。模板引擎通过一个可选的 `delimiters` 参数 (parameter)来适应这种情况，该参数接受一个包含开头和结尾分隔符的元组。

比如，要使用单大括号占位符，您可以这样做：

```python
from kerb.prompt import render_template

template = "将 '{text}' 从 {source_lang} 翻译成 {target_lang}。"
variables = {
    "text": "你好！",
    "source_lang": "英语",
    "target_lang": "西班牙语"
}

# 将单大括号指定为分隔符
final_prompt = render_template(template, variables, delimiters=("{", "}"))

print(final_prompt)
```

这种灵活性保证了即使您的目标模型或平台具有不同的模板语法，您也可以使用相同的渲染逻辑。

### 构建可靠的应用程序

在生产环境中，您无法总是保证在渲染提示词 (prompt)时每个变量都可用。缺少变量通常会导致错误，可能会使您的应用程序崩溃。

为了构建更具韧性的系统，您可以使用 `render_template_safe`。此函数与 `render_template` 类似，但允许您为任何缺失的变量指定一个默认回退值，从而避免错误。

```python
from kerb.prompt import render_template_safe

template = "用户: {{user_name}}\n情境: {{context}}"

# 字典中缺少 'context' 变量
variables = {
    "system_role": "AI 助手",
    "user_name": "Bob"
}

# 使用安全渲染，并指定一个默认回退字符串
safe_prompt = render_template_safe(template, variables, default="[未提供]")

print(safe_prompt)
```

该函数不是抛出错误，而是用默认值替换缺失的占位符，生成一个可用的提示词：

```text
用户: Bob
情境: [未提供]
```

## 参考资料

- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2023)
  Publisher: OpenAI
  为设计大型语言模型的有效提示词提供了实用建议和最佳实践，强调了动态模板所促进的清晰指令和结构化输入的重要性。

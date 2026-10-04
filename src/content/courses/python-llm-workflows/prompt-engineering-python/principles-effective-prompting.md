---
course: "python-llm-workflows"
chapter: "prompt-engineering-python"
lesson: "principles-effective-prompting"
sourceId: 5196
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-8-prompt-engineering-python/principles-effective-prompting"
title: "有效提示的原则"
description: "编写提示的基本指导方针，旨在产生所需的LLM响应。"
order: 1
plots: []
sourceHash: "f391b97434b1422f76ca304b67f04fa4c7bcf122d4dc858b27c1fb3033b95482"
sourceCorrections: []
---

与大型语言模型（LLM）进行有效交流，取决于对提示工程 (prompt engineering)艺术与技巧的掌握。提示是指导模型行为的主要工具。把它看作是为一位能力很强但有时会逐字理解的助手提供指令或设定任务，而不是简单的搜索查询。编写清晰、具体并提供足够背景信息的提示，是你实现Python应用中可靠和有用结果的根本保证。

让我们分析支撑有效提示的要点原则。遵循这些指导方针将显著提升LLM响应的质量和可预测性。

### 清晰具体

模糊是良好LLM输出的敌人。模糊的提示常导致宽泛、无用甚至错误的响应，因为模型有太多解释空间。

- **明确说明任务：** 清晰地阐明你希望LLM做什么。与其说“告诉我关于Python列表的信息”，不如试试“解释Python列表，包括如何创建它们，以及`append()`和`pop()`等常用方法，并提供一个简单的代码示例。”
- **定义范围和限制：** 如果任务范围较广，请缩小范围。与其说“写一封营销邮件”，不如说“为一门面向中级开发者的新Python在线课程写一封营销邮件。强调LangChain和RAG部分。字数限制在150字以内，并包含访问课程页面的行动号召。”
- **避免使用行话（除非合适）：** 如果预期受众或所需角色理解特定行话，请使用它。否则，请选择更清晰、更普遍理解的术语。

思考一下区别：

- **模糊：** `prompt = "Summarize this article."`
- **具体：** `prompt = "Summarize the main findings of the following article in three bullet points, suitable for a non-technical audience."`

第二个提示就所需输出格式、目标受众和侧重，为模型提供了更清晰的指令。

### 提供充足背景信息

LLM不具备你具体情况或对话前几部分的固有知识，除非你提供。背景信息能帮助模型定位，使其能生成更相关和准确的响应。

- **背景信息：** 包含模型理解请求所需的必要细节。如果请求代码修改，请提供原始代码。如果讨论特定话题，请提供简要总结或相关摘录。
- **对话历史：** 在多轮交互（如聊天机器人）中，包含相关的过往消息。许多框架会自动处理此部分，但理解这个原则很重要。
- **相关数据：** 如果任务涉及特定数据（例如，总结用户反馈），请将这些数据直接包含在提示中，或确保模型可以访问它（正如我们将在RAG系统中看到的那样）。

示例：

- **无背景信息：** `prompt = "Is this approach suitable?"` （模型不知道“这种方法”是什么）。
- **有背景信息：**

  ```python
  original_code = """
  def process_data(data):
      # ... 复杂处理 ...
      return result
  """
  prompt = f"""
  给定以下Python代码：
  {original_code}
  我计划重构它，通过将其拆分为更小的辅助函数来提高可读性。
  这种方法适合提升可维护性吗？请解释原因。
  """
  ```

  这个提示提供了必要的代码和用户目标，使LLM能够提供有见地的建议。

### 定义角色或身份

指导LLM扮演特定角色或身份，能显著影响其响应的风格、语气、深度和侧重。这有助于使输出符合预期。

- **专家角色：** “扮演一位高级数据科学家...”，“你是一位严谨的代码审查员...”，“以宪法专家身份回应...”
- **目标受众适应：** “像是在和一位高中生交谈一样解释这个概念。”，“面向工程领域受众描述技术细节。”
- **创意角色：** “以莎士比亚的风格写作...”，“以一位机智的海盗身份回应...”

示例：

- **通用：** `prompt = "Explain the benefits of using virtual environments in Python."`
- **基于角色：** `prompt = "Act as a Python programming instructor. Explain the benefits of using virtual environments in Python to a beginner programmer, emphasizing why it prevents dependency conflicts."`

角色设定鼓励使用教学语气，并将解释侧重于与目标受众相关的一个特定好处。

### 指定输出格式

LLM可以生成多种格式的文本。明确要求特定结构，能使输出更可预测，也更容易解析或直接在你的应用代码中使用。

- **结构化数据：** 请求JSON、CSV、XML或YAML等格式。对模式（键、数据类型）保持精确。
- **文本格式：** 请求要点、编号列表、Markdown表格、特定标题，或带语言标注的代码块。
- **长度限制：** 指定所需输出长度（例如，“一个段落”，“少于100字”，“提供一个简明摘要”）。

示例：

- **非结构化：** `prompt = "Extract the main points from the meeting notes."`
- **结构化：**

  ```python
  meeting_notes = """
  会议记录 - 凤凰项目启动
  参会者：爱丽丝、鲍勃、查理
  日期：2023-10-27
  - 爱丽丝介绍了项目目标：2024年第一季度前启动。
  - 鲍勃讨论了资源分配。还需要2名工程师。
  - 查理概述了初始技术栈：Python, FastAPI, PostgreSQL。
  - 行动项：鲍勃下周前完成工程师分配。联系方式：bob@example.com
  """
  prompt = f"""
  从以下会议记录中：
  ---
  {meeting_notes}
  ---
  提取项目名称、参会者（字符串列表形式），以及任何行动项（包含负责人和截止日期，如果提及）。
  以JSON对象形式提供输出，键为：'project_name', 'attendees', 'action_items'（一个对象列表，每个对象包含'task', 'owner', 'deadline'）。
  如果未找到某个值，则使用null。
  """
  ```

这指导LLM生成易于解析的JSON：

```json
{
  "project_name": "凤凰项目",
  "attendees": ["爱丽丝", "鲍勃", "查理"],
  "action_items": [
    {
      "task": "完成工程师分配",
      "owner": "鲍勃",
      "deadline": "下周"
    }
  ]
}
```

### 设定限制和界限

通过定义模型*应该*做什么和*不应该*做什么来指导它。这有助于防止不相关、不合需要或不安全的内容。

- **语气和风格：** “使用正式语气。”，“避免技术行话。”，“保持鼓励。”
- **内容包含/排除：** “只侧重软件方面。”，“不要提及具体价格。”，“包含欧洲历史的例子。”
- **安全/道德准则：** “不要生成有害内容。”，“避免偏见性语言。”（虽然模型有内置的安全措施，但加强限制也会有帮助）。

示例：

- **开放式：** `prompt = "Write a comparison between LangChain and LlamaIndex."`
- **受限：** `prompt = "Write a balanced comparison between LangChain and LlamaIndex for building RAG systems. Focus on their core strengths in data indexing and workflow orchestration. Keep the tone objective and avoid declaring one as definitively 'better'. Limit the response to two paragraphs."`

### 从简开始，逐步完善

尽管复杂的提示可以获得复杂的结果，但通常最好从简单的提示开始，然后逐步完善。

1. **起草基本提示：** 为核心任务编写一个直接指令。
2. **测试并观察：** 运行提示并分析输出。它符合基本要求吗？它在哪里不足？
3. **完善并添加元素：** 根据观察到的输出和你的目标，逐步增加具体性、背景信息、角色、格式指令或限制。
4. **重新测试：** 评估改进后的提示。重复完善过程，直到输出持续满足你的需求。

这个迭代过程比试图一次性完善一个高度复杂的提示更易于管理。它能够有条理地进行调试和改进。

> 提示完善的迭代方法。

掌握这些原则，为提示工程 (prompt engineering)打下了坚实的底子。通过在指令中做到清晰、提供背景、注意角色、指定格式和设定限制，并采用迭代完善过程，你将显著提高你从LLM获取高质量、可预测响应的能力，在你的Python应用中。这些要点为少样本提示等更高级技术铺平了道路，我们接下来将分析这些技术。

## 参考资料

- [Prompt engineering](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFQ-6ez4eBZFxqXDsgeaNyFCvpMDo1NOdJRvK2ziS9W67BXCZSezOJB8cXJyb4KMR_wojHaGOnBDQ6Ays1Mg3C7InL22xWQkC0X_ys4fywH0TbfMf02knyDKA==) — OpenAI (2023)
  Publisher: OpenAI
  介绍设计有效提示语的基本技术和最佳实践，涵盖清晰度、具体性、上下文和迭代改进。
- [Introduction to prompt design](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEjT2eL5WDrIV1J5f_2z80DaqOR7CkoX1tBBMC-tIX1NU9f5L9jJuLI00MCGh9r3fhFteKos7bqgxykxFRMTTcs-XwINsK0DsSfDsWNIakWG0dEwmNrvwcbzVr7Nyvd5fRRlKRp4KELcdgMOYWGWijkOm344FDHPw7iTHlgadmSNdzKPAU_WDJBupV0VqKzy2pwxT6Rsdxp_g=) — Google Cloud (2025)
  Publisher: Google Cloud
  提供制作清晰有效提示语的指南和示例，强调清晰度、指令和提供上下文等原则。

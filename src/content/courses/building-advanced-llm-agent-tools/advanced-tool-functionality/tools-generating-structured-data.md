---
course: "building-advanced-llm-agent-tools"
chapter: "advanced-tool-functionality"
lesson: "tools-generating-structured-data"
sourceId: 6610
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-5-advanced-tool-functionality/tools-generating-structured-data"
title: "构建生成结构化数据的工具"
description: "创建协助LLM生成良好格式的结构化数据输出（如JSON、XML或CSV）的工具。"
order: 6
plots: []
sourceHash: "10c94e2ec18b9eab3dac45c5df97c2157c073bf05755c933de835e6003e87cd1"
sourceCorrections: []
---

大型语言模型（LLM）擅长生成人类可读文本，但有时难以持续生成严格定义、机器可读格式的数据，如JSON、XML或CSV。当LLM代理需要与其他软件系统、API或数据库交互时，或者当输出需要以程序方式存储和处理时，这种精确性通常是必要的。因此，构建协助LLM生成良好结构化数据的工具是提升代理能力的一个重大进展，使代理能够与其环境进行更可靠、更复杂的互动。

这些工具不会取代LLM的生成能力；相反，它们指导并约束LLM，以确保其输出符合预定义模式。这通常结合了对LLM的清晰指令、模式定义和验证机制。

### 明确目标：指定模式

构建结构化数据生成工具的第一步是明确定义您希望LLM生成的数据结构。这个定义，即模式，是LLM的蓝图，也是验证的基础。

对于**JSON**数据（这是一项常见需求），Pydantic等Python库非常有用。Pydantic允许您使用标准Python类型提示定义数据模式。

```python
from pydantic import BaseModel, EmailStr, PositiveInt, field_validator
from typing import Optional, List

class UserProfile(BaseModel):
    username: str
    email: EmailStr
    age: Optional[PositiveInt] = None
    is_active: bool = True
    tags: List[str] = []
    score: float

    @field_validator('username')
    @classmethod
    def username_must_be_alphanumeric(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError('用户名必须是字母数字且不含空格。')
        return v

    @field_validator('score')
    @classmethod
    def score_must_be_between_zero_and_one(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError('分数必须在0.0到1.0之间。')
        return v
```

在这个`UserProfile`模型中：

- 标准类型如 `str`、`PositiveInt`、`bool` 和 `float` 定义了预期数据类型。
- `EmailStr` 是 Pydantic 提供的类型，用于验证电子邮件格式。
- `Optional` 指示并非严格必需的字段。
- `List[str]` 定义一个字符串列表。
- 自定义验证器（`@field_validator`）可以强制执行更具体的规则，例如 `username` 的格式或 `score` 的范围。

对于**CSV**数据，模式可以像标题名称列表及其预期数据类型（如果需要）的指示一样简单。对于**XML**，您可以使用DTD或XSD，尽管指导LLM生成有效且复杂的XML可能更具挑战性，并且通常需要更详细的提示或后期处理。

### 引导LLM：提示策略

一旦有了模式，工具就需要指导LLM生成符合该模式的数据。有效的提示在这里很重要。您给LLM的提示应包含：

1. **任务的明确描述：** “生成一个代表用户资料的JSON对象。”
2. **模式定义（或摘要）：** 您可以将Pydantic模型结构（或JSON模式表示）序列化到提示中，或者提供字段、其类型和任何约束的简洁文本描述。例如：“该JSON对象应包含：`username`（字符串，字母数字）、`email`（字符串，有效电子邮件格式）、`age`（整数，正数，可选）、`is_active`（布尔值，默认为true）、`tags`（字符串列表，可选）和`score`（浮点数，0.0到1.0之间）。”
3. **输入数据（如果有）：** 如果LLM是根据自然语言输入生成结构化数据，请清晰地提供该输入。“为'testuser123'生成一个用户资料，电子邮件'[test@example.com](mailto:test@example.com)'，分数为0.75，标签为['beta', 'tester']。”
4. **示例（小样本提示）：** 包含一两个有效输出JSON对象的示例可以显著提高LLM的性能和格式依从性。

目标是为LLM提供足够的上下文 (context)，使其理解所需内容和结构约束。

### 验证与优化：确保正确性

即使有仔细的提示，LLM有时可能会生成与模式不完全匹配的输出。因此，结构化数据生成工具的一个重要组成部分是验证步骤。

使用我们的Pydantic示例，在LLM生成JSON字符串后，工具会尝试解析并验证它：

```python
import json

# 假设 'llm_generated_json_string' 是LLM的输出
# llm_generated_json_string = '{"username": "test_user", "email": "test@example.com", "score": 0.5}' # 用户名无效
# llm_generated_json_string = '{"username": "testuser", "email": "invalid-email", "score": 1.5}' # 电子邮件和分数无效

def parse_and_validate_profile(json_string: str) -> (Optional[UserProfile], Optional[str]):
    try:
        # Pydantic 可以直接使用 model_validate_json 解析 JSON 字符串
        profile = UserProfile.model_validate_json(json_string)
        return profile, None
    except ValueError as e:
        # 构建一个有用的错误消息。
        # 在实际工具中，此错误消息可以反馈给LLM进行另一次尝试。
        error_details = json.loads(e.json()) if hasattr(e, 'json') else str(e)
        return None, f"验证错误：{error_details}"

# 使用示例：
# profile_instance, error = parse_and_validate_profile(llm_generated_json_string)
# if error:
#     print(f"创建资料失败：{error}")
#     # 此处，工具可以带着错误消息重新提示LLM进行修正。
# else:
#     print("成功创建资料：")
#     print(profile_instance.model_dump_json(indent=2))
```

如果验证失败，工具可以实现一个优化循环：

1. 将验证器中的错误消息反馈给LLM。
2. 要求LLM根据错误修正其先前的输出。
3. 重新验证新输出。
   这种迭代过程可以显著提高生成有效结构化数据的成功率。

以下图表描绘了使用LLM生成结构化数据的常见工作流程：

> 此图表展示了一个代理请求结构化数据的情况。该工具使用LLM生成数据，然后验证输出。如果无效，则将反馈提供给LLM以进行优化。

### 特定格式的注意事项

虽然JSON应用广泛，但您的工具可能需要生成其他格式：

- **CSV (逗号分隔值)：** 对于表格数据，如果在给出清晰的标题和少量示例的情况下，LLM通常可以很好地生成CSV内容。验证可能涉及检查每行的列数和基本数据类型。
- **XML (可扩展标记 (token)语言)：** 由于其语法（标签、属性、嵌套），生成良好格式且有效的XML可能更复杂。通常需要提供一个模板或对预期XML结构的非常详细的描述。根据XSD或DTD进行验证很重要。
- **YAML (YAML不是标记语言)：** 其数据建模能力与JSON相似，但语法更具可读性。LLM可以生成YAML，但在提示和验证中需要仔细注意缩进和语法。

### 优点与设计考量

结构化数据生成工具有以下几个优点：

- **可靠性：** 它们增加了获得可被其他系统使用的正确格式数据的可能性。
- **互操作性：** 它们使LLM代理能够更顺畅地与现有API和数据处理流程集成。
- **减少后期处理：** 通过在生成阶段就正确处理结构，您可以减少后续复杂且可能脆弱的解析和校正逻辑的需求。

在设计此类工具时，请考虑：

- **模式的复杂性：** 从更简单的模式开始。非常复杂或深度嵌套的结构对于LLM一次性正确生成可能具有挑战性。
- **LLM指令的清晰度：** 提供给代理框架的工具本身的描述应清楚说明它生成哪种结构化数据（例如，“根据输入详细信息生成代表用户资料的JSON对象”），以及它需要从代理或用户那里获得哪些输入。
- **错误处理：** 在验证错误后，考虑如果LLM反复未能生成有效输出会发生什么。实施重试限制或备用机制。

通过整合协助结构化数据生成的工具，您可以使LLM代理执行更广泛的任务，这些任务依赖于精确、机器可读的数据格式，使它们更具通用性和效率。

## 参考资料

- [Pydantic Documentation](https://docs.pydantic.dev/) — The Pydantic Developers (2024)
  提供使用Python类型提示定义数据模式和验证的完整指南，直接支持内容中介绍的JSON模式定义和验证方法。
- [Function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2023)
  描述了如何使大型语言模型定义和调用外部工具或函数，这是LLM智能体需要结构化数据输出以与API和系统交互的基石。
- [Instructor: A Library for Structured Generation with LLMs](https://github.com/jxnl/instructor) — Jason Liu (2024)
  一个Python库，它扩展了OpenAI的函数调用功能，强制大型语言模型生成经过Pydantic验证的结构化输出，提供了对所讨论的结构化数据生成概念的实际实现。
- [Tool Learning with Large Language Models: A Survey](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGd2lce2FQVaKTUlsn1OGSnl3mfnr5c_ny7FSkmc8Lj9SCkoPjjmPxeC4-VicTUG7f8ot39AG93c04I84OaMEK3v0mu-Mpr-ZH7997zN_kIQCGw6-lXytai0XzrAIQ=) — Yujia Qin, Shihao Liang, Yikang Shen, Yuxuan Wang, Xin Cong, Yunsheng Jiang, Ruiying Geng, Ran Song, Jingfeng Yang, Longhui Zhang, Junteng Liu, Zhiding Liu, Chao Zhang, Qiang Yang, Huadong Wang, Kaijie Li, Zhiyuan Liu, Maosong Sun (2023)
  Journal: arXiv preprint arXiv:2309.17361
  一篇关于大型语言模型学习使用外部工具这一快速变化的领域的调查，为构建结构化数据生成工具作为一种LLM智能体工具提供了学术背景。

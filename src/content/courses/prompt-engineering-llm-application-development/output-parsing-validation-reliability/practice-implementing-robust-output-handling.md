---
course: "prompt-engineering-llm-application-development"
chapter: "output-parsing-validation-reliability"
lesson: "practice-implementing-robust-output-handling"
sourceId: 3086
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-7-output-parsing-validation-reliability/practice-implementing-robust-output-handling"
title: "实践：实现可靠的输出处理"
description: "动手练习为LLM应用添加输出解析、验证和错误处理功能。"
order: 8
plots: []
sourceHash: "e512cb249bbfba287961e81f7d2bc092ee2ac7b131fead6279908153dfa0f8b3"
sourceCorrections: []
---

LLM输出不一致带来了挑战，要求进行有效的解析、验证和错误处理。现在，您将编写一个函数，尝试使用LLM从文本中提取结构化数据，确保输出足够可靠以供应用程序使用。

想象一下，您有一个应用程序需要从用户提交的文本片段中提取联系信息（姓名和电子邮件）。目的是以JSON格式可靠地获取这些信息。

### 场景：提取联系信息

我们从一些示例文本开始：

```text
From: Sarah Chen <s.chen@example.com>
Subject: Project Update

Hi team,

Just wanted to share the latest updates...

Best,
Sarah
```

我们的目标输出格式是一个简单的JSON对象：

```json
{
  "name": "Sarah Chen",
  "email": "s.chen@example.com"
}
```

### 初次尝试：提示词 (prompt)编写和基本解析

首先，我们定义一个旨在获得所需JSON输出的提示词。基于前几章介绍的方法，我们将明确规定输出格式。

```python
def create_extraction_prompt(text_input):
  """创建用于将姓名和电子邮件提取为JSON的提示词。"""
  prompt = f"""
  从以下文本中提取全名和电子邮件地址。
  请严格按照以下JSON格式提供输出：
  {{"name": "...", "email": "..."}}

  如果未找到姓名或电子邮件，请返回值为空字符串的JSON：
  {{"name": "", "email": ""}}

  请勿在JSON结构之外包含任何解释或介绍性文本。

  文本：
  \"\"\"
  {text_input}
  \"\"\"

  JSON输出：
  """
  return prompt

# 假设 call_llm_api 是一个接受提示词的函数
# 并返回LLM的文本响应。
# def call_llm_api(prompt: str) -> str:
#   # 实际API调用逻辑的占位符
#   # 在真实场景中，这会与OpenAI、Anthropic等交互。
#   # 在本例中，让我们模拟一些可能的输出。
#   pass
```

现在，让我们尝试处理响应。一种简单的方法可能仅仅假设LLM返回完美的JSON。

```python
import json

# 实际文本输入的占位符
input_text = """
From: Sarah Chen <s.chen@example.com>
Subject: Project Update
...
"""

# 模拟一个成功的LLM响应
simulated_good_response = '{"name": "Sarah Chen", "email": "s.chen@example.com"}'

# 尝试解析
try:
  prompt = create_extraction_prompt(input_text)
  # raw_output = call_llm_api(prompt) # 实际调用会在这里
  raw_output = simulated_good_response # 使用模拟输出
  extracted_data = json.loads(raw_output)
  print(f"成功提取：{extracted_data}")
except json.JSONDecodeError:
  print("错误：无法从LLM响应中解码JSON。")
except Exception as e:
  print(f"发生了一个意外错误：{e}")
```

如果LLM表现完美，这就能奏效。但如果响应略有偏差呢？

```python
# 模拟一个格式错误的响应（例如，末尾逗号、缺少引号）
simulated_bad_response = '{"name": "Sarah Chen", "email": "s.chen@example.com",}'

try:
  # ... (提示词创建)
  raw_output = simulated_bad_response # 使用模拟的错误输出
  extracted_data = json.loads(raw_output)
  print(f"成功提取：{extracted_data}")
except json.JSONDecodeError as e:
  print(f"错误：无法解码JSON。详情：{e}") # 捕获错误
except Exception as e:
  print(f"发生了一个意外错误：{e}")
```

`json.loads`调用会引发`JSONDecodeError`，我们的基本`try...except`块会捕获它。这可以防止应用程序崩溃，但我们仍然无法获取数据。

### 使用Pydantic添加数据验证

即使JSON在语法上有效，它可能也不具备我们期望的*结构*或*数据类型*。我们引入Pydantic来进行模式验证。

首先，如果您尚未安装Pydantic，请执行以下操作：
`pip install pydantic`

现在，定义一个代表我们所需结构的Pydantic模型：

```python
from pydantic import BaseModel, EmailStr, ValidationError

class ContactInfo(BaseModel):
  name: str
  email: EmailStr # Pydantic 会验证这是否是有效的电子邮件格式
```

我们将其集成到处理逻辑中：

```python
import json
from pydantic import BaseModel, EmailStr, ValidationError

# --- Pydantic 模型 ---
class ContactInfo(BaseModel):
  name: str
  email: EmailStr

# --- LLM调用的占位符 ---
def call_llm_api(prompt: str) -> str:
  # 模拟不同的输出以进行演示
  # return '{"name": "Sarah Chen", "email": "s.chen@example.com"}' # 好的
  # return '{"name": "Sarah Chen", "email": "not-an-email"}' # 电子邮件格式错误
  return '{"contact_name": "Sarah Chen", "address": "s.chen@example.com"}' # 字段名称错误

# --- 处理逻辑 ---
input_text = "From: Sarah Chen <s.chen@example.com> ..." # 示例文本
prompt = create_extraction_prompt(input_text) # 之前已定义

try:
  raw_output = call_llm_api(prompt)
  print(f"原始LLM输出：\n{raw_output}")

  # 尝试直接使用Pydantic解析和验证
  contact_info = ContactInfo.model_validate_json(raw_output)
  print(f"\n成功验证数据：{contact_info.model_dump()}")

except json.JSONDecodeError as e:
  print(f"\n错误：无法解码JSON。详情：{e}")
  print("LLM输出可能不是有效的JSON。")
except ValidationError as e:
  print(f"\n错误：数据验证失败。详情：\n{e}")
  print("LLM输出是JSON，但与所需模式不匹配（例如，字段名称错误、电子邮件格式无效）。")
except Exception as e:
  print(f"\n发生了一个意外错误：{e}")
```

使用不同的模拟`call_llm_api`返回值运行此代码。您会发现`model_validate_json`可以处理JSON解码错误和模式验证错误，并在验证失败时提供详细信息（例如，哪个字段错误，为什么电子邮件无效）。

### 实现重试机制

有时，LLM可能会因为暂时性问题而失败，或者在第一次尝试时生成略微不正确的输出。一个简单的重试机制通常可以解决这些间歇性问题。

我们把API调用和处理逻辑封装在一个重试循环中：

```python
import json
import time
from pydantic import BaseModel, EmailStr, ValidationError

# --- Pydantic 模型 ---
class ContactInfo(BaseModel):
  name: str
  email: EmailStr

# --- LLM调用占位符（修改为有时会失败） ---
import random
_call_count = 0
def call_llm_api_flaky(prompt: str) -> str:
    global _call_count
    _call_count += 1
    print(f"LLM API 调用尝试 #{_call_count}")
    # 模拟第一次失败，第二次成功
    if _call_count == 1 and random.random() < 0.7: # 初始失败率为70%
        # return '{"name": "Sarah Chen", "email": "s.chen@example.com",}' # 格式错误的JSON
        return '{"name": "Sarah Chen"}' # 缺少字段
    else:
        return '{"name": "Sarah Chen", "email": "s.chen@example.com"}' # 好的响应

# --- 带重试的处理函数 ---
def extract_contact_info_robust(text_input: str, max_retries: int = 2) -> ContactInfo | None:
  global _call_count
  _call_count = 0 # 为每次新的提取尝试重置计数器
  prompt = create_extraction_prompt(text_input)

  for attempt in range(max_retries):
    print(f"\n--- 第 {attempt + 1} 次尝试，共 {max_retries} 次 ---")
    try:
      raw_output = call_llm_api_flaky(prompt)
      print(f"原始LLM输出：{raw_output}")

      contact_info = ContactInfo.model_validate_json(raw_output)
      print("验证成功！")
      return contact_info # 成功！退出循环并返回数据

    except (json.JSONDecodeError, ValidationError) as e:
      print(f"尝试失败：{e.__class__.__name__}")
      if attempt < max_retries - 1:
        print("正在重试...")
        # 可选：重试前添加少量延迟
        # time.sleep(0.5)
      else:
        print("已达到最大重试次数。无法提取有效数据。")
        # 记录最终错误和有问题的输出，以便调试
        # logger.error(f"尝试 {max_retries} 次后失败。最后输出：{raw_output}", exc_info=True)
        return None # 表示失败

    except Exception as e:
      # 处理意外错误（例如，如果是实际API调用，则为网络问题）
      print(f"发生了一个意外错误：{e}")
      # 认真记录此错误
      # logger.exception("提取过程中发生意外错误")
      return None # 表示失败

  return None # 如果循环逻辑正确，这里应该无法到达，但以防万一

# --- 示例用法 ---
input_text = "From: Sarah Chen <s.chen@example.com> ..."
result = extract_contact_info_robust(input_text)

if result:
  print(f"\n最终提取数据：{result.model_dump_json(indent=2)}")
else:
  print("\n无法可靠地提取联系信息。")
```

在此版本中：

1. `call_llm_api_flaky`函数模拟了不可靠的行为。
2. `extract_contact_info_robust`函数会循环最多`max_retries`次。
3. 它捕获`JSONDecodeError`和`ValidationError`。
4. 如果尝试失败但仍有重试次数，它会打印一条消息并继续循环。
5. 如果所有重试次数都用尽，它会记录失败（在生产环境中您会使用适当的日志库）并返回`None`。
6. 它还包含一个捕获所有`Exception`的块，以应对真正意外的问题。

### 考虑备用方案和后续步骤

如果`extract_contact_info_robust`返回`None`怎么办？您的应用程序需要一个方案：

1. **默认值：** 使用空字符串或预定义默认值。
2. **日志记录和警报：** 记录失败和输入文本以供后续审查。如果失败次数超过阈值，向开发人员发出警报。
3. **人工审查队列：** 将有问题的文本片段路由给人工进行手动提取。
4. **更简单的方法：** 尝试一种复杂度较低的提取方法（例如，正则表达式）作为备用方案，尽管它可能不那么准确。

此外，您可以在将文本发送到LLM进行提取之前，或者在接收到响应之后，集成**内容审查API**（如本章前面讨论的），为*内容*本身增加一层安全保障。

本次练习演示了一种使LLM交互更具恢复力的实用工作流程。通过结合精心设计的提示词 (prompt)、解析、验证和错误处理（如重试和备用方案），您可以显著提高基于大型语言模型的应用程序的可靠性。请记住，这通常是一个迭代过程；监控生产中的失败将指导您进一步完善提示词和处理逻辑。

## 参考资料

- [Function calling](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGCZBS2tvNweDRJ6SEvSFiSNowOfPAax_yxho-Sfm99FYPNZ1FSZSVUoawB0tmUnqHUvmP1fvWlsWOJt00GOEB_Mjxx06lkt0ig0_1TMe0wofsleqJ9eHGGdGeczwYw6bvus-PAdmBjfk8BZzjhEhhp76P4RsCbHjmp4SemO1EgPPEmqeG4tXIR) — OpenAI (2024)
  Publisher: OpenAI
  展示了一种使用预定义模式从大型语言模型（LLM）中获取结构化JSON输出的方法，适用于健壮的解析和验证。
- [json - JSON encoder and decoder](https://docs.python.org/3/library/json.html) — Python Software Foundation (2024)
  Python标准库中用于解析和序列化JSON数据的模块官方文档，对于处理LLM输出至关重要。
- [Pydantic Documentation](https://docs.pydantic.dev/latest/) — Pydantic Developers (2024)
  Pydantic的官方指南，这是一个用于数据验证和使用类型提示进行设置管理的Python库，对于确保模式和数据类型正确性非常重要。
- [Site Reliability Engineering: How Google Runs Production Systems](https://sre.google/sre-book/table-of-contents/) — Betsy Beyer, Chris Jones, Jennifer Petoff, and Niall Richard Murphy (2016)
  Publisher: O'Reilly Media, Inc.
  关于构建和维护高可靠性软件系统的基础文献，涵盖错误处理、重试机制和系统弹性策略，适用于大型语言模型（LLM）应用程序开发。

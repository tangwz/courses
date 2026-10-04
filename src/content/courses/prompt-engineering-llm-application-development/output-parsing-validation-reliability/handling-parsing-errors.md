---
course: "prompt-engineering-llm-application-development"
chapter: "output-parsing-validation-reliability"
lesson: "handling-parsing-errors"
sourceId: 3081
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-7-output-parsing-validation-reliability/handling-parsing-errors"
title: "处理解析错误"
description: "应对LLM输出无法正确解析的情况的策略。"
order: 5
plots: []
sourceHash: "9fb72c9dc5178a5f5b070f10f74380c305515fb6870140bc0e516e0311ff6da2"
sourceCorrections: []
---

构建可靠的大型语言模型（LLM）应用需要谨慎处理输出，因为LLM并不总是能生成完全符合期望结构的内容。即使通过精心的提示词 (prompt)工程和使用输出解析器——这些都是引导LLM响应的常用技术——错误仍然可能出现。网络问题、模型暂时性故障，或者仅仅是生成过程的概率性质，都可能导致响应无法根据您定义的模式（如 JSON 或 Pydantic 模型）正确解析。优雅地处理这些解析错误对于构建可靠的应用非常重要。忽视它们可能导致应用崩溃、数据处理不当或用户体验不佳。

当LLM的输出不符合预期格式时，您的解析代码（无论是 `json.loads()`、框架的解析器，还是Pydantic模型的验证）很可能会抛出异常。您需要一个应对故障的策略，而不是任由它导致应用崩溃。

### 解析失败的常见原因

了解解析可能失败的原因有助于选择正确的恢复方法：

1. **格式不正确的结构：** LLM可能生成不完整或语法错误的结构。例如，JSON中缺少闭合括号、引用不正确，或主体结构外有额外文本。
2. **数据类型不正确：** 预期为整数的字段可能包含字符串，或者布尔值可能表示为“True”而不是 `true`。
3. **意外内容：** LLM可能包含解释性文本、抱歉（“我无法提供该信息...”），或拒绝回答，这些都不符合预期的数据模式。
4. **生成不完整：** 输出可能因为在结构完全生成之前达到最大token限制（`max_tokens`）而被截断。

### 处理解析错误的策略

当解析尝试失败时，请考虑以下方法：

1. **重试请求（简单重试）：** 有时，失败是暂时的。简单地重试*完全相同的API调用*，第二次尝试可能就会得到正确的响应。这通常是第一步也是最简单的步骤。将此与短暂延迟（退避）相结合，以避免API过载，特别是当错误是由于速率限制引起时。这在“实现重试机制”一节中有更详细的讨论，但它在这里也是一种基本方法。
2. **使用纠正性提示重试：** 如果简单重试失败，问题可能出在LLM对格式指令的理解或遵循上。您可以修改重试时的提示词 (prompt)。策略包括：

   - **明确的错误反馈：** 在*下一次*提示中包含失败的输出和解析错误信息，要求LLM纠正其之前的尝试。例如：“您之前的响应 `{previous_llm_output}` 解析失败，错误为：`{error_message}`。请再次提供响应，严格遵循请求的JSON格式：{schema description}。”
   - **简化请求：** 如果任务复杂，请尝试将其分解或先请求一个更简单的输出版本。
   - **再次强调格式说明：** 在提示中添加更强的说明或示例，说明所需的格式。
3. **备用机制：** 如果重试（无论是否修改提示）持续失败，您需要一个备用方案：

   - **默认值：** 返回一个安全的默认值或结构。如果部分或默认状态可接受以使应用逻辑继续运行，则此方法适用。
   - **通知用户：** 通知用户请求未能完全处理，并可能提供替代方案或请求澄清。
   - **记录日志并警报：** 记录错误详情（提示、失败输出、错误信息）供后续分析。如果失败次数超过某个阈值，向开发者发送警报。
   - **人工审查队列：** 对于关键任务，将失败的交互标记 (token)出来以供人工审查和修正。

### 实现示例（Python）

我们来用Python的 `try-except` 块演示一个基本流程，假设您正在尝试解析 JSON 并可能使用 Pydantic 这样的验证库。

````python
import json
from pydantic import BaseModel, ValidationError
import time
import random

# 假设 'llm_api_call' 是一个函数，接受提示词并返回 LLM 的文本响应
# 假设 'YourDataModel' 是一个定义预期结构的 Pydantic 模型

MAX_RETRIES = 3
INITIAL_BACKOFF = 1 # seconds

def process_llm_request_with_parsing(prompt: str):
    """尝试获取和解析 LLM 输出，包含重试和错误处理。"""

    current_prompt = prompt
    for attempt in range(MAX_RETRIES):
        try:
            raw_output = llm_api_call(current_prompt)

            # 尝试 1：直接 JSON 解析
            try:
                parsed_data = json.loads(raw_output)
                # 可选：使用 Pydantic 验证
                # validated_data = YourDataModel(**parsed_data)
                # print("Successfully parsed and validated.")
                # return validated_data
                print("成功解析 JSON。")
                return parsed_data # 如果不需要验证，返回原始解析数据

            except json.JSONDecodeError as e:
                print(f"尝试 {attempt + 1}：JSON 解析失败：{e}")
                error_message = str(e)
                failed_output = raw_output # 保留以备纠正性提示

            # except ValidationError as e: # 如果使用 Pydantic
            #     print(f"尝试 {attempt + 1}：Pydantic 验证失败：{e}")
            #     error_message = str(e)
            #     failed_output = raw_output

            # 如果解析/验证失败，准备重试
            if attempt < MAX_RETRIES - 1:
                # 策略：使用纠正性提示重试（简化示例）
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response failed parsing: `{error_message}`.\n"
                    f"Previous response snippet: ```{failed_output[:200]}...```\n"
                    f"请严格以正确的 JSON 格式提供响应。"
                )
                # 添加带抖动的指数退避
                backoff_time = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                print(f"{backoff_time:.2f} 秒后重试...")
                time.sleep(backoff_time)
            else:
                print("达到最大重试次数。解析失败。")
                # 在此处实现备用机制
                log_error(prompt, failed_output, error_message)
                return None # 或者抛出自定义异常，返回默认值等。

        except Exception as api_error: # 捕获潜在的 API 调用错误
            print(f"第 {attempt + 1} 次尝试时 API 调用失败：{api_error}")
            if attempt < MAX_RETRIES - 1:
                backoff_time = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(backoff_time)
            else:
                print("达到最大重试次数后 API 调用失败。")
                log_error(prompt, None, str(api_error))
                return None

    return None # 如果逻辑正确，这里理论上不应被执行

def log_error(prompt, failed_output, error_message):
    # 实际日志实现（例如，写入文件，发送到日志服务）的占位符
    print(f"记录错误：\n提示词：{prompt}\n失败输出：{failed_output}\n错误：{error_message}")

# 示例用法：
my_prompt = "Extract the name and age from the text 'John Doe is 30 years old' as JSON."
# 从文本 'John Doe is 30 years old' 中提取姓名和年龄，以 JSON 格式。
result = process_llm_request_with_parsing(my_prompt)

if result:
    print("最终结果：", result)
else:
    print("未能获得有效结果。")
````

### 可视化处理流程

我们可以用图表表示这个决策过程：

> 流程图说明了处理潜在LLM输出解析错误的过程，包括重试尝试和备用操作。

选择重试、纠正性提示和备用方案的正确组合取决于具体的应用需求、对失败的容忍度以及预期LLM输出的性质。错误处理不是事后才考虑的事情；它是构建可靠LLM驱动应用的核心组成部分。

## 参考资料

- [Release It! Design and Deploy Production-Ready Software](https://books.google.com/books/about/Release_It_Design_and_Deploy_Production.html?id=l643DwAAQBAJ) — Michael T. Nygard (2018)
  Publisher: Pragmatic Bookshelf; Pages: 376
  一本关于设计稳健和弹性软件系统的经典书籍，涵盖了重试、指数退避和熔断等模式，这些对于处理LLM集成中的瞬时故障至关重要。
- [Pydantic Documentation](https://docs.pydantic.dev/) — Samuel Colvin and Pydantic Contributors (2024)
  Pydantic的官方文档，Pydantic是一个在Python中广泛使用的数据验证和解析库，是本节讨论的验证LLM结构化输出的核心工具。

---
course: "building-advanced-llm-agent-tools"
chapter: "developing-custom-python-tools"
lesson: "tool-input-validation-sanitization"
sourceId: 6561
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-2-developing-custom-python-tools/tool-input-validation-sanitization"
title: "验证和净化工具输入"
description: "在你的工具中实施输入验证和净化，以确保数据完整性并防止常见漏洞。"
order: 4
plots: []
sourceHash: "ac90016b99089ddc1c898a9d5a367e3034a19fd34c3c8cd1bd3fd7a4f89df05c"
sourceCorrections: []
---

当一个LLM智能体决定使用你的某个自定义Python工具时，它将尝试根据工具的描述和对任务的理解来提供所需的输入。然而，LLM，特别是在为工具输入生成JSON等结构化数据时，有时会产生意料之外、格式不正确或甚至潜在不安全的值。因此，在对工具收到的任何输入进行操作之前，对其进行彻底的验证和净化，是构建可靠和安全智能体系统的基本组成部分。这不仅仅是为了防止崩溃；更是为了保证行为可预测并保护你的系统。

你可以把输入验证和净化看作是数据进入你的工具的两个不同但相辅相成的安全检查点。

- **验证**确认输入数据符合工具的预期：类型是否正确？是否在可接受的范围或长度内？是否遵循特定格式？
- **净化**修改输入数据，以移除或中和潜在有害元素，使其可以安全地由你的工具或与之交互的任何下游系统（如数据库或外部API）进行处理。

下图说明了验证和净化在工具调用流程中的位置：

> LLM工具内部输入处理流程，着重说明验证和净化步骤。

### 输入验证的策略

有效的输入验证是第一道防线。如果输入不符合基本要求，通常就没有必要进行净化或执行。

#### 基本检查：类型、范围和长度

在最基本的层面，Python的内置功能可以提供帮助。

- **类型检查**：使用`isinstance()`来确保输入是预期的Python类型（例如，`str`、`int`、`list`）。

  ```python
  def process_item_id(item_id: int):
      if not isinstance(item_id, int):
          raise TypeError("项目ID必须是整数。")
      # ... 进一步处理
  ```
- **范围检查**：对于数值输入，验证它们是否在可接受的范围内。

  ```python
  def set_brightness(level: int):
      if not (0 <= level <= 100):
          raise ValueError("亮度级别必须在0到100之间。")
      # ... 设置亮度
  ```
- **长度检查**：对于字符串或集合，确保它们符合最小或最大长度要求。

  ```python
  def submit_comment(text: str):
      if not (10 <= len(text) <= 1000):
          raise ValueError("评论长度必须在10到1000个字符之间。")
      # ... 提交评论
  ```

#### 使用正则表达式进行模式匹配

对于必须遵循特定格式（例如，电子邮件地址、特定ID模式、日期）的输入，正则表达式是一种强效工具。Python的`re`模块提供了所需功能。

```python
import re

def validate_product_code(code: str):
    pattern = r"^[A-Z]{3}-\d{5}$" # 例如, ABC-12345
    if not re.match(pattern, code):
        raise ValueError("产品代码格式无效。预期格式：XXX-NNNNN")
    return True
```

尽管灵活，复杂的正则表达式可能变得难以维护。对清晰、明确的模式谨慎使用它们。

#### 采用数据验证库：Pydantic

对于需要结构化输入，特别是类JSON对象（这在LLM工具参数 (parameter)中很常见）的工具，Pydantic等库可以大大简化验证。Pydantic使用Python类型提示来定义数据模式，并执行验证、数据解析和错误报告。

考虑一个需要搜索查询和可选的最大结果数量的工具：

```python
from pydantic import BaseModel, Field, validator, ValidationError
from typing import Optional

class SearchToolInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=200, description="搜索查询字符串。")
    max_results: Optional[int] = Field(default=10, gt=0, le=50, description="要返回的最大结果数量。")

    @validator('query')
    def query_must_be_alphanumeric_or_spaces(cls, v):
        # 允许字母数字字符和空格的自定义验证器
        if not re.match(r"^[a-zA-Z0-9\s]+$", v):
            raise ValueError('查询只能包含字母数字字符和空格。')
        return v.strip() # 这也是一种轻量级的净化：去除空白字符

# 在你的工具中的示例用法
def search_documents(raw_input: dict):
    try:
        validated_input = SearchToolInput(**raw_input)
        # 现在使用 validated_input.query 和 validated_input.max_results
        print(f"正在搜索：'{validated_input.query}'，最大结果数量：{validated_input.max_results}")
        # ... 实际搜索逻辑 ...
        return {"status": "success", "results": []} # 占位符
    except ValidationError as e:
        # Pydantic的错误消息非常有用
        return {"status": "error", "message": f"输入验证失败：{e.errors()}"}

# 模拟LLM输入
llm_provided_input_valid = {"query": "  Python best practices  ", "max_results": 5}
llm_provided_input_invalid_query = {"query": "Python!", "max_results": 5} # 包含 '!'
llm_provided_input_invalid_max_results = {"query": "Data Science", "max_results": 100} # > 50

print(search_documents(llm_provided_input_valid))
print(search_documents(llm_provided_input_invalid_query))
print(search_documents(llm_provided_input_invalid_max_results))
```

使用Pydantic有几项优势：

- **声明式**：模式使用类型提示清晰定义。
- **提供信息的错误**：验证错误信息详细，明确指出哪个字段失败以及原因。这对于调试和LLM可能进行的自我修正很有帮助。
- **数据强制转换**：Pydantic通常可以将输入强制转换为正确类型（例如，将字符串“5”转换为整数5），但这应在理解其影响的情况下使用。

### 输入净化的方法

净化是关于转换输入以确保其安全，特别是当该输入可能用于敏感环境，如数据库查询、shell命令或HTML输出时。核心原则是将所有来自外部源（包括LLM）的输入视为不可信。

#### 最小权限原则

你的工具应以最低限度的必要权限运行。如果工具写入文件，请确保它只能写入预期的安全位置。如果它查询数据库，请使用具有受限权限的数据库用户（例如，如果只需要对特定表的只读访问权限）。

#### 常见净化场景和方法

- **SQL注入**：如果你的工具使用输入数据构建SQL查询，它将非常脆弱。

  - **最佳防御**：始终使用数据库库提供的参数 (parameter)化查询（预处理语句）。输入与SQL命令分开发送，并由数据库驱动程序安全处理。
  - **避免字符串拼接**：绝不要通过将用户输入直接拼接到SQL字符串来构建SQL查询。

  ```python
  # 错误：易受SQL注入攻击
  # cursor.execute(f"SELECT * FROM users WHERE username = '{username}'")

  # 正确：使用参数化查询（sqlite3示例）
  # import sqlite3
  # conn = sqlite3.connect(':memory:')
  # cursor = conn.cursor()
  # username = "admin" # 可能来自LLM
  # cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
  ```
- **命令注入**：如果你的工具需要执行shell命令，直接将用户输入插入到命令字符串中是非常危险的。

  - **最佳防御**：如果可能，避免使用用户输入调用shell命令。如果必要，使用`subprocess.run()`等函数，将命令和参数作为列表而不是单个字符串传递。
  - **转义**：像`shlex`这样的库提供了函数（例如`shlex.quote()`）来转义用于shell的字符串中的字符，但这应该是对谨慎命令构建的次要防御。

  ```python
  import subprocess
  import shlex

  def list_directory_contents(user_path: str):
      # 验证路径（例如，确保它在允许的基本目录内）
      # 为简化起见，此示例省略了复杂的路径验证。
      # 假设 user_path 类似于 "my_safe_subdir/data"

      # 如果构建字符串（不太安全），则对shell参数进行净化
      # safe_path_arg = shlex.quote(user_path)
      # command_string = f"ls -l {safe_path_arg}"
      # result = subprocess.run(command_string, shell=True, capture_output=True, text=True)

      # 更好：将参数作为列表传递（更安全）
      try:
          # 理想情况下，此处应验证 user_path 以防止 '..' 或绝对路径
          # 对于此示例，我们假设已发生基本验证。
          # 如果 user_path 包含可能对 ls 命令本身有害的元素，则需要更多检查。
          # 示例：user_path = "."
          # 示例：user_path = "some_dir"
          # 避免 user_path = "-la" 或类似可能被解释为选项的值。
          # 对路径组件采用严格的允许列表甚至更好。

          if not re.match(r"^[a-zA-Z0-9_/\.-]+$", user_path) or ".." in user_path:
               return {"status": "error", "message": "无效路径字符或路径遍历尝试。"}

          result = subprocess.run(["ls", "-l", user_path], capture_output=True, text=True, check=True)
          return {"status": "success", "output": result.stdout}
      except subprocess.CalledProcessError as e:
          return {"status": "error", "message": f"命令失败：{e.stderr}"}
      except FileNotFoundError:
          return {"status": "error", "message": "ls 命令未找到或路径无效。"}

  # print(list_directory_contents(".")) # 安全
  # print(list_directory_contents("nonexistent_dir; rm -rf /")) # 验证应该捕获这一点。
                                                            # 如果没有，shlex.quote 或基于列表的参数会阻止 ; rm -rf
  ```
- **HTML/脚本注入（跨站脚本攻击 - XSS）**：如果你的工具输出可能在网页浏览器中渲染，并且该输出包含LLM生成的输入，你必须净化它以防止XSS攻击。

  - **使用库**：像`bleach`这样的库旨在清理HTML，只允许安全的标签和属性子集。

  ```python
  import bleach

  def format_comment_for_html(user_comment: str):
      # 只允许粗体和斜体标签，并转义所有其他HTML
      allowed_tags = ['b', 'i']
      sanitized_comment = bleach.linkify(bleach.clean(user_comment, tags=allowed_tags, strip=True))
      return f"<p>{sanitized_comment}</p>"

  # unsafe_comment = "This is <b>great</b>! <script>alert('XSS')</scrip>"
  # print(format_comment_for_html(unsafe_comment))
  # 输出：<p>This is <b>great</b>! &lt;script&gt;alert('XSS')&lt;/script&gt;</p>
  ```

#### 允许列表优于拒绝列表

在可能的情况下，定义 *允许* 的内容（允许列表），而不是试图列出所有 *不* 允许的内容（拒绝列表）。拒绝列表众所周知难以正确实现，并且随着新攻击载体的发现而经常被规避。例如，对于需要国家代码的参数，请根据已知有效国家代码列表进行验证。

### 处理验证失败

当验证或净化规则被违反时，你的工具不应仅仅崩溃或继续处理受污染的数据。它需要：

1. **明确拒绝输入。**
2. **返回提供信息的错误消息。** 这对LLM智能体尤为重要。一个好的错误消息可以帮助智能体理解哪里出了问题，并可能在后续尝试中修正其输入。

例如，如果Pydantic验证失败，其`ValidationError`包含关于错误的结构化信息。你可以将其格式化为LLM可以解析或理解的字符串。

```python
# （Pydantic SearchToolInput 示例的延续）
# ...
# except ValidationError as e:
#     error_details = e.errors() # 这是一个字典列表
#     # 为LLM构建一个用户友好的消息
#     messages = []
#     for error in error_details:
#         field = " -> ".join(map(str, error['loc'])) # loc 可以是嵌套模型的路径
#         msg = error['msg']
#         messages.append(f"字段 '{field}'：{msg}")
#     error_summary = "输入验证失败。 " + "; ".join(messages)
#     return {"status": "error", "message": error_summary, "details": error_details}

# llm_provided_input_very_wrong = {"query": "Q", "max_results": 200, "extra_field": "test"}
# result = search_documents(llm_provided_input_very_wrong)
# print(json.dumps(result, indent=2))
# 预期输出可能如下所示：
# {
#   "status": "error",
#   "message": "Input validation failed. Field 'query': ensure this value has at least 3 characters; Field 'max_results': ensure this value is less than or equal to 50; Field 'extra_field': extra fields not permitted",
#   "details": [
#     { /* Pydantic 错误详情 */ }
#   ]
# }
```

目标是为LLM（或监控智能体的开发者）提供足够的信息，以便更好地理解输入要求。

通过认真应用输入验证和净化，你为你的Python工具构建了信任和可靠的基础。这不仅能防止错误和安全漏洞，还有助于LLM智能体与其扩展功能之间更可预测和有效的交互。请记住，任何来自你直接控制范围之外的输入，包括LLM生成的输入，都需要仔细检查。

## 参考资料

- [OWASP Top 10](https://owasp.org/www-project-top-10/) — OWASP Foundation (2021)
  一份关于最重要的Web应用程序安全风险的指南，提供理解和缓解漏洞（包括注入缺陷和跨站脚本）的框架。
- [Pydantic Documentation](https://docs.pydantic.dev/) — Samuel Colvin (2024)
  Pydantic的官方文档，这是一个数据验证和设置管理库，对Python中结构化输入的定义和验证至关重要。

# 数据验证技术（例如 Pydantic）

来源：[原文](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-7-output-parsing-validation-reliability/data-validation-techniques)

[返回章节目录](README.md) · [返回课程目录](../README.md)

成功解析大型语言模型（LLM）的输出，例如从其响应字符串中提取 JSON 对象，是一个重要步骤。然而，解析仅确认输出*看起来*符合预期格式；它不保证*内容*正确或可用。LLM 可能会返回一个 JSON 对象，但却省略了必需字段，使用了错误的数据类型（例如，在预期数字的地方返回字符串），或者提供了在您的应用约束下没有意义的值。这就是数据验证变得必不可少的地方。

数据验证在解析后充当质量关卡。它检查实际数据是否符合预定义的模式或一组规则，确保其完整性，然后才能在您的应用中进一步使用。实施验证有助于防止意外的运行时错误，维持数据一致性，并最终有助于构建更可靠的应用。

### 在 Pydantic 中使用 Python 类型提示

对于使用 Python 构建的应用，Pydantic 库提供了一种优雅而强大的数据验证方式。Pydantic 使用 Python 的类型提示（在 PEP 484 中引入）来定义数据模式。您可以使用标准 Python 类和类型注解声明预期数据的结构和类型，Pydantic 会根据这些定义处理解析和验证。

我们来看一个例子。假设您已提示 LLM 提取用户信息并以 JSON 格式返回，预期结构如下：

```json
{
  "name": "Alice",
  "user_id": 12345,
  "is_active": true,
  "email": "alice@example.com"
}
```

您可以定义一个 Pydantic 模型来表示这个结构：

```python
from pydantic import BaseModel, EmailStr, ValidationError
from typing import Optional

class UserProfile(BaseModel):
    name: str
    user_id: int
    is_active: bool
    email: EmailStr # Pydantic 提供了像 EmailStr 这样的特定类型
    location: Optional[str] = None # 一个可选字段，带有默认值
```

在这个 `UserProfile` 模型中：

- 每个字段都定义为带有类型提示 (`str`, `int`, `bool`) 的属性。
- Pydantic 会自动强制执行这些类型。如果 LLM 返回 `user_id` 为 `"12345"`（字符串），Pydantic 会尝试将其强制转换为整数。如果失败，或者类型根本不兼容（例如为 `user_id` 提供 `true`），它会引发验证错误。
- 我们为 `email` 字段使用 `EmailStr`。这是 Pydantic 提供的一种特殊类型，用于验证字符串是否符合标准电子邮件格式。
- `location` 字段被标记 (token)为 `Optional[str]`，这意味着它在输入数据中不是必需的。我们还为其分配了默认值 `None`。

现在，假设您已收到来自 LLM 的响应，并将其解析为 Python 字典 `parsed_data`。您可以像这样对照模型验证此数据：

```python
# 假设 llm_response 是 LLM 返回的原始字符串
# 假设 parse_llm_json(llm_response) 将其解析为字典
# parsed_data = parse_llm_json(llm_response)

parsed_data = {
  "name": "Bob",
  "user_id": "67890", # 注意：是字符串而非整数
  "is_active": "yes", # 注意：是字符串而非布尔值
  "email": "bob@domain" # 注意：可能是无效的电子邮件格式
  # 'location' 字段缺失，这是可以的，因为它是可选的
}

try:
    # 尝试验证数据
    user_profile = UserProfile.model_validate(parsed_data)
    
    # 如果验证成功，user_profile 将是 UserProfile 的一个实例
    # 且类型已强制转换（例如，user_id 现在是整数）
    print("验证成功！")
    print(f"用户 ID: {user_profile.user_id} (类型: {type(user_profile.user_id)})")
    print(f"是否活跃: {user_profile.is_active} (类型: {type(user_profile.is_active)})")
    # 在应用中使用已验证的数据
    # ... process_user(user_profile) ...

except ValidationError as e:
    # 如果验证失败，Pydantic 会引发 ValidationError
    print("验证失败！")
    print(e.json()) # 获取详细的 JSON 格式错误信息
    
    # 在此处实现错误处理逻辑：
    # - 记录错误
    # - 请求 LLM 使用修正后的指令重试
    # - 回退到默认值
    # - 通知用户或管理员
```

如果 `parsed_data` 包含 `"user_id": "67890"` 和 `"is_active": "yes"`，Pydantic 的默认行为可能会成功地将它们强制转换为整数 `67890` 和布尔值 `True`。然而，如果 `email` 是 `"bob@domain"`，`EmailStr` 验证很可能会失败。如果 `user_id` 是 `"not-a-number"`，整数转换会失败。在这些失败情况下，会引发 `ValidationError`。

### 处理验证错误

`ValidationError` 异常包含有关问题发生原因的详细信息。您可以检查其 `errors()` 方法或 `json()` 表示形式，以了解哪些字段验证失败以及原因。

`e.json()` 对于无效电子邮件的示例输出：

```json
[
  {
    "type": "值错误",
    "loc": ["电子邮件"],
    "msg": "值不是有效的电子邮件地址",
    "input": "bob@domain",
    "ctx": {"reason": "电子邮件地址无效。"}
  }
]
```

这精确地告诉您，`email` 字段失败是因为输入 `"bob@domain"` 未被识别为有效的电子邮件地址。有了这些信息，您的应用可以决定下一步行动：

1. **重试：** 重新向 LLM 提示，可能强调之前遗漏的特定格式要求。
2. **回退：** 如果合适，对无效字段使用默认值。
3. **通知：** 记录错误以供监控，或告知用户提供的信息无效。
4. **请求澄清：** 如果与用户交互，请他们提供正确的信息。

### 应用流程中的验证

数据验证应在您将 LLM 的原始输出解析为初步结构（例如字典）后立即进行。

> 该流程图显示了数据验证在解析 LLM 输出后、应用逻辑使用数据之前的位置。

Pydantic 提供了更多高级功能，例如自定义验证器（用于强制执行基本类型的规则，例如检查数字是否落在特定范围内）、字段别名（JSON 字段名和 Python 属性名之间的映射），以及可辨识联合（用于处理可以有多种结构的对象）。虽然讨论这些超出本节范围，但理解定义模式并对照模式验证数据的核心思想对于构建稳定的应用非常重要。

虽然 Pydantic 是一个流行且便捷的选择，但其他库如 Marshmallow 或 Cerberus 也存在，并且您始终可以实现自定义验证函数。原则保持不变：定义您预期的数据结构和规则，并在继续之前严格检查 LLM 的输出是否符合这些规则。这种结构化验证是从实验性 LLM 交互转向可靠的、可用于生产的应用的重要方法。

## 参考资料

- [Pydantic Documentation](https://docs.pydantic.dev/) — Pydantic Developers (2024)
  提供 Pydantic 模型定义、数据验证和错误处理的指导。
- [PEP 484 -- Type Hints](https://peps.python.org/pep-0484/) — Guido van Rossum, Jukka Lehtosalo, Łukasz Langa (2014)
  Journal: Python Enhancement Proposals
  定义了 Python 中类型提示的标准，是 Pydantic 数据模式定义的基础。
- [Fluent Python: Clear, Concise, and Effective Programming](https://play.google.com/store/books/details/Fluent_Python_Clear_Concise_and_Effective_Program?id=L8YZEAAAQBAJ) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  讨论了 Python 特性，例如类型提示和数据建模，有助于构建可靠的应用程序。
- [OpenAI Cookbook: Structured output](https://cookbook.openai.com/examples/structured_outputs_intro) — OpenAI (2024)
  Publisher: OpenAI
  展示了指导 LLM 生成结构化输出的示例，经常说明数据验证的必要性。

---

[上一节](03-%E4%BD%BF%E7%94%A8%E8%BE%93%E5%87%BA%E8%A7%A3%E6%9E%90%E5%99%A8.md) · [下一节](05-%E5%A4%84%E7%90%86%E8%A7%A3%E6%9E%90%E9%94%99%E8%AF%AF.md)

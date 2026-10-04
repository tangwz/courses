---
course: "getting-started-with-llm-toolkit"
chapter: "application-safety-and-reliability"
lesson: "implementing-content-moderation"
sourceId: 7793
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-10-application-safety-and-reliability/implementing-content-moderation"
title: "实施内容审核"
description: "使用安全模块过滤输入和输出中的有害或不当内容。"
order: 2
plots: []
sourceHash: "71bedb905cefba5645e723897ae0b1d69f9bbbf589e5b7a6bc6e37d17c17c98b"
sourceCorrections: []
---

随着应用程序与用户互动并生成内容，构建一个安全层来过滤有害文本是迈向投入使用阶段的必要一步。未经审核的输入可能被用于攻击您的应用程序，而未经审核的输出则可能损害您的品牌声誉并伤害用户。`safety` 模块提供简单明了的工具来检测和管理不当内容。

### 全面的审核检查

检查文本最直接的方式是使用 `moderate_content` 函数。它会运行全面的分析，一次性检查多种类别的有害内容，包括毒性、仇恨言论和脏话。它返回一个 `SafetyResult` 对象，提供清晰、可操作的评估结果。

让我们审视它在安全和不安全文本上的表现：

```python
from kerb.safety import moderate_content

# 一条安全、专业的 H 消息
safe_text = "Thank you for your inquiry. I'll be happy to assist you."
safe_result = moderate_content(safe_text)

print(f"文本: '{safe_text}'")
print(f"是否安全: {safe_result.safe}")
print(f"总分: {safe_result.overall_score:.3f}")

# 一条明显有害的消息
toxic_text = "I hate dealing with stupid questions like this!"
toxic_result = moderate_content(toxic_text)

print(f"\n文本: '{toxic_text}'")
print(f"是否安全: {toxic_result.safe}")
print(f"总分: {toxic_result.overall_score:.3f}")
print(f"标记类别: {[cat.value for cat in toxic_result.flagged_categories]}")
```

`safe` 属性提供一个简单的布尔值用于快速判断，而 `flagged_categories` 则准确告诉您文本违反了哪些规则。这使得处理可以更细致，例如记录阻止消息的具体原因。

### 调整审核灵敏度

并非所有应用程序对敏感内容的容忍度都相同。一个通用聊天机器人可能需要更严格的筛选，而一个用于分析社交媒体数据的工具则可能需要更宽松的设置。您可以使用 `SafetyLevel` 枚举来控制审核检查的灵敏度。

共有三个级别：

- `SafetyLevel.STRICT`: 最敏感的级别，会标记 (token)任何潜在的有害内容。将其用于面向大众或年轻受众的应用程序。
- `SafetyLevel.MODERATE`: 一个平衡的默认值，适用于大多数应用程序。
- `SafetyLevel.PERMISSIVE`: 一个更宽松的级别，只标记明显有害的内容。

我们来看这些级别如何处理一个边界性语句：

```python
from kerb.safety import SafetyLevel

borderline_text = "This is ridiculous and annoying."

print(f"文本: '{borderline_text}'")

for level in [SafetyLevel.PERMISSIVE, SafetyLevel.MODERATE, SafetyLevel.STRICT]:
    result = moderate_content(borderline_text, level=level)
    print(f"\n使用 {level.value.upper()} 级别进行测试:")
    print(f"  是否安全: {result.safe}")
    print(f"  总分: {result.overall_score:.3f}")
```

正如您所见，相同的文本在 `PERMISSIVE` 级别可能被认为是安全的，但在 `STRICT` 级别则会被标记。这使您能够调整安全防护机制，以符合您应用程序的特定要求，而无需更改代码逻辑。

### 构建审核防护机制

在实际应用中，内容审核应作为一种“防护机制”，应用于大型语言模型（LLM）的输入和输出。

1. **输入审核**：防止用户提交有害、有毒或恶意的提示。这可以保护您的系统免受操纵，并过滤掉滥用行为。
2. **输出审核**：作为最终检查，确保大型语言模型（LLM）没有生成不当、不安全或有害的内容，即使输入是良性的。

这建立了一个双向的安全检查，确保交互流程的安全性。

> 一个典型的审核流程在提供最终回应之前，会检查用户输入和大型语言模型（LLM）的输出。

您可以这样实现这种防护机制模式。下面的函数首先审核用户输入，如果安全，则调用大型语言模型（LLM）。然后它会审核大型语言模型（LLM）的输出，最后向用户返回一个安全的响应。

```python
def simulate_llm_response(prompt: str) -> str:
    """模拟LLM响应以进行演示。"""
    responses = {
        "greeting": "您好！今天有什么可以帮助您的吗？",
        "angry": "我讨厌处理这种愚蠢的问题！",
        "professional": "感谢您的询问。我很乐意为您提供帮助。",
        "offensive": "如果你不明白，你就是个白痴。",
    }
    return responses.get(prompt, "我在这里提供帮助！")

def get_safe_response(user_prompt: str) -> str:
    """处理带有输入和输出审核的用户提示。"""
    
    # 1. 审核用户输入
    input_check = moderate_content(user_prompt, level=SafetyLevel.MODERATE)
    if not input_check.safe:
        # 阻止有害输入
        return "很抱歉，我无法处理该请求。"

    # 2. 获取LLM响应（仅当输入安全时）
    llm_output = simulate_llm_response(user_prompt)

    # 3. 审核LLM输出
    output_check = moderate_content(llm_output, level=SafetyLevel.MODERATE)
    if not output_check.safe:
        # 阻止有害输出并返回备用消息
        print(f"操作: 过滤 - 因 {[cat.value for cat in output_check.flagged_categories]} 被标记")
        return "抱歉，我无法提供该回应。"

    # 4. 返回安全输出
    return llm_output

# 用一个生成不安全响应的安全提示进行测试
response = get_safe_response("angry")
print(f"\n用户提示: 'angry'")
print(f"最终回应: {response}")

# 用一个生成安全响应的安全提示进行测试
response = get_safe_response("professional")
print(f"\n用户提示: 'professional'")
print(f"最终回应: {response}")
```

这种模式确保您的应用程序在其边界处受到保护，提供更安全的用户体验。

### 定向类别检查

如果您的应用程序只需要检查特定类型的内容，您可以使用专门的检查函数或将类别列表传递给 `moderate_content`。如果您不需要全面的分析，这样做会更高效。例如，您可能希望允许脏话，但严禁仇恨言论。

可用类别在 `ContentCategory` 枚举中定义。

```python
from kerb.safety import ContentCategory

text_with_profanity = "This damn feature is so frustrating to use!"

# 只检查毒性和仇恨言论，忽略脏话
result = moderate_content(
    text_with_profanity,
    categories=[ContentCategory.TOXICITY, ContentCategory.HATE_SPEECH]
)

print(f"文本: '{text_with_profanity}'")
print("仅检查: 毒性和仇恨言论")
print(f"是否安全: {result.safe}")

# 单个分数仍然可用
print("\n类别分数:")
for category, score in result.categories.items():
    print(f"  {category.value}: {score:.3f}")
```

这种定向方法为您提供精细控制，让您能够构建一个精确符合您应用程序环境和用户社区标准的审核策略。

## 参考资料

- [Building Safer AI Systems](https://openai.com/research) — OpenAI (Ongoing)
  此资源详细介绍了OpenAI开发安全AI的方法，包括内容审核和防止滥用的策略。
- [Automated Content Moderation: A Survey](https://jist.nit.ac.ir/article_1276_0.html) — Shahriar, B. and Nandi, A. (2020)
  Journal: Journal of Information Systems and Telecommunication; Publisher: Academic Center for Education, Culture and Research (ACECR); Volume: 8; Pages: 21-36; DOI: [10.22061/JIST.2020.10.155.1504](https://doi.org/10.22061/JIST.2020.10.155.1504)
  这篇综述文章回顾了用于自动化内容审核的各种技术和系统，涵盖了不同类型的有害内容。
- [Google AI Principles](https://ai.google/principles/) — Google (2018)
  Publisher: Google
  本文档概述了Google的AI开发道德准则，强调安全和负责任的使用，这支持了内容审核的必要性。
- [Perspective API](https://perspectiveapi.com/) — Jigsaw (2017)
  Publisher: Jigsaw (a Google company)
  这是一个用于识别有毒及其他有害评论的API官方网站，提供多类别内容分类。

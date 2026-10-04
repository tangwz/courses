# 内容审核与内容过滤API

来源：[原文](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-7-output-parsing-validation-reliability/moderation-content-filtering-apis)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然提示技巧和输出解析器有助于构建LLM的响应，但它们本身并不能保证生成内容的*安全性*或*适当性*。LLM在海量互联网文本上训练，有时会生成有害、不道德、带有偏见或存在其他问题的内容。仅依靠LLM进行自我审查不足以构建负责任和值得信赖的应用。集成专门的内容审核和内容过滤服务可提供必要的安全保障。

### 外部内容审核的必要性

LLM可能会生成包含以下内容的文本：

- 仇恨言论或骚扰
- 色情内容
- 煽动暴力或非法行为
- 宣扬自残
- 错误信息或虚假信息（尽管审核API更侧重于安全类别）

此外，恶意用户可能会尝试提示注入攻击，旨在绕过LLM的安全训练并引出有害响应。仅依靠提示设计或期望LLM完美地自我监管通常是不够的。对于生产系统而言，进行单独的专业检查是标准做法。

### 引入审核API

一些平台提供专门设计用于根据预定义安全类别对文本进行分类的API。这些服务通常使用其专门优化的微调 (fine-tuning)模型来检测有问题的内容。常见的例子包括：

- **OpenAI 审核端点：** OpenAI提供的专用API端点，用于根据其安全政策对文本进行分类。
- **Google Cloud 自然语言API（内容分类）：** Google Cloud AI服务的一部分，能够识别敏感主题。
- **Google Perspective API：** 专门设计用于对文本的毒性、威胁、侮辱以及其他常与在线评论相关的属性进行评分。
- **Amazon Comprehend（毒性检测）：** AWS服务，用于检测文本中的有害内容。
- **第三方服务：** 各种专业公司提供内容审核API。

### 审核API的工作原理

典型的工作流程是将一段文本（无论是用户输入还是LLM输出）发送到审核API端点。API处理文本并返回响应，通常指示：

1. **标记 (token)状态：** 布尔值（true/false），指示内容是否违反任何政策。
2. **类别分数：** 针对各种有害内容类别（例如，`hate`、`sexual`、`violence`、`self-harm`）的置信分数。分数通常在0到1之间。
3. **标记类别：** 哪些特定类别超过了某个内部阈值而被触发。

以下是使用Python的`requests`库与审核端点交互的示例：

```python
import requests
import os
import json

# 假设API密钥已安全存储，例如环境变量
MODERATION_API_KEY = os.environ.get("MODERATION_API_KEY")
MODERATION_ENDPOINT_URL = "https://api.example-moderation.com/v1/check" # 替换为实际URL

def check_content_safety(text_to_check: str) -> dict:
    """
    将文本发送到审核API并返回分类结果。
    """
    if not MODERATION_API_KEY:
        print("警告：未找到审核API密钥。跳过安全检查。")
        # 根据策略返回一个默认的安全响应或引发错误
        return {"flagged": False, "categories": {}, "scores": {}}

    headers = {
        "Authorization": f"Bearer {MODERATION_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = json.dumps({"text": text_to_check})

    try:
        response = requests.post(MODERATION_ENDPOINT_URL, headers=headers, data=payload, timeout=10)
        response.raise_for_status() # 对不良响应（4xx或5xx）引发HTTPError
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"调用审核API时出错：{e}")
        # 决定如何处理API错误：安全失败（假设不安全？）、记录日志、重试？
        # 返回潜在不安全的内容可能存在风险。
        return {"flagged": True, "error": str(e), "categories": {}, "scores": {}} # 示例：安全失败

# 示例用法：
user_input = "一些可能存在问题的用户文本"
llm_output = "LLM生成的输出"

# 在将用户输入发送给LLM之前进行检查
user_input_safety = check_content_safety(user_input)
if user_input_safety.get("flagged"):
    print(f"用户输入被标记：{user_input_safety.get('categories')}")
    # 处理被标记的输入（例如，拒绝、要求用户重新措辞）
else:
    # 继续生成LLM响应...
    # llm_output = generate_llm_response(user_input) # 占位符

    # 在向用户显示LLM输出之前进行检查
    output_safety = check_content_safety(llm_output)
    if output_safety.get("flagged"):
        print(f"LLM输出被标记：{output_safety.get('categories')}")
        # 处理被标记的输出（例如，显示通用消息、记录日志、不显示）
    else:
        # 显示安全的LLM输出
        print("LLM输出（安全）：", llm_output)
```

### 将审核集成到应用流程中

您可以在两个主要环节集成审核检查：

1. **输入审核：** 在用户提供的输入发送到LLM*之前*对其进行检查。这可以防止用户故意尝试引出有害内容或探测LLM的安全边界。
2. **输出审核：** 在LLM生成的响应显示给用户或被下游应用组件使用*之前*对其进行检查。这可以捕获LLM本身生成的潜在有害内容。

在这两个环节都进行审核可以提供更全面的保护。

> 一个典型的应用流程，包含了可选的输入和输出审核检查。

### 处理被标记 (token)内容和阈值

当内容被标记时，您的应用需要一个响应策略。简单地阻止内容在某些情况下可能合适，但其他选项包括：

- **记录日志：** 记录被标记的内容和审核结果以供审查。
- **返回通用消息：** 告知用户输入/输出不适当，但不显示有害内容。
- **要求重新措辞：** 要求用户修改其输入。
- **管理员通知：** 提醒管理员注意潜在的恶意使用或问题输出模式。

大多数审核API提供每个类别的分数。您需要为这些分数决定合适的**阈值**。设置一个非常低的阈值可能会标记无害内容（误报），而一个非常高的阈值可能会遗漏真正有害的内容（漏报）。恰当的平衡取决于您的应用风险承受能力和具体情况。通常需要通过实验和监控来微调 (fine-tuning)这些阈值。

使用审核API是构建更安全、更可靠LLM应用的重要一步。它作为独立的保障措施，补充了仔细的提示设计和输出验证，有助于保护您的用户和应用声誉。

## 参考资料

- [Moderation](https://platform.openai.com/docs/guides/moderation) — OpenAI (2022)
  Publisher: OpenAI
  OpenAI Moderation API的官方文档，说明了其在LLM应用中进行内容安全检查的功能。
- [Perspective API](https://www.perspectiveapi.com/) — Jigsaw and Google (2024)
  Google Perspective API的官方网站和文档，专门用于检测文本中的毒性和其他有害属性。

---

[上一节](06-%E5%AE%9E%E6%96%BD%E9%87%8D%E8%AF%95%E6%9C%BA%E5%88%B6.md) · [下一节](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%8F%AF%E9%9D%A0%E7%9A%84%E8%BE%93%E5%87%BA%E5%A4%84%E7%90%86.md)

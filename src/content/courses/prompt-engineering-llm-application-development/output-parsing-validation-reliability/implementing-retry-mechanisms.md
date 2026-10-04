---
course: "prompt-engineering-llm-application-development"
chapter: "output-parsing-validation-reliability"
lesson: "implementing-retry-mechanisms"
sourceId: 3082
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-7-output-parsing-validation-reliability/implementing-retry-mechanisms"
title: "实施重试机制"
description: "通过重试失败的LLM调用或解析尝试，提高应用程序的韧性。"
order: 6
plots: []
sourceHash: "50f258a432b0a579921cf3163edf80cadc864a4c7e96221c11a85bc57625ccbe"
sourceCorrections: []
---

即使精心设计了提示词 (prompt)和输出解析，与大型语言模型（LLMs）或周围基础设施的交互有时仍可能出错。网络故障、API暂时不可用、速率限制，甚至偶尔出现未能通过初步检查的格式错误输出，都可能扰乱您的应用程序运行。一个可靠的应用程序不应直接失败，而应尝试从这些暂时性问题中恢复。实施重试机制是构建与外部服务（包括LLM API）交互的更可靠软件的标准做法。

### 为什么需要重试

与LLM API的交互包含几个潜在的故障点：

1. **网络问题：** 连接API端点的暂时性问题。
2. **API服务器错误：** LLM提供商可能遇到暂时性问题（例如，HTTP 500内部服务器错误，503服务不可用）。
3. **速率限制：** 超出每分钟/天允许的请求数量（HTTP 429请求过多）。延迟后重试通常是所需操作。
4. **暂时性输出问题：** 有时，LLM可能因随机性产生未能通过您的解析或验证逻辑的输出，即使提示词 (prompt)通常有效。用相同的提示词重试可能会得到正确回应。
5. **资源限制：** 提供商方面暂时缺乏可用的计算资源。

简单的重试逻辑可以自动处理许多这些暂时性问题，无需人工干预，从而极大提高您应用程序的正常运行时间和用户体验。

### 简单重试策略

最基本的方法是，如果发生错误，在短暂的固定延迟后重试操作固定次数。

```python
import time
import random

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1

def make_llm_api_call_with_simple_retry(prompt):
    """使用简单重试逻辑进行LLM API调用。"""
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            # 替换为您的实际API调用函数
            response = call_your_llm_api(prompt)

            # 可选：在此处添加解析/验证，如果失败则抛出错误
            parsed_output = parse_and_validate(response)
            if parsed_output is None: # 验证失败示例
                raise ValueError("输出验证失败")

            return parsed_output # 成功
        except (requests.exceptions.RequestException, ValueError, RateLimitError) as e:
            print(f"尝试 {attempt + 1} 失败: {e}")
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                print(f"在 {RETRY_DELAY_SECONDS} 秒后重试...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print("达到最大重试次数。失败。")
                raise last_exception # 所有重试失败后重新抛出最后一次异常

# 示例用法（假设 call_your_llm_api、
# parse_and_validate 和 RateLimitError 等必要函数存在）
# try:
#     result = make_llm_api_call_with_simple_retry("总结这段文本...")
#     print("成功接收并解析响应:", result)
# except Exception as e:
#     print(f"操作在重试后最终失败: {e}")
```

这适用于非常短暂、不常见的问题，但可能存在问题。如果API过载或速率限制生效，立即多次重试可能会使情况恶化或只是浪费资源。

### 带抖动（Jitter）的指数退避

一种更被广泛采用的策略是指数退避。它不是等待固定的时间，而是每次失败尝试后，延迟时间呈指数级增长。这让外部服务（如LLM API）有更多时间恢复，如果它正经历持续负载或问题。

此外，在延迟中加入“抖动”（一个小的随机时间量）有助于避免“群起效应”问题，即许多客户端可能在普遍的暂时性故障后同时重试，再次使服务过载。

以下是其逻辑：

1. 尝试操作。
2. 如果失败，等待一个基础延迟（例如，1秒）。
3. 如果再次失败，等待 `base_delay * 2 + random_jitter`。
4. 如果再次失败，等待 `base_delay * 4 + random_jitter`。
5. 继续将等待时间加倍（直到一个合理的最高延迟），直到达到最大重试次数。

```python
import time
import random
import math

# 假设已导入必要的requests和错误类型（例如 RateLimitError）

def make_llm_api_call_with_backoff(prompt, max_retries=5, base_delay=1, max_delay=60):
    """使用指数退避和抖动进行LLM API调用。"""
    last_exception = None
    for attempt in range(max_retries):
        try:
            # 替换为您的实际API调用和验证
            response = call_your_llm_api(prompt)
            parsed_output = parse_and_validate(response)
            if parsed_output is None:
                 raise ValueError("输出验证失败")
            return parsed_output # 成功

        except (requests.exceptions.RequestException, ValueError, RateLimitError) as e:
            print(f"尝试 {attempt + 1} 失败: {e}")
            last_exception = e

            if attempt < max_retries - 1:
                # 计算指数退避延迟
                backoff_time = min(max_delay, base_delay * (2 ** attempt))
                # 添加抖动（退避时间的一个随机部分，例如，最多1秒）
                jitter = random.uniform(0, 1)
                sleep_time = backoff_time + jitter
                print(f"在 {sleep_time:.2f} 秒后重试...")
                time.sleep(sleep_time)
            else:
                print("达到最大重试次数。失败。")
                raise last_exception # 最后一次尝试后重新抛出异常

# 示例用法
# try:
#     result = make_llm_api_call_with_backoff("生成一首关于编程的诗。")
#     print("成功接收并解析响应:", result)
# except Exception as e:
#     print(f"操作在重试后最终失败: {e}")
```

> 说明指数退避重试逻辑的流程图。

### 配置重试行为

实施重试时，请考虑以下几点：

- **哪些错误应触发重试？** 您可能只希望对特定的HTTP状态码（如429、500、503）或网络错误进行重试，而不是对验证错误（400错误请求）进行重试，后者表明您的请求本身存在问题。有时，如果您怀疑是LLM的随机性导致了验证失败，您可能*会*希望重试。根据具体的错误和场景进行决定。
- **最大重试次数：** 设置一个限制（例如，3-5次重试），以防止错误持续存在时出现无限循环。
- **最大延迟：** 限制指数退避延迟，以避免过长的等待（例如，60秒）。
- **总超时：** 您也可以实施一个包括所有重试在内的整个操作的整体超时。
- **日志记录：** 记录重试尝试和失败原因。这对于调试和监测系统健康非常有帮助。

许多HTTP客户端库（例如 `requests` 及其 `HTTPAdapter` 和 `Retry` 类）或LLM框架组件（例如LangChain中的）都内置了对配置重试策略的支持，这与从头编写逻辑相比可以简化实施。

通过加入智能重试机制，您能极大提高LLM应用程序应对分布式系统和API交互中固有暂时性故障的弹性，从而带来更可靠和稳定的用户体验。

## 参考资料

- [Error retries and exponential backoff](https://docs.aws.amazon.com/general/latest/gr/api-retries.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  提供了关于实现指数退避和抖动重试逻辑以进行弹性API交互的详细指南。
- [urllib3.util.retry.Retry](https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.retry.html) — The urllib3 Project Developers (2024)
  介绍了`Retry`类，展示了如何在`requests`库中配置有效的重试策略，包括退避和状态码处理。
- [Release It!: Design and Deploy Production-Ready Software](https://pragprog.com/titles/mbret2/release-it-second-edition/) — Michael T. Nygard (2018)
  Publisher: Pragmatic Bookshelf
  一本构建可靠系统的经典指南，涵盖了重试、熔断器和超时等对生产可靠性必不可少的模式。

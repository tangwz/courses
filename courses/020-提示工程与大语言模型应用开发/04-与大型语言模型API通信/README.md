# 第 4 章：与大型语言模型API通信

来源：[原章节](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-4-interacting-with-llm-apis)

[返回课程目录](../README.md)

在掌握了设计有效提示词的技巧后，我们现在转向如何在软件应用中实际使用这些提示词。本章主要讲解如何通过编程方式，借助大型语言模型（LLM）的应用程序编程接口（API）与其通信。您将学习连接代码到LLM服务、发送提示词以及处理生成的回应的必要步骤。

具体而言，本章内容包括：
*   常见LLM API及其功能概览。
*   安全API认证的方法。
*   使用Python构建和发送API请求。
*   了解模型选择和生成控制等主要请求参数。
*   在您的应用中解析和使用API回应。
*   管理潜在API错误和使用限制的策略。
*   实现流式传输以增量接收回应。

通过实际例子，包括构建一个基础问答应用，您将获得进行API调用和处理结果的实际操作经验，从而为构建集成LLM的软件奠定根基。

## 小节

- 1. [常见大型语言模型API概览（OpenAI、Anthropic等）](01-%E5%B8%B8%E8%A7%81%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8BAPI%E6%A6%82%E8%A7%88%EF%BC%88OpenAI%E3%80%81Anthropic%E7%AD%89%EF%BC%89.md)
- 2. [API 认证与安全](02-API%20%E8%AE%A4%E8%AF%81%E4%B8%8E%E5%AE%89%E5%85%A8.md)
- 3. [使用 Python 发送 API 请求](03-%E4%BD%BF%E7%94%A8%20Python%20%E5%8F%91%E9%80%81%20API%20%E8%AF%B7%E6%B1%82.md)
- 4. [理解API请求参数](04-%E7%90%86%E8%A7%A3API%E8%AF%B7%E6%B1%82%E5%8F%82%E6%95%B0.md)
- 5. [处理API响应](05-%E5%A4%84%E7%90%86API%E5%93%8D%E5%BA%94.md)
- 6. [处理 API 错误和速率限制](06-%E5%A4%84%E7%90%86%20API%20%E9%94%99%E8%AF%AF%E5%92%8C%E9%80%9F%E7%8E%87%E9%99%90%E5%88%B6.md)
- 7. [流式响应](07-%E6%B5%81%E5%BC%8F%E5%93%8D%E5%BA%94.md)
- 8. [动手实践：搭建一个简单的问答机器人](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%90%AD%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%E9%97%AE%E7%AD%94%E6%9C%BA%E5%99%A8%E4%BA%BA.md)

章节测验：[在线测验](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-4-interacting-with-llm-apis/quiz)

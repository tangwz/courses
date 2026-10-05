# 第 8 章：应用程序开发考量

来源：[原章节](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-8-application-development-considerations)

[返回课程目录](../README.md)

在尝试了提示词、与API进行交互并使用了框架来构建大型语言模型应用的核心逻辑后，我们现在将关注在实际开发和有效部署这些系统时的考量。构建一个可用的原型只是第一步；为长期使用做准备需要关注其架构、安全、成本和运行维护等方面。

本章涵盖了一些重要的做法，旨在帮助您将LLM应用从想法阶段带到一个更易于管理和部署的状态。您将了解到：

*   组织应用程序代码，使其清晰且易于维护。
*   安全地管理API密钥和其他敏感凭证。
*   估算和监控LLM API使用相关成本的方法。
*   实施基本缓存以优化性能并降低开销。
*   测试包含LLM组件的应用程序的方法。
*   适合初步使用的简单部署选项概览，例如无服务器函数或容器化。

这里的重点是实际的工程方面，这些方面支持您的LLM驱动软件的可靠运行。

## 小节

- 1. [大语言模型应用代码组织](01-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E4%BB%A3%E7%A0%81%E7%BB%84%E7%BB%87.md)
- 2. [管理 API 密钥和机密信息](02-%E7%AE%A1%E7%90%86%20API%20%E5%AF%86%E9%92%A5%E5%92%8C%E6%9C%BA%E5%AF%86%E4%BF%A1%E6%81%AF.md)
- 3. [成本估算与监控](03-%E6%88%90%E6%9C%AC%E4%BC%B0%E7%AE%97%E4%B8%8E%E7%9B%91%E6%8E%A7.md)
- 4. [基本缓存策略](04-%E5%9F%BA%E6%9C%AC%E7%BC%93%E5%AD%98%E7%AD%96%E7%95%A5.md)
- 5. [大型语言模型应用测试](05-%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E6%B5%8B%E8%AF%95.md)
- 6. [简单部署选项（无服务器、容器）](06-%E7%AE%80%E5%8D%95%E9%83%A8%E7%BD%B2%E9%80%89%E9%A1%B9%EF%BC%88%E6%97%A0%E6%9C%8D%E5%8A%A1%E5%99%A8%E3%80%81%E5%AE%B9%E5%99%A8%EF%BC%89.md)
- 7. [动手实践：一个简单LLM应用的容器化](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95LLM%E5%BA%94%E7%94%A8%E7%9A%84%E5%AE%B9%E5%99%A8%E5%8C%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-8-application-development-considerations/quiz)

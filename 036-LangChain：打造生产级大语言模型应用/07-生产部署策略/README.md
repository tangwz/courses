# 第 7 章：生产部署策略

来源：[原章节](https://apxml.com/zh/courses/langchain-production-llm/chapter-7-deployment-strategies-production)

[返回课程目录](../README.md)

您已经搭建并优化了LangChain应用。现在，下一步是将其从开发环境迁移到生产环境，使其能够可靠运行。本章关注LangChain应用在实际使用中的打包、部署和管理等实际方面。

我们将介绍使您的应用投入运行的必要做法。您将学习如何：

*   组织项目代码和配置，以便维护和部署。
*   使用Docker等容器化工具打包您的应用。
*   评估不同部署环境，包括传统服务器、Kubernetes和无服务器平台。
*   实施无服务器部署的特定模式。
*   安全管理API密钥和配置等敏感信息。
*   使用持续集成和持续部署（CI/CD）流水线，自动化构建、测试和部署流程。
*   采用安全的部署策略，例如蓝绿部署或金丝雀发布，以在更新时减少停机时间和风险。

在本章结束时，您将掌握使LangChain应用投入实际运行的常见技术和注意事项。

## 小节

- 1. [为部署构建 LangChain 项目结构](01-%E4%B8%BA%E9%83%A8%E7%BD%B2%E6%9E%84%E5%BB%BA%20LangChain%20%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84.md)
- 2. [使用 Docker 将 LangChain 应用容器化](02-%E4%BD%BF%E7%94%A8%20Docker%20%E5%B0%86%20LangChain%20%E5%BA%94%E7%94%A8%E5%AE%B9%E5%99%A8%E5%8C%96.md)
- 3. [部署方案：服务器、Kubernetes、无服务器](03-%E9%83%A8%E7%BD%B2%E6%96%B9%E6%A1%88%EF%BC%9A%E6%9C%8D%E5%8A%A1%E5%99%A8%E3%80%81Kubernetes%E3%80%81%E6%97%A0%E6%9C%8D%E5%8A%A1%E5%99%A8.md)
- 4. [LangChain 的无服务器部署模式](04-LangChain%20%E7%9A%84%E6%97%A0%E6%9C%8D%E5%8A%A1%E5%99%A8%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%BC%8F.md)
- 5. [管理环境变量和敏感信息](05-%E7%AE%A1%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E5%92%8C%E6%95%8F%E6%84%9F%E4%BF%A1%E6%81%AF.md)
- 6. [搭建 CI/CD 流水线](06-%E6%90%AD%E5%BB%BA%20CI-CD%20%E6%B5%81%E6%B0%B4%E7%BA%BF.md)
- 7. [蓝绿部署与金丝雀部署策略](07-%E8%93%9D%E7%BB%BF%E9%83%A8%E7%BD%B2%E4%B8%8E%E9%87%91%E4%B8%9D%E9%9B%80%E9%83%A8%E7%BD%B2%E7%AD%96%E7%95%A5.md)
- 8. [动手实践：通过 Docker 部署 LangChain 应用](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E9%80%9A%E8%BF%87%20Docker%20%E9%83%A8%E7%BD%B2%20LangChain%20%E5%BA%94%E7%94%A8.md)

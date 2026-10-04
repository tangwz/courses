# API 测试简介

来源：[原文](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-4-structuring-testing-fastapi/introduction-api-testing)

[返回章节目录](README.md) · [返回课程目录](../README.md)

确保 FastAPI API 如预期般运行是一项主要事项，尤其是在整合机器学习 (machine learning)模型时。编写代码只是过程的一部分；通过测试来验证其正确性，对于构建可靠的应用非常必要。在为机器学习模型提供服务时，意外输入或行为可能导致不准确的预测或服务故障。测试能为应用的后续维护和未来开发提供必要的信心。

自动化测试充当安全网。它们让您能够对代码库进行修改（例如重构端点逻辑、更新依赖项，甚至替换机器学习模型），同时确保您不会无意中破坏现有功能。对于机器学习 API，测试验证多个重要方面：

- **契约一致性：** 验证您的 API 根据定义的 Pydantic 模式正确处理输入，并以预期格式返回输出。这确保与您的 API 交互的客户端收到可预测的响应。测试确认您在第 2 章中定义的数据结构在 API 边界得到正确执行。
- **输入验证逻辑：** 确认您设定的验证规则（例如数据类型、值范围、字符串格式）得到执行，阻止格式错误或无意义的数据抵达您的模型推断逻辑。这对于机器学习模型非常重要，因为它们通常要求输入具有非常特定的格式或范围。
- **集成正确性：** 确保 API 端点正确预处理输入，调用已加载的机器学习模型（如第 3 章所述），并对输出进行后处理。尽管 API 测试通常不评估模型本身的*统计准确性*（那是模型验证，一个通常在模型开发期间执行的独立过程），但它们会验证模型周围的流程在 API 环境中正确运行。API 是否调用了模型的 `predict` 方法？结果在发送回之前是否正确转换？
- **错误处理：** 检查您的 API 是否对无效请求（例如那些 Pydantic 验证失败的请求）或内部问题（例如加载模型文件失败）做出适当响应，提供信息丰富的错误消息和正确的 HTTP 状态码，而不是崩溃或返回难以理解的错误。
- **回归预防：** 捕捉代码更改的意外副作用，确保以前正常运行的功能在修改后继续正常运行。随着应用的演变，这非常有用。

在 FastAPI 应用中，我们主要关注模拟 HTTP 请求到 API 端点并对响应断言条件的自动化测试。这些测试可以从简单的*单元测试*（在隔离状态下检查特定的辅助函数或 Pydantic 模型），到*集成测试*（验证应用中的整个请求-响应流程，包括与可能已加载的机器学习模型的互动）。

FastAPI 在设计时就考虑了可测试性。它使用标准的 Python 类型提示和 Pydantic 有助于数据结构的清晰定义，使针对这些定义编写测试变得更简单。此外，FastAPI 提供一个基于 `httpx` 库的 `TestClient`，让您能够在测试套件中直接向应用发送请求，无需运行服务器。这使得编写快速、可靠的测试变得简单直接。

以下章节将引导您使用 `TestClient` 为您的机器学习 API 端点编写有效的单元测试和集成测试，侧重于验证输入、测试预测逻辑，并确保您的应用结构支持可测试性。这种实践将巩固本章前面介绍的应用结构，使您的机器学习部署服务更具弹性且更易于管理。

## 参考资料

- [Testing FastAPI Applications](https://fastapi.tiangolo.com/tutorial/testing/) — Sebastián Ramírez (2024)
  FastAPI官方测试指南，展示了如何使用`TestClient`和`httpx`进行单元和集成测试。
- [Python Testing with pytest: A Hands-On Guide to Test-Driven Development](https://pragprog.com/titles/bopytest4/python-testing-with-pytest-fourth-edition/) — Brian Okken (2022)
  Publisher: The Pragmatic Programmers
  使用`pytest`测试Python应用的指南，涵盖了适用于FastAPI开发的单元、集成和功能测试方法。
- [Practical MLOps: Operationalizing Machine Learning Models for Production](https://www.oreilly.com/library/view/practical-mlops/9781098103002/) — Noah Gift and Alfredo Deza (2021)
  Publisher: O'Reilly Media; Pages: 458
  为机器学习模型投入生产提供建议，包括测试和监控已部署的ML系统及其API的方法。

---

[上一节](03-%E7%AE%A1%E7%90%86%E4%BE%9D%E8%B5%96%E9%A1%B9.md) · [下一节](05-%E4%BD%BF%E7%94%A8%20TestClient%20%E8%BF%9B%E8%A1%8C%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95.md)

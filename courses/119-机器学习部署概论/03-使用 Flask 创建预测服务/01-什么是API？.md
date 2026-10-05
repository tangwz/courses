# 什么是API？

来源：[原文](https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask/what-is-an-api)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在机器学习 (machine learning)模型训练并保存之后，接下来的挑战通常是：外部应用程序（例如一个网络应用或另一项服务）如何实际*使用*该模型来获取预测结果？直接访问存储 `model.pkl` 或 `model.joblib` 文件的文件系统通常不是一个可行的方法。相反，不同软件组件之间需要一种明确的通信方式。这正是应用程序编程接口（API）所要解决的问题。

可以将API视为一种约定或一套规则，它允许一个软件组件向另一个组件请求服务或数据。它规定了可以提出哪些类型的请求、如何提出请求、应使用的数据格式以及预期会收到哪些响应。

### 服务员类比

想象一下您在一家餐馆里。您（客户端应用程序）想要食物（一个预测结果）。您不会直接进入厨房（模型逻辑）开始烹饪。相反，您会与服务员（API）互动。

1. **菜单（API定义）：** 菜单告诉您有哪些菜品（操作/预测）可用，以及厨房需要哪些信息来准备它们（输入数据格式）。
2. **点餐（发出请求）：** 您告诉服务员您想从菜单上点什么（“我想要一份包含输入数据X的意大利面”）。您以预期的格式提供必要的详细信息。
3. **厨房（服务器/模型）：** 服务员将您的订单送到厨房。厨房工作人员（运行您的模型代码的服务器）准备您的菜品（根据输入X生成预测）。
4. **上菜（返回响应）：** 服务员将准备好的菜品（预测结果）送回您的餐桌。

在这个类比中：

- 您是**客户端**：需要预测结果的应用程序。
- 服务员是**API**：定义如何进行交互的中间方。
- 菜单是**API规范**：它记录了可用的功能和所需的输入/输出。
- 厨房是**服务器**：托管模型和逻辑的系统。
- 您的订单是**请求**：向API端点发送数据。
- 准备好的菜品是**响应**：从API接收预测结果。

### API在网络开发和模型服务中的应用

在网络服务的背景下，API通常使用超文本传输协议（HTTP），这与您的网络浏览器获取网页所用的协议相同。当我们构建预测服务时，我们通常会创建一个*网络API*。这意味着我们的模型将通过网络在特定的URL（通常称为*端点*）上监听传入的请求。

A客户端应用程序向特定的端点发送HTTP请求。此请求通常包含：

- 一个**HTTP方法**：例如 `POST`（常用于发送数据以创建或更新内容，适合发送用于预测的输入特征）或 `GET`（通常用于获取数据）。
- **端点URL**：预测函数的特定地址（例如，`http://yourserver.com/predict`）。
- **请求头（Headers）**：包含请求的元数据（例如，正在发送的数据格式）。
- **请求体（Body）**：通常包含模型的实际输入数据，常格式化为JSON（JavaScript对象表示法），这是一种用于数据交换的可读文本格式。

托管API的服务器接收此请求，处理输入数据，可能会加载已保存的模型，将数据提供给模型以获得预测结果，然后将HTTP响应发送回客户端。此响应通常包含预测结果，同样常格式化为JSON。

> 一个简单的交互流程：客户端通过HTTP请求向API服务器发送输入数据，服务器使用已保存的模型生成预测结果，并通过HTTP响应将其发送回去。

为什么这对于机器学习 (machine learning)部署很有用？

1. **解耦：** 需要预测结果的应用程序无需了解模型内部的工作方式，甚至无需知道模型文件存储在哪里。它只需要知道API的地址以及如何格式化请求。
2. **可访问性：** 您的模型可以通过网络进行访问，从而允许各种应用程序（网络前端、移动应用、其他后端服务）使用它。
3. **可扩展性：** 您可以独立于客户端应用程序扩展API服务器。如果预测请求增加，您可以为处理API的服务器增加更多资源。
4. **技术无关性：** 只要客户端应用程序（无论是用Java、JavaScript还是任何其他语言编写的）和您的基于Python的预测API都“说”HTTP，并且在数据格式（如JSON）上达成一致，它们就可以进行交互。

在本章中，我们将侧重于使用Flask（一个流行且轻量级的Python网络框架）来构建这样的网络API。您将学会如何创建端点、处理传入数据、加载您之前保存的模型、生成预测结果，并将这些预测结果作为响应发送回去。这个API将作为您的训练模型与需要其功能的应用程序之间的桥梁。

## 参考资料

- [RESTful Web APIs](https://www.oreilly.com/library/view/restful-web-apis/9781449358068/) — Leonard Richardson, Mike Amundsen, Sam Ruby (2013)
  Publisher: O'Reilly Media
  理解REST架构风格、Web API设计和HTTP交互的指南。
- [Flask Documentation](https://flask.palletsprojects.com/) — Pallets (2024)
  Flask Web框架的官方文档，详细介绍了其功能和API开发。
- [The JSON Data Interchange Format](https://www.ietf.org/rfc/rfc8259.txt) — T. Bray, Ed. (2017)
  Journal: IETF RFC 8259; Publisher: Internet Engineering Task Force (IETF)
  JSON（JavaScript对象表示法）的正式规范，它是Web API常用的数据格式。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready ML Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  涵盖机器学习模型部署的过程以及API在向应用程序提供预测方面的作用。

---

[上一节](../02-%E4%B8%BA%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%81%9A%E5%87%86%E5%A4%87/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E7%AE%80%E5%8D%95%E6%A8%A1%E5%9E%8B.md) · [下一节](02-Web%E6%A1%86%E6%9E%B6%E7%AE%80%E4%BB%8B.md)

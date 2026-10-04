---
course: "intro-data-engineering"
chapter: "foundational-concepts"
lesson: "introduction-apis-data-retrieval"
sourceId: 5517
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-2-foundational-concepts/introduction-apis-data-retrieval"
title: "数据获取的API简介"
description: "学习如何使用应用程序编程接口（API）来获取数据。"
order: 6
plots: []
sourceHash: "4b8bfc6f374b696d9d8cb73e0330c584161fe7ac2deee826c206722cc19b1910"
sourceCorrections: []
---

假设您需要的数据未存储在您自己的文件或数据库中。例如，您可能需要实时的天气信息、股票市场价格，或在线商店的产品详情。这些数据通常位于其他人的服务器上。您如何获取这些数据呢？通常，答案是使用应用程序编程接口，简称API。

可以把API想象成餐厅的菜单。菜单列出了您可以点的菜肴（数据），并提供了点菜的说明。您（客户端应用程序）不需要知道厨房（服务器）是如何准备食物的。您只需按照菜单规则下订单（发出请求），服务员（API）就会把菜肴（数据）送给您。

### 为什么使用API获取数据？

API充当中间层，使不同的软件应用程序能够相互通信。对于数据工程师而言，API是获取数据的重要途径，因为它们提供了一种规范的方式来：

1. **访问外部数据：** 获取第三方服务（如天气服务、社交媒体平台、金融数据提供商）托管的数据。
2. **系统集成：** 连接组织内部不同的系统，使数据能在它们之间流动。
3. **标准化数据访问：** 提供一种一致的数据请求方法，无论源系统的底层有多复杂。

### 请求-响应模型

与大多数Web API的交互遵循一种简单的方式：

1. **请求：** 您的应用程序（客户端）向特定的API端点（一个URL）发送请求。此请求指定您想要什么数据以及如何获取。
2. **响应：** 托管API的服务器处理请求。如果请求有效，它会返回包含所需数据的响应，通常采用JSON或XML等结构化格式。

> 一个简单的图示，说明了客户端应用程序与服务器之间的API请求-响应循环。

### API数据获取请求的重要组成部分

当您发出API请求以获取数据时，通常会涉及以下几个组成部分：

- **端点URL：** 这是您发送请求的特定网络地址。它通常指明您请求的资源种类。例如，一个在线商店的API可能有一个端点，如 `https://api.store.com/v1/products`。
- **HTTP方法：** 这个动词告诉服务器您希望执行什么操作。对于获取数据，最常用的方法是 `GET`。其他方法如 `POST`、`PUT` 和 `DELETE` 分别用于创建、更新和删除数据，但 `GET` 是我们获取数据时关注的。
- **参数 (parameter)：** 有时，您需要在请求中提供更具体的详情。这些通常作为*查询参数*附加在URL的问号 (`?`) 之后。例如，要获取特定产品的详情，URL可能是 `https://api.store.com/v1/products?id=456`。多个参数通常用连接符 (`&`) 分隔，例如 `https://api.store.com/v1/products?category=electronics&in_stock=true`。
- **请求头：** 这些提供有关请求的元数据，例如预期的响应格式（例如，`Accept: application/json`）。
- **认证：** 许多API要求您表明身份，以确认您有权限访问数据。这通常是通过随请求一起发送的*API密钥*或*令牌*完成的，通常在请求头中。这有助于防止未经授权的访问，并协助API提供方记录使用情况。

### API响应中的数据样式

当API返回数据时，它需要是您的应用程序能够理解的样式。虽然存在多种样式（如XML或CSV），但\*\*JSON（JavaScript对象表示法）\*\*在Web API中占据主导地位。JSON轻量、易读，并且机器解析起来很方便。

以下是请求产品数据时JSON响应可能呈现的示例：

```json
{
  "product_id": 456,
  "name": "Wireless Noise-Cancelling Headphones",
  "category": "Electronics",
  "price": 249.99,
  "in_stock": true,
  "features": [
    "Bluetooth 5.0",
    "Active Noise Cancellation",
    "20-hour battery life"
  ]
}
```

您的应用程序会接收此文本，对其进行解析，然后获取所需信息，例如产品名称或价格。

### API作为数据工作中的数据源

了解API对数据工程师来说非常重要，因为它们是*获取*外部或内部服务数据的主要方式。从API中获取的数据通常作为数据管道的起始数据。数据工程师需要知道如何稳定地与这些API进行交互，处理可能出现的错误（如网络问题或无效响应），安全地保管认证密钥，并在将其加载到数据湖或数据仓库等存储系统进行后续处理和分析之前，解析接收到的数据。

## 参考资料

- [API Handbook: A Guide to APIs, Web Services, and Microservices](https://www.oreilly.com/library/view/api-handbook/9781098106972/) — Robert K. O'Connell (2021)
  Publisher: O'Reilly Media
  提供了关于API、其类型及其工作原理的广泛而易于理解的概述，是API新手的优秀入门读物。
- [JSON (JavaScript Object Notation)](https://www.json.org/json-en.html) — Douglas Crockford (2023)
  JSON的官方权威规范，它是本节讨论的API响应中主要的数据格式。
- [Requests: HTTP for Humans™ Documentation](https://docs.python-requests.org/en/latest/) — Kenneth Reitz and the Requests Community (2023)
  为使用Python进行HTTP请求提供了实用指导和示例，展示了如何实现请求-响应模型，包括GET方法、参数和用于数据检索的请求头。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本在数据工程领域广受推崇的书籍，提供了数据集成和外部数据源的背景，强化了API在可靠数据获取中的作用。

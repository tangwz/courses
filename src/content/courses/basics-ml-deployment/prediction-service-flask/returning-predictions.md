---
course: "basics-ml-deployment"
chapter: "prediction-service-flask"
lesson: "returning-predictions"
sourceId: 3994
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-3-prediction-service-flask/returning-predictions"
title: "返回预测结果"
description: "将模型的预测结果格式化并作为响应返回，通常采用JSON格式。"
order: 8
plots: []
sourceHash: "a8d1efe820ad538a2bbbb7c5aa96f59ffa6f9d70b38841e0faeb83881806975e"
sourceCorrections: []
---

当Flask应用接收到输入数据并使用其模型生成预测结果后，将该预测发回给客户端是必不可少的一步。仅仅在Python函数中得到预测结果是不够的；它需要被打包并通过网络传输，格式需客户端能够理解。

### 为何要构造响应？

设想一下调用你的API的客户端应用。它可能是一个Web前端、一个移动应用，或者另一个后端服务。该客户端需要准确知道如何解析它接收到的数据。只返回原始数字或字符串对非常简单的情况可能适用，但这种方式不可靠也不灵活。构造化的响应可以让你：

1. **提供上下文 (context)：** 你不仅可以包含预测结果本身，还可以包含元数据，例如预测的单位、置信度分数（如果适用），或所用模型的版本。
2. **确保一致性：** 通过定义特定格式，你可以让客户端应用更容易可靠地解析响应，即使你稍后更新模型或添加更多信息。
3. **支持复杂输出：** 一些模型可能返回多个值或构造化的数据（例如不同类别的概率）。构造化的格式可以轻松处理这种情况。

### 使用JSON作为响应格式

Web API中构造数据最常用的格式是JSON (JavaScript 对象表示法)。它轻量、易于阅读（大部分情况下），并且几乎所有用于Web开发的编程语言和工具都可以轻松解析。

Python对JSON有优秀的内置支持，Flask提供了一个方便的辅助函数 `jsonify`，用于将Python字典转换为符合规范的JSON HTTP响应。

### 构建JSON响应

假设你的模型的 `predict()` 方法返回预测结果，可能是一个单一值或一个列表/数组。在你的Flask路由处理函数中（即带有 `@app.route(...)` 装饰器的函数），获取预测结果后，你通常会创建一个Python字典，包含结果以及任何你想发回的其他信息。

```python
# Flask路由处理函数内的示例

# ... (加载模型和获取输入数据的代码) ...

# 假设 'prediction_result' 保存了 model.predict() 的输出
# 例如，prediction_result 可能是 [1] 或 ['spam'] 或 [0.85]
prediction_result = model.predict(processed_input_data)

# 通常好的做法是提取主要的预测值
# 这取决于你模型的输出格式
# 如果 prediction_result 是一个只包含一个元素的 NumPy 数组或列表：
final_prediction = prediction_result[0] 

# 为响应创建一个 Python 字典
response_data = {
    'prediction': final_prediction,
    'model_version': 'v1.0' # 示例额外信息
}

# 使用 Flask 的 jsonify 创建 JSON 响应
# 这也会设置正确的 Content-Type 头部（application/json）
return jsonify(response_data)
```

在此示例中：

1. 我们从模型获取 `prediction_result`。
2. 我们可能将核心预测值提取到 `final_prediction` 中。你如何做到这一点完全取决于你的 `model.predict()` 函数返回什么。如果它直接返回一个单一值，你可能不需要这一步。
3. 我们创建一个Python字典 `response_data`。我们使用一个描述性键 `'prediction'` 来存放实际的预测值。我们可以根据需要添加其他键值对，例如 `'model_version'`。使用清晰的键使API更易于理解和使用。
4. `jsonify(response_data)` 接收该字典，将其转换为JSON字符串（例如，`{"prediction": 1, "model_version": "v1.0"}`），并将其封装在一个Flask响应对象中，附带正确的 `Content-Type: application/json` HTTP头部。这会告知客户端应用，响应体中包含JSON数据。

### 考虑HTTP状态码

除了JSON响应体，标准的HTTP响应还包括一个状态码。默认情况下，如果路由处理函数成功返回一个值，Flask会返回 `200 OK`。这通常是成功预测的正确代码。如果出现问题（例如，无效输入数据、模型加载失败），你可能返回不同的状态码（如 `400 Bad Request` 或 `500 Internal Server Error`），并在JSON响应体中附带错误消息。我们在这里不会深入探讨错误处理，但将其纳入考虑是好的做法。

目前，我们专注于成功情况，使用 `jsonify` 足以以一种良好构造的、标准格式返回模型的预测结果，客户端应用可以轻松使用这种格式。下一节将提供一个动手实践的例子，结合了模型加载、输入处理、进行预测以及返回JSON响应。

## 参考资料

- [The JSON Data Interchange Format](https://www.ecma-international.org/publications-and-standards/standards/ecma-404/) — Douglas Crockford (2017)
  Publisher: Ecma International
  定义了JavaScript对象表示法（JSON）数据交换格式，对结构化API响应至关重要。
- [Designing Web APIs: Building Readily Adopted APIs](https://www.oreilly.com/library/view/the-design-of-web-apis/9781617295102/) — Arnaud Lauret (2019)
  Publisher: Manning Publications; Pages: 392
  提供了API设计原则、使用JSON进行响应结构化以及恰当使用HTTP状态码的指导。
- [Building Machine Learning Powered Applications: Going from Idea to Product](https://www.abebooks.com/9781492045113/Building-Machine-Learning-Powered-Applications-Idea-149204511X/plp) — Emmanuel Ameisen (2020)
  Publisher: O'Reilly Media
  关于部署机器学习模型的指南，涵盖API端点创建和有效的预测返回策略。

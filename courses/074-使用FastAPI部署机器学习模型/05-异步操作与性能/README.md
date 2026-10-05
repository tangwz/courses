# 第 5 章：异步操作与性能

来源：[原章节](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-5-async-operations-performance)

[返回课程目录](../README.md)

FastAPI 基于异步原理构建，使其能够同时高效地处理大量连接。本章侧重于将这些原理应用到您的机器学习API。您将学习如何使用 `async` 和 `await` 定义异步路由处理程序，并了解它们在何处提供最大的优势，通常在I/O密集型预处理或后处理步骤中。

我们将解决一个常见问题：如何将CPU密集型任务（如模型推理）整合到异步应用程序中，并介绍避免阻塞服务器事件循环的方案，例如使用 `run_in_threadpool` 等方法。此外，您还将学习实现后台任务，用于在响应发送后可以执行的操作，例如记录详细的预测结果或发送通知。本章最后将审视专门针对通过API提供机器学习模型服务时的性能要点，确保您的应用程序在负载下保持响应性。

## 小节

- 1. [理解 FastAPI 路由中的 async 和 await](01-%E7%90%86%E8%A7%A3%20FastAPI%20%E8%B7%AF%E7%94%B1%E4%B8%AD%E7%9A%84%20async%20%E5%92%8C%20await.md)
- 2. [机器学习推理何时适用异步](02-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%8E%A8%E7%90%86%E4%BD%95%E6%97%B6%E9%80%82%E7%94%A8%E5%BC%82%E6%AD%A5.md)
- 3. [运行阻塞型机器学习操作](03-%E8%BF%90%E8%A1%8C%E9%98%BB%E5%A1%9E%E5%9E%8B%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%93%8D%E4%BD%9C.md)
- 4. [使用后台任务](04-%E4%BD%BF%E7%94%A8%E5%90%8E%E5%8F%B0%E4%BB%BB%E5%8A%A1.md)
- 5. [ML I/O 异步请求的优势](05-ML%20I-O%20%E5%BC%82%E6%AD%A5%E8%AF%B7%E6%B1%82%E7%9A%84%E4%BC%98%E5%8A%BF.md)
- 6. [API 端点性能考量](06-API%20%E7%AB%AF%E7%82%B9%E6%80%A7%E8%83%BD%E8%80%83%E9%87%8F.md)
- 7. [实践：实现异步操作](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%BC%82%E6%AD%A5%E6%93%8D%E4%BD%9C.md)

章节测验：[在线测验](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-5-async-operations-performance/quiz)

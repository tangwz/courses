---
course: "fastapi-ml-deployment"
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-5-async-operations-performance"
sourceId: 983
chapter: "async-operations-performance"
title: "异步操作与性能"
order: 5
description: "学习如何使用 async/await、后台任务，并理解在FastAPI中用于机器学习推理API的性能影响。"
hasQuiz: true
---

FastAPI 基于异步原理构建，使其能够同时高效地处理大量连接。本章侧重于将这些原理应用到您的机器学习API。您将学习如何使用 `async` 和 `await` 定义异步路由处理程序，并了解它们在何处提供最大的优势，通常在I/O密集型预处理或后处理步骤中。

我们将解决一个常见问题：如何将CPU密集型任务（如模型推理）整合到异步应用程序中，并介绍避免阻塞服务器事件循环的方案，例如使用 `run_in_threadpool` 等方法。此外，您还将学习实现后台任务，用于在响应发送后可以执行的操作，例如记录详细的预测结果或发送通知。本章最后将审视专门针对通过API提供机器学习模型服务时的性能要点，确保您的应用程序在负载下保持响应性。

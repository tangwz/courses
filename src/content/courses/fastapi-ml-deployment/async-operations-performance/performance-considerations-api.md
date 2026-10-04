---
course: "fastapi-ml-deployment"
chapter: "async-operations-performance"
lesson: "performance-considerations-api"
sourceId: 5332
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-5-async-operations-performance/performance-considerations-api"
title: "API 端点性能考量"
description: "影响 API 性能的因素，包括模型加载时间、推理速度和网络延迟。"
order: 6
plots: ["plots/5332-0.json"]
sourceHash: "099253b06cf1bad30e623ba43ad38c5c557f3e85846825317175f9f620efc544"
sourceCorrections: []
---

构建响应迅速、高效的机器学习 (machine learning) API 需要认真考虑性能问题。尽管 FastAPI 的异步能力有助于有效管理并发连接，尤其是在 I/O 密集型任务中，但机器学习工作负载的独特要求会带来特定的瓶颈。了解这些因素对于优化您的预测端点非常重要。

### 模型加载时间

在您的 API 进行预测之前，机器学习 (machine learning)模型必须加载到内存中。这可能是一个耗时的步骤，特别是对于大型或复杂的模型。

- **启动加载：** 在 FastAPI 应用程序启动时加载模型（例如，在全局变量中或使用带 `yield` 的依赖项）会增加初始启动时间，但能确保模型已为第一个请求做好准备。这通常是生产环境的首选方法，以避免用户初次交互时的延迟。
- **按需加载：** 仅当第一个预测请求到达时才加载模型可以减少启动时间，但会给第一个请求带来显著的延迟。这在开发或低流量场景中可能是可接受的。
- **内存消耗：** 大型模型消耗更多内存。请确保您的部署环境有足够的 RAM 来容纳模型，而不会影响其他进程或导致过度交换。

根据您的特定模型和应用程序需求，权衡启动时间、内存使用和首个请求延迟。

### 推理 (inference)延迟

已加载模型处理输入数据并生成预测所需的时间就是推理延迟。这通常是机器学习 (machine learning) API 中最主要的性能瓶颈。

- **CPU 密集型：** 大多数传统机器学习模型推理操作（如深度学习 (deep learning)中的矩阵乘法或集成方法中的树遍历）都是计算密集型的，并且是 CPU 密集型的。如前所述，直接在 `async` 函数中运行这些操作会阻塞事件循环。使用 `run_in_threadpool` 对于卸载这些任务来说非常必要，但推理本身仍需要时间。
- **模型复杂性：** 更复杂的模型（例如，更深层的神经网络 (neural network)、更大的集成模型）通常具有更高的推理延迟。
- **输入数据大小：** 处理更大的输入数据负载（例如，高分辨率图像、长文本 (long context)序列）自然需要更长时间。
- **批处理：** 某些模型和机器学习框架支持批处理，即同时处理多个输入样本。如果您的 API 预计会并发接收多个预测请求，那么在将输入发送到模型之前进行批处理（通常在线程池中处理）可以显著提高吞吐量 (throughput)，尽管这可能会稍微增加批处理中单个请求的延迟。
- **硬件加速：** 尽管这本身属于 FastAPI 配置的范围，但底层硬件（CPU 速度、GPU/TPU 可用性）会显著影响推理速度。您的部署策略应考虑硬件要求，以实现可接受的性能。

### 数据预处理和后处理

原始输入数据在送入模型之前通常需要转换，模型输出在返回给客户端之前可能需要格式化。

- **I/O 密集型步骤：** 如果预处理涉及从外部源（数据库、其他 API）获取数据，或后处理涉及保存结果，这些都是 I/O 密集型操作。对这些步骤使用 `async` 和 `await` 对性能非常有益，它允许服务器在等待时处理其他请求。
- **CPU 密集型步骤：** 复杂的特征工程、图像转换或文本分词 (tokenization)可能是 CPU 密集型的。与推理 (inference)类似，这里的繁重计算步骤应谨慎处理，如果它们有可能长时间阻塞事件循环，可以考虑使用 `run_in_threadpool`。
- **Pydantic 验证：** 尽管 Pydantic 提供了非常有用的数据验证功能，但解析和验证复杂的输入/输出模型会给每个请求增加少量开销。对于极高吞吐量 (throughput)的场景，Pydantic 模型的结构和复杂性可能会成为一个次要的性能影响因素。

### 网络延迟和负载大小

数据在客户端和您的 API 服务器之间传输所需的时间，以及序列化/反序列化数据所花费的时间，都会影响整体感知性能。

- **负载大小：** 发送大型输入特征（例如，base64 编码图像）或接收大型预测输出会增加网络传输时间。考虑高效的数据表示形式，并考虑客户端是否确实需要所有返回的数据。
- **序列化/反序列化：** FastAPI 和 Pydantic 可以高效地处理 JSON 序列化/反序列化。然而，对于非常大或复杂的嵌套对象，此过程仍会消耗 CPU 周期并增加请求/响应时间。
- **地理位置：** 将您的 API 部署到更靠近用户的位置可以减少网络延迟。内容分发网络 (CDN) 也可以帮助缓存静态资产（如果适用），尽管这通常与动态预测端点关系不大。

### 并发和服务器资源

FastAPI 的异步特性使其能够高效处理许多并发连接，但在负载下的性能取决于阻塞任务的管理方式和可用的服务器资源。

- **事件循环阻塞：** 正如本章反复强调的，使用长时间运行的同步代码（如没有 `run_in_threadpool` 的 CPU 密集型推理 (inference)）阻塞事件循环是异步框架的主要性能杀手。它会阻止服务器处理其他传入请求。
- **线程池大小：** 使用 `run_in_threadpool` 时，阻塞任务的性能取决于底层线程池的大小。过小的线程池会导致请求排队等待空闲线程。默认大小可能需要根据预期负载和阻塞任务的持续时间进行调整。这通常由 ASGI 服务器（如 Uvicorn）管理。
- **服务器工作进程：** 对于生产部署，您通常使用 Uvicorn 等 ASGI 服务器运行 FastAPI，并通常由 Gunicorn 等进程管理器进行管理。运行多个工作进程（通常与 CPU 核心数量相关）可以实现请求的真正并行处理，使您的容量倍增，超出单个事件循环（即使有线程池）所能处理的范围。正确配置工作进程数量对于扩展非常必要。

下图展示了单个机器学习 (machine learning) API 请求的延迟分解，并标示了潜在的瓶颈：



![机器学习 API 请求延迟分解](plots/5332-0.json)



> 机器学习 API 请求在不同阶段所花费的时间分解。模型推理通常占据主导，但预处理、网络和序列化也贡献较大。

优化机器学习 API 包括确定特定用例中造成延迟的最大因素，并采用适当的策略，无论是对 I/O 使用 `async`，对 CPU 密集型任务使用 `run_in_threadpool`，优化模型本身，管理数据大小，还是扩展服务器资源。持续监控和性能分析对于查明生产环境中的瓶颈非常重要。

## 参考资料

- [Concurrency and async/await](https://fastapi.tiangolo.com/async/) — Sebastián Ramírez (2024)
  详细说明FastAPI如何利用异步编程、事件循环以及`run_in_threadpool`来有效管理并发的I/O密集型和CPU密集型任务。
- [Designing Machine Learning Systems: New Ways of Thinking About Data, Programs, Experiments, and Products](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  探讨构建生产级机器学习系统的架构模式和工程实践，其中包含与模型服务性能、可伸缩性和资源管理相关的内容。
- [Fluent Python, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056340/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  深入解释Python的并发模型，包括`asyncio`、协程和事件循环，这对优化像FastAPI这样的异步应用程序至关重要。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://learning.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  深入理解构建可靠和可伸缩数据系统背后的基本原理，包括对延迟、吞吐量、序列化和分布式系统问题的详细讨论，这些对高性能API至关重要。

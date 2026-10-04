---
course: "deploying-diffusion-models-scale"
chapter: "building-scalable-inference-apis"
lesson: "handling-long-running-tasks"
sourceId: 5435
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-4-building-scalable-inference-apis/handling-long-running-tasks"
title: "处理长时间运行的生成任务"
description: "实施Webhook或轮询等策略进行异步API操作。"
order: 2
plots: []
sourceHash: "8be01ee56aa1bc2fe23858d27420644a52e1421dbbf693eadbde7e2dea696e2f"
sourceCorrections: []
---

扩散模型推理 (inference)，特别是高分辨率图像生成，是计算密集型任务。与通常在毫秒内完成的典型网络请求不同，生成一张图像可能需要几秒、几十秒，甚至几分钟，这取决于模型大小、扩散步数（推理质量）和可用硬件（特别是GPU）。这种固有的延迟在通过标准同步API将这些模型集成到交互式应用或后端服务时，构成了一项重要挑战。

同步API请求通常要求客户端在等待服务器响应时保持开放连接。大多数网络服务器、负载均衡器和客户端都有默认的超时时间（通常为30-60秒）。如果图像生成过程超过此超时时间，连接就会中断，导致错误和糟糕的用户体验。此外，在等待长时间的GPU计算时保持连接开放会不必要地占用API服务器资源，限制其有效处理新传入请求的能力。

为了解决此问题，我们必须采用异步处理模式。API不会让客户端等待完整的生成过程，而是立即确认请求，为其分配一个唯一标识符，并将此标识符返回给客户端。实际计算在后台进行，与初始客户端交互分离。客户端随后可以使用该标识符检查任务状态或在完成后收到通知。

我们来审视处理这些长时间运行任务的常用模式：

### 轮询

轮询可能是从客户端角度来看最简单的异步模式。

1. **请求：** 客户端发送图像生成请求（例如，`POST /generate`），包含必要的参数 (parameter)（提示词 (prompt)、设置等）。
2. **确认与任务ID：** API服务器立即验证请求，在后台系统（稍后详细介绍队列）中将生成任务入队，生成一个唯一的`job_id`，并返回包含`job_id`和可能的状态查询URL（例如，`/jobs/{job_id}/status`）的响应（例如，`202 Accepted`）。
3. **状态检查：** 客户端定期向状态端点发送请求（例如，`GET /jobs/{job_id}/status`）。
4. **响应：** 状态端点返回任务的当前状态（`PENDING`待处理、`PROCESSING`处理中、`COMPLETED`已完成、`FAILED`失败）。如果`COMPLETED`，响应包含生成结果的位置（例如，图像的URL），如果结果足够小，也可以包含结果本身。如果`FAILED`，则包含错误详情。
5. **重复：** 客户端持续轮询，直到状态为`COMPLETED`或`FAILED`。

> 此图展示了客户端轮询机制。客户端启动任务，接收任务ID，并重复查询状态端点，直到后台任务完成。

**优点：**

- 客户端逻辑相对简单。
- 在标准HTTP上运行良好。
- 保持了无状态API设计。

**缺点：**

- **低效：** 可能产生大量状态检查请求，占用网络带宽和服务器资源。
- **延迟：** 客户端仅在任务完成后下次轮询时才得知完成，可能引入获取结果的延迟。
- **轮询间隔的权衡：** 短间隔会增加负载；长间隔会增加感知延迟。

### Webhook

Webhook（或回调）反转了通知方向。服务器在任务完成时告知客户端，而不是客户端主动询问。

1. **请求：** 客户端发送生成请求（`POST /generate`），并在有效负载中包含`callback_url`。此URL是客户端应用托管的端点，能够接收HTTP POST请求。
2. **确认与任务ID：** 与轮询类似，API服务器验证并入队任务，生成`job_id`，存储与任务关联的`callback_url`，并返回带有`job_id`的`202 Accepted`。
3. **后台处理：** 任务在后台进行处理。
4. **通知：** 完成（或失败）后，后台工作器或专门的通知服务向客户端提供的`callback_url`发起HTTP POST请求。此请求的有效负载包含`job_id`、最终状态（`COMPLETED`或`FAILED`），以及结果（或指向结果的链接）或错误详情。

> 此图展现了webhook模式。客户端提供回调URL；当后台任务完成时，服务器直接通知此URL。

**优点：**

- **高效：** 无需轮询；通知是事件驱动的，任务完成后近乎实时。
- 与轮询相比，减少了不必要的网络流量和服务器负载。

**缺点：**

- **客户端复杂度：** 客户端必须对外暴露一个可公开访问的HTTP端点以接收回调。这可能因防火墙、NAT和安全考量而变得复杂。
- **可靠性：** API服务器需要有稳健的机制来处理回调失败（例如，客户端端点宕机、网络错误）。带退避的重试是必不可少的。
- **安全性：** 回调端点必须经过保护以防未经授权的请求。方法包括使用签名请求（例如，HMAC）或秘密令牌。

### WebSocket

WebSocket通过单个TCP连接在客户端和服务器之间提供持久、全双工的通信通道。这非常适合实时更新。

1. **连接：** 客户端与API服务器建立WebSocket连接。
2. **请求：** 客户端通过已建立的WebSocket连接发送生成请求消息。
3. **确认与任务ID：** 服务器通过WebSocket发回确认消息，可能包含`job_id`。
4. **后台处理：** 任务异步处理。
5. **更新与完成：** 服务器可以在状态更新（`PROCESSING`处理中、进度百分比等）和最终结果（或失败通知）发生时，立即通过持久WebSocket连接直接推送给客户端。

**优点：**

- **实时性：** 通知延迟最低，适合生成过程中细粒度的进度更新。
- **高效：** 一旦连接建立，可避免多次请求的HTTP开销。

**缺点：**

- **有状态连接：** 服务器需要管理大量潜在的持久WebSocket连接，这比无状态HTTP请求消耗更多内存资源。
- **基础设施复杂性：** 相比标准HTTP，需要支持WebSocket的服务器以及可能不同的负载均衡策略（例如，粘性会话）。
- **扩展性挑战：** 管理大量持久连接在可靠扩展方面可能比无状态HTTP架构更复杂。
- **客户端支持：** 尽管得到广泛支持，但WebSocket在客户端的实现可能比简单的HTTP轮询略复杂。

### 运用消息队列

无论客户端通知模式（轮询或Webhook）如何，消息队列（MQ）几乎总是实现后端可靠异步处理的必要组成部分。例如RabbitMQ、Apache Kafka、AWS SQS、Google Cloud Pub/Sub和Azure Service Bus。

> 此架构使用消息队列将API服务器与后端GPU工作器解耦。这允许独立伸缩并提高韧性。客户端可以使用轮询或Webhook（通过通知服务）来获取结果。

其工作方式如下：

1. API服务器接收请求。
2. 它进行快速验证，然后将包含任务详情（提示词 (prompt)、参数 (parameter)、如果提供则包含回调URL、任务ID）的消息发布到消息队列中。
3. 它更新一个状态数据库（例如，Redis、DynamoDB、PostgreSQL），表明任务处于`PENDING`待处理状态。
4. 它立即将`job_id`（以及用于轮询的状态URL）返回给客户端。
5. 一个单独的工作器实例池（运行扩散模型，通常在GPU硬件上）监听队列。
6. 当工作器空闲时，它从队列中拉取一条消息（任务）。
7. 工作器将任务状态更新为`PROCESSING`处理中。
8. 它执行计算密集型的图像生成。
9. 完成后，它将结果上传到持久存储（如云对象存储）。
10. 它将任务状态更新为`COMPLETED`已完成（或`FAILED`失败），并存储结果位置或错误详情。
11. 如果请求了Webhook，工作器（或由状态更新触发的独立通知服务）发送回调。

这种解耦架构提供了：

- **扩展性：** 您可以根据负载独立扩展API服务器和工作器实例的数量。如果队列增长，就增加更多工作器。
- **韧性：** 如果工作器在处理期间崩溃，消息通常可以返回到队列（取决于配置和确认机制），由另一个工作器接收。如果API服务器重启，队列中已有的任务不受影响。
- **缓冲：** 队列充当缓冲区，吸收暂时的请求高峰，而不会使工作器超载。

### 选择合适的方法

- **轮询：** 最适合那些不需要近乎实时通知且客户端不易托管Webhook端点的情况。适用于内部工具或周期性检查可接受的简单网页前端。
- **Webhook：** 服务器到服务器集成或需要即时通知且客户端能可靠托管端点的应用。比轮询更高效。
- **WebSocket：** 最适合需要实时更新的高度交互式应用，包括生成过程中的潜在进度反馈。会带来更高的基础设施复杂性。

实际中，许多大型系统同时提供轮询（通过状态端点）和可选的Webhook，允许客户端选择最适合其需求的方法。在任一模式的底层，消息队列系统都是可靠且可扩展地管理实际后台工作的标准。有效处理长时间运行的任务不仅仅是选择一种通知机制；它关乎设计一个韧性好、解耦的后端架构。

## 参考资料

- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781449373320/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  本书全面讨论了分布式系统概念，包括异步处理、消息队列以及构建可扩展和可靠应用程序的架构模式。
- [What is Amazon Simple Queue Service (Amazon SQS)?](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/welcome.html) — Amazon Web Services (2024)
  一种广泛使用的消息队列服务的官方文档，详细介绍了其功能、用法以及如何实现解耦和可扩展的架构。
- [WebSockets](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API) — MDN Contributors (2024)
  WebSockets API的权威指南，解释了其协议、建立过程以及在Web应用程序中用于实时双向通信。

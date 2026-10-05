# 生成式模型的API设计模式

来源：[原文](https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-4-building-scalable-inference-apis/api-design-generative-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

部署扩散模型不仅仅是运行推理 (inference)代码。它需要通过稳定、可预测且高性能的应用程序编程接口（API）来提供模型的生成能力。扩散模型带来了一些特殊问题，例如推理时间可能很长（几秒到几分钟），且计算步骤密集，因此需要细致的API设计。专门针对图像生成等生成任务的API常见构建方式进行分析。

选择合适的API方法并恰当组织请求和响应，是影响可伸缩性、客户端集成及整体系统可维护性的基础性决定。我们将考察REST和gRPC等标准Web API方法如何适应扩散模型推理的特殊要求。

### 主要API模式：REST对比gRPC

当前构建Web API的两种主要模式是表征性状态传输（REST）和gRPC。两者在生成模型服务中均有其优点。

#### RESTful API

REST因其简洁性、无状态性以及对标准HTTP方法（GET、POST、PUT、DELETE）的依赖，依然是Web API广泛采用的标准。对于生成任务，RESTful方法通常包含以下内容：

- **端点：** 定义特定URI以执行操作。常见的方式是向`/generate`或`/images`等端点发送POST请求，以启动生成任务。状态检查可能使用GET请求到`/status/{job_id}`或`/jobs/{job_id}`。
- **请求体（POST）：** 通常是JSON格式，包含文本提示、负面提示、采样步数、指导比例（CFG）、随机种子、所需尺寸等参数 (parameter)，以及可能的高级选项，如ControlNet输入或LoRA配置。

  ```json
  {
    "prompt": "A photorealistic cat astronaut exploring Mars",
    "negative_prompt": "cartoon, drawing, illustration, sketch, low quality",
    "steps": 30,
    "cfg_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "seed": 12345
  }
  ```
- **响应体：** 根据操作是同步还是异步（稍后详细讨论），响应会有所不同。
  - **同步（仅限快速推理 (inference)）：** 可能直接返回图像数据（例如Base64编码）或临时URL。对于标准扩散模型，由于延迟，这通常不切实际。
  - **异步（推荐）：** 通常立即返回一个任务标识符。

    ```json
    {
      "job_id": "a7f3b1c9-e4d8-4bfa-8a1e-7d0c9e1a2b3d",
      "status": "queued"
    }
    ```

    客户端随后轮询`/status/{job_id}`端点，以检查进度并在准备就绪后获取最终结果（例如图像URL）。

#### gRPC API

由Google开发的gRPC使用HTTP/2作为传输协议，并以Protocol Buffers（protobuf）作为其接口定义语言（IDL）和消息交换格式。其在扩散模型服务方面的潜在优势包括：

- **性能：** HTTP/2提供多路复用和头部压缩等功能。Protobuf序列化通常比JSON更快，并产生更小的负载。
- **流式传输：** gRPC原生支持双向流式传输，这可能用于扩散采样步骤中的中间进度更新，尽管这会增加复杂性。
- **强类型：** 使用`.proto`文件定义服务和消息提供了强类型，可以生成更好的代码并减少不同客户端/服务器语言之间的集成错误。

一个gRPC服务定义可能如下所示（简化版）：

```protobuf
syntax = "proto3";

package diffusion.v1;

service DiffusionService {
  rpc GenerateImage(GenerateImageRequest) returns (GenerateImageResponse);
  rpc GetJobStatus(GetJobStatusRequest) returns (JobStatusResponse);
}

message GenerateImageRequest {
  string prompt = 1;
  string negative_prompt = 2;
  int32 steps = 3;
  float cfg_scale = 4;
  int32 width = 5;
  int32 height = 6;
  optional int64 seed = 7;
}

message GenerateImageResponse {
  string job_id = 1;
}

// ... 其他用于状态请求/响应的消息定义
```

尽管gRPC能带来性能优势，但REST/JSON通常更易于实现、调试并与现有Web基础设施和工具集成。选择取决于具体的性能要求、团队专业知识和期望的生态系统兼容性。

### 复杂输入的结构化

扩散模型除了简单的文本提示外，通常接受多种参数 (parameter)。控制信号（如深度图、姿态或ControlNet的Canny边缘）、带权重 (weight)的多个提示、LoRA标识符和采样器选择都增加了复杂性。

- **嵌套JSON (REST)：** 使用嵌套JSON对象对相关参数进行逻辑分组。例如，ControlNet输入可以是对象数组，每个对象指定一个控制图像和类型。
- **Protobuf消息 (gRPC)：** 为复杂输入定义特定消息类型，凭借Protobuf的结构和类型安全。

输入验证非常重要。API层在将请求排队给模型工作器之前，应严格验证所有传入参数（类型、范围、允许值）。这能避免格式不正确的请求占用宝贵的计算资源。

### 管理输出负载：图像和元数据

在API响应中直接返回生成的图像会带来问题。图像可能很大（兆字节），尤其是在高分辨率下。将Base64编码的图像直接嵌入 (embedding)JSON响应会显著增加负载大小，并可能耗尽网络带宽和客户端内存。



[交互图表：API响应负载大小对比](https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-4-building-scalable-inference-apis/api-design-generative-models#plot-bkl7qe)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "API响应负载大小对比",
    "xaxis": {
      "title": "输出方法"
    },
    "yaxis": {
      "title": "大致负载大小 (MB)"
    },
    "barmode": "group"
  },
  "data": [
    {
      "type": "bar",
      "name": "1024x1024 PNG (无损)",
      "x": [
        "Base64 in JSON",
        "URL/Job ID"
      ],
      "y": [
        4.5,
        0.001
      ],
      "marker": {
        "color": "#4dabf7"
      }
    },
    {
      "type": "bar",
      "name": "1024x1024 JPG (质量90)",
      "x": [
        "Base64 in JSON",
        "URL/Job ID"
      ],
      "y": [
        0.8,
        0.001
      ],
      "marker": {
        "color": "#3bc9db"
      }
    }
  ]
}
```

</details>



> 返回图像数据（Base64编码）与仅返回URL或任务ID时，API响应负载的大致大小对比。直接嵌入会显著增加大小。

首选模式，尤其对于异步操作，是：

1. 让生成工作器将完成的图像保存到持久对象存储服务（例如，AWS S3、Google Cloud Storage、Azure Blob Storage）。
2. 返回以下其中一项：
   - 指向已存储图像的**预签名URL**。此URL授予临时访问权限，客户端无需复杂的身份验证。
   - **任务ID**，要求客户端向状态端点发出单独请求，一旦任务完成，该端点将提供图像URL。

这种方法将图像存储和检索与主要API请求流程分离，使API响应保持轻量级，并利用可伸缩的云存储。

> 示意图展示了使用队列和独立状态端点的典型异步图像生成API流程。

### 幂等性

网络问题可能导致客户端重试API请求。幂等API确保多次发起相同请求会产生与一次请求相同的效果（或状态变化）。对于生成API，这可以防止意外的重复图像生成和费用产生。实现幂等性的方法有：

- 允许客户端在其POST请求中提供唯一的`request_id`或`idempotency_key`。
- 在服务器端存储最近的请求ID（例如，在Redis或数据库中）。
- 在处理新请求之前，检查所提供的ID是否近期已出现。如果是，则返回之前的结果（或任务ID），而不是启动新的生成。

### 版本控制

模型、参数 (parameter)和API契约会发展变化。从一开始就实施API版本控制策略对于平稳管理变更很重要。常见方法包括：

- **URL路径版本控制：** `/v1/generate`, `/v2/generate`
- **Header版本控制：** 使用自定义请求头，如`X-API-Version: 2`
- **查询参数版本控制：** `/generate?version=2` （对于重大更改较不常见）

路径版本控制通常是最清晰且最广为人知的方法。版本控制使您能够在`v2`中引入新功能或破坏性变更，同时保持对仍使用`v1`的客户端的兼容性。

细致设计API契约是重要的一步。通过考虑REST或gRPC方法，有效组织输入和输出，处理大型负载，确保幂等性，并规划版本控制，您可以为您的扩散模型部署创建可靠且可伸缩的接口。接下来的章节将在此基础上进行论述，考察如何处理长时间运行任务的异步特性以及如何实现请求队列等支持性基础设施。

## 参考资料

- [Building Microservices: Designing Fine-Grained Systems](https://www.oreilly.com/library/view/building-microservices-2nd/9781492034018/) — Sam Newman (2021)
  Publisher: O'Reilly Media; Pages: 612
  微服务架构的实用指南，涵盖分布式系统的API设计模式（REST、异步通信）、幂等性和版本控制策略。
- [gRPC Concepts](https://grpc.io/docs/what-is-grpc/core-concepts/) — The gRPC Authors (2024)
  gRPC的官方文档，提供gRPC、其架构、Protocol Buffers和通信模式的基础信息，是gRPC API开发的主要资源。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  本书全面考察数据密集型系统，讨论异步处理、消息队列和持久存储等主题，这些对于构建可扩展的生成模型API至关重要。

---

[上一节](../03-%E5%8F%AF%E6%89%A9%E5%B1%95%E9%83%A8%E7%BD%B2%E6%9E%B6%E6%9E%84/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%9C%A8%20Kubernetes%20%E4%B8%8A%E9%83%A8%E7%BD%B2.md) · [下一节](02-%E5%A4%84%E7%90%86%E9%95%BF%E6%97%B6%E9%97%B4%E8%BF%90%E8%A1%8C%E7%9A%84%E7%94%9F%E6%88%90%E4%BB%BB%E5%8A%A1.md)

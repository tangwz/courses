# NVIDIA Triton 推理服务器的多模型服务

来源：[原文](https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-4-high-performance-model-inference/serving-models-triton-server)

[返回章节目录](README.md) · [返回课程目录](../README.md)

高效地提供机器学习 (machine learning)模型服务是一个重大的工程难题。模型通常使用 TensorRT 或 ONNX Runtime 等工具进行优化，但高效服务仍然是一个独立的挑战。生产环境通常需要管理多个不同模型，而不仅仅是单个模型。将每个模型部署为独立的隔离服务可能会导致很大的运营开销和较低的资源使用率，尤其是在昂贵的 GPU 硬件上。NVIDIA Triton 推理 (inference)服务器正是为应对这一难题而设计的，它提供了一个统一的服务解决方案，能够同时承载来自不同框架的多个模型。

Triton 作为独立的服务器进程运行，它从指定的模型库加载模型，并通过 HTTP/gRPC 端点提供这些模型的访问。其优势在于其灵活的架构，旨在复杂的、多模型场景下最大化吞吐量 (throughput)和硬件使用率。

### 模型库

Triton 的主要组织结构是其模型库。这是一个简单的文件系统目录，Triton 会监控此目录以查找模型。要添加、删除或更新模型，只需修改此目录的内容。Triton 会自动检测并加载这些更改，无需重启服务器。

每个模型都必须有自己的子目录，其中包含模型文件和名为 `config.pbtxt` 的配置文件。这种结构是标准化的，使得 Triton 能够以统一的方式管理来自完全不同框架的模型。

考虑一个包含两个不同模型的库：一个以 ONNX 格式保存的 DenseNet 计算机视觉模型和一个使用 PyTorch 后端的 BERT 语言模型。目录结构如下所示：

```text
/path/to/model_repository/
├── densenet_onnx/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── bert_pytorch/
    ├── config.pbtxt
    └── 1/
        └── model.pt
```

编号的子目录（例如 `1/`）代表一个模型版本，允许原子更新和版本化发布。`config.pbtxt` 文件用于定义模型的元数据、输入、输出，以及最重要的是其执行和优化设置。

### GPU 动态批处理以提高使用率

推理 (inference)服务中一个重要的低效来源是硬件使用不足。一个现代神经网络 (neural network)的单次推理请求可能只占用 GPU 几毫秒，导致处理器在其余时间处于空闲状态。逐个发送请求无法发挥 GPU 很强的并行能力。

Triton 的动态批处理器通过在执行前透明地将传入的单个推理请求分组为更大的批次来解决此问题。此过程对客户端是不可见的，客户端仍然发送一个请求并接收一个响应。通过处理更大的批次，GPU 可以更高效地执行计算，大幅提升整体吞吐量 (throughput)。

> 多个独立的请求被动态批处理器拦截，组合成一个批次，然后发送到 GPU 进行并行处理。

你可以在模型的 `config.pbtxt` 文件中启用和配置此功能。主要参数 (parameter)是 `max_batch_size`，它定义了要分组的最大请求数；以及 `max_queue_delay_microseconds`，它设置了服务器等待填充批次的最长时间。这会产生一个权衡：更长的延迟允许形成更大、更高效的批次，但会增加单个请求的延迟。

这里是 `densenet_onnx` 模型启用动态批处理的配置片段：

```protobuf
name: "densenet_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 64

dynamic_batching {
  preferred_batch_size: [16, 32]
  max_queue_delay_microseconds: 5000
}

input [
  {
    name: "input_image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
# ... 输出配置
```

在此示例中，Triton 将尝试创建 16 或 32 大小的批次，但如果请求快速到达，它将形成最大 64 大小的批次。它将等待不超过 5 毫秒（5000 微秒），然后将队列中的所有内容发送到模型进行执行。

### 模型集成与业务逻辑管道

通常，业务应用需要的不仅仅是单次模型调用。一种常见模式涉及一个管道：

1. **预处理：** 将原始输入（如 JPEG 图像）转换为数值张量。
2. **推理 (inference)：** 在张量上运行主要模型。
3. **后处理：** 将模型的输出张量转换为人类可读的格式（如带有类别标签的 JSON）。

在客户端实现此逻辑会增加复杂性，并在每个步骤之间引入网络延迟。Triton 的**集成调度器**允许你在服务器上将整个管道定义为单个“集成”模型。客户端对集成模型发出一个请求，Triton 则处理组成模型之间的数据内部路由。

> 客户端向单个集成端点发送原始图像。Triton 在内部将数据导向预处理、推理和后处理模型，然后返回最终结果。

这在集成模型的 `config.pbtxt` 中配置。你需要将平台指定为 `ensemble`，并在 `step` 块中定义数据流。

```protobuf
name: "image_classifier_pipeline"
platform: "ensemble"
max_batch_size: 64

input [
  {
    name: "RAW_IMAGE"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }
]
output [
  {
    name: "PREDICTIONS"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocessor"
      model_version: -1
      input_map {
        "IMAGE_IN"
        value: "RAW_IMAGE"
      }
      output_map {
        "important": "TENSOR_OUT"
        value: "preprocessed_tensor"
      }
    },
    {
      model_name: "resnet_tensorrt"
      model_version: -1
      input_map {
        key: "INPUT__0"
        value: "preprocessed_tensor"
      }
      output_map {
        key: "OUTPUT__0"
        value: "logit_tensor"
      }
    },
    {
      model_name: "postprocessor"
      model_version: -1
      input_map {
        "LOGITS_IN"
        value: "logit_tensor"
      }
      output_map {
        "LABELS_OUT"
        value: "PREDICTIONS"
      }
    }
  ]
}
```

此配置定义了一个三步管道。每个步骤的 `input_map` 和 `output_map` 声明了张量如何从一个模型传递到下一个模型。中间张量（`preprocessed_tensor`、`logit_tensor`）仅存在于 Triton 内部，从而最小化数据移动并简化客户端代码。

### 并发执行与资源管理

通过将多个独立的模型加载到单个 Triton 实例中，可以有效地共享硬件资源。Triton 可以将多个模型放置在同一个 GPU 上，管理内存以确保它们可以并发运行。当混合了高流量和低流量模型，或具有不同资源占用量的模型时，这对于最大化使用率特别有效。例如，一个轻量级的基于 CPU 的预处理模型可以与一个重量级的基于 GPU 的视觉模型并行运行，所有这些都由同一个服务器实例管理并从同一个端点提供服务，从而简化了基础设施管理并降低了闲置资源成本。

## 参考资料

- [NVIDIA Triton Inference Server Documentation](https://developer.nvidia.com/triton-inference-server/) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  理解 Triton 功能、架构和配置细节的权威来源，包括模型仓库、动态批处理和模型组合。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供了在生产环境中部署和提供机器学习模型的工程挑战和最佳实践的更广泛背景，包括效率和资源管理的考虑因素。

---

[上一节](03-%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96%E6%8A%80%E6%9C%AF%EF%BC%9AINT8%E5%92%8CFP8.md) · [下一节](05-%E6%A8%A1%E5%9E%8B%E7%9A%84A-B%E6%B5%8B%E8%AF%95%E4%B8%8E%E9%87%91%E4%B8%9D%E9%9B%80%E9%83%A8%E7%BD%B2.md)

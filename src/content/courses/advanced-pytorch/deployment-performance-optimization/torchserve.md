---
course: "advanced-pytorch"
chapter: "deployment-performance-optimization"
lesson: "torchserve"
sourceId: 2751
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-4-deployment-performance-optimization/torchserve"
title: "使用 TorchServe 提供模型服务"
description: "使用 TorchServe 将 PyTorch 模型部署为生产就绪的Web服务。"
order: 7
plots: []
sourceHash: "6fc269591c4d6451e6688259428b082b4745da579c10143842b905f96476b534"
sourceCorrections: []
---

将机器学习 (machine learning)模型部署到最终用户或下游应用程序，需要通过服务基础设施使其可用。这通常发生在模型使用 TorchScript、量化 (quantization)或剪枝等技术优化推理 (inference)性能之后。手动构建和管理此类服务基础设施可能很复杂，涉及 API 开发、请求处理、扩缩容和监控。TorchServe 是一个专门为简化 PyTorch 模型这一过程而开发的工具。它提供了一种标准化的方式，用于在生产环境中打包、部署、管理和提供训练好的模型。

TorchServe 充当了桥梁，连接优化后的 PyTorch 模型与需要使用其预测结果的应用程序。它处理模型服务的操作方面，使您能够专注于模型开发和集成。

### 理解 TorchServe 架构

TorchServe 在设计时充分考虑了灵活性和性能。其主要组件共同协作以提供一个方案：

1. **模型归档器 (`torch-model-archiver`):** 这个命令行工具是为 TorchServe 准备模型的第一步。它将所有必需的工件打包成一个独立的归档文件，扩展名为 `.mar`。这些工件通常包括：

   - 序列化模型文件（例如，TorchScript `.pt` 文件或标准的 `state_dict`）。
   - 定义 **处理器** 逻辑（预处理、推理 (inference)、后处理）的 Python 脚本。
   - 可选的辅助文件（例如，词汇文件、配置 JSON、标签映射文件）。
   - 清单文件（自动生成），描述模型、版本、处理器等。
2. **TorchServe 运行时:** 这是核心服务器进程。它侦听预定义网络端口上的传入请求。它管理已部署模型的生命周期，包括将它们加载到内存中、按模型扩展工作进程数量，以及将推理请求路由到适当的工作进程。
3. **处理器:** 处理器是 Python 脚本或类，它们定义 TorchServe 如何与您的特定模型交互。它们封装了以下逻辑：

   - `initialize(context)`: 模型加载时调用一次。用于将模型加载到内存和进行一次性设置。
   - `preprocess(data)`: 将传入请求数据转换为模型 `forward` 方法期望的格式（例如，解码图像、文本分词 (tokenization)）。
   - `inference(model_input)`: 通过调用模型的预测函数来执行实际推理。
   - `postprocess(inference_output)`: 将模型的原始输出转换为用户友好的格式（例如，将类别索引映射到标签，格式化 JSON）。
     TorchServe 提供了用于常见任务（图像分类、对象检测、文本分类）的多个内置处理器，但您可以轻松地为特殊的模型输入/输出或复杂的工作流创建自定义处理器。
4. **API:** TorchServe 公开了两个主要 REST API：

   - **推理 API (默认端口: 8080):** 用于向已加载的模型发送推理请求并接收预测结果。
   - **管理 API (默认端口: 8081):** 用于管理 TorchServe 提供的模型。操作包括注册新模型、注销现有模型、设置模型的默认版本以及按模型扩展工作进程数量。
   - **指标 API (默认端口: 8082):** 以 Prometheus 兼容的格式公开有关服务器和模型的运行指标（例如，请求延迟、错误率、CPU/内存使用情况）。

gRPC 支持也可用，用于低延迟通信，在微服务架构中尤其有用。

### The TorchServe 工作流程

使用 TorchServe 部署模型通常遵循以下步骤：

1. **准备模型工件：** 确保您的训练模型已保存（例如，使用 `torch.jit.save()` 保存 TorchScript 模型或使用 `torch.save()` 保存 `state_dict`），并收集所有必需的辅助文件。
2. **编写处理器（如果需要）：** 如果内置处理器不适合您的模型，请实现一个自定义处理器脚本（`.py` 文件）。
3. **归档模型：** 使用 `torch-model-archiver` 将模型、处理器和其他文件打包成一个 `.mar` 归档。
4. **启动 TorchServe：** 启动 TorchServe 运行时，将其指向一个目录（“模型存储”），`.mar` 文件将位于或注册到该目录。
5. **注册模型：** 使用管理 API 告知 TorchServe 从其 `.mar` 文件加载您的模型并准备好服务。您可以指定初始工作进程数量。
6. **发送推理 (inference)请求：** 使用推理 API 将数据发送到注册模型的端点并接收预测结果。
7. **管理与监控：** 使用管理 API 扩缩工作进程或更新模型，并使用指标 API 监控性能。

> TorchServe 部署工作流程的高级概述，描绘了准备步骤和运行时请求处理。

### 创建模型归档

`torch-model-archiver` 工具对于模型打包非常核心。以下是典型的命令结构：

```bash
torch-model-archiver \
  --model-name my_transformer_model \
  --version 1.0 \
  --serialized-file traced_transformer.pt \
  --handler transformer_handler.py \
  --extra-files "vocab.txt,config.json" \
  --export-path /path/to/model-store \
  --force
```

让我们分析一下这些参数 (parameter)：

- `--model-name`: 在 API 调用中使用的模型逻辑名称（例如，`my_transformer_model`）。
- `--version`: 模型的版本字符串（例如，`1.0`）。TorchServe 可以管理同一模型的多个版本。
- `--serialized-file`: 已保存模型文件的路径（例如，`torch.jit.save` 的输出）。如果使用 `state_dict`，您还需要 `--model-file` 指向您的模型定义 Python 文件。
- `--handler`: 您的处理器脚本（`.py`）的路径。这可以是内置处理器之一（例如，`image_classifier`，`text_classifier`），也可以是您的自定义脚本。
- `--extra-files`: 模型或处理器所需的额外文件的逗号分隔列表（例如，分词 (tokenization)器 (tokenizer)配置、词汇文件、标签映射）。这些文件将通过上下文 (context)对象在处理器的 `initialize` 方法中访问。
- `--export-path`: 生成的 `.mar` 文件将保存到的目录。这通常设置为 TorchServe 模型存储目录。
- `--force`: 如果 `.mar` 文件已存在，则覆盖它。

执行此命令将创建 `/path/to/model-store/my_transformer_model.mar`。

### 自定义处理器

虽然 TorchServe 的内置处理器涵盖了许多常见用例，但您通常会需要自定义逻辑。自定义处理器是一个 Python 脚本，其中包含一个类（通常继承自 `BaseHandler`），它实现了以下部分或全部方法：

```python
# custom_handler.py
import torch
import json
from ts.torch_handler.base_handler import BaseHandler
import logging
import os

logger = logging.getLogger(__name__)

class MyCustomHandler(BaseHandler):
    """
    自定义处理器，用于处理特定输入/输出格式。
    """
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None
        self.mapping = None

    def initialize(self, context):
        """
        加载模型和额外文件。模型加载时调用一次。
        """
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir") # 包含解压后的 MAR 内容的目录

        # 确定设备
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu")
        logger.info(f"处理器在设备上初始化: {self.device}")

        # 加载模型（假设为 TorchScript 的示例）
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("缺少 model.pt 文件")

        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        self.model.eval()
        logger.info(f"模型 {self.manifest['model']['modelName']} 加载成功。")

        # 加载额外文件（例如，标签映射）
        mapping_file_path = os.path.join(model_dir, "index_to_name.json") # 假设通过 --extra-files 传入
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
            logger.info("标签映射加载成功。")
        else:
            logger.warning("映射文件未找到。")

        self.initialized = True

    def preprocess(self, data):
        """
        将原始输入数据转换为模型输入张量。
        'data' 是一个字典列表，每个字典包含原始请求数据。
        """
        # 示例：假设输入是 JSON，如 {'text': 'some input string'}
        # 这在很大程度上取决于您的具体应用程序
        processed_inputs = []
        for row in data:
            request_body = row.get("data") or row.get("body") # 处理不同的输入源
            if isinstance(request_body, (bytes, bytearray)):
                request_body = request_body.decode('utf-8')

            # 在此处添加您的特定预处理逻辑（分词、张量创建等）
            # 为简单起见，我们假设 request_body 是所需的直接输入
            # 实际上，您会进行文本分词、图像解码/大小调整等操作。
            # 这部分必须返回 self.model() 期望格式的数据。
            logger.info(f"收到输入: {request_body}")
            # 虚拟预处理：直接传递（替换为真实逻辑）
            processed_inputs.append(request_body)

        # 示例：如果模型需要，将处理后的输入转换为批处理张量
        # input_tensor = self.tokenizer(processed_inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # return input_tensor
        return processed_inputs # 返回列表作为虚拟示例

    def inference(self, model_input):
        """
        使用模型运行推理。
        'model_input' 是 preprocess() 的输出。
        """
        # 示例：运行已加载的 TorchScript 模型
        # 输出取决于您的模型结构
        # 确保 model_input 在正确的设备上
        # output = self.model(model_input.to(self.device))

        # 虚拟推理：只回显输入（替换为真实模型调用）
        logger.info(f"正在对: {model_input} 运行虚拟推理")
        with torch.no_grad(): # 对推理来说必不可少
             # 将此替换为：output = self.model(model_input)
            output = [f"Processed: {item}" for item in model_input]
        return output

    def postprocess(self, inference_output):
        """
        将模型输出转换为用户友好的格式。
        'inference_output' 是 inference() 的输出。
        返回预测结果列表，每个输入请求一个。
        """
        # 示例：将原始模型输出（例如，logits）转换为标签/分数
        # predictions = torch.softmax(inference_output, dim=1).argmax(dim=1).cpu().tolist()
        # result = [self.mapping[str(pred)] if self.mapping else str(pred) for pred in predictions]

        # 虚拟后处理：只返回推理输出
        logger.info(f"后处理: {inference_output}")
        return inference_output # 应该是一个列表

# 注意：处理器文件名必须与 --handler 参数值匹配（不带 .py）
# 如果处理器类名与首字母大写的文件名不同，
# 则在 MANIFEST.json 中或通过归档器指定。
```

此示例显示了基本结构。您将用实际的预处理（例如，图像转换、文本分词 (tokenization)）、模型调用和后处理（例如，将输出索引映射到类别名称、格式化 JSON 响应）来替换虚拟逻辑。`context` 对象提供对系统属性（如 GPU 可用性）、模型目录和清单详情的访问。

### 运行 TorchServe 和管理模型

一旦您的 `.mar` 文件在模型存储中，您就可以启动 TorchServe：

```bash
# 如果模型存储目录不存在，则创建它
mkdir /path/to/model-store

# 启动 TorchServe，指向模型存储
torchserve --start \
           --model-store /path/to/model-store \
           --models my_model=/path/to/model-store/my_transformer_model.mar \
           --ts-config /path/to/config.properties
```

- `--start`: 在后台启动 TorchServe 服务器。使用 `torchserve --stop` 停止它。
- `--model-store`: 指定包含 `.mar` 文件的目录。
- `--models`:（可选）在启动时预加载并注册特定模型。格式为 `model_name=model_archive.mar`，如果文件在模型存储中，也可以是简单的 `model_name.mar`。您可以指定多个模型。
- `--ts-config`:（可选）用于自定义端口、JVM 参数 (parameter)、日志等的配置文件（`.properties`）路径。

启动后，您将与 API 交互，通常使用 `curl` 或 Python 中的 `requests` 等客户端库。

**注册模型（管理 API）：**

```bash
curl -X POST "http://localhost:8081/models?url=my_transformer_model.mar&model_name=transformer&initial_workers=1&synchronous=true"
```

此命令告知 TorchServe 从其 `.mar` 文件加载 `my_transformer_model.mar`（假设它在模型存储中），将其注册为逻辑名称 `transformer`，为其启动 1 个工作进程，并等待注册完成。

**发送推理 (inference)请求（推理 API）：**

具体格式取决于您的处理器的 `preprocess` 方法。如果它期望原始图像字节：

```bash
curl -X POST http://localhost:8080/predictions/transformer -T image.jpg
```

如果它期望 JSON：

```bash
curl -X POST http://localhost:8080/predictions/transformer -H "Content-Type: application/json" -d '{"text": "This is an example sentence."}'
```

响应将包含来自您的处理器的 `postprocess` 方法的输出。

**扩缩工作进程（管理 API）：**

如果您需要特定模型的更高吞吐量 (throughput)，您可以增加工作进程的数量：

```bash
curl -X PUT "http://localhost:8081/models/transformer?min_worker=4"
```

这会将 `transformer` 模型的工作进程数量扩缩到 4（TorchServe 处理它们之间的负载均衡）。

**检查状态和指标：**

- 列出已注册的模型：`curl http://localhost:8081/models`
- 描述特定模型：`curl http://localhost:8081/models/transformer`
- 获取指标：`curl http://localhost:8082/metrics`

### 性能与可扩展性

TorchServe 专为生产负载而设计。有助于提高性能的特点包括：

- **工作进程：** 在独立的工作进程中运行推理 (inference)可以实现并行请求处理和隔离。
- **批处理：** 处理器可以在 `preprocess` 和 `inference` 中实现批处理逻辑，以同时处理多个请求，从而提高 GPU 利用率。TorchServe 还具有可通过管理 API 或配置文件配置的内置动态批处理能力。
- **异步后端：** TorchServe 使用异步后端（基于 Netty）来高效处理并发连接。
- **指标端点：** 提供详细指标，用于使用 Prometheus 和 Grafana 等工具监控性能并发现瓶颈。
- **集成：** 可以轻松容器化 (Docker) 并使用 Kubernetes 等编排工具部署，以实现自动扩缩容和高可用性。

通过使用 TorchServe，您能显著减少部署和管理 PyTorch 模型所需的工程工作量，且确保可靠高效。它提供了一个标准化、功能丰富的平台，与 PyTorch 生态系统和常见的 MLOps 实践良好集成，使其成为将您的先进模型投入生产的重要工具。

## 参考资料

- [TorchScript and Deployment](https://pytorch.org/docs/stable/jit.html) — PyTorch Development Team (2024)
  Publisher: PyTorch Foundation
  概述了 TorchScript 等技术，用于优化和序列化 PyTorch 模型，这是高效生产部署的准备步骤。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://books.google.com/books?isbn=9781098107963) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供构建、部署和维护机器学习系统的架构模式和实践，包括模型服务基础设施。

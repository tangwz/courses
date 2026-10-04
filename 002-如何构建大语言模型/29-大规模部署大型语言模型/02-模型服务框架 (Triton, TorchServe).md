# 模型服务框架 (Triton, TorchServe)

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-29-serving-llms-at-scale/model-serving-frameworks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然使用Flask或FastAPI等框架的简单Web服务器可能足以部署较小的机器学习 (machine learning)模型，但高效地服务大型语言模型（LLM）则会带来一系列不同的工程问题。LLM参数 (parameter)的庞大体积会给内存资源带来压力，自回归 (autoregressive)生成对延迟很敏感，高效处理并发用户请求需要精密的批处理和资源管理。标准Web服务器并未针对这些GPU密集型、有状态的推理 (inference)工作进行优化。这正是专用模型服务框架不可或缺的原因。

这些框架专门设计用于大规模部署机器学习模型，为LLM提供优化的性能、更好的硬件利用率以及运行的稳定性。它们屏蔽了管理推理请求、模型加载、硬件加速和并发处理的许多底层复杂性。让我们考察两个重要的例子：NVIDIA Triton 推理服务器和PyTorch TorchServe。

### NVIDIA Triton 推理 (inference)服务器

NVIDIA Triton 是一个高性能推理服务平台，旨在将来自各种框架（包括PyTorch、TensorFlow、ONNX Runtime、TensorRT和自定义后端）的模型部署到GPU和CPU上。其架构旨在最大限度地提高吞吐量 (throughput)和硬件利用率，使其成为应对严苛LLM工作负载的有力选择。

与LLM服务相关的重要特点包括：

1. **多框架支持：** Triton 可以同时服务在不同框架中训练的模型，这为您的开发和部署流程带来了灵活性。您可以使用PyTorch模型，也可以同时使用经过TensorRT优化的专用版本或用于预处理的自定义C++后端。
2. **并发模型执行：** Triton 可以在单个GPU或跨多个GPU上同时运行多个模型或同一模型的多个实例。这提高了利用率，特别是对于内存容量大的GPU，它允许不同的用户请求或甚至一个集成模型的不同部分并行处理。
3. **动态批处理：** 这是LLM推理的一个重要功能。Triton在服务器端自动将来自不同客户端的传入请求进行批处理。通过批量处理请求，它更有效地利用GPU的并行处理能力，显著提高吞吐量。Triton根据观察到的请求负载和延迟限制动态调整批处理大小，以优化当前的流量模式。
4. **模型组合与流水线：** Triton 允许您定义流水线或组合模型，一个模型的输出作为另一个模型的输入。这对于需要独立分词 (tokenization)/反分词步骤或复杂预处理/后处理逻辑的LLM非常有用，这些可以作为由Triton管理的组合中的独立模型实现。
5. **多种后端：** 它支持多种执行后端。对于PyTorch模型，您可以使用原生的PyTorch后端 (`libtorch`)。为了获得最佳性能，尤其是在NVIDIA GPU上，模型通常可以转换为TensorRT并通过TensorRT后端提供服务，这可能会提供更低的延迟和更高的吞吐量。
6. **HTTP/gRPC 端点：** 为客户端应用程序发送推理请求提供标准化的网络接口。
7. **性能分析工具：** Triton 包含用于分析模型性能、查看延迟/吞吐量瓶颈以及优化部署配置的工具。

Triton采用声明式配置方法。您通常定义一个模型仓库结构，其中每个模型都含有一个`config.pbtxt`文件，指定其平台、后端、输入/输出张量、版本策略和实例组设置（控制有多少实例在哪些设备上运行）。动态批处理也在此处配置。

这里是一个使用动态批处理的PyTorch LLM的`config.pbtxt`简化示例：

```protobuf
# Triton中PyTorch LLM模型的config.pbtxt
name: "my_llm_model"
platform: "pytorch_libtorch" # 指定后端
max_batch_size: 64           # Triton可形成的最大批处理大小

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ] # 变长序列维度
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ] # 变长序列维度
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, 50257 ] # 示例：序列长度，词汇表大小
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ] # Triton偏好形成的批次大小
  max_queue_delay_microseconds: 10000    # 请求等待批处理的最大时间 (10ms)
}

instance_group [
  {
    count: 1 # 此模型的实例数量
    kind: KIND_GPU
    gpus: [ 0 ] # 分配给GPU 0
  }
]

# 可选：指定默认模型文件名 (如果不是 model.pt)
# default_model_filename: "my_llm_scripted.pt"
```

> NVIDIA Triton 推理服务器内部的基本交互流程。

Triton的优势在于其广泛的兼容性、动态批处理和TensorRT集成等性能优化功能，以及对并发请求的处理，这使其非常适合大规模、多模型的部署。

### PyTorch TorchServe

TorchServe是一个开源的模型服务框架，专门为PyTorch模型开发。它旨在提供一种简单且高效的方式将PyTorch模型部署到生产环境。由于它是PyTorch原生框架，对于已经大量使用PyTorch生态系统的团队来说，它通常提供了一条更直接的路径。

与LLM服务相关的重要特点包括：

1. **原生PyTorch支持：** 从头开始为PyTorch设计，使用`torch-model-archiver`工具使模型打包和部署过程相对简单。
2. **模型打包：** `torch-model-archiver`实用程序将您的模型代码（例如`model.py`）、序列化权重 (weight)（`.pt`或`.pth`文件）和自定义处理程序文件（`handler.py`）打包成一个`.mar`（模型归档）文件，这是TorchServe的部署单位。
3. **自定义处理程序：** 这是TorchServe的一个重要方面。您在Python中定义一个处理程序类（通常继承自`BaseHandler`），以指定精确的预处理（例如，分词 (tokenization)）、推理 (inference)调用和后处理（例如，反分词，从logits生成文本）逻辑。这使您可以直接在Python中对请求生命周期进行细致的控制。
4. **批量推理：** TorchServe支持在服务器端进行请求批处理，类似于Triton，以提高吞吐量 (throughput)。
5. **管理API：** 提供用于注册、注销和动态扩展模型的API，无需重启服务器。
6. **模型版本控制：** 允许同时加载模型的多个版本，便于A/B测试或逐步发布。
7. **日志和指标：** 提供可配置的日志记录，并发出与Prometheus和CloudWatch等监控工具兼容的运行指标（例如，延迟、请求计数、错误率）。

使用TorchServe部署LLM通常涉及创建自定义处理程序来管理分词和生成过程。

以下是一个生成式LLM的自定义处理程序（`handler.py`）的示例片段：

```python
# handler.py (TorchServe 示例)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class LLMHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.tokenizer = None
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        加载模型和分词器。模型加载时调用一次。
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        # 根据CUDA可用性确定设备
        use_cuda = (
            torch.cuda.is_available()
            and properties.get("gpu_id") is not None
        )
        if use_cuda:
            self.device = torch.device(
                "cuda:" + str(properties.get("gpu_id"))
            )
        else:
            self.device = torch.device("cpu")

        logger.info(f"将模型加载到设备：{self.device}")

        # 从模型目录加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval() # 将模型设置为评估模式

        # 如果缺失，添加padding token（一些模型常见）
        if self.tokenizer.pad_token is None:
             self.tokenizer.pad_token = self.tokenizer.eos_token
             self.model.config.pad_token_id = self.model.config.eos_token_id

        logger.info(
            "Transformer模型和分词器加载成功。"
        )
        self.initialized = True

    def preprocess(self, requests):
        """
        对输入提示进行分词。requests 是请求列表。
        """
        input_texts = [req.get("data") or req.get("body") for req in requests]
        # 如有必要则解码（假设JSON输入包含'prompt'字段）
        prompts = []
        for item in input_texts:
            if isinstance(item, (bytes, bytearray)):
                prompt = item.get("prompt", item.decode('utf-8'))
            else:
                prompt = item.get("prompt")
            prompts.append(prompt)

        logger.info(f"收到提示：{prompts}")
        # 对批处理提示进行分词
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(self.device)
        # inputs 是一个包含 'input_ids' 和 'attention_mask' 的字典
        return inputs

    def inference(self, inputs):
        """
        执行模型推理（生成）。
        """
        # 示例生成参数（可在请求数据中传递）
        max_new_tokens = inputs.pop("max_new_tokens", 50) # 如果未提供，则为默认值
        do_sample = inputs.pop("do_sample", True)
        temperature = inputs.pop("temperature", 0.7)
        top_p = inputs.pop("top_p", 0.9)

        with torch.no_grad():
            # 模型生成调用
            outputs = self.model.generate(
                **inputs, # 传递 input_ids 和 attention_mask
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        # 包含生成token ID的张量
        return outputs

    def postprocess(self, outputs):
        """
        对生成的序列进行反分词。
        """
        # 将生成的token ID解码回文本
        # 跳过特殊token，只解码新生成的部分
        generated_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        logger.info(f"生成的文本：{generated_texts}")
        # 返回生成的字符串列表
        return generated_texts

# 打包此模型（假设模型权重保存在 ./my_llm_model/ 中）：
# $ torch-model-archiver --model-name my_llm \
#   --version 1.0 \
#   --serialized-file ./my_llm_model/pytorch_model.bin \
#   --model-file ./my_llm_model/modeling_utils.py \
#   --handler handler.py \
#   --extra-files "./my_llm_model/{config,tokenizer,tokenizer_config,special_tokens_map}.json" \
#   --export-path ./model_store
# 
# $ torchserve --start --ncs --model-store ./model_store --models my_llm=my_llm.mar
```

TorchServe为部署PyTorch模型提供了一条简化路径，通过基于Python的自定义处理程序提供灵活性，并与PyTorch生态系统的工具和实践良好集成。

### 框架选择

Triton和TorchServe之间的选择通常取决于具体的项目需求和现有基础设施：

- **选择Triton，如果：**
  - 您需要服务来自多个机器学习 (machine learning)框架（PyTorch, TensorFlow, ONNX, TensorRT）的模型。
  - 实现最大可能的吞吐量 (throughput)和最低延迟，可能通过TensorRT优化，是主要目标。
  - 您需要共享GPU上的并发模型执行或声明式定义的复杂模型组合等功能。
  - 您的团队有C++经验，或者习惯通过文本文件（`config.pbtxt`）配置部署。
- **选择TorchServe，如果：**
  - 您的主要关注点是部署PyTorch模型。
  - 您更喜欢直接在Python自定义处理程序中编写预处理/后处理逻辑的灵活性和便捷性。
  - 与PyTorch开发工作流程的良好集成很重要。
  - 内置模型版本控制和简单快照等功能符合您的运维需求。

这两个框架都能够高效地服务大型模型。它们提供了必要的抽象和优化（如批处理和硬件加速集成），这些是难以从头构建且耗时的工作，使工程团队能够专注于模型开发和应用逻辑，而不是低级别的服务基础设施。可靠且可扩展地部署LLM需要超越基本的Web服务器，转向这些专业的推理 (inference)服务解决方案。

## 参考资料

- [NVIDIA Triton Inference Server Documentation](https://developer.nvidia.com/triton-inference-server/documentation) — NVIDIA Corporation (2023)
  NVIDIA Triton Inference Server 的官方文档，详细介绍了其架构、功能以及高性能模型部署的配置。
- [Accelerate Inference with Dynamic Batching on NVIDIA Triton Inference Server](https://developer.nvidia.com/blog/accelerate-inference-with-dynamic-batching-on-nvidia-triton-inference-server/) — Andrew P. Kim (2021)
  Journal: NVIDIA Developer Blog; Publisher: NVIDIA
  这篇来自 NVIDIA 的博客文章解释了 Triton 内部动态批处理的工作原理，以优化 GPU 利用率和推理工作负载的吞吐量，尤其对大型语言模型有益。

---

[上一节](01-LLM%20%E4%BA%A4%E4%BA%92%E7%9A%84%20API%20%E8%AE%BE%E8%AE%A1.md) · [下一节](03-%E5%A4%84%E7%90%86%E5%B9%B6%E5%8F%91%E8%AF%B7%E6%B1%82.md)

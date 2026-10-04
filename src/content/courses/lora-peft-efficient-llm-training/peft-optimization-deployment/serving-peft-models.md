---
course: "lora-peft-efficient-llm-training"
chapter: "peft-optimization-deployment"
lesson: "serving-peft-models"
sourceId: 4169
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-5-peft-optimization-deployment/serving-peft-models"
title: "使用 PEFT 适配器部署模型"
description: "讨论用于推理的可互换 PEFT 适配器模型的部署方案。"
order: 7
plots: []
sourceHash: "6ab09e8035948b50e8d159ddf838b412a473b803b146e2e8c39cd71aa8a6e530"
sourceCorrections: []
---

与部署完全微调 (fine-tuning)的模型相比，部署使用参数 (parameter)高效微调（PEFT）方法微调的模型具有显著的优势和一些需要考虑的地方。由于PEFT方法通常只修改模型参数的一小部分，因此生成的成果（适配器权重 (weight)）远小于基础模型。这使得高效部署多个专门模型成为可能，且无需复制大型基础模型的权重。

### 小适配器在模型部署中的好处

设想一下，您需要部署大型语言模型的多个版本，每个版本都为不同的任务或客户量身定制。如果您使用完全微调 (fine-tuning)，则可能需要将整个数十亿参数 (parameter)模型的多个实例加载到内存中，每个任务一个。这在GPU内存和基础设施成本方面通常是极其昂贵的。

PEFT 方法，特别是 LoRA 或适配器微调等增益方法，提供了一种有吸引力的替代方案。主要思路是只加载一次大型预训练 (pre-training)基础模型。然后，对于每个特定任务，您只需加载一小部分适配器权重 (weight)（通常只有几兆字节大小）。这显著减少了内存占用，并允许单个模型实例潜在地处理需要不同微调行为的请求。

### 使用 PEFT 适配器部署模型的方案

有几种方法可以部署使用 PEFT 微调 (fine-tuning)的模型，每种方法在灵活性、性能和复杂性方面都有自己的取舍。

#### 方案 1：静态部署（合并权重 (weight)）

对于 LoRA 等方法，其适应性调整涉及向现有权重矩阵 ($W = W_0 + s \cdot B A$) 添加低秩更新，因此可以在部署*之前*将适配器权重直接合并到基础模型权重中。

1. **离线合并：** 使用库函数（例如 Hugging Face `peft` 中的 `model.merge_and_unload()`）来计算最终权重 $W$。
2. **保存合并模型：** 保存生成的模型，该模型现在包含了微调调整。这个保存的模型在结构上与原始基础模型相同，只是权重有所修改。
3. **部署标准模型：** 使用标准部署设施部署此合并模型。在推理 (inference)时无需对 PEFT 适配器进行特殊处理。

- **优点：**
  - 与原始基础模型相比，推理开销为零，因为在前向传递过程中无需额外的计算或逻辑。
  - 部署流程简化；使用标准的模型部署技术。
- **缺点：**
  - 如果您需要许多不同的适配器，则会失去参数 (parameter)效率*在部署时*的好处。您最终会存储和管理多个大型模型检查点，每个合并的适配器一个。
  - 不适用于需要在同一部署实例上动态切换不同任务适配器的场景。

#### 方案 2：动态适配器加载与切换

此方法利用适配器的小尺寸，使得从单个基础模型实例进行多任务或多租户部署成为可能。

1. **加载基础模型：** 将原始预训练 (pre-training)基础模型加载到部署设施上（例如，GPU）。
2. **加载适配器：** 加载一个或多个 PEFT 适配器配置。这些适配器权重可以保存在 CPU 内存中，并根据需要移动到 GPU，或者如果经常使用且空间允许，则直接缓存在 GPU 内存中。
3. **请求路由：** 在您的部署应用程序中（例如，FastAPI 后端）实现逻辑，以根据传入请求识别需要哪个适配器（基于用户ID、请求负载中的任务标识符等）。
4. **激活适配器：** 在运行推理之前，动态配置基础模型以使用相应的适配器权重。Hugging Face `peft` 等库为此提供了函数，例如 `model.set_adapter("adapter_name")` 或通过 `model.enable_adapters(["adapter_name"])` 管理活动适配器。然后，前向传递在内部进行修改，以包含活动适配器的权重。对于 LoRA，这意味着计算 $h = W_0 x + s \cdot B_{active} A_{active} x$。

> 此图显示了动态适配器切换。单个基础模型实例根据传入请求使用切换逻辑来应用不同的小型适配器（A、B、C）。

- **优点：**
  - 与加载多个完整模型相比，内存节省很多。只需大型基础模型权重的一个副本。
  - 使得在共享基础设施上进行灵活的多任务或多租户部署成为可能。
  - 如果适配器易于获取（例如，在 GPU 内存中），任务之间切换快速。
- **缺点：**
  - 部署应用程序中管理适配器加载、缓存和路由的复杂性增加。
  - 与合并模型相比，推理开销略微增加，这是由于前向传递中额外的适配器计算 ($s \cdot B A x$) 所致，尽管通常很小。
  - 内存使用量仍然会随着主动加载的适配器数量而增加，特别是当它们缓存在 GPU 上时。需要仔细管理。

#### 方案 3：混合方法

您可以组合多种方案。例如，您可以将默认或最常用的适配器合并到基础模型中，用于一般请求（方案 1），并为特定、不那么频繁的任务动态加载其他适配器（方案 2）。

### 实施时需要考虑的事项

- **框架支持：** 使用为 PEFT 设计的库。Hugging Face 的 `peft` 库与 `transformers` 很好地结合，并提供了用于加载、保存、合并和动态切换适配器的 API（`load_adapter`、`set_adapter`、`merge_and_unload`）。
- **推理 (inference)服务器：** Triton Inference Server、TorchServe 或 TensorFlow Serving 等标准推理服务器可能需要自定义后端或处理逻辑来实现动态适配器切换。使用 FastAPI 或 Flask 等框架的更简单的自定义服务器通常提供更直接的控制来实现此逻辑。
- **适配器存储和缓存：** 决定在哪里存储适配器权重 (weight)（例如，S3、本地磁盘），如果适配器按需加载，则实施缓存策略（如最近最少使用 LRU）以最小化延迟。
- **性能监控：** 分析您的部署设置。测量请求延迟、吞吐量 (throughput)和 GPU 内存利用率。评估动态切换的开销与合并模型的额外实例成本之间的关系。对于动态切换，确保选择和应用适配器权重的开销对于您的应用程序的延迟要求是可接受的。

选择正确的部署方案在很大程度上取决于您的具体应用程序需求。如果您只有少数几个固定任务，合并适配器可能是最简单的。如果您需要动态支持许多任务或为大量用户个性化响应，动态切换可以提供很好的资源效率，尽管实现复杂性会增加。通过理解这些权衡，您可以设计一个部署架构，有效利用 PEFT 的优点，实现高效且可扩展的模型部署。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (2021)
  Journal: arXiv; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA），一种广泛使用的参数高效微调（PEFT）方法。
- [Parameter-Efficient Fine-Tuning (PEFT) library](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  Hugging Face PEFT库的官方文档，提供了管理和部署PEFT适配器的实用API细节。

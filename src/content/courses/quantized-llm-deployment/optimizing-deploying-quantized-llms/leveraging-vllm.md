---
course: "quantized-llm-deployment"
chapter: "optimizing-deploying-quantized-llms"
lesson: "leveraging-vllm"
sourceId: 4653
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-4-optimizing-deploying-quantized-llms/leveraging-vllm"
title: "借助 vLLM 实现高吞吐量推理"
description: "运用 vLLM 库实现量化 LLM 的高吞吐量服务。"
order: 4
plots: ["plots/4653-0.json"]
sourceHash: "a2cb2c163f1b1e721234072bdf9dbece0916468ae40f63a68a52ce98829badce"
sourceCorrections: []
---

量化 (quantization)虽然显著减少了内存占用，并能加速单个操作的计算，但在服务大型语言模型 (LLM) 时，要实现高吞吐量 (throughput)会带来额外难题，尤其是在并发请求的内存管理方面。传统方法常遇到内存碎片和低效批处理的问题。在此，vLLM 等专用推理 (inference)引擎显得尤其有价值。vLLM 是一个开源库，专门用于快速且内存高效的 LLM 推理，使其成为在重负载下部署量化模型的理想选择。

### LLM 服务中的内存瓶颈

服务 LLM 涉及管理大型张量，特别是注意力机制 (attention mechanism)所需的键值 (KV) 缓存。每个用户请求都会生成自己的 KV 缓存，该缓存会随生成的序列长度增加。在高并发环境里，高效管理这些缓存很困难：

1. **内存碎片：** 传统系统通常会为每个序列的 KV 缓存预分配大块的连续内存空间。这可能导致严重的内部碎片（已分配块中未使用的内存）和外部碎片（已分配块之间无法使用的空闲内存），限制系统能处理的并发请求数量。
2. **低效批处理：** 静态批处理将请求分组，但需要将较短序列填充到批处理中最长序列的长度。整个批处理必须等待最慢的序列完成其生成步骤，导致 GPU 利用率不足。

### vLLM 的解决方案：分页注意力

vLLM 通过其核心创新：**分页注意力 (PagedAttention)**，直接解决了这些内存难题。受操作系统中虚拟内存和分页技术的启发，分页注意力将 KV 缓存管理在称为“页”的非连续内存块中。

与为每个序列分配一个大块内存不同，序列的 KV 缓存存储在可能许多更小的、固定大小的块中。块表将逻辑块（序列缓存中的位置）映射到物理块（GPU 内存中的实际位置）。

这种方法有以下几个优点：

- **接近零碎片：** 由于块是按需分配且无需连续，因碎片造成的内存浪费大大减少。一个 4MB 的块可能只在末尾浪费几 KB，而连续分配方案中可能浪费数兆字节。
- **高效内存共享：** 分页注意力促进了高级内存共享策略，如写时复制。例如，从相同提示生成的多个输出可以共享与提示 KV 缓存对应的内存块，直到某个序列出现分歧，此时只需复制不同的块。

### 实现连续批处理

分页注意力的高效内存管理直接促成了一种更动态、更有效的批处理策略，称为**连续批处理**。

与静态批处理不同，连续批处理允许推理 (inference)引擎以更细粒度的步骤运行。当当前批处理中的序列完成生成时，它会立即从批处理中移除，其内存资源（物理块）被回收。调度器可以立即将新的等待请求插入批处理中，确保 GPU 尽可能持续接近其最大能力进行处理。

这消除了与等待静态批处理中最慢序列相关的空闲时间，并大幅提升了整体 GPU 利用率，从而提高吞吐量 (throughput)。

> 静态批处理常导致 GPU 空闲，等待最长序列完成，而 vLLM 的连续批处理通过使用分页注意力动态管理请求，使 GPU 保持忙碌。

### vLLM 在量化 (quantization)模型中的使用

vLLM 原生支持与 LLM 相关的常见量化方法，例如激活感知权重 (weight)量化 (AWQ) 和 GPTQ。这意味着您可以将量化带来的模型大小减小和潜在的计算加速，与分页注意力和连续批处理带来的吞吐量 (throughput)提升结合起来。

在 vLLM 中加载量化模型通常很简单。该库通常会根据模型文件自动检测量化类型，或允许明确指定。

这是一个使用 vLLM Python API 加载并运行 AWQ 量化模型推理 (inference)的示例：

```python
from vllm import LLM, SamplingParams

# 指定量化模型的路径或 Hugging Face 标识符
# vLLM 通常会自动检测 AWQ/GPTQ 格式
model_id = "your-org/your-quantized-model-awq"

# 定义生成采样参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

# 初始化 vLLM 引擎
# 如果自动检测失败，明确设置 quantization='awq'
# tensor_parallel_size 可用于多 GPU 推理
llm = LLM(model=model_id,
          quantization="awq", # 通常可选，取决于模型格式
          trust_remote_code=True, # 某些模型必需
          # tensor_parallel_size=2 # 2 块 GPU 示例
         )

# 准备提示（可以是列表用于批处理）
prompts = [
    "vLLM 中 PagedAttention 的原理是什么？",
    "量化 LLM 提供的好处包括",
]

# 运行推理
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")

# 对于服务，vLLM 还提供一个兼容 OpenAI 的服务器：
# python -m vllm.entrypoints.openai.api_server --model your-org/your-quantized-model-awq --quantization awq
```

此示例演示了如何加载 AWQ 模型。GPTQ 模型的处理过程类似，通常只需更改 `quantization` 参数 (parameter)或依赖自动检测。

### 性能提升与注意事项

通过采用分页注意力和连续批处理，vLLM 在处理高并发和可变序列长度时，通常能实现显著更高的吞吐量 (throughput)（按每秒请求数或每秒 token 数衡量），与基线 Hugging Face `transformers` 实现或 TGI 等其他优化服务器相比。在服务量化 (quantization)模型时，其优势尤为明显，因为每个序列的内存占用更低，允许同时批处理更多请求。



![示意性吞吐量：量化 LLM 服务](plots/4653-0.json)



> 示意性比较，说明了 vLLM 潜在的吞吐量提升，特别是在高并发情况下，服务量化模型时。实际性能因模型、硬件和工作负载而异。

请注意：

- **兼容性：** 确保您使用的 vLLM 版本支持特定的量化格式和模型架构。
- **硬件：** 性能提升在很大程度上取决于可用的 GPU 硬件。
- **工作负载：** vLLM 在提示和生成长度可变、请求量大的动态工作负载下表现出最显著的提升。

总而言之，vLLM 为服务 LLM 提供了一个强大的引擎，其通过分页注意力和连续批处理实现的复杂内存管理使其非常适合部署量化模型。通过使用 vLLM，您可以最大化量化 LLM 的吞吐量，同时服务更多用户并有效使用您的硬件资源。量化与高级服务技术的这种结合对于构建可扩展且经济高效的 LLM 应用程序必不可少。

## 参考资料

- [vLLM: Universal and Efficient Engine for Large Language Model Serving](https://doi.org/10.48550/arXiv.2309.06180) — Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023)
  Journal: SOSP 2023; DOI: [10.48550/arXiv.2309.06180](https://doi.org/10.48550/arXiv.2309.06180)
  介绍vLLM及其核心创新PagedAttention，用于高吞吐量LLM推理，解决了内存碎片和低效批处理问题。
- [vLLM Documentation](https://docs.vllm.ai/en/latest/) — vLLM Developers (2024)
  官方文档，提供vLLM的全面指南、API参考和实际示例，包括对量化模型的支持。
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: MLSys 2024; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  介绍了激活感知权重量化（AWQ），一种专为大型语言模型设计的后训练量化方法，用于减少内存占用和加速推理。
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2023)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍了GPTQ，一种后训练量化方法，能够为生成式预训练Transformer实现低位宽但高精度的量化。

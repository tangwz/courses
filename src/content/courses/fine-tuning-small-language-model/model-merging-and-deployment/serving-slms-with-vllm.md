---
course: "fine-tuning-small-language-model"
chapter: "model-merging-and-deployment"
lesson: "serving-slms-with-vllm"
sourceId: 8529
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-7-model-merging-and-deployment/serving-slms-with-vllm"
title: "使用 vLLM 部署 SLM"
description: "配置 vLLM 以托管微调后的模型，实现高吞吐量的本地推理。"
order: 3
plots: []
sourceHash: "758fc8cd2386d72ce6e2372c77b92db462775115360181065f7d0436a37ab41d"
sourceCorrections: []
---

在以 Safetensors 格式保存合并后的权重 (weight)后，首要目标就是让模型可用于推理 (inference)。虽然将权重加载到标准的 PyTorch 和 Transformers 脚本中可以进行本地测试，但在处理多个并发请求时效率极低。标准推理脚本的吞吐量 (throughput)低且延迟高，因为它们在连续文本生成过程中无法高效管理内存。为了在生产环境中有效地运行模型，需要一个专门的推理引擎。

vLLM 是一个专为高吞吐量和内存高效利用而设计的开源库。与标准的 Hugging Face 流水线相比，它显著提升了推理速度。vLLM 背后的核心机制是 PagedAttention，这是一种解决文本生成中常见的内存瓶颈的算法。

### PagedAttention 的工作机制

在文本生成过程中，自回归 (autoregressive)模型逐个生成标记（token）。为了避免重复计算过去标记的注意力分数，模型将其键（key）和值（value）向量 (vector)存储在 GPU 显存 (VRAM)中。这些存储的数据被称为 KV 缓存（KV cache）。

在标准实现中，KV 缓存是在内存中连续分配的。由于无法预先知道生成的序列长度，系统通常会为了防止序列达到最大限制而过度分配内存。这会导致严重的内存碎片和显存浪费。

PagedAttention 通过像操作系统管理虚拟内存一样管理 KV 缓存来解决这个问题。它将序列划分为固定大小的块。这些块在物理内存中不需要连续。这允许系统随着序列的增长动态分配内存，几乎消除了浪费，并允许服务器将更多请求组合在一起进行批处理。

> 使用 PagedAttention 进行内存分配，其中输入提示词 (prompt)被动态分配到不连续的 KV 缓存块中。

### 安装与启动服务器

使用标准的 Python 包管理器安装 vLLM 非常简单。

```bash
pip install vllm
```

安装完成后，vLLM 可以作为独立服务器运行，也可以作为 Python 模块导入。对于部署的应用，将其作为兼容 OpenAI 的服务器运行是最实用的方法。这种兼容性意味着任何设计用于 OpenAI API 的客户端或库都可以直接与您本地托管的小语言模型配合使用。

您可以直接从命令行启动 vLLM 服务器。将其指向包含合并后的 Safetensors 模型的目录。

```bash
python -m vllm.entrypoints.openai.api_server --model ./my-merged-slm --host 0.0.0.0 --port 8000
```

您可以配置几个参数 (parameter)，根据硬件限制来优化服务器：

- **gpu-memory-utilization**：这定义了允许 vLLM 分配的 GPU 显存 (VRAM)比例。默认设置为 $0.90$。如果您在同一块 GPU 上运行其他进程，应调低此值，例如设置为 $0.50$。
- **max-model-len**：这限制了最大上下文 (context)长度，包括输入和输出标记 (token)。如果您的模型是在短指令上微调 (fine-tuning)的，可以降低此值以节省内存。例如，将其设置为 $1024$ 会减少初始内存预留。
- **tensor-parallel-size**：如果您有多个 GPU，此参数允许您在这些 GPU 之间拆分模型执行。对于小语言模型，单个 GPU 通常足够，因此默认值为 $1$。

### 与服务器交互

服务器运行时，它会在 8000 端口监听传入的 HTTP 请求。您可以使用标准的 `curl` 命令与其交互。因为它模拟了 OpenAI API 结构，请求格式依赖于标准的 JSON 字段。

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./my-merged-slm",
    "prompt": "Explain the difference between a list and a tuple in Python.",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

服务器会处理请求，在后台应用 PagedAttention 优化，并返回包含生成文本的 JSON 响应。

### 用于批处理的离线推理 (inference)

有时您不需要持久化的服务器，而是希望以最大吞吐量 (throughput)在本地处理大量数据。vLLM 提供了一个 `LLM` 类用于离线推理，原生支持这种处理方式。

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(model="./my-merged-slm", gpu_memory_utilization=0.8)

# 定义生成参数
sampling_params = SamplingParams(temperature=0.7, max_tokens=150)

# 要处理的提示词列表
prompts = [
    "Write a short function to calculate the factorial of a number.",
    "What are the benefits of using a dictionary in Python?"
]

# 生成输出
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示词: {prompt}\n输出: {generated_text}\n")
```

这种离线方法完全绕过了网络开销。`generate` 方法会自动对提示词 (prompt)进行批处理，从而最大化 GPU 利用率并缩短处理大型数据集的总时间。现在，您已经拥有了一套快速且可扩展的机制来运行定制的语言模型。

## 参考资料

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://dl.acm.org/doi/10.1145/3600006.3613165) — Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023)
  Journal: Proceedings of the 29th Symposium on Operating Systems Principles (SOSP '23); Publisher: ACM; Pages: 611–626; DOI: [10.1145/3600006.3613165](https://doi.org/10.1145/3600006.3613165)
  介绍了 PagedAttention 算法及其在 vLLM 引擎中的实现原始研究论文。
- [vLLM Documentation](https://docs.vllm.ai/) — vLLM Team (2024)
  官方文档，涵盖安装、配置参数以及与 OpenAI 兼容的服务器部署。

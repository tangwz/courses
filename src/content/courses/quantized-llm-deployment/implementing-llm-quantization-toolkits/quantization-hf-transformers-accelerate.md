---
course: "quantized-llm-deployment"
chapter: "implementing-llm-quantization-toolkits"
lesson: "quantization-hf-transformers-accelerate"
sourceId: 4616
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-2-implementing-llm-quantization-toolkits/quantization-hf-transformers-accelerate"
title: "使用 Hugging Face Transformers 和 Accelerate 实现量化"
description: "运用 Hugging Face 库，轻松整合 bitsandbytes 等量化方法。"
order: 3
plots: ["plots/4616-0.json"]
sourceHash: "e8ae58a4f58d1321b8e90a72e86c9e0a7ed04b2e2085ff62df619ab591be2153"
sourceCorrections: []
---

Hugging Face 生态系统，尤其是 `Transformers` 库，提供了方便且标准化的接口，用于操作大量预训练 (pre-training)模型。意识到模型效率日益增长的需求，Hugging Face 已将量化 (quantization)功能直接整合到其核心工作流程中，通常在后台依赖于 `bitsandbytes` 等库。这种整合极大地简化了加载和运行量化模型的流程。

`Accelerate` 库对 `Transformers` 进行了补充，它简化了 PyTorch 代码在各种硬件设置（包括多 GPU 或 GPU 与 CPU 混合）上的执行。处理大型模型时，即使经过量化，`Accelerate` 也能自动管理模型的加载和设备分配。

### 使用 `Transformers` 加载量化 (quantization)模型

在 `Transformers` 中使用量化的主要方法是，在使用 `from_pretrained` 方法加载模型时直接指定量化参数 (parameter)。这种方法通常会调用 `bitsandbytes` 库，在模型加载时即时执行量化。

配置通过特定的量化配置类进行管理，最值得关注的是 `BitsAndBytesConfig`。我们来了解一下主要参数：

- `load_in_8bit` (bool): 如果设置为 `True`，模型将被加载并量化为 8 位整数。
- `load_in_4bit` (bool): 如果设置为 `True`，模型将被加载并量化为 4 位整数。
- `bnb_4bit_quant_type` (str, optional): 指定 4 位量化类型。常见选项包括 `"nf4"` (归一化 (normalization)浮点 4) 和 `"fp4"` (浮点 4)。NF4 通常被推荐用于更好地保留性能。默认为 `"nf4"`。
- `bnb_4bit_compute_dtype` (torch.dtype, optional): 设置量化后用于计算（例如矩阵乘法）的数据类型。使用 `bfloat16` 等更高精度类型可以提高准确性，但与 `float16` 相比，会牺牲一些性能和内存。默认为 `torch.float32`，但如果硬件支持，通常更推荐 `torch.bfloat16`。
- `bnb_4bit_use_double_quant` (bool, optional): 启用嵌套量化方案，其中量化常数本身也进行量化。这可以节省少量额外内存（每个参数约 0.4 位），但可能会轻微影响准确性。默认为 `False`。
- `llm_int8_threshold` (float, optional): 仅用于 `load_in_8bit`。此参数与 LLM.int8() 算法中的异常值阈值有关，该算法使用混合精度分解。

### 示例：加载一个 4 位量化 (quantization)模型

这里是一个典型示例，展示如何使用 `bfloat16` 计算精度和 4 位 NF4 量化来加载 `meta-llama/Llama-2-7b-chat-hf` 这样的模型：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-2-7b-chat-hf" # 示例模型 ID

# 配置 4 位量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False, # 可选
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 使用量化配置加载模型
# device_map="auto" 使用 Accelerate 在可用的 GPU/CPU 上分发模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto", # 需要 accelerate
    # torch_dtype=torch.bfloat16 # 通常设置为与计算数据类型匹配
)

print(f"Model loaded on devices: {model.hf_device_map}")
print(f"Memory footprint: {model.get_memory_footprint()} bytes")

# 现在模型已准备好进行推理，已量化为 4 位
# 推理示例：
prompt = "What is quantization in deep learning?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # 确保输入在正确的设备上

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

在此代码中：

1. 我们从 `torch` 和 `transformers` 导入必要的类。
2. 我们创建一个 `BitsAndBytesConfig` 对象，指定所需的 4 位量化设置（NF4 类型，`bfloat16` 计算）。
3. 我们照常加载分词 (tokenization)器 (tokenizer)。
4. 重要的是，我们将 `quantization_config` 对象传递给 `AutoModelForCausalLM.from_pretrained`。
5. 我们使用 `device_map="auto"`。这会指示 `Accelerate` 判断如何最好地在可用 GPU 之间分割模型层，并在需要时可能将部分层卸载到 CPU RAM。这对于即使量化后也可能不适合单个 GPU 的大型模型非常有帮助。需要安装 `Accelerate`（`pip install accelerate`）。
6. 随后我们可以检查设备映射和内存占用。与以原生精度（例如 FP16 或 BF16）加载模型相比，内存占用将明显降低。



![70 亿参数 LLM 的大致 GPU 内存使用情况](plots/4616-0.json)



> 通过 Hugging Face Transformers 整合，以不同精度级别加载典型 70 亿参数 (parameter) LLM 时，GPU 内存需求的大致降低情况。实际内存使用量可能因模型架构和硬件而异。

### `Accelerate` 的作用

如示例所示，`Accelerate` 紧密集成。`device_map="auto"` 参数 (parameter)由 `Accelerate` 提供支持。如果没有它，您需要手动指定设备（`.to('cuda')`），这在量化 (quantization)模型仍然超出单个 GPU 内存时很可能会失败。`Accelerate` 会检查可用硬件和模型大小，以智能地分配层。这可能涉及将不同层放置在不同 GPU 上，甚至将某些层卸载到 CPU 内存中（尽管这会显著影响推理 (inference)速度）。

### 考量与权衡

- **易用性：** 这种方法的主要优势是简单。量化 (quantization)在加载时发生，只需极少的代码更改。
- **灵活性：** 尽管 `bitsandbytes`（通常是此处的后端）功能强大，但此方法主要支持在线性层上统一应用 4 位和 8 位量化。更复杂的方案，如混合精度或将量化应用于特定的非线性层，可能需要自定义代码或不同的工具包。
- **性能：** 量化显著减少内存使用。由于更快的内存访问和专业硬件上可能更快的计算，推理 (inference)速度通常会提高，尽管计算 `dtype`（`bfloat16` vs `float16` vs `float32`）也发挥作用。应考虑 `bitsandbytes` 操作的开销。
- **准确性：** 4 位量化，特别是 NF4，通常能为许多任务保持良好的准确性，但与 FP16/BF16 相比，预计会有一定的性能下降。8 位量化提供了一个中间点，准确性损失较小，但内存和速度提升也较少。`bnb_4bit_compute_dtype` 的选择会影响最终的准确性。
- **保存量化模型：** 通过标准 `save_pretrained` 方法保存使用 `bitsandbytes` 量化的模型有时会比较复杂，或者可能无法以在所有硬件上都能直接使用相同量化配置加载的格式保存模型。它通常保存原始权重 (weight)，在加载时需要重新量化。对于持久化的量化格式，通常使用 `AutoGPTQ` 等专用库或涉及 ONNX/TensorRT 的方法，这将在后面介绍。

这种整合为应用量化提供了一个方便的入口。只需在模型加载期间添加 `QuantizationConfig`，您就可以立即从减少的内存占用中获益，从而可以在消费级硬件上运行更大的模型。后续章节将介绍 `AutoGPTQ` 和 `AutoAWQ` 等库，它们实现了特定的训练后量化算法（GPTQ 和 AWQ），并且通常会生成可直接共享和加载的持久化量化模型。

## 参考资料

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/main/en/quantization) — Hugging Face (2024)
  Hugging Face Transformers库的官方文档，详细介绍了其量化功能，包括BitsAndBytesConfig以及与bitsandbytes集成以加载量化模型。
- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Accelerate库的官方文档，解释了其简化分布式执行和自动设备放置的能力，这对于使用device_map='auto'加载大型量化模型至关重要。
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 35; DOI: [10.48550/arXiv.2208.07339](https://doi.org/10.48550/arXiv.2208.07339)
  介绍了LLM.int8()量化方法，该方法专门用于处理大型语言模型中的异常特征，并通过llm_int8_threshold等参数用于8位量化。

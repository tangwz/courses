---
course: "quantized-llm-deployment"
chapter: "implementing-llm-quantization-toolkits"
lesson: "applying-autogptq"
sourceId: 4618
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-2-implementing-llm-quantization-toolkits/applying-autogptq"
title: "使用 AutoGPTQ 应用 GPTQ"
description: "使用 AutoGPTQ 库实现 GPTQ 量化的实际操作。"
order: 4
plots: []
sourceHash: "1e89161ed85cf7c138b62bae17c22bec9e54d21b15f67a55d57b1d5679ad2dc5"
sourceCorrections: []
---

GPTQ 是一种有效的训练后量化 (quantization)（PTQ）方法，尤其擅长保持大型语言模型在 INT4 等低位宽下的精度。尽管该算法涉及复杂的逐层量化和基于 Hessian 的权重 (weight)更新，但其实际应用通过专门的库得到了显著简化。`AutoGPTQ` 库为将 GPTQ 应用于与 Hugging Face Transformers 生态系统兼容的模型提供了一个方便高效的接口。

### AutoGPTQ 库

AutoGPTQ 简化了使用 GPTQ 算法量化 (quantization) LLM 的过程。它提供优化的 CUDA 内核，以加快量化和推理 (inference)速度，与 `transformers` 库顺畅集成，并提供了一个直接的 API 来量化模型、保存模型并重新加载模型进行推理。这使其成为从业者使用 GPTQ 而无需管理算法实现的复杂细节的一个热门选择。

它的主要功能是接受一个预训练 (pre-training)的浮点模型，使用校准数据集应用 GPTQ 过程，并生成一个量化模型的状态字典以及加载优化模型所需的配置文件。

### 使用 AutoGPTQ 的主要步骤

使用 AutoGPTQ 量化 (quantization)模型的典型操作流程包含几个不同的步骤：

1. **安装：** 确保 AutoGPTQ 及其依赖项已安装。
2. **模型和分词 (tokenization)器 (tokenizer)加载：** 使用 `transformers` 加载目标 LLM 及其对应的分词器。
3. **校准数据准备：** 准备用于校准量化过程的具有代表性的文本数据样本。
4. **量化配置：** 定义量化参数 (parameter)（位宽、组大小等）。
5. **量化执行：** 使用校准数据在模型上运行 GPTQ 算法。
6. **保存量化模型：** 存储量化权重 (weight)和配置以备后续使用。

让我们更详细地查看这些步骤。

### 安装

您通常可以使用 pip 安装 AutoGPTQ。它通常需要特定版本的 PyTorch 才能与其 CUDA 内核兼容，因此建议查阅库的文档以了解确切要求。

```bash
# 示例安装命令（请查阅 AutoGPTQ 仓库获取最新信息）
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/ # 根据您的 CUDA 版本调整 cuXXX

# 确保 transformers、torch 和 datasets 也已安装
pip install transformers torch datasets accelerate
```

### 量化 (quantization)准备

在开始量化之前，您需要基础模型和校准数据集。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 加载模型和分词器
model_id = "facebook/opt-125m" # 示例模型
model_fp32 = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, # 如果可能，以 float16 格式加载以提高效率
    device_map="auto" # 使用 device_map 来处理大型模型
)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# 2. 准备校准数据集
# 需要一个小型、具有代表性的数据集。
# 这里，我们将使用 'wikitext' 中的一个样本
from datasets import load_dataset

dataset_name = "wikitext"
dataset_config = "wikitext-2-raw-v1"
dataset_split = "train"
num_samples = 128 # 校准样本数量
seq_len = 512 # 校准数据的序列长度

dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
calibration_data = [
    tokenizer(sample["text"], return_tensors="pt", max_length=seq_len, truncation=True).input_ids
    for sample in dataset.shuffle(seed=42).select(range(num_samples))
]

# 如果需要，过滤掉长度小于最小值的样本
# calibration_data = [data for data in calibration_data if data.shape[1] >= seq_len]
# 确保数据位于正确的设备上（如果使用 GPU）
# calibration_data = [data.to(model_fp32.device) for data in calibration_data]

# 为 AutoGPTQ 重新格式化（字符串列表或分词后的字典列表）
# 为了简单起见，这里我们使用字符串列表
calibration_text = [sample["text"] for sample in dataset.shuffle(seed=42).select(range(num_samples))]
```

`calibration_data` 的质量和代表性是影响量化模型最终精度的重要因素。它理想情况下应反映模型在推理 (inference)过程中将遇到的数据类型。虽然 128 个样本通常已足够，但可能需要尝试不同大小和来源的数据才能获得最佳结果。

### 使用 `GPTQQuantizer` 执行量化 (quantization)

AutoGPTQ 中用于量化的核心组件是 `GPTQQuantizer`。

```python
from auto_gptq import GPTQQuantizer, AutoQuantizeModel

# 3. 配置量化参数
quantizer = GPTQQuantizer(
    bits=4,               # 目标位宽（例如 4、3、2）
    dataset=calibration_text, # 校准数据（字符串列表）
    model_seqlen=seq_len, # 校准使用的序列长度
    group_size=128,       # 量化组大小（-1 表示按通道）
    damp_percent=0.1,     # Hessian 计算稳定性所需的对角偏移百分比
    desc_act=False,       # 激活顺序处理（通常 False 效果很好）
    use_cuda_fp16=True    # 如果可能，使用 float16 进行计算
)

# 4. 运行量化
# 此过程所需时间取决于模型大小和校准数据
print("开始量化...")
quantized_model = quantizer.quantize_model(model_fp32, tokenizer)
print("量化完成。")

# 5. 保存量化模型
quantized_model_dir = f"{model_id.replace('/', '_')}-4bit-gptq"
print(f"正在将量化模型保存到 {quantized_model_dir}...")
quantizer.save(model_fp32, quantized_model_dir, use_safetensors=True) # 使用量化器对象的 save 方法
tokenizer.save_pretrained(quantized_model_dir) # 同时保存分词器
print("模型已保存。")
```

**参数 (parameter)：**

- `bits`：确定量化权重 (weight)的精度。常用值为 4、3，甚至 2。较低的值会导致模型更小，推理 (inference)速度可能更快，但精度下降的风险也更大。
- `group_size`：控制量化的粒度。权重被分组在一起，并为每组计算共享的量化参数（缩放因子和零点）。常见默认值为 128。较小的组大小（例如 64、32）有时可以通过允许更细粒度的调整来提高精度，但与较大的组相比，可能会略微增加模型大小开销。将 `group_size` 设置为 -1 通常对应于按通道量化。
- `dataset`：提供的校准数据。AutoGPTQ 使用它来计算量化参数，并根据 Hessian 矩阵信息确定权重处理顺序，这对于 GPTQ 保持精度至关重要。
- `damp_percent`：添加到 Hessian 矩阵逆计算对角线上的一个小数（例如 0.01 到 0.1）。这有助于稳定过程，尤其是在 Hessian 矩阵条件不佳的情况下。
- `desc_act`（激活降序）：GPTQ 提出根据相应激活的幅度来量化权重列。虽然理论上有益，但实证结果各异，并且 `desc_act=False`（按权重顺序量化）通常能在较低复杂度下产生良好结果。如果需要，AutoGPTQ 允许启用此功能。
- `model_seqlen`：校准期间考虑的最大序列长度。它通常应与用于准备校准数据的序列长度相匹配。

`quantize_model` 函数遍历模型的层（具体来说，是目标量化的线性层），根据校准数据应用 GPTQ 更新规则。`save` 方法随后序列化量化权重（通常使用 `safetensors` 以提高安全性和效率），并将必要的量化配置 (`quantize_config.json`) 保存到指定目录中。

### 加载和使用量化 (quantization)模型

保存后，量化模型可以使用 AutoGPTQ 的专用类加载，该类理解量化格式和配置。

```python
from auto_gptq import AutoGPTQForCausalLM

# 加载已保存的量化模型
# 如果在同一会话中运行，请确保清空内存
# del model_fp32
# torch.cuda.empty_cache()

print(f"正在从 {quantized_model_dir} 加载量化模型...")
model_quantized = AutoGPTQForCausalLM.from_quantized(
    quantized_model_dir,
    device_map="auto", # 处理设备放置
    use_safetensors=True,
    use_triton=False # Triton 内核可能提供速度提升，但需要兼容的硬件/设置
)
print("量化模型已加载。")

# 照常执行推理
prompt = "Quantization is a technique to"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_quantized.device)

with torch.no_grad():
    generated_ids = model_quantized.generate(
        inputs=input_ids,
        max_new_tokens=50
    )

decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\n生成文本：")
print(decoded_text)
```

`AutoGPTQForCausalLM.from_quantized` 方法自动读取保存目录中的 `quantize_config.json` 文件以理解量化参数 (parameter)（位、组大小等），并加载量化权重 (weight)。然后使用标准 `transformers` API（`generate` 方法）进行推理 (inference)。

### 工作流程图

该过程可可视化如下：

> 使用 AutoGPTQ 库量化 (quantization) LLM 的基本工作流程。

### 考量与权衡

- **精度与压缩：** 由 `bits` 和 `group_size` 控制的主要权衡。4 位量化 (quantization)与 128 的组大小是一个常见的起点，它在显著减小模型大小和提高性能的同时，通常能保持可接受的精度损失。更激进的量化（3 位、2 位、更小的组大小）需要仔细评估（第 3 章会涉及）。
- **校准数据：** 校准数据的选择对结果有显著影响。使用能代表下游任务的数据很重要。
- **量化时间：** GPTQ 量化可能计算密集，尤其是对于非常大的模型。AutoGPTQ 中使用优化的内核有所帮助，但它仍然比舍入等更简单的量化方法耗时得多。
- **硬件支持：** 尽管 AutoGPTQ 提供优化的内核，但推理 (inference)速度的提升在高效支持低位矩阵乘法的硬件上最为明显（例如，最新的 NVIDIA GPU）。

AutoGPTQ 提供了一种直接且相对用户友好的方式来应用强大的 GPTQ 算法。通过理解其工作流程和重要参数 (parameter)，您可以有效地量化 LLM，为更高效的部署场景做好准备。在接下来的部分，我们将查看 AutoAWQ 等其他工具包，并比较它们的方法和结果。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍GPTQ算法的原始研究论文，为本节讨论的实际应用提供了理论基础。
- [Model quantization](https://huggingface.co/docs/accelerate/usage_guides/quantization) — Hugging Face (2025)
  Publisher: Hugging Face
  Hugging Face官方文档，讨论了多种量化技术，以及它们如何集成到更广泛的Transformers生态系统中，以实现高效的模型推理。

---
course: "quantized-llm-deployment"
chapter: "implementing-llm-quantization-toolkits"
lesson: "applying-autoawq"
sourceId: 4621
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-2-implementing-llm-quantization-toolkits/applying-autoawq"
title: "使用 AutoAWQ 应用 AWQ"
description: "使用 AutoAWQ 库实现 AWQ 量化的实际操作。"
order: 5
plots: ["plots/4621-0.json"]
sourceHash: "eda3995d86bd8b0158e3a98e2268be52321cb9adb465a3c09ada3c29f685d77f"
sourceCorrections: []
---

`AutoAWQ` 库提供了一个易于使用的界面，用于将激活感知权重 (weight)量化 (quantization) (AWQ) 算法应用于 Hugging Face Transformers 模型。AWQ 的主要思路是，通过观察校准数据中对应激活值的量级，来识别对模型输出影响显著的权重，并尽量减少这些权重上的量化误差。这种选择性方法旨在更有效地保持模型精度，尤其是在 4 比特等非常低的比特率下。

### 理解 AutoAWQ 工作流程

`AutoAWQ` 库显著简化了这一过程。它负责计算量化 (quantization)因子并根据 AWQ 方法量化权重 (weight)，与 Hugging Face 生态系统良好兼容。常见步骤包括加载模型、使用校准数据执行量化，以及保存量化后的模型文件。

### 安装

首先，请确保您已安装 `AutoAWQ` 及其所需的依赖项。它通常需要 PyTorch 和 Transformers。您通常可以通过 pip 进行安装：

```bash
pip install autoawq
# 如果需要，可能需要安装特定版本的 torch，请查看 AutoAWQ 文档
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # CUDA 11.8 示例
```

始终参考 `AutoAWQ` 官方仓库获取最新的安装说明和兼容性要求（例如，CUDA 版本、PyTorch 版本）。

### 使用 AutoAWQ 量化 (quantization)模型

让我们一步步地演示使用 `AutoAWQ` 量化预训练 (pre-training)语言模型的过程。

1. **加载模型和分词 (tokenization)器 (tokenizer)**：首先使用 `AutoAWQForCausalLM` 类和标准 Transformers `AutoTokenizer` 加载预训练模型及其对应的分词器。
2. **定义量化配置**：指定 AWQ 量化的参数 (parameter)。常见参数包括：

   - `w_bit`：权重 (weight)的目标比特宽度（例如，4）。
   - `q_group_size`：共享相同量化参数（缩放因子和零点）的权重数量。较小的分组大小可以提高精度，但可能会略微增加模型大小和计算开销。常见值为 64、128。
   - `zero_point`：一个布尔值，指示是否使用零点（非对称量化）或不使用（对称量化）。AWQ 在 `zero_point=True` 时通常表现良好。
3. **准备校准数据**：AWQ 依靠校准数据来识别重要的权重。正如第一章所讨论的，这应该是一个小型的、具有代表性的数据集样本，它反映了模型在推理 (inference)时将遇到的数据。在将其传递给 `quantize` 方法之前，您需要对这些数据进行分词。通常，字符串列表就足够了，然后您再对其进行分词。
4. **执行量化**：在已加载的模型实例上调用 `quantize` 方法，并传入分词器、量化配置和准备好的校准数据。`AutoAWQ` 将会就地执行分析和量化。
5. **保存量化模型**：使用 `save_quantized` 方法存储量化后的权重和必要的配置文件。分词器也应与模型一起保存。

以下是一个 Python 代码片段，演示了这些步骤：

```python
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 定义模型路径和量化模型的保存位置
model_path = "meta-llama/Llama-2-7b-hf" # Hugging Face Hub 中的模型 ID 示例
quant_path = "models/llama-2-7b-awq"   # 量化模型保存的本地路径
quant_config = {
    "w_bit": 4,              # 目标权重比特宽度
    "q_group_size": 128,     # 量化参数的分组大小
    "zero_point": True       # 使用非对称量化
}

# 加载未量化模型和分词器
# 如果模型需要权限/认证，请确保您已获得
print(f"加载模型: {model_path}...")
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    # 通常推荐使用 Safetensors 加载
    safetensors=True,
    # 使用合适的设备映射；'auto' 会在可用 GPU 上分布
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("模型和分词器已加载。")

# --- 准备校准数据 ---
# 使用小型、有代表性的数据集。这里我们使用简单的占位符文本。
# 对于实际用例，请从 C4 或 WikiText 等数据集中加载约 128 个样本。
calibration_texts = [
    "The field of large language models is rapidly evolving.",
    "Quantization helps reduce the computational cost of inference.",
    "AWQ is a post-training quantization method.",
    "Calibration data is important for accurate quantization."
]
# 对校准数据进行分词
# 注意：AutoAWQ 的 quantize 方法通常在内部处理分词
# 如果您传递原始文本，但在某些版本/场景中可能需要提供已分词的数据。
# 这里我们传递原始文本，因为 `quantize` 通常需要它。
print("正在准备校准数据...")

# --- 执行量化 ---
print("开始 AWQ 量化...")
model.quantize(
    tokenizer=tokenizer,
    quant_config=quant_config,
    calibration_data=calibration_texts # 直接传递原始文本列表
)
print("量化完成。")

# --- 保存量化模型 ---
print(f"正在保存量化模型到: {quant_path}")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path) # 将分词器与模型一起保存
print("量化模型和分词器保存成功。")

# 可选：如果需要，清理内存
# del model
# torch.cuda.empty_cache()
```

此脚本使用 AWQ 算法量化指定的模型并保存结果，通常包括权重的 `.safetensors` 文件和 JSON 配置文件。模型大小的减小是主要的优势。



![近似模型尺寸减小（70 亿参数模型）](plots/4621-0.json)



> 70 亿参数模型从 FP16 量化到 AWQ 4 比特时，磁盘占用的大致减少量。实际情况取决于分组大小和元数据。

### 加载和使用量化 (quantization)模型

保存后，您可以使用 `AutoAWQForCausalLM.from_quantized` 方法加载 AWQ 量化模型进行推理 (inference)。此方法需要您保存量化文件的路径。

```python
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, pipeline

# 量化模型保存的路径
quant_path = "models/llama-2-7b-awq"
device_map = "auto" # 或者指定 "cuda:0" 等

# 加载量化模型和分词器
print(f"正在从 {quant_path} 加载量化模型")
model = AutoAWQForCausalLM.from_quantized(quant_path, device_map=device_map, safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
print("量化模型和分词器已加载。")

# 设置推理管道（可选，用于演示）
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device_map)

# 推理示例
prompt = "What is Activation-Aware Weight Quantization (AWQ)?"
print(f"\n正在使用提示词运行推理: '{prompt}'")
outputs = pipe(prompt, max_new_tokens=60, do_sample=True, temperature=0.7)
print("生成的文本:")
print(outputs[0]['generated_text'])

# 可选：清理内存
# del model
# del pipe
# torch.cuda.empty_cache()
```

这演示了如何将优化后的模型加载回内存并用于文本生成等任务。性能优势（延迟、吞吐量 (throughput)、内存使用）将在第三章中具体介绍。

### 使用 AutoAWQ 的考量

- **校准数据质量**：AWQ 的有效性取决于校准数据是否准确反映了推理 (inference)过程中遇到的激活模式。使用多样化和有代表性的样本对于维持模型精度很重要。
- **硬件和计算核**：从 AWQ 量化 (quantization)模型获得最大加速通常需要针对所选分组大小的低比特矩阵乘法而优化的专用计算核。`AutoAWQ` 通常包含优化后的计算核，vLLM 等部署框架（第四章讨论）也提供了高度优化的实现，适配 AWQ 格式。请确保您的部署环境使用这些计算核。
- **模型兼容性**：虽然 `AutoAWQ` 支持许多流行的 LLM 架构，但请始终查阅库的文档，了解与特定模型的兼容性或任何已知限制。
- **AWQ 与 GPTQ 对比**：AWQ 和 GPTQ（将在下文介绍）都是有效的 PTQ 方法。AWQ 的量化过程通常比 GPTQ 快，因为它避免了每层解决复杂的优化问题。然而，最终的精度和推理速度权衡可能因模型架构、校准数据和目标硬件而异。通常需要使用您的特定模型和任务进行直接比较（如本章后续实践部分所述）。

通过使用 `AutoAWQ` 库，您可以掌握一种量化大型语言模型的有效技术，显著降低其资源需求，同时旨在保持高精度。这为模型更高效的部署做好了准备，这也是后续章节的主要内容。

## 参考资料

- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: MLSys 2024; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  介绍了用于压缩大型语言模型并通过保留重要权重来提高模型性能的激活感知权重量化（AWQ）算法。
- [AutoAWQ GitHub Repository](https://github.com/casper-hansen/AutoAWQ) — Casper Hansen and contributors (2024)
  Publisher: GitHub
  AutoAWQ库的官方仓库，提供安装说明、API文档和将AWQ应用于Hugging Face模型的示例。
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍了GPTQ，这是另一种重要的LLM训练后量化方法，在课程内容中与AWQ进行了直接比较。

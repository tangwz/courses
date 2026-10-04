---
course: "practical-llm-quantization"
chapter: "quantization-formats-tooling"
lesson: "quantization-conversion-tools"
sourceId: 4663
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/quantization-conversion-tools"
title: "模型转换与加载工具"
description: "常用实用程序和脚本概览，用于模型格式转换及加载。"
order: 7
plots: []
sourceHash: "5457a816e877b7c8eb0ea16edce38d74e7639993afdab2c7ba9a25bc40e03340"
sourceCorrections: []
---

量化 (quantization)模型虽然更小、更快，但其实际部署需要专门的工具。标准的 PyTorch 或 TensorFlow 保存机制通常不够，因为它们本身不理解如何存储和重建低位宽权重 (weight)，以及缩放因子、零点或组大小等必要元数据。常用的实用程序和库函数可以将标准模型转换为量化格式，并随后加载进行推理 (inference)。

### 模型转换为量化 (quantization)格式

转换模型通常指获取预训练 (pre-training)模型（通常为 FP32 或 FP16 等标准格式），并对其进行处理，以生成量化格式所需的特定文件结构和数据表示。

#### GGUF 转换

GGUF 格式由 `llama.cpp` 推广，被设计为包含模型架构、元数据、词汇表 (vocabulary)和量化权重 (weight)的自包含文件。转换通常涉及使用 `llama.cpp` 项目或相关工具提供的特定脚本。

常见的工作流程包括：

1. **获取原始模型：** 从标准格式的模型权重开始，这些权重通常来自 Hugging Face（例如 PyTorch 的 `.bin` 或 `safetensors` 文件）。
2. **使用转换脚本：** 主要工具通常是 Python 脚本（例如 `llama.cpp` 仓库中的 `convert.py`）。该脚本读取原始模型权重和词汇表。
3. **指定量化类型：** 在转换过程中，您选择 GGUF 支持的所需量化方案（例如 `Q4_K_M`、`Q5_K_S`、`Q8_0`）。脚本将所选的量化方法（通常是带有缩放因子的基本舍入等简单方法）逐层应用。
4. **输出 GGUF 文件：** 脚本将模型的架构定义、词汇表、张量（现已量化）和量化参数 (parameter)整合到一个 `.gguf` 文件中。

```bash
# 使用典型 llama.cpp 转换脚本的示例命令
python llama.cpp/convert.py \
  path/to/original/model \
  --outfile path/to/output/model.gguf \
  --outtype q4_k_m # 指定所需的 GGUF 量化类型
```

该过程实质上是将模型转换为针对 `llama.cpp` 等推理 (inference)引擎优化的格式。

#### GPTQ 和 AWQ 模型保存

对于 GPTQ 和 AWQ 等方法，“转换”步骤与量化过程本身紧密结合。`AutoGPTQ`（用于 GPTQ）或 `AutoAWQ`（用于 AWQ）等库执行量化算法，然后提供函数以兼容 `transformers` 或其自身生态系统加载的格式保存模型。

通常，保存 GPTQ 或 AWQ 量化模型涉及：

1. **运行量化算法：** 使用相应的库（例如 `AutoGPTQ` 或 `AutoAWQ`）将量化逻辑应用于基础模型，这通常需要校准数据。
2. **保存量化模型：** 调用库提供的 `save_quantized()` 方法。这通常会保存：
   - 量化权重（通常打包成整数类型），格式为 `.bin` 或 `.safetensors`。
   - 一个修改后的模型配置文件 (`config.json`)，表明模型使用 GPTQ/AWQ。
   - 一个特定的量化配置文件（例如 `quantization_config.json` 或嵌入 (embedding)在 `config.json` 中），详细说明了位宽（例如 4 位）、组大小、对称/非对称量化以及可能的算法特定细节等参数。

```python
# 使用保存函数的示例
# 假设 'model' 是应用 GPTQ/AWQ 量化后的模型对象
# 'tokenizer' 是相应的分词器

output_dir = "./quantized_model_directory"
model.save_quantized(output_dir)
tokenizer.save_pretrained(output_dir)

# 'output_dir' 现在将包含以下文件：
# - config.json（已为量化更新）
# - quantization_config.json（或类似文件，包含详细信息）
# - pytorch_model.bin / model.safetensors（包含打包的权重）
# - 分词器文件（tokenizer.json 等）
```

输出通常是一个目录，其结构类似于标准 Hugging Face 模型，但包含量化权重和额外的配置文件。

### 加载量化 (quantization)模型进行推理 (inference)

模型保存为量化格式后，需要合适的工具将其加载到内存并运行推理。

#### 加载 GGUF 模型

GGUF 文件专为特定推理引擎设计。常见的加载方式包括：

- **`llama.cpp`：** 主要的 C++ 引擎，专为 GGUF 设计。它能直接在 CPU 和 GPU 上高效加载并执行这些文件。
- **Python 绑定：** `ctransformers` 等库或 `llama.cpp` 的官方/社区 Python 绑定允许您在 Python 环境中直接加载 GGUF 模型并进行交互。

```python
# 使用 ctransformers 库的示例
from ctransformers import AutoModelForCausalLM

# 加载 GGUF 模型（本例中使用 Q4_K_M 量化）
llm = AutoModelForCausalLM.from_pretrained(
    "path/to/your/model.gguf",
    model_type="llama", # 如果需要，指定模型架构
    gpu_layers=50 # 要卸载到 GPU 的层数（如果可用）
)

# 现在使用 'llm' 对象进行推理
prompt = "What is model quantization?"
print(llm(prompt))
```

主要优点是简洁性：单个 GGUF 文件几乎包含所需的一切。

#### 使用 Transformers 加载 GPTQ/AWQ 模型

使用 `AutoGPTQ` 或 `AutoAWQ` 等库量化并保存的模型通常设计为可使用 Hugging Face `transformers` 库加载，有时需要安装原始量化库。

加载过程通常使用标准的 `AutoModelForCausalLM.from_pretrained()` 方法。`transformers` 库会检查保存目录中的 `config.json` 和 `quantization_config.json`（或等效文件），以了解如何加载和反量化（或执行）权重 (weight)。

```python
# 使用 transformers 加载 GPTQ 模型的示例
# 需要安装 AutoGPTQ 和可能的 Optimum
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "./quantized_model_directory" # GPTQ 模型保存的目录

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载量化模型
# device_map="auto" 有助于在可用硬件（CPU/GPU）上分配层
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)

# 模型已准备好进行推理
input_text = "Explain the GPTQ algorithm briefly."
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

`transformers` 库，通常借助 `accelerate` 进行设备映射，并可能在后台使用 `optimum` 或特定的量化库（`AutoGPTQ`、`AutoAWQ`），处理使用量化层设置模型的复杂性。

#### 通过 `bitsandbytes` 集成进行加载

如前一节“使用 bitsandbytes 进行量化”所述，该库通常直接集成到 `transformers` 的加载过程中。您无需加载预量化模型格式，而是加载一个标准模型（FP16/BF16），并指示 `transformers` 使用 `bitsandbytes` 实时将线性层量化为 8 位或 4 位精度。

```python
# 使用 bitsandbytes 4 位量化加载的示例
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "meta-llama/Llama-2-7b-chat-hf" # 基础模型示例

# 配置 bitsandbytes 量化
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.bfloat16 # 可选：计算数据类型
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 在加载时应用 4 位量化来加载模型
model = AutoModelForCausalLM.from_pretrained(
   model_id,
   quantization_config=quantization_config,
   device_map="auto" # 自动分配层
)

# 模型已准备好使用量化线性层进行推理
```

这种方法很方便，因为它不需要单独的转换步骤，而是在模型加载到目标设备时动态执行量化。

### 工具生态系统总结

该方面包含针对特定格式和工作流程定制的各种工具：

- **转换脚本 (`llama.cpp/convert.py`)：** 主要用于从标准格式创建 GGUF 文件。
- **量化 (quantization)库 (`AutoGPTQ`, `AutoAWQ`)：** 执行高级量化（如 GPTQ、AWQ），并以与 `transformers` 兼容的目录格式保存模型。
- **Hugging Face `transformers`：** 加载多种模型类型的核心库，包括使用 GPTQ/AWQ 量化或集成 `bitsandbytes` 的模型。
- **Hugging Face `optimum`：** 扩展 `transformers`，以提供与各种硬件加速器和量化后端更好的集成和优化。
- **`bitsandbytes`：** 通过 `transformers` 在模型加载期间实现实时量化（NF4，INT8）。
- **推理 (inference)引擎 (`llama.cpp`, `ctransformers`)：** 专为加载和执行 GGUF 等特定格式而优化的运行时环境。

选择合适的工具取决于目标格式（GGUF、GPTQ、AWQ）、所需的量化方法，以及推理环境（使用 `transformers` 的 Python，专用 C++ 引擎）。了解这些工具对于从拥有训练好的模型到部署高效的量化版本之间的实际步骤非常重要。

## 参考资料

- [llama.cpp: Inference of LLaMA model in pure C/C++](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov and the llama.cpp community (2023)
  引入GGUF格式并提供模型转换和推理工具的开源项目。
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍GPTQ算法，一种在不降低性能的情况下对大型语言模型进行训练后量化的方法。
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: MLSys 2024; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  提出AWQ权重量化方法，通过激活感知调整来保持模型精度。
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face team (2023)
  `transformers`库的官方文档，一个用于加载和使用包括量化模型在内的各种模型的标准。
- [TimDettmers/bitsandbytes: 8-bit and 4-bit quantization for PyTorch](https://github.com/bitsandbytes-foundation/bitsandbytes) — Tim Dettmers and bitsandbytes contributors (2023)
  `bitsandbytes`的GitHub仓库，一个为PyTorch模型提供高效8位和4位量化，并与`transformers`集成的库。

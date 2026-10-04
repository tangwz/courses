---
course: "practical-llm-quantization"
chapter: "quantization-formats-tooling"
lesson: "awq-format-details"
sourceId: 4657
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/awq-format-details"
title: "AWQ 格式说明"
description: "涵盖了如何存储和加载使用 AWQ 方法量化的模型的具体内容。"
order: 4
plots: []
sourceHash: "825b83bc01746d79e08893a0f0f4d68fc203b40424697bc64ee43634f0e8641d"
sourceCorrections: []
---

激活值感知权重 (weight) 量化 (quantization) (AWQ) 表现突出，因为它根据相应的激活值大小选择性地保留权重精度，旨在提高准确度，特别是在极低位深下，如 4 位 (INT4)。具体来说，这涉及的不仅是根据权重本身计算权重组的缩放因子，还通过分析校准数据集中的激活模式来评估其重要性，进而确定缩放因子。

这种激活值感知的方法使得不仅需要存储打包的低位权重。精心计算的缩放因子在推理 (inference)时对于正确地反量化或使用量化权重进行计算必不可少。因此，AWQ 模型的“格式”围绕着这些量化权重及其关联缩放因子如何保存和组织以便后续使用展开。

### AWQ 量化 (quantization)模型的结构

不同于 GGUF，AWQ 没有单一、标准化的文件扩展名或独立容器来定义其“AWQ 格式”。相反，AWQ 实现通常会调整现有的模型存储约定，例如 Hugging Face Transformers 所用的约定，通过在标准模型组件旁保存特定文件。

当您使用 `autoawq` 等库量化模型时，输出目录通常包含：

1. **量化权重 (weight)：** 量化模型的核心。这些是低精度权重（例如，INT4、INT3）。由于标准张量格式原生不支持 INT4 等类型，这些权重通常被“打包”成标准整数类型（如 INT8，其中两个 INT4 值可放入一个 INT8 字节）。这些打包的权重通常以常见的张量序列化格式保存，如 `.safetensors` 或 `.bin` 文件（PyTorch 的 `pytorch_model.bin`）。
2. **缩放因子：** 这是 AWQ 模型的一个显著特点。这些浮点值在 AWQ 处理过程中根据激活统计数据计算得出，在推理 (inference)过程中正确缩放量化权重所需。它们通常按通道或按组保存，与量化时使用的粒度匹配（例如，组大小为 128 很常见）。这些缩放因子可以存储在与权重相同的张量文件中，也可以存储在单独的专用文件中。
3. **零点：** 尽管 AWQ 通常采用对称量化（零点为零），如果使用非对称量化，相应的零点（整数）也必须与缩放因子一起保存。这些通常与缩放因子以相同的粒度（按通道或按组）存储。
4. **更新的配置文件：** 模型的标准配置文件（例如，`config.json`）通常会被修改或增补。它需要包含元数据，表明模型已进行 AWQ 量化，并注明所用参数 (parameter)：
   - 量化方法标识符（例如，`"quant_method": "awq"`）。
   - 位深（例如，`"bits": 4`）。
   - 组大小（例如，`"group_size": 128`）。
   - 是否使用/保存零点。
   - 将量化层映射回其相应缩放因子和零点的信息。
   - 有时，它包含架构提示或指定自定义模型类（如 `"auto_map": { "AutoModelForCausalLM": "awq.models.auto.AutoAWQForCausalLM" }`）以引导加载库。

> 包含 AWQ 量化模型的目录中常见的组件，源自原始 FP32 模型。主要新增内容是打包的低位权重、激活值感知的缩放因子（和可选的零点），以及已更新的配置文件，其中指定了 AWQ 参数。

### 加载与使用 AWQ 模型

加载 AWQ 量化 (quantization)模型需要理解此结构的特定逻辑。通常无法直接使用标准的 `transformers` `from_pretrained` 调用加载 AWQ 模型，除非有适当的集成或自定义代码。专为 AWQ 设计的库会处理此过程：

1. **识别：** 加载代码首先检查 `config.json` 以将模型识别为 AWQ 量化，并获取位深和组大小等参数 (parameter)。
2. **权重 (weight)和参数加载：** 它加载打包的低位权重、必需的缩放因子和零点（如果有）。
3. **模型实例化：** 创建基础模型架构的实例。
4. **层替换：** 标准层（如 `torch.nn.Linear`）被替换为专用的 AWQ 层。这些自定义层旨在利用量化权重进行计算，并在前向传播过程中高效地整合缩放因子（和零点）。这通常涉及即时解包权重或使用自定义计算核。
5. **核设置（可选）：** 为实现最佳性能，尤其是在 GPU 上，AWQ 库通常依赖自定义 CUDA 核进行包含缩放的低位矩阵乘法。加载过程确保这些核可用并已配置。

使用 `autoawq` 等库的典型加载模式大大简化了用户的操作：

```python
# 示例：使用 autoawq 库
from awq import AutoAWQForCausalLM

# 包含 config.json、量化权重等文件的目录路径
quantized_model_dir = "path/to/your/awq_model" 

# 加载量化模型和分词器
# 库会处理读取配置、加载权重/缩放因子、设置层等操作
model = AutoAWQForCausalLM.from_quantized(quantized_model_dir, device_map="auto")
# 分词器通常单独标准加载
# tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir) 

# 模型现在已可用于推理
```

### 实际考量

- **库依赖性：** 由于没有单一的文件标准，AWQ 模型通常与创建它们的库紧密关联（例如，使用 `autoawq` 量化 (quantization)的模型最好使用 `autoawq` 加载）。请确保在量化和推理 (inference)时使用兼容的库版本。
- **核要求：** 高性能通常依赖于自定义 CUDA 核。这意味着兼容的 GPU、CUDA 工具包版本以及正确安装的库组件是必需的。如果核不可用，推理可能会退回到较慢的实现方式。
- **格式差异：** 尽管原理相似，但不同 AWQ 实现或库版本之间，缩放因子/权重 (weight)存储或命名方式可能存在微小差异。务必参阅所用量化工具的文档。

了解 AWQ 模型保存的组件有助于管理这些模型，并认识到专用库在高效加载和运行它们方面的作用。主要收获是存储和正确使用激活值感知缩放因子的必要性，这使得 AWQ 的存储要求有别于更简单的量化方法。

---
course: "practical-llm-quantization"
chapter: "quantization-formats-tooling"
lesson: "gptq-format-support"
sourceId: 4656
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/gptq-format-support"
title: "GPTQ格式：库支持与应用"
description: "阐述GPTQ量化模型常用的格式和支持库。"
order: 3
plots: []
sourceHash: "5d17a4044f917aef004c803662f50793836b03f1d0f5e1eb77a071363c6b9193"
sourceCorrections: []
---

尽管GGUF等技术旨在提供独立的文件格式，采用GPTQ算法量化 (quantization)的模型通常遵循一系列约定，而非单一的标准化文件结构。GPTQ主要侧重于将模型（通常是线性层）的权重 (weight)量化为INT4或INT3等极低位宽，同时通常将激活值保持在较高精度（如FP16）。这种仅对权重进行量化的方法需要存储和正确加载特定信息。

### GPTQ模型存储约定

GPTQ量化 (quantization)模型通常由以下部分组成，而非单个巨型文件：

1. **量化权重 (weight)：** 模型的核心。原始的FP32或FP16权重被转换为低位整数（例如4位）。由于标准数据类型无法高效地原生支持4位或3位存储，这些低位权重被打包到更大的标准整数类型中，如INT8或INT32。例如，八个4位整数可以打包到一个32位整数中。
2. **量化参数 (parameter)：** 这些参数对于在推理 (inference)期间重建近似权重值非常重要。对于每组权重（由量化过程中使用的`group_size`参数定义），会存储以下信息：
   - **缩放因子：** 通常以FP16格式存储。这些缩放因子用于将低位整数范围映射回原始权重的近似浮点范围。
   - **零点：** 通常也以FP16或低位整数格式存储。零点将整数表示的零值与浮点表示的零值对齐 (alignment)。
3. **元数据：** 加载库正确解析量化权重和参数所需的配置信息。这通常包括：
   - `bits`：用于量化的位宽（例如4、3、8）。
   - `group_size`：每列（或行，取决于具体实现）中共享相同缩放因子和零点的权重数量。常见值为32、64或128。较小的组大小通常会带来更高的精度，但需要存储更多元数据。
   - `damp_percent`：GPTQ校准过程中使用的阻尼百分比（与Hessian计算相关）。
   - `desc_act`：一个布尔值，指示GPTQ算法中使用了激活顺序还是权重顺序（`True`表示激活顺序，通常更受青睐）。
   - 如果列在量化过程中被重新排序，可能会包含关于置换顺序的信息（在较简单的GPTQ实现中较不常见）。
4. **原始模型结构：** 模型的其余架构（非量化层、`config.json`等配置文件、分词 (tokenization)器 (tokenizer)文件）与原始FP16/FP32模型基本保持不变。

这些文件集合（打包权重、缩放因子、零点、元数据和原始结构文件）在实际中构成了“GPTQ格式”。在Hugging Face Hub等平台上，这些组件通常一起存储在模型仓库中。一个特定文件，通常命名为`quantize_config.json`，经常用于存储量化元数据（`bits`、`group_size`等）。

### 加载与推理 (inference)的库支持

在推理过程中解包权重 (weight)和应用量化 (quantization)参数 (parameter)的复杂性由专门的库处理。手动实现GPTQ的底层解包和优化矩阵乘法具有挑战性。以下是一些主要的库：

1. **AutoGPTQ：** 这个库是执行GPTQ量化以及使用所得模型进行推理最常用的工具之一。它提供优化的CUDA内核，以便在NVIDIA GPU上进行快速推理。当您使用依赖AutoGPTQ的库加载GPTQ模型时，它会即时处理反量化（或更准确地说，量化计算）。
2. **Hugging Face Transformers：** `transformers`库与`auto-gptq`提供了流畅的集成。通过将`auto-gptq`作为依赖项安装，您通常可以直接使用熟悉的`AutoModelForCausalLM.from_pretrained()`方法加载GPTQ模型。该库会检测GPTQ参数的存在（通常通过`quantize_config.json`），并自动使用`auto-gptq`后端进行加载和推理。这提供了一个高级且用户友好的界面。

   ```python
   # 示例：使用Hugging Face Transformers加载GPTQ模型
   # （假设已安装'auto-gptq'和'optimum'）

   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_id = "TheBloke/Llama-2-7B-Chat-GPTQ" # 示例GPTQ模型ID

   # 像往常一样加载分词器
   tokenizer = AutoTokenizer.from_pretrained(model_id)

   # 加载量化模型
   # Transformers检测到GPTQ并使用auto-gptq后端
   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       device_map="auto" # 自动将层分布到可用的GPU/CPU上
   )

   # 现在'model'已准备好使用优化的GPTQ内核进行推理
   # 示例推理：
   # prompt = "什么是量化？"
   # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
   # outputs = model.generate(**inputs, max_new_tokens=50)
   # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

   > 这个代码片段展示了从Hugging Face Hub加载GPTQ模型。`transformers`库结合`auto-gptq`，处理了加载打包权重和量化参数的底层复杂性。
3. **Hugging Face Optimum：** 尽管`transformers`提供了主要接口，但`optimum`通常与其搭配使用，提供硬件加速优化。对于GPTQ，它帮助`transformers`与`auto-gptq`等后端衔接。
4. **ExLlama / ExLlamaV2：** 这些是高度优化的推理库，专门设计用于在NVIDIA GPU上高效运行GPTQ和类似的权重量化模型（如EXL2格式）。通过使用为GPTQ推理中涉及的特定操作（如解包位和执行量化矩阵乘法）精心调整的自定义CUDA内核，它们通常比通用库实现更高的吞吐量 (throughput)和更低的延迟。使用ExLlama通常涉及与标准`transformers` API略有不同的加载过程，但能带来可观的性能优势。

### 实施注意事项

- **依赖项：** 运行GPTQ模型需要在`transformers`之外安装相应的后端库（例如`auto-gptq`）。这些后端通常有特定的CUDA版本要求。
- **硬件：** GPTQ推理 (inference)的性能高度依赖CUDA。尽管某些库可能存在CPU实现，但它们通常慢得多，且并非主要应用场景。
- **兼容性：** 确保推理库版本与用于量化 (quantization)模型的工具兼容。使用新版`auto-gptq`量化的模型可能需要相应更新的版本才能加载。
- **加载时间：** GPTQ模型初始加载时间可能比FP16模型稍长，因为推理库需要处理并可能重新排列权重 (weight)。然而，推理速度和内存使用量的减少是其主要优势。

了解这些约定和支持库的作用，可以有效使用GPTQ量化模型，以高效地部署LLM，主要是在GPU硬件上。Hugging Face周围的生态系统使得这些模型的加载和使用相对简单，屏蔽了大部分底层复杂性。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍了原始的GPTQ算法，该算法是所描述量化格式和实现规范的基础。
- [Quantization with Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/quantization) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Transformers文档中的官方指南，内容涵盖如何加载和使用量化模型，包括采用GPTQ格式的模型。
- [ExLlamaV2 GitHub Repository](https://github.com/turboderp/exllamav2) — turboderp and ExLlamaV2 contributors (2024)
  Publisher: turboderp-org
  ExLlamaV2的仓库，这是一个经过优化的推理库，专门用于在NVIDIA GPU上快速且内存高效地执行GPTQ和类似的权重量化模型。

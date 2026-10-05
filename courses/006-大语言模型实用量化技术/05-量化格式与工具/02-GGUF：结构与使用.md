# GGUF：结构与使用

来源：[原文](https://apxml.com/zh/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/gguf-format)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准模型保存方法在处理量化 (quantization)模型的一些特定需求时，往往不足，例如存储低位宽权重 (weight)及其相关的缩放因子和零点。GGUF（Georgi Gerganov Universal Format）作为一种实用方案出现，它专为满足这些需求而设计，尤其是在`llama.cpp`生态系统内以及用于高效的CPU推理 (inference)，尽管它也支持GPU加速。

GGUF 从早期的 GGML 格式演变而来，目标是为分发和运行量化大型语言模型创造一个更具扩展性的文件格式。它的主要目的是将运行模型所需的一切，包括架构细节、分词 (tokenization)器 (tokenizer)信息、量化参数 (parameter)以及量化权重本身，打包到一个可移植的单个文件中。

### GGUF 文件结构

A GGUF 文件是一种二进制格式，由三个主要部分组成：文件头、元数据和张量数据。

1. **文件头：** 包含一个魔数（`GGUF`）来标识文件类型和格式版本号。这使得工具能够快速验证它们是否正在读取有效的 GGUF 文件，并处理潜在的版本差异。
2. **元数据：** 一个灵活的键值存储，存放关于模型的重要信息。这是对旧格式的显著改进，因为它允许存储加载和正确运行模型所需的任意细节，且没有歧义。常见的元数据键包括：

   - `general.architecture`：指定模型架构（例如，`llama`、`mistral`）。
   - `general.name`：模型的一个易读名称。
   - `[architecture].context_length`：模型支持的最大序列长度。
   - `[architecture].embedding_length`：模型嵌入 (embedding)的维度。
   - `[architecture].block_count`：transformer 块/层的数量。
   - `tokenizer.ggml.model`：使用的分词 (tokenization)器 (tokenizer)模型（例如，`llama`、`gpt2`）。
   - `tokenizer.ggml.tokens`：词汇表 (vocabulary)中的分词列表。
   - `tokenizer.ggml.scores`：与分词相关的可选分数（例如，用于 SentencePiece）。
   - `tokenizer.ggml.merges`：BPE 分词器的合并规则。
   - `tokenizer.chat_template`：用于格式化聊天提示的 Jinja2 模板（可选但非常实用）。
   - 张量信息：文件中每个张量的名称、维度、数据类型和偏移量。这包括每个量化 (quantization)张量使用的量化类型（例如，`Q4_0`、`Q8_K_S`）。

   这些丰富的元数据使得 GGUF 文件可以独立运行，减少了对单独配置文件或关于模型细节的先验知识的需求。
3. **张量数据：** 一个连续的二进制数据块，包含实际的模型参数 (parameter)（权重 (weight)、偏置 (bias)、归一化 (normalization)参数）。对于量化模型，此部分存储低精度整数值以及反量化所需的任何相关缩放因子或元数据，它们根据元数据中指定的量化类型（如`Q4_0`、`Q5_K`、`Q8_0`、`F16`、`F32`等）进行打包。数据通常对齐 (alignment)到特定的字节边界，以便高效地进行内存映射和加载。

> GGUF 文件简化结构，呈现文件头、元数据部分和张量数据块。

### GGUF 中的量化 (quantization)类型

GGUF 在其规范中直接定义了各种量化类型。这表示一个文件可以包含使用不同方法量化的张量。常见例子包括：

- `F32`：标准 32 位浮点（未量化）。
- `F16`：16 位浮点。
- `Q8_0`：使用块大小 32 的 8 位量化，每个块一个浮点缩放。
- `Q4_0`：4 位量化（类型 0），块大小 32，一个浮点缩放。
- `Q4_K`：使用 K-Quant 策略的 4 位量化，通常提供优于 `Q4_0` 的质量。使用 256 个块，FP16 缩放/最小值。
- `Q5_K`：使用 K-Quant 的 5 位量化。
- `Q6_K`：使用 K-Quant 的 6 位量化。
- `IQ2_XS`：2 位量化（实验性/较新类型）。

每个张量使用的具体类型在元数据中列出，使得推理 (inference)引擎能够应用正确的反量化逻辑。“K-Quant”类型（`_K`）通常指那些旨在提高精度、优于简单量化方案的实现，它们通常涉及更复杂的块结构或缩放因子。

### GGUF 的优势

- **可移植性：** 将模型、配置和分词 (tokenization)器 (tokenizer)数据打包到单个文件中，使分发和使用变得简单。
- **可扩展性：** 键值元数据系统允许添加新信息而不会破坏向后兼容性。也可以加入新的量化 (quantization)类型。
- **快速加载：** 为高效加载而设计，通常通过内存映射（`mmap`）来按需加载张量数据，减少启动时间和内存占用，尤其对于大型模型。
- **广泛支持：** 主要由 `llama.cpp` 推动，它在专注于本地推理 (inference)的开源 LLM 社区中的各种工具和库中得到了采用。

### 与 llama.cpp 及其他工具的使用

GGUF 文件是流行的 `llama.cpp` 推理 (inference)引擎的本地格式。使用 `llama.cpp` 运行 GGUF 模型通常很简单：

```bash
# 使用 llama.cpp 运行推理的示例命令
./main -m path/to/your/model.gguf -p "What is the capital of Malaysia?" -n 50
```

在这里，`-m` 指定 GGUF 模型文件的路径。`llama.cpp` 读取元数据，确定架构和量化 (quantization)方案，加载张量，并执行推理。

创建 GGUF 文件通常涉及转换脚本。例如，`llama.cpp` 仓库包含 Python 脚本（`convert.py`、`convert-hf-to-gguf.py`），它们可以将 Hugging Face Transformers 格式（PyTorch `.bin` 或 `safetensors` 文件）的模型转换为 GGUF，并在该过程中应用所需的量化。在此转换步骤中，通常会指定目标量化类型（例如，`q4_K_M`、`q8_0`）。

```bash
# 将 Hugging Face 模型转换为 GGUF 的示例
python llama.cpp/convert.py models/my-hf-model --outfile models/my-model.q4_K_M.gguf --outtype q4_K_M
```

在使用本地运行的 LLM 时，了解 GGUF 格式很重要，因为它提供了一种标准化且高效的方式来处理量化模型，尤其是在典型的以 Python/GPU 为中心的深度学习 (deep learning)框架之外。它的设计选择优先考虑了在更广泛的硬件上进行推理的易用性和性能。

## 参考资料

- [GGUF File Format Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) — Georgi Gerganov and `llama.cpp` Contributors (2024)
  定义了 GGUF 文件的二进制格式、头、元数据和张量数据结构。
- [\`llama.cpp\`](https://github.com/ggerganov/llama.cpp) — Georgi Gerganov and `llama.cpp` Contributors (2024)
  Publisher: ggml-org
  创建 GGUF 的项目，提供 GGUF 模型在不同硬件上高效推理和转换的工具。
- [\`ggml\` - Machine learning in C/C++](https://github.com/ggml-org/ggml) — Georgi Gerganov and `ggml` Contributors (2024)
  Publisher: GitHub
  `llama.cpp` 和 GGUF 底层的核心 C/C++ 张量库，实现低精度计算。
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 2022; DOI: [10.48550/arXiv.2208.07339](https://doi.org/10.48550/arXiv.2208.07339)
  介绍了大型语言模型中 8 位量化的方法，展示了部署时内存和效率方面的优势。

---

[上一节](01-%E5%B8%B8%E8%A7%81%E9%87%8F%E5%8C%96%E6%A8%A1%E5%9E%8B%E6%A0%BC%E5%BC%8F%E6%A6%82%E8%BF%B0.md) · [下一节](03-GPTQ%E6%A0%BC%E5%BC%8F%EF%BC%9A%E5%BA%93%E6%94%AF%E6%8C%81%E4%B8%8E%E5%BA%94%E7%94%A8.md)

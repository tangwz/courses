# 使用bitsandbytes进行量化

来源：[原文](https://apxml.com/zh/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/using-bitsandbytes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`bitsandbytes`库提供了执行高效低比特计算所需的软件引擎，尤其是在模型推理 (inference)期间。这个库与GGUF和GPTQ等专用格式相辅相成，后者定义了量化 (quantization)模型如何在磁盘上存储和构成。它是CUDA自定义函数的一个轻量级封装，使得可以使用8比特或4比特精度的权重 (weight)进行矩阵乘法，大幅减少了大型模型的内存占用。

可以将`bitsandbytes`视为运行时加速器，而非文件格式本身。它使得加载那些原本可能因GPU内存不足而无法加载的模型成为可能，通过使用更少比特表示权重。该库与流行框架良好集成，最显著的是Hugging Face Transformers，使得以更低精度加载和运行模型相对简单。

### 为什么要用`bitsandbytes`？

运用`bitsandbytes`的主要原因在于**内存占用减少**。例如，以4比特精度加载模型权重 (weight)，相比标准16比特浮点格式（如FP16或BF16），可以将内存需求减少近四倍。这使得以下成为可能：

1. 在显存 (VRAM)有限的硬件上（例如消费级GPU）**运行大型模型**。
2. 如果内存是瓶颈，则在推理 (inference)期间**增加批处理大小**以获得更高吞吐量 (throughput)。

尽管内存节省是主要优点，但`bitsandbytes`有时可以提供推理加速，但这并非总能保证。该过程涉及将权重量化 (quantization)为低精度（INT8、NF4），但通常以更高精度格式（例如BF16或FP16）执行实际计算（如矩阵乘法）。这种混合精度方法需要即时反量化权重，这会带来一些计算开销。然而，对于内存受限的情况，能够将模型适配到硬件上是最大的好处。

### 核心功能和原理

`bitsandbytes`引入了几种方法来使低比特量化 (quantization)有效：

1. **混合精度分解：** 核心思想是将权重 (weight)存储为低比特格式（INT8或FP4/NF4），但执行矩阵乘法（$A \times B$），其中一个矩阵（例如激活$A$）采用FP16/BF16格式，而另一个矩阵（权重$B$）在计算前从低比特格式动态反量化。这在保持合理精度的同时，为权重实现了显著的内存节省。
2. **8比特量化（`LLM.int8()`）：** 这是`bitsandbytes`早期推广的一项突破。它使用向量 (vector)级量化方案结合混合精度分解。它识别并分离激活中的系统性异常特征，以FP16处理这些特征，同时将其余部分量化为INT8。相比简单的INT8量化，这能更好地保持精度。
3. **4比特量化（NF4和双重量化）：** 这提供了更大的内存节省。

   - **NF4（NormalFloat 4比特）：** `bitsandbytes`支持NF4数据类型，而非标准整数量化。这种格式的设计基于模型权重通常服从正态分布的假设。NF4对正态分布数据是信息理论上的最佳选择，意味着它比标准整数或浮点格式能以每比特更高的效率表示此类数据。它使用分位数创建非对称数据桶，能更好地捕捉权重的典型分布。
   - **双重量化（DQ）：** 为了进一步减少内存开销，DQ对量化常数本身进行量化。在典型的量化中，每个块或张量都有相关的缩放因子和可能的零点，通常存储为FP32或FP16。DQ对这些常数应用第二层量化，通常使用8比特浮点数，大幅减少了这种元数据开销，对于4比特模式下使用的较小量化块大小尤其重要。

### 与Hugging Face Transformers的集成

运用`bitsandbytes`最常见的方式是通过Hugging Face `transformers`库。它允许直接将模型加载为8比特或4比特精度，只需几个配置标志。

#### 以8比特加载

要使用8比特量化 (quantization)加载模型，只需在调用`from_pretrained`时设置`load_in_8bit`标志：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B" # 示例模型ID
# 确保bitsandbytes已安装：pip install bitsandbytes

try:
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 以8比特加载模型
    model_8bit = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto" # 将模型分布到可用设备（GPU/CPU）上
    )

    print(f"模型 {model_id} 已以8比特加载。")
    print(f"内存占用：{model_8bit.get_memory_footprint() / 1e9:.2f} GB")

except Exception as e:
    print(f"加载8比特模型时出错：{e}")
    print("检查是否有足够的GPU内存和CUDA设置。")
```

设置`device_map="auto"`很重要，因为它会自动处理将模型层放置到可用GPU上（如果没有找到GPU或内存不足，则放置到CPU上），运用Accelerate库的集成。

#### 以4比特加载

以4比特加载需要更多配置，使用`BitsAndBytesConfig`类：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B" # 示例模型ID
# 确保bitsandbytes和accelerate已安装：
# pip install bitsandbytes accelerate transformers torch

try:
    # 配置4比特量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,              # 启用4比特加载
        bnb_4bit_quant_type="nf4",      # 使用NF4量化类型
        bnb_4bit_use_double_quant=True, # 启用双重量化
        bnb_4bit_compute_dtype=torch.bfloat16 # 设置矩阵乘法的计算数据类型
    )

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 使用4比特配置加载模型
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto" # 自动映射模型层
    )

    print(f"模型 {model_id} 已以4比特加载（NF4加DQ）。")
    print(f"内存占用：{model_4bit.get_memory_footprint() / 1e9:.2f} GB")

    # 示例：检查量化线性层权重的Ddata类型
    # 注意：直接访问权重可能显示封装对象
    # 示例：找到一个线性层并检查其属性（可能因模型架构而异）
    # layer_name = model_4bit.model.layers[0].self_attn.q_proj # 示例路径，根据模型调整
    # print(f"权重属性（示例层）：{type(layer_name.weight)}，{layer_name.weight.dtype}")

except Exception as e:
    print(f"加载4比特模型时出错：{e}")
    print("检查GPU兼容性（CUDA计算能力）、内存和库版本。")
```

在此配置中：

- `load_in_4bit=True` 启用4比特模式。
- `bnb_4bit_quant_type="nf4"` 指定NormalFloat 4比特数据类型。另一个选项是标准4比特浮点的`"fp4"`，但NF4通常能获得更好的精度。
- `bnb_4bit_use_double_quant=True` 启用双重量化优化，以降低内存开销。
- `bnb_4bit_compute_dtype=torch.bfloat16` (或 `torch.float16`) 设置内部计算中使用的数据类型。在较新硬件（Ampere+）上，BF16通常更受青睐，因为它具有更宽的动态范围，与FP16相比可能提高稳定性和精度，同时内存成本相同。

### 性能考量

运用`bitsandbytes`的主要好处是将模型适配到有限的显存 (VRAM)中。对于Llama 3 8B这样的模型，差异很显著：

- FP16/BF16：约16 GB显存
- 8比特：约9 GB显存
- 4比特：约5 GB显存

这使得在以前显存不足的GPU上运行模型成为可能。然而，推理 (inference)速度可能并非总是按比例提高。动态反量化 (quantization)会增加开销。在内存和带宽充足的高端GPU上，运行原生FP16/BF16可能仍然更快。但在内存受限的系统上，4比特或8比特量化往往是运行模型的唯一方式，因此速度比较变得次要。`bnb_4bit_compute_dtype`的选择也会影响速度和精度；BF16计算在兼容硬件上通常更快。

总之，`bitsandbytes`是LLM量化工具集中必不可少的库，提供运行时机制来执行加载了8比特或4比特权重 (weight)的模型。它与Hugging Face Transformers的顺畅集成，使希望在现有硬件上更高效运行大型模型的从业者易于使用。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: Advances in Neural Information Processing Systems; Volume: 36; Pages: 44907-44923; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  这篇基础论文介绍了NormalFloat 4位（NF4）数据类型和双量化技术，它们是`bitsandbytes`用于4位模型量化的关键技术。
- [bitsandbytes: Lightweight Custom CUDA functions for 8-bit and 4-bit Quantization](https://github.com/TimDettmers/bitsandbytes) — The bitsandbytes team (2024)
  Publisher: bitsandbytes-foundation
  `bitsandbytes`库的官方仓库和主要文档，涵盖了其低位计算功能的安装、特性和使用方法。
- [Quantization](https://huggingface.co/docs/transformers/main/en/quantization) — The Hugging Face team (2024)
  Publisher: Hugging Face
  Hugging Face Transformers的官方文档提供了在`transformers`库中集成和使用`bitsandbytes`进行8位和4位模型加载的实用指南。

---

[上一节](05-%E4%BD%BF%E7%94%A8%20Hugging%20Face%20Transformers%20%E5%92%8C%20Optimum.md) · [下一节](07-%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2%E4%B8%8E%E5%8A%A0%E8%BD%BD%E5%B7%A5%E5%85%B7.md)

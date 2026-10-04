---
course: "quantized-llm-deployment"
chapter: "implementing-llm-quantization-toolkits"
lesson: "comparing-toolkit-outputs"
sourceId: 4623
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-2-implementing-llm-quantization-toolkits/comparing-toolkit-outputs"
title: "比较工具包的输出和性能"
description: "分析使用不同量化工具包时，输出和性能的差异。"
order: 6
plots: ["plots/4623-0.json", "plots/4623-1.json"]
sourceHash: "0bd5b1215b2789f147ba33f8da7c26ceef76efcb6f02f5a171bc569b6b661d3e"
sourceCorrections: []
---

量化 (quantization)工具包，包括 `bitsandbytes`、`AutoGPTQ` 和 `AutoAWQ`，在对 LLM 应用量化技术方面发挥着重要作用。虽然使用这些工具并针对相同算法（例如 GPTQ 4 比特）进行处理时，其结果可能看似可互换，但实际情况更为复杂。不同的工具包通常具有不同的实现、输出格式和性能特点，即使它们基于相同的底层量化原理。评估这些差异是选择合适工具并优化部署流程的重要一步。

本节将阐述如何比较各种LLM量化库的输出和性能影响。我们将查看量化模型的结构、准确性指标的变化，以及推理 (inference)速度和内存使用的基准。理解这些细节有助于您做出明智的决定，选择最适合您的特定模型、硬件目标和性能需求的工具包。

### 比较量化 (quantization)模型文件

当您使用不同的工具包量化模型时，生成的文件（常称为文件产物）可能会有很大差异。这些差异影响模型在推理 (inference)期间的存储、加载和使用方式。

- **文件结构和格式：**

  - **`bitsandbytes` (通过Hugging Face `Transformers`)：** 量化参数 (parameter)通常直接集成到模型的状态字典或配置文件（`config.json`，`quantization_config.json`）中。使用`transformers`库加载模型时，配合正确的标志（`load_in_4bit=True`，`load_in_8bit=True`）可以动态处理`bitsandbytes`核心的运用。保存的模型可能类似于标准的Hugging Face模型检查点，但附加了量化元数据。
  - **`AutoGPTQ`：** 通常将量化权重 (weight)保存为特定格式（例如`.safetensors`或`.pt`），并附带一个配置文件（`quantize_config.json`），其中详细说明GPTQ参数（比特数、组大小、对称/非对称等）。加载通常需要使用`AutoGPTQ`库本身或专门设计用于处理其输出格式和核心的推理引擎。
  - **`AutoAWQ`：** 与`AutoGPTQ`类似，它通常生成量化权重和指定AWQ参数的配置文件。推理性能通常依赖于`vLLM`等库提供或支持的定制核心，或理解AWQ格式的专用Triton核心。
- **元数据：** 与量化权重一同存储的元数据很重要。它包含量化比特宽度（$w_{bits}$）、组大小（$g$）、量化方案（对称/非对称），以及可能的缩放因子（$s$）和零点（$z$）等信息。此元数据存储和解释方式的差异可能会影响工具包和推理服务器之间的兼容性。
- **兼容性：** 一个主要考虑点是兼容性。使用`AutoGPTQ`量化的模型可能无法直接使用标准的PyTorch `load_state_dict`函数加载，或者无法在没有特定转换步骤或对该格式的支持下立即被TensorRT-LLM等推理服务器使用。另一方面，通过`Transformers`集成的`bitsandbytes`通常在该生态系统中提供更流畅的体验，但可能需要特定版本或硬件支持其优化的核心。

### 分析量化 (quantization)保真度和准确性

即使应用相同的名义量化方法（例如4比特GPTQ），不同工具包在模型准确性方面也可能产生略微不同的结果。

- **实现差异：** 像GPTQ或AWQ等算法实现中的微小差异，例如校准期间的数值精度、边缘情况的处理，或应用量化比例和零点的具体方法，都可能导致结果的不同。
- **校准敏感性：** 训练后量化（PTQ）方法（如GPTQ和AWQ）依赖于校准数据。尽管您可能使用相同的数据集，但工具包处理或使用它的方式可能略有不同，从而影响最终的量化参数 (parameter)。
- **评估指标：** 为比较保真度，请使用标准指标评估量化模型：
  - **困惑度：** 在保留的验证数据集上衡量困惑度。较低的困惑度通常表示模型语言建模能力保留得更好。
  - **下游任务准确性：** 使用相关准确性指标（ROUGE、F1分数、准确率）评估LLM预期用于的特定任务（例如摘要、问答、分类）的性能。

工具包之间在困惑度或任务准确性上的微小差异很常见。与原始FP16/BF16模型相比出现显著下降，或工具包之间存在较大差异，可能表示量化过程或实现细节存在问题。



![困惑度比较（越低越好）](plots/4623-0.json)



> 两种不同模型使用各种工具包量化为4比特后的困惑度分数。虽然分数接近，但存在细微差异，如果差异较大，则需要进一步检查。

### 基准测试性能：速度和内存

量化 (quantization)的主要目的通常是提升性能。比较使用不同工具包量化的模型的推理 (inference)速度和内存占用量是必要的。

- **指标：**

  - **延迟：** 单次推理请求所需的时间（例如，生成固定数量的token）。通常以每token毫秒数或总生成时间来衡量。
  - **吞吐量 (throughput)：** 单位时间内处理的请求数量或输出token数量（例如，每秒token数）。对于处理并发请求的服务器部署尤其重要。
  - **内存使用：**
    - **磁盘大小：** 保存的模型文件产物的大小。量化模型应比其全精度版本小很多。
    - **运行时内存（VRAM）：** 推理期间的GPU内存峰值使用量。这通常是运行大型模型的瓶颈。
- **基准测试考虑因素：**

  - **推理引擎：** 性能受到所用推理引擎的显著影响。使用`AutoGPTQ`量化的模型，在使用为其专门构建的优化核心加载时可能表现最佳，这可能发生在支持`AutoGPTQ`的`vLLM`或`TGI`中。`bitsandbytes`量化模型则依赖于集成到`Transformers`中的核心的效率。请使用预期的部署框架进行基准测试。
  - **硬件：** 性能因硬件（例如，A100与H100等不同代GPU）而异。确保在目标硬件上进行比较。
  - **工作负载：** 使用真实的工作负载进行测试（例如，典型的输入长度、输出长度、批处理大小）。



![推理性能（7B模型，A100 GPU）](plots/4623-1.json)



> 示例基准测试结果，比较了使用不同工具包量化并在NVIDIA A100 GPU上运行兼容且优化推理核心的7B参数 (parameter)模型的延迟、吞吐量和VRAM使用情况。性能可能因所使用的具体核心而异。

### 工具包特点和权衡

选择工具包需要同时考虑这些比较以及可用性和生态系统因素：

- **通过Hugging Face使用`bitsandbytes`：**
  - **优点：** 与Hugging Face生态系统出色集成，相对易于使用（`load_in_4bit=True`），支持NF4等流行格式。
  - **缺点：** 性能可能很大程度上取决于您的硬件上可用并优化的具体`bitsandbytes`核心。与专用库相比，可能提供较少的配置选项。
- **`AutoGPTQ`：**
  - **优点：** GPTQ算法的专用实现，通常使用专门核心（如ExLLama）可以获得良好的性能，社区活跃支持。
  - **缺点：** 加载和推理 (inference)需要特殊处理，模型格式可能不太标准化，性能与兼容推理核心的可用性和质量相关。
- **`AutoAWQ`：**
  - **优点：** 实现AWQ，理论上通过保留重要权重 (weight)提供更好的性能，与`vLLM`等高性能引擎集成。
  - **缺点：** 与`AutoGPTQ`类似，依赖于特定核心和格式以获得最佳性能。与GPTQ相比，可能稍新或模型兼容性有所不同。

最终，“最佳”工具包取决于您的目标。如果与Hugging Face的顺畅集成很重要，`bitsandbytes`可能是起点。如果目标是使用`vLLM`或特定硬件核心追求最大吞吐量 (throughput)，那么`AutoGPTQ`或`AutoAWQ`可能更合适，前提是您能管理相关的格式和核心依赖。

系统地进行这些比较使您能够选择最能在准确性、性能和集成便捷性之间取得平衡的量化 (quantization)工具包和生成模型，以适应您的特定LLM部署场景。这种实证评估通常是必要的，因为理论优势并非总能直接转化为所有模型和硬件平台上的实际性能提升。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  描述了GPTQ算法，一种用于AutoGPTQ的训练后量化方法。
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  介绍了AWQ算法，一种由AutoAWQ实现的激活感知权重量化方法。
- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了NF4量化及其他技术，它们是bitsandbytes中4位量化的基础。

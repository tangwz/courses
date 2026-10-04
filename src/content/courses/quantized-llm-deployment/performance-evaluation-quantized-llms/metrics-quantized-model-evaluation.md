---
course: "quantized-llm-deployment"
chapter: "performance-evaluation-quantized-llms"
lesson: "metrics-quantized-model-evaluation"
sourceId: 4630
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-3-performance-evaluation-quantized-llms/metrics-quantized-model-evaluation"
title: "量化模型评估指标"
description: "界定合适的指标：困惑度、任务专用准确率、延迟、吞吐量、内存占用。"
order: 1
plots: ["plots/4630-0.json"]
sourceHash: "47c40684ffdd2ebfb00ba6d83cd64a244e333e126338343eca394d7d0b59b2ef"
sourceCorrections: []
---

对大型语言模型（LLM）进行量化 (quantization)，在效率方面具有重要益处，但对这些近似的影响进行彻底评估非常重要。选择合适的指标，可以客观衡量性能提升，并了解模型质量可能出现的任何下降。这项评估是决定量化模型是否满足特定应用要求的重要依据。

我们可以将评估指标大致分为两类：效率指标和质量指标。

## 衡量效率提升

量化 (quantization)主要旨在提升计算速度和内存使用。通常通过以下方式衡量这些方面：

### 推理 (inference)延迟

延迟衡量处理单个输入请求并生成输出所需的时间。对于响应速度很重要的面向用户应用来说，这是一个重要指标。

- **定义：** 从发送输入到接收单个推理实例的完整输出所经过的时间。
- **测量：** 通常报告为许多请求的平均延迟。然而，平均值可能会掩盖异常情况。报告P95（95百分位）或P99（99百分位）等百分位数可以更好地了解最差情况下的性能，这对于用户体验通常更具参考价值。
- **影响因素：** 延迟受模型大小、量化 (quantization)级别（更低的精度通常带来更快的计算）、输入/输出序列长度、批处理大小（通常，延迟随批处理大小增加，但吞吐量 (throughput)可能提升）以及底层硬件能力的影响。衡量延迟通常涉及直接对推理调用进行计时，如果只评估模型计算本身，则排除任何网络开销。

### 吞吐量 (throughput)

吞吐量衡量模型处理请求的速率。这对于服务大量并发用户的应用尤其重要。

- **定义：** 单位时间内处理的推理 (inference)请求数量（例如，每秒请求数、每秒token数）。
- **测量：** 通常通过在持续负载下运行模型（例如，有多个并发请求或大批量处理）并测量处理速率来计算。对于生成文本的LLM，吞吐量通常以每秒生成的输出token数衡量。
- **与延迟的关系：** 吞吐量和延迟相关但不同。批处理等技术可以通过并行处理多个请求来提高吞吐量，但通常会增加单个请求的延迟。优化高吞吐量通常涉及最大限度地发挥硬件效能。

### 内存占用

量化 (quantization)直接减少了模型所需的内存，包括存储和执行时的内存。

- **磁盘大小：** 这是最直接的优势。降低模型权重 (weight)的精度（例如，从16位浮点数到4位整数）可以大幅减小模型文件大小。这以兆字节（MB）或千兆字节（GB）衡量。更小的磁盘占用简化了模型分发并降低了存储成本。
- **运行时内存使用：** 这指的是推理 (inference)过程中消耗的RAM量，对于LLM而言更常见的是GPU显存 (VRAM)。更低精度的权重和激活值减少了加载模型和存储中间状态（如Transformer中的KV缓存）所需的内存。衡量峰值运行时内存使用通常需要特定硬件的分析工具（如NVIDIA GPU的`nvidia-smi`）或深度学习 (deep learning)框架提供的工具。减少运行时内存使得更大的模型能适应现有硬件，支持更大的批处理量以可能提高吞吐量 (throughput)，或有助于在内存有限的设备上部署。

## 评估模型质量

虽然效率提升是可取的，但它们不应以模型预测能力不可接受的代价为前提。评估质量涉及衡量量化 (quantization)模型在其预定语言任务上的表现如何。

### 困惑度（PPL）

困惑度是评估语言模型的标准内在指标。它衡量概率模型预测给定文本样本的准确程度。

- **定义：** 在数学上，PPL是序列的平均负对数似然的指数化。直观地讲，较低的困惑度分数表明模型对测试数据的“惊讶”程度较低，这表示它为观测到的序列分配了更高的概率，并且对语言结构有更好的把握。
  $\text{困惑度}(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log p(w_i | w_1, \dots, w_{i-1}) \right)$
  其中 $W = (w_1, w_2, \dots, w_N)$ 是测试序列，$N$ 是序列长度。
- **测量：** 通过在保留的验证数据集（例如WikiText、C4）上运行模型来计算。在比较原始模型及其量化 (quantization)版本的困惑度时，使用相同的数据集和分词 (tokenization)方法非常重要。
- **解释：** 尽管PPL很有用，但它有局限性。它侧重于预测下一个token，可能无法与特定下游任务的性能完全相关。如果在目标应用上的性能保持较高，量化后PPL的小幅增加可能是可接受的。

### 任务专用准确性和基准

评估LLM将用于的特定任务的性能，可以更直接地衡量质量下降。

- **方法：** 使用与您的应用相关的既定基准数据集。示例包括：
  - **问答：** SQuAD (F1分数, 准确匹配)
  - **摘要：** CNN/Daily Mail (ROUGE分数: ROUGE-1, ROUGE-2, ROUGE-L)
  - **情感分析/分类：** GLUE, SuperGLUE 子集 (准确率, F1分数)
  - **一般推理 (inference)/知识：** MMLU (大规模多任务语言理解), HellaSwag, ARC
- **评估框架：** 像EleutherAI LM Evaluation或Hugging Face Evaluate这样的框架，可以简化在各种任务上运行这些基准测试。
- **人工评估：** 对于创意写作、对话生成或复杂指令遵循等任务，自动化指标可能不足。人工评估虽然资源消耗大，但可以在连贯性、相关性、安全性以及整体实用性等方面提供非常有用的信息，这些方面难以通过自动化方法衡量。

## 分析权衡

评估很少呈现出明确的胜利。量化 (quantization)通常涉及权衡：效率提升（更低延迟、更高吞吐量 (throughput)、更小占用）通常会伴随一定程度的质量下降（更高困惑度、更低任务准确率）。

可视化这些权衡很有帮助。例如，绘制不同量化级别（FP16、INT8、INT4）的准确率与延迟或内存使用情况的图表，可以显示效率边界。



![量化权衡：延迟 vs 准确率](plots/4630-0.json)



> 这是一个权衡图示例，展示了与原始FP16模型相比，不同量化方法下的相对延迟降低与潜在准确率下降。目标通常是接近“拐点”，在准确率损失最小的情况下实现明显的效率提升。

选择合适的量化策略在很大程度上取决于特定应用的约束。实时聊天机器人可能优先考虑低延迟，容忍轻微的准确率下降，而离线文档摘要工具可能优先考虑最大准确率，即使推理 (inference)时间更长。因此，采用一套适用的指标进行全面评估，对于做出关于部署量化LLM的明智决策是必要的。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2023)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍了一种广泛采用的大语言模型训练后量化方法，展示了在保持最小精度损失的情况下实现效率提升。
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: Proceedings of the 40th International Conference on Machine Learning (ICML); Volume: 202; Pages: 20443-20462; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  提出了一种利用激活统计数据来保持模型性能的量化方法，为高效部署大型语言模型提供了一种稳健的途径。
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) — Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han (2023)
  Journal: ICML 2023; Volume: 202; Pages: 39031-39048; DOI: [10.48550/arXiv.2211.10438](https://doi.org/10.48550/arXiv.2211.10438)
  提出了一种通过显著提高训练后量化精度来解决大型语言模型量化挑战的技术，对平衡效率和质量很重要。
- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) — Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2009.03300](https://doi.org/10.48550/arXiv.2009.03300)
  提供了一个综合性基准，用于评估语言模型在广泛主题上的知识和推理能力，这对于评估模型质量至关重要。

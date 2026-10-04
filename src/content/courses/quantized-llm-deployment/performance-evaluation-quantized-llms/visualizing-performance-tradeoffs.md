---
course: "quantized-llm-deployment"
chapter: "performance-evaluation-quantized-llms"
lesson: "visualizing-performance-tradeoffs"
sourceId: 4642
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-3-performance-evaluation-quantized-llms/visualizing-performance-tradeoffs"
title: "可视化性能权衡"
description: "用于可视化量化级别、性能和准确性之间关系的方法。"
order: 7
plots: ["plots/4642-0.json", "plots/4642-1.json"]
sourceHash: "c4e5648a44b36dc12cfc155b69a22f4969d7c081d18da128c41aa48baa351d55"
sourceCorrections: []
---

理解之前讨论过的延迟、吞吐量 (throughput)、内存使用和准确性等各项性能指标，提供了重要的数据点。然而，量化 (quantization)很少能同时改进所有方面。通常，激进的量化会以牺牲一些准确性为代价，提升效率（更低的延迟，更小的占用空间）。要明智地决定采用哪种量化策略，需要理解这些指标*之间*的关系。可视化提供了一种直观理解这些复杂权衡的有力方式。

通过将性能指标与准确性一同绘制，您可以快速找出哪些量化方法能为您的特定需求提供最佳平衡。这种视觉分析有助于理解不同量化选择的影响。

### 准确性-性能边界的可视化

可视化这些权衡的最常见且有效的方式是通过散点图。这些图通常将模型质量的衡量（如困惑度或基准任务上的准确性）放在一个轴上，将性能指标（如推理 (inference)延迟或模型大小）放在另一个轴上。图中的每个点代表一个特定的模型版本，通常对应不同的量化 (quantization)技术或位精度。

考虑一个比较准确性与推理延迟的图：



![不同量化方案的准确性与延迟对比](plots/4642-0.json)



> 针对使用各种方法量化的LLM，准确性分数与每个词元 (token)的平均推理延迟的对比图。延迟越低越好（向左），准确性越高越好（向上）。

一个表现出高准确性和低延迟的理想模型，在性能权衡的可视化中通常位于左上角。FP16 基准通常位于右侧（延迟较高），但准确性最高。不同的量化方法（INT8、INT4 变体）将操作点推向左侧，理想情况下准确性下降最小。像 AWQ 这样的技术在相似位宽或相似延迟下，可能比 GPTQ 达到略高的准确性，在这样的可视化中近似相同延迟下位置更高。极低位方法（如 INT3）可能提供最低延迟，但通常会带来显著的准确性损失，使它们位于更下方。

同样，您可以可视化准确性与模型大小之间的权衡：



![不同量化方案的准确性与模型大小对比](plots/4642-1.json)



> 准确性分数与模型磁盘大小的对比图。大小越小越好（向左），准确性越高越好（向上）。

此图突出了通过量化实现的内存节省。与 FP16 或 INT8 相比，INT4 方法大幅减小了模型大小，使得在内存受限设备上的部署变得可行。

### 解读权衡曲线

这些可视化有助于确定*帕累托前沿*，这是一个借鉴自多目标优化的想法。帕累托前沿代表了一组点（量化 (quantization)配置），在这些点上，你无法在不降低另一个目标（例如，降低准确性）的情况下改进一个目标（例如，减少延迟）。位于此前沿的模型代表了可用的最有效权衡。

分析这些图时：

- **确定基准：** 找到未量化模型（通常是 FP16 或 BF16）作为参考点。
- **评估效率提升：** 观察相对于基准，点向期望角落（例如，更低的延迟，更小的模型大小）移动了多远。
- **评估准确性影响：** 注意每次效率提升相关的垂直下降（准确性损失）。
- **比较方法：** 不同的量化算法（如 GPTQ 与 AWQ 与简单舍入）会在图上产生不同的点。比较它们的位置，以查看哪种为您的目标位宽提供了更好的权衡。
- **选择操作点：** 选择最符合您应用程序特定要求的量化模型。例如，实时应用程序可能优先考虑延迟，接受较小的准确性下降，而离线处理任务即使延迟略高也可能优先考虑最高准确性。

### 有效可视化的考量

这些权衡图上点的具体形状和位置在很大程度上取决于几个因素：

- **模型架构：** 不同的 LLM 架构（例如 Llama、Mistral、T5）对量化 (quantization)的响应不同。
- **模型大小：** 大型模型有时比小型模型对量化的容忍度更高。
- **硬件：** 性能指标（延迟、吞吐量 (throughput)）与硬件相关。在一块 GPU 上生成的图在另一块 GPU 或 CPU 上可能看起来不同。
- **量化参数 (parameter)：** 量化期间做出的选择，例如校准数据集大小或 GPTQ/AWQ 中的组大小，会影响结果。
- **评估基准：** 选择的准确性指标（困惑度、特定任务准确性如 MMLU、BoolQ）影响点的垂直位置。

因此，重要的是在与您的目标部署环境和评估标准紧密匹配的条件下生成这些可视化。它们不是普遍真理，而是特定情况下的性能快照。

通过系统地测量性能和准确性，并可视化由此产生的权衡，您可以获得必要的理解，从而有效选择和部署量化 LLM，平衡计算效率与预测质量。这些可视化是优化过程中重要的决策工具。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2023)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  介绍了GPTQ，一种广泛用于大型语言模型的训练后量化方法，详细阐述了其方法论和性能影响。
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: arXiv preprint arXiv:2306.00978; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  介绍了AWQ，一种为高效大型语言模型推理设计的激活感知权重量化技术，详细说明了其在平衡准确性和硬件效率方面的方案。

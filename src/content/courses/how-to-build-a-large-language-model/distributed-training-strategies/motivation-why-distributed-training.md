---
course: "how-to-build-a-large-language-model"
chapter: "distributed-training-strategies"
lesson: "motivation-why-distributed-training"
sourceId: 6021
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-15-distributed-training-strategies/motivation-why-distributed-training"
title: "动机：为什么要进行分布式训练？"
description: "重申内存和计算限制，使得分布式方法成为必然。"
order: 1
plots: ["plots/6021-0.json"]
sourceHash: "81ef146d6d0bc4ab54ac6c0832561b03248eef95e8c4acec5a36b50026a610cf"
sourceCorrections: []
---

目前训练巨型语言模型，这些模型通常包含数百亿甚至数千亿参数 (parameter)，带来了严峻的工程挑战，仅靠单个加速设备（如GPU或TPU）是无法应对的。

虽然较小的深度学习 (deep learning)模型可以轻松地在一块GPU内存中运行，但大型语言模型（LLMs）的规模使得这变得不再可行。尝试在单个设备上加载和训练这类巨型模型，会迅速遭遇硬件方面的基本限制，这些限制涉及内存容量和计算吞吐量 (throughput)。

我们来分析一下为什么分布式训练过程变得绝对必要。

### 内存限制：模型及其附属数据的存放问题

最直接的挑战是内存。一块高端加速卡，例如NVIDIA A100或H100 GPU，通常提供约40GB或80GB的高带宽内存（HBM）。这看起来容量很大，但LLM在训练期间的内存占用包括以下几个主要部分：

1. **模型参数 (parameter)：** 这些是模型学习到的权重 (weight)和偏置 (bias)。存储它们需要大量空间，尤其是在使用标准32位浮点精度（FP32）时。即使使用FP16或BFloat16 (BF16)等16位格式进行混合精度训练，仅参数存储就可能超出单个GPU的内存容量。例如，一个包含700亿参数的模型，若使用BF16（每个参数2字节）存储，则需要140GB内存，这已经远远超过了80GB GPU的容量。

   ```python
   # 参数的大致内存占用（单位GB）
   num_parameters = 70_000_000_000
   bytes_per_parameter_mixed_precision = 2 # 使用BF16或FP16

   param_memory_gb = (num_parameters * bytes_per_parameter_mixed_precision) / (1024**3)
   print(f"Approx. parameter memory (BF16/FP16): {param_memory_gb:.2f} GB")
   # 输出：BF16/FP16格式下大致的参数内存占用：130.39 GB
   # 注意：实际内存占用可能会因对齐/框架开销而略有不同。
   # 为了文本描述一致性，我们使用140GB这个数字。
   ```
2. **梯度：** 在反向传播 (backpropagation)过程中，会为每个参数计算梯度。这些梯度通常与参数本身具有相同的维度和精度要求。因此，在我们700亿参数的混合精度示例中，仅存储梯度就需要额外140GB内存。
3. **优化器状态：** Adam或AdamW等现代优化器会维护内部状态，以调整每个参数的学习率。AdamW是LLM训练中常用的优化器，它为每个参数存储两种状态：一阶矩（动量）和二阶矩（方差）。为了保持稳定性，即使在混合精度训练中，这些状态也常以更高精度（FP32，4字节）保存。这意味着优化器状态还需要额外$70 \text{ 亿} \times 2 \text{ 个状态} \times 4 \text{ 字节/状态} = 560$ GB。

   ```python
   # AdamW优化器状态的大致内存占用（单位GB）
   # 通常以FP32（4字节）存储以保持稳定性
   bytes_per_state_fp32 = 4
   num_states_per_param_adamw = 2

   optimizer_memory_gb = (num_parameters * num_states_per_param_adamw * 
                      bytes_per_state_fp32) / (1024**3)
   print(f"Approx. AdamW optimizer state memory (FP32): {optimizer_memory_gb:.2f} GB")
   # 输出：FP32格式下AdamW优化器状态的大致内存占用：521.54 GB
   # 为了文本中计算一致性，使用560GB。
   ```
4. **激活值：** 在正向传播过程中，每一层的中间输出（激活值）必须存储起来，以便在反向传播计算梯度时使用。这些激活值的大小取决于批处理大小、序列长度和模型隐藏维度。对于Transformer模型，自注意力 (self-attention)机制 (attention mechanism)尤其占用内存，在其朴素实现中可能需要$O(批处理大小 \times 序列长度^2 \times 隐藏维度)$的内存。虽然激活检查点（在反向传播时重新计算激活值而非存储它们）等技术可以减少内存占用，但仍然需要大量的激活值内存，通常根据配置不同，可能达到数十或数百GB。

仅将参数、梯度和优化器状态加起来，对于我们700亿参数的示例，总计为$140 + 140 + 560 = 840$ GB。这已经超过了单个80GB GPU容量的十倍，而且我们甚至还没有完全计入激活值或cuDNN等库所需的额外工作内存。



![内存估算：约700亿参数模型 vs. 单块GPU](plots/6021-0.json)



> 训练约700亿参数模型时的内存估算细分，与高端GPU典型的80GB HBM内存对比。激活值和工作内存还会增加额外的需求。

显然，内存需求使得单个设备无法满足要求。

### 计算能力限制：使训练变为可能

除了内存问题，训练LLM所需的极高的计算成本（以浮点运算，即FLOPs衡量）非常高昂。对一个包含数十亿参数 (parameter)的模型进行一次前向和反向传播 (backpropagation)，涉及数万亿次计算。

- **FLOPs需求：** 训练涉及在多个训练周期中处理海量数据集（数万亿个token）。所需的总计算量会随模型大小和数据集大小的增长而大幅增加。经验性的缩放法则表明，要获得好的性能，需要大量的计算预算，这通常以PetaFLOP-天（1 PetaFLOP = $10^{15}$ FLOPs）来衡量。
- **训练时间：** 即使是最快的单个加速器，每秒也只能执行有限数量的FLOPs（例如，稠密运算可达数百TeraFLOPs）。在单个设备上完成一次完整的LLM预训练 (pre-training)，需要ExaFLOPs（$10^{18}$ FLOPs）或更多的计算量，这将耗时数月、数年甚至数十年，使其完全不切实际。研究和开发需要更快的迭代周期。
- **注意力机制 (attention mechanism)的复杂度：** 自注意力 (self-attention)机制是Transformer模型的根本组成部分，具有$O(序列长度^2 \times 隐藏维度)$的计算复杂度。随着模型在更长上下文 (context)窗口（数千个token）上进行训练，注意力计算的成本变得越来越主导，进一步减慢了单个设备上的训练速度。

### 必由之路：分布式训练

面对这些内存和计算瓶颈，一个有效的方法是将训练负载**分布式**到多个互联加速器组成的集群上。通过在多个并行工作的设备上划分模型的参数 (parameter)、数据或计算图，我们可以做到：

1. 汇集内存容量以容纳整个模型状态。
2. 汇集计算能力，大幅缩短训练时间，从数年降至数周或数天。

接下来的章节会介绍为实现这种分布而发展起来的主要策略：数据并行、张量并行和流水线并行，以及它们的组合和相关的通信考量。理解这些技术对于任何参与构建和训练大型语言模型的人员来说都非常重要。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR 2015); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化器，由于其优化器状态，该优化器的内存占用在该节中进行了讨论。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS 2017); Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  Transformer架构的奠基性论文，它是现代大型语言模型的基础，其注意力机制对计算和内存挑战有显著贡献。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08058) — Mohammad Shoeybi, Mostafa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro, Greg Bernhardt, Vaibhav Gadi, Dale Green, Vijay Korthikoti, Michael Houston (2019)
  Journal: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '19); DOI: [10.1145/3295500.3356183](https://doi.org/10.1145/3295500.3356183)
  介绍了用于高效训练超大型基于Transformer的语言模型的框架和技术，直接解决了文中讨论的内存和计算挑战。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  探讨了语言模型的经验性缩放法则，展示了模型大小、数据集大小和计算预算如何影响性能以及对巨大计算资源的需求。

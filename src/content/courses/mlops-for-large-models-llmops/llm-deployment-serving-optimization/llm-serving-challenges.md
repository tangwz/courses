---
course: "mlops-for-large-models-llmops"
chapter: "llm-deployment-serving-optimization"
lesson: "llm-serving-challenges"
sourceId: 3164
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/llm-serving-challenges"
title: "大型模型服务中的挑战"
description: "分析部署大型语言模型的具体难点，包括延迟、吞吐量和资源需求。"
order: 1
plots: []
sourceHash: "87d8330f1785d84c563f289ea5d558e14dac249480059359f5647057fad0e2f5"
sourceCorrections: []
---

将大型语言模型部署到生产环境，与传统机器学习 (machine learning)模型相比，带来了一系列独特的运行难题。训练阶段侧重于在大量数据集和计算集群上优化学习，而服务阶段则将重心转向实时性能、资源利用率和实际流量下的成本控制。大型语言模型庞大的体量和高计算强度加剧了对小型模型来说可能只是轻微不便的问题。理解这些具体难题是制定有效部署方案的根本。

### 内存占用

也许最直接的挑战是大型语言模型庞大的规模。拥有数百亿甚至数千亿参数 (parameter)的模型仅用于存放模型权重 (weight)就需要大量内存。

- **GPU显存 (VRAM)需求：** 像Llama 2 70B这样的模型，使用16位精度（FP16或BF16）时，仅权重就需要大约 $70 \times 2 = 140$ GB的GPU显存。这通常会超出即使是高端单GPU（如NVIDIA A100 80GB或H100 80GB）的容量，因此即使是推理 (inference)也需要多GPU配置或专用硬件。将这些权重从存储加载到GPU内存中，还会给推理服务器带来显著的启动延迟。
- **激活缓存（KV缓存）：** 在自回归 (autoregressive)生成过程中，每个新生成的token都依赖于之前生成的token，因此必须存储一个名为键值（KV）缓存的中间状态。该缓存包含每个层的注意力键和值，并随着序列长度（上下文 (context) + 生成的token）和批次大小线性增长。对于长上下文或大批次，KV缓存可能消耗数十甚至数百GB的内存，甚至可能比模型权重本身需要更多的显存。这种动态内存需求使资源分配更加复杂，并可能成为一个主要瓶颈。

> 大型语言模型的自回归生成过程。每一步都使用之前的上下文（部分存储在不断增长的KV缓存中）来生成下一个token。KV缓存的大小随着每个生成的token而增加，消耗大量内存。

### 计算成本与延迟

使用大型语言模型生成文本计算量很大。每个生成的token都需要通过模型参数 (parameter)进行一次完整的正向传播，这涉及数十亿或数万亿次浮点运算（FLOPs）。

- **每个Token的高FLOPs：** 计算成本随着模型大小显著增加。这直接转化为硬件（GPU/TPU）上的处理时间。即使使用强大的加速器，生成token也需要相当长的时间。
- **序列生成延迟：** 由于每个token通常依赖于前一个token，因此生成过程主要是顺序的。虽然处理输入提示（预填充阶段）通常可以在token之间并行化，但生成长度为 $N$ 的输出序列（解码阶段）通常需要 $N$ 次顺序的正向传播通过模型。这从根本上限制了生成完整响应所能达到的最小延迟。提示处理后，*首个token生成时间* 可能相对较快，但*最后一个token生成时间*（总生成时间）会随着所需输出长度线性增加。许多交互式应用（如聊天机器人）对这种可感知的延迟非常敏感。
- **延迟与吞吐量 (throughput)的权衡：** 优化单个请求的最低延迟（例如，使用批次大小为1）通常会导致硬件利用率低下，因为现代GPU是为大规模并行处理设计的。这导致整体吞吐量低（所有并发请求每秒生成的总token数）和每个token的成本高。相反，最大化吞吐量通常涉及以大批次处理请求。虽然这提高了硬件利用率并降低了成本，但它增加了单个请求的延迟，因为这些请求可能需要等待批次填满或批次中其他请求完成。有效地平衡这种权衡是大型语言模型服务中的一个核心问题。

### 吞吐量 (throughput)与并发性

服务系统必须有效地处理可能存在的许多并发用户。实现高吞吐量对于经济高效地支持大量用户至关重要。

- **批处理复杂性：** 简单的静态批处理（在处理前等待固定数量的请求）会引入显著的延迟。动态批处理策略旨在更灵活地对传入请求进行分组。然而，请求之间输入和输出长度的差异使得高效批处理变得复杂。通常需要连续批处理或分页注意力等高级技术来有效地打包计算，并高效管理不同序列长度并发请求的KV缓存。
- **资源分配与调度：** 有效地将各种请求（有些需要短响应，有些需要长响应）调度到可能分布在多个节点上的可用GPU资源上，需要复杂的请求调度器和资源管理。目标是在满足延迟和可用性服务水平目标（SLO）的同时，最大化资源利用率。

### 成本控制

大型语言模型服务所需的专用硬件（通常是具有大量显存 (VRAM)和内存带宽的高端GPU）代表着巨大的资本或运营支出。

- **高昂的基础设施成本：** 像NVIDIA H100或A100这样的GPU价格昂贵，持续运行它们会产生巨大的能源、散热和云实例使用费。
- **成本效率压力：** 由于高昂的基础成本，实现高硬件利用率不仅是性能目标，更是经济上的需要。每一个未充分利用的GPU周期都代表着浪费的开支。这推动了对积极优化的需求，通过高效的服务引擎、模型压缩技术（如量化 (quantization)）和智能自动扩缩容。

这些与内存限制、计算需求、延迟预期、吞吐量 (throughput)要求和运营成本相关的相互关联的挑战，需要本章中讨论的专门部署和优化技术。成功应对这些难题是构建由大型语言模型驱动的高性能、可扩展且经济可行的应用程序的必要条件。

## 参考资料

- [vLLM: Efficient LLM Serving with PagedAttention](https://arxiv.org/pdf/2309.06180.pdf) — Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lichao Yu, Daniel Friedman, Xin Jin, Joseph E. Gonzalez (2023)
  Journal: Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023); Publisher: NeurIPS; DOI: [10.48550/arXiv.2309.06180](https://doi.org/10.48550/arXiv.2309.06180)
  介绍了PagedAttention，这是一种对LLM服务中KV缓存管理至关重要的内存优化技术，并提出了vLLM，一个高效的服务系统。
- [A Survey of Large Language Model Acceleration](https://arxiv.org/pdf/2312.15166.pdf) — Lianmin Zheng, Ying Sheng, Hanwen Chang, Wei-Ming Chen, Xiangru Lian, Zhaoyang Zhang, Zhiqiang Xie, Puxin Xu, Yiyuan Dong, Renjie Liu, Xingyao Chen, Hao Zhang, Kaiwen Zhang, Zhuohan Li, Zixuan Wu, Siyuan Zhuang, Joseph Gonzalez, Yi Wu, Michael Mahoney, Archita Sharma, Fan Lai, Yinghui Li, Junjie Liu, Chris Van Durme, Guangxuan Song, Shangguang Wang, Wen-mei Hwu, Yonghong Yan, Zhi Yang, Zhenglian Wu, Yuandong Tian, Zhiruo Wang, Haotian Tang, Hantian Ding, Michael Jordan, Dawn Song, Michael I Jordan (2024)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2312.15166](https://doi.org/10.48550/arXiv.2312.15166)
  全面回顾了加速大型语言模型的技术，涵盖了内存优化、计算效率和系统设计，与服务挑战相关。
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323.pdf) — Elias Frantar, Shuming Ma, Saleh Ashkboos, Oleg Rybakov, Torsten Hoefler (2023)
  Journal: Proceedings of the 40th International Conference on Machine Learning (ICML 2023); DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  提出了一种专为大型生成模型设计的训练后量化方法，显著降低了推理的内存占用和计算成本。
- [Large Language Model Inference: The Cost of Waiting](https://huggingface.co/blog/llm-inference-costs) — Quentin Lhoest, Lysandre Neis (2023)
  Publisher: Hugging Face Blog
  审视了LLM推理的挑战，侧重于延迟和吞吐量之间的权衡，并阐明了持续批处理和高效KV缓存管理等概念。

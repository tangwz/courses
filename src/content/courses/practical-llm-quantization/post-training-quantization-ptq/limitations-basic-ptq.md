---
course: "practical-llm-quantization"
chapter: "post-training-quantization-ptq"
lesson: "limitations-basic-ptq"
sourceId: 4620
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-2-post-training-quantization-ptq/limitations-basic-ptq"
title: "基础PTQ的局限性"
description: "识别简单PTQ方法可能导致显著准确性下降的场景。"
order: 7
plots: ["plots/4620-0.json"]
sourceHash: "803109410a8ad601ca2d2a79c3ae36ceabdd1990599146933d8a30cefa44f5cc"
sourceCorrections: []
---

尽管训练后量化 (quantization)（PTQ）提供了一种无需重新训练即可获得更小、更快模型的有效途径，但理解其固有局限性非常重要，尤其是在使用简单或“基础”PTQ算法（如直接的MinMax量化）时。这些局限性通常表现为模型准确性的下降，有时这种下降会非常显著，以至于量化模型无法胜任其预定任务。

### 主要挑战：准确性下降

量化 (quantization)的核心权衡在于计算效率（较低精度）与表示保真度（准确性）之间。将连续浮点值映射到离散整数集不可避免地会引入误差，这被称为量化误差。对于一个值 $x$，其量化表示 $x_q$ 涉及一个映射函数 $Q(\cdot)$ 和一个反量化函数 $D(\cdot)$：

$x_q = D(Q(x))$

该值的量化误差为 $e = x - x_q$。基础PTQ方法旨在统计性地最小化模型权重 (weight)和激活中的这种误差，但它们并非总能成功，尤其是在激进量化（例如，使用像INT4这样更少比特）的情况下。

有几个因素会导致基础PTQ中的准确性下降：

1. **对异常值的敏感性：** 大型语言模型中的权重和激活通常表现出非均匀分布，带有长尾或明显的异常值。基础的范围设定算法，如MinMax，根据校准期间观察到的绝对最小值和最大值来确定量化比例。如果存在显著的异常值，它们会极大地拉伸这个范围。这使得大部分值（它们位于窄得多的区间内）被迫量化到仅有的几个可用整数级别，从而导致大量数据损失显著精度。尽管存在处理异常值的方法（如前所述），但*基础*PTQ实现可能不包含复杂的裁剪或范围调整方法，使其易受影响。
2. **均匀量化与非均匀数据：** 大多数基础PTQ方案采用*均匀*量化。这意味着连续整数级别之间的间隔（步长）在原始浮点尺度上表示相同的变化。然而，模型参数 (parameter)和激活很少均匀分布；它们通常遵循钟形或拉普拉斯分布。均匀量化在表示这些分布时不是最优的，因为它对均值附近密集区域和稀疏尾部区域赋予相同的精度。
3. **层敏感度差异：** 大型语言模型中不同层对量化噪声的敏感度可能差异很大。例如，注意力机制 (attention mechanism)或负责细粒度预测的后期层可能比早期层更容易受到精度损失的影响。基础PTQ通常采用统一的量化策略（例如，将所有线性层量化为INT8），而不考虑这种变化的敏感度。这可能会不成比例地影响模型的关键部分，导致整体性能明显下降。
4. **激进的低精度量化：** 尽管量化到INT8通常能通过基础PTQ获得良好效果，但推向像INT4这样更低的比特宽度会显著增加量化误差。每个整数级别必须代表更大范围的浮点值。基础PTQ方法在这些较低比特级别下通常难以保持可接受的准确性，除非采用专门解决增加误差的更高级技术。下图展示了一个典型趋势，即随着比特精度的降低，准确性下降加速。

   
   
   ![准确性与比特精度（基础PTQ）](plots/4620-0.json)
   
   

   > 随着比特精度的降低，准确性通常呈非线性下降，使用基础PTQ方法时，INT8以下常见显著下降。

"5. **校准数据依赖性：** PTQ的有效性在很大程度上依赖于用于确定量化参数（比例因子和零点）的校准数据集。如果校准数据量小、不能代表实际推理 (inference)数据分布或缺乏多样性，则生成的量化参数将不理想。当模型处理输入时，这会导致量化误差增加。尽管这会影响所有PTQ方法，但与调整权重或使用更精细范围分析的技术相比，基础方法对校准中的不完善之处可能不那么稳定。"

这些局限性表明，尽管基础PTQ是模型优化的一种有价值的工具，但它并非万能药。当准确性保持非常重要，或当目标是非常低的比特宽度时，由这些因素引起的性能下降可能是不可接受的。这促使需要下一章中讨论的更高级的PTQ技术（如GPTQ、AWQ），或使用量化感知训练（QAT），后者允许模型在训练或微调 (fine-tuning)过程中适应量化噪声。了解这些局限性有助于为您的特定模型和应用需求选择正确的量化策略。

## 参考资料

- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://www.frontiersin.org/articles/10.3389/fnins.2021.647915/full) — Amir Gholami, Song Han, Sijia Liu, Jianbo Liu, Minhao Cheng, Sehoon Kim, Zhongnan Qu, Hesham Mostafa, William J. Dally, Mahmoud Naghshvari, Kurt Keutzer (2021)
  Journal: Frontiers in Neuroscience; Publisher: Frontiers Media SA; Volume: 15; DOI: [10.3389/fnins.2021.647915](https://doi.org/10.3389/fnins.2021.647915)
  一篇关于神经网络各种量化技术的全面综述，讨论了它们的动机以及简单方法固有的挑战，包括对异常值的敏感性问题。
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2208.07339](https://doi.org/10.48550/arXiv.2208.07339)
  本文识别并解决了大型语言模型激活中异常值严重降低量化精度的重要问题，解释了基本PTQ为何在LLM上表现不佳。
- [GPTQ: Accurate Post-training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2023)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  通过阐述基本PTQ在大型语言模型上，特别是在激进的低位宽（例如INT4）下精度显著下降的问题，介绍了一种先进的PTQ方法。

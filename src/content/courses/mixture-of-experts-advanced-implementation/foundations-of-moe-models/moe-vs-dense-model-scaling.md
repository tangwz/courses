---
course: "mixture-of-experts-advanced-implementation"
chapter: "foundations-of-moe-models"
lesson: "moe-vs-dense-model-scaling"
sourceId: 7050
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-1-foundations-of-moe-models/moe-vs-dense-model-scaling"
title: "与密集模型扩展的比较"
description: "比较专家混合模型与传统密集模型架构在参数和计算方面的扩展特性。"
order: 7
plots: ["plots/7050-0.json"]
sourceHash: "e5cc059e70550d366ade952b9f820012a9a2e3a9bb27e11a4210eaa1d9a287fb"
sourceCorrections: []
---

获得更强模型的传统方法是一种简单但代价大的策略：统一增大模型。当标准Transformer模型通过增加其前馈网络（FFN）的宽度或添加更多层来扩展时，总参数 (parameter)量和计算成本（即浮点运算次数，FLOPs）会紧密关联地一同增加。专家混合模型（MoE）打破了这种僵硬的关系，从根本上改变了扩展的经济性。

### 将参数 (parameter)与计算分离

在密集模型中，给定层中的每个输入token都由每个参数处理。例如，Transformer中的FFN块，它通常是MoE层替换的主要对象，通常包含两个大型线性变换。单个token所需的参数数量和计算量与模型的维度成正比，具体来说是 `d_model` 和 `d_ffn`。


$$
\text{密集模型的FLOPs} \approx 2 \times \text{批大小} \times \text{序列长度} \times d_{\text{model}} \times d_{\text{ffn}}
$$


这个等式表明，通过增加 `d_ffn` 让模型“更智能”，你必须同时付出直接且不可避免的计算代价。

MoE模型引入了稀疏激活模式，分离了这两个因素。尽管模型可能包含大量总参数并分布在许多专家中，但对于任何给定token，只有其中一小部分参数被激活。例如，门控网络从可能的64个专家中选择前两个专家。

MoE层的总参数量是其所有专家的总和：


$$
\text{MoE参数} = N_{\text{专家}} \times \text{专家参数}
$$


然而，计算成本仅与*激活*的专家数量 $k$ 成正比：


$$
\text{MoE模型的FLOPs} \approx k \times (2 \times \text{批大小} \times \text{序列长度} \times d_{\text{model}} \times d_{\text{专家}})
$$


这是MoE架构的核心优势。通过增加更多专家（$N_{\text{专家}}$）同时保持 $k$ 固定不变，你可以大幅提升模型的容量（总参数），同时保持每个token的计算成本不变。这使得拥有数万亿参数的模型成为可能，它们可以用与小得多密集模型相当的计算预算进行训练。

> 这是数据流的比较。在密集模型中，所有token都通过一个单一的大型网络。在MoE模型中，门控网络将token路由到专业化、更小的专家网络的一个小子集。

### 扩展趋势与系统影响

扩展理念上的差异不仅仅是理论上的；它对硬件利用率、训练时间、和系统设计有深刻影响。密集模型的增长受限于单个设备的内存和计算能力，而MoE模型的增长则受限于设备集群的聚合内存。



![参数量与计算成本对比](plots/7050-0.json)



> 在密集模型中，参数 (parameter)和计算成本几乎呈线性增长。MoE模型打破了这一趋势，使得参数可以大幅增加，而训练所需的FLOPs计算量增长慢得多，呈亚线性关系。

这个图表说明了核心的权衡。对于给定的计算预算（y轴），你可以训练一个参数量（x轴）远超密集模型的MoE模型。这与以下发现一致：一旦达到特定阈值，模型性能与参数量的关系优于与额外训练数据或训练步骤的关系。

### 训练动态与推理 (inference)特点

这种架构差异在模型生命周期中导致不同的挑战。

#### 训练

对于**密集模型**，扩展意味着获取具有更大内存和处理能力的更强加速器。主要挑战是将一个更大的、整体的模型加载到一个设备上，并在合理时间内完成训练步骤。

对于**MoE模型**，挑战从单节点能力转向多节点通信和内存聚合。由于总参数 (parameter)量很容易超过任何单个加速器的内存，模型*必须*分布式部署。这使得专家并行等技术（即不同专家托管在不同设备上）成为必要而非可选。瓶颈通常从原始计算转向“全对全”操作的通信开销，这些操作需要将token发送到其指定的专家并收集结果。

#### 推理

在推理期间，**密集模型**相对来说部署简单。其整个参数集都处于激活状态，一旦加载到内存中，它会以可预测的延迟处理请求。

**MoE模型**的推理更为复杂。庞大的参数量带来了显著的内存挑战。将一个数万亿参数的模型加载到单个推理服务器的GPU内存中通常不可行。这催生了*专家卸载*等专门技术，即不活跃的专家存储在更便宜、更慢的内存（如CPU RAM或NVMe存储）中，并且仅在需要时才加载到GPU上。虽然这解决了内存问题，但可能引入显著的延迟，使得高效的批处理和调度对于获得可接受的性能极为重要。

总之，MoE的扩展并非免费午餐。它用密集模型的暴力计算扩展，换取了一种更复杂、更注重系统的方法。你可以在给定FLOP预算下获得极大的模型容量，但作为交换，你必须管理分布式系统、通信瓶颈以及复杂的推理策略。后续章节将提供应对这一新局面的工具。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: arXiv preprint arXiv:1701.06538; DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了神经网络中稀疏门控专家混合层，使模型能够拥有大量参数同时保持每个样本的计算成本不变。这是参数与计算解耦概念的奠基性工作。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  一篇里程碑式的论文，探讨了密集神经网络语言模型的性能如何随模型大小、数据集大小和计算预算扩展，为解释MoE参数效率的优势提供了背景。

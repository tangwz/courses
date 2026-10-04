---
course: "how-to-build-a-large-language-model"
chapter: "introduction-large-scale-language-modeling"
lesson: "significance-of-scale"
sourceId: 5949
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-1-introduction-large-scale-language-modeling/significance-of-scale"
title: "规模的重要性"
description: "讨论模型大小、数据量与涌现能力之间的关系。"
order: 3
plots: ["plots/5949-0.json"]
sourceHash: "42c8d6c7955b729b1f1e266975ca2e5c39427fc18ff961d89df958dad5ad3a64"
sourceCorrections: []
---

谈及大型语言模型时，“大”不仅仅是一个定性描述；它定量指代着庞大的参数 (parameter)数量、海量的训练数据集以及所需的大量计算资源。这种规模并非偶然特性，而是其能力的基本推动力。与早期模型性能提升可能相对较快达到平台期不同，现代大型语言模型展现出与规模增长直接相关的不同现象。

### 缩放定律：可预测的改进

大型语言模型研究中一个重要的发现是*缩放定律*的存在。这些是经验观察结果，表明模型性能（通常通过在独立数据集上的交叉熵损失来衡量）会随着模型大小（参数 (parameter)数量）、数据集大小和训练所用计算量的增加而可预测地提升。

大型语言模型中的性能与规模之间的关系通常被建模为幂律，尤其在对数-对数坐标图上查看时。例如，损失$L$可以近似地与非嵌入 (embedding)参数的数量$N$、数据集大小$D$和计算预算$C$（以FLOPs计）关联如下：


$$
L(N) \approx \left( \frac{N_c}{N} \right)^{\alpha_N}
$$


$$
L(D) \approx \left( \frac{D_c}{D} \right)^{\alpha_D}
$$


这里，$N_c$ 和 $D_c$ 代表特征尺度，而 $\alpha_N$ 和 $\alpha_D$ 是缩放指数（通常是小于1的正值，对于$N$和$D$通常在0.05-0.1左右）。类似的关联也适用于计算量 $C$。这些定律表明，投入更多资源（参数、数据、计算）会在主要训练目标上带来边际递减但持续的改进。



![大型语言模型缩放定律](plots/5949-0.json)



> 验证损失倾向于随着模型大小、数据集大小或计算预算的增加而呈幂律下降。

这些缩放定律对于规划训练过程非常有帮助。它们使研究人员和工程师能够估算在给定预算下可达到的性能提升，反之，也能估算达到目标性能水平所需的资源，从而避免投入昂贵且耗时长的实验。

### 涌现 (emergence)能力

规模最吸引人的方面之一，或许是*涌现能力*的出现。这些能力在较小的模型中不存在或无法衡量，但一旦模型大小、数据量或计算量超过特定阈值，它们就会相对突然地显现。它们不仅仅是现有指标的渐进式改进，而是性质上全新的行为。

实例包括：

- **少样本推理 (inference)：** 能够仅根据提示中提供的少量示例执行新任务，而无需任何梯度更新。较小的模型通常需要大量微调 (fine-tuning)才能执行新任务。
- **思维链推理：** 提示大型模型“一步步思考”可以大幅提升其在需要算术、符号推理或常识逻辑的任务上的表现。这种能力在较小的模型中通常不存在。
- **指令遵循：** 大型模型在遵循自然语言中给出的复杂指令方面变得更好。

这些能力出现的阈值是经验性的且依赖于具体任务，但它们的存在有力地推动了对更大模型的追求。这表明，仅仅扩大现有架构的规模就能产生根本性的新功能。

> 说明了规模的增加如何带来更复杂的能力，包括涌现的能力。

### 参数 (parameter)、数据和计算的关系

规模化不仅仅是使某一个方面变大；它涉及平衡三个主要组成部分：模型大小($N$)、数据集大小($D$)和训练计算量($C$)。研究，特别是DeepMind的“Chinchilla”论文（Hoffmann et al.，2022），指出在固定的计算预算下，仅通过最大化模型大小并不能获得最佳性能。相反，存在一个最佳分配方案，即模型大小和数据集大小应大致按比例进行缩放。

以前的模型通常使用相对较小的数据集进行训练，与它们的参数数量相比（计算受限状态）。Chinchilla的研究结果表明，许多大型模型存在显著的训练不足；对于已使用的计算量，通过在更多数据上训练一个较小的模型，本可以提升性能。这表明，数据规模与模型规模同样重要，对于在给定计算范围内达到最佳结果而言。

计算参数数量提供了衡量模型规模的具体方法。对于一个典型的Transformer块，参数主要来自自注意力 (self-attention)投影（Query、Key、Value、Output）和前馈网络层。

```python
import torch
import torch.nn as nn
from math import prod

def count_parameters(model: nn.Module) -> int:
    """计算PyTorch模型中可训练参数的总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 示例：简化的Transformer层组件
hidden_dim = 768
ffn_dim = hidden_dim * 4 # 常见做法
num_heads = 12
head_dim = hidden_dim // num_heads # 通常 d_model / num_heads

# 单个注意力机制 + FFN的粗略估算
# Q、K、V 投影（每个 hidden_dim x hidden_dim）
qkv_params = 3 * hidden_dim * hidden_dim
# 输出投影（hidden_dim x hidden_dim）
attn_output_params = hidden_dim * hidden_dim
# FFN 第一层（hidden_dim x ffn_dim）
ffn1_params = hidden_dim * ffn_dim
# FFN 第二层（ffn_dim x hidden_dim）
ffn2_params = ffn_dim * hidden_dim

# 注意：为简化起见，此处忽略了偏置和归一化层
approx_params_per_layer = (qkv_params + attn_output_params +\
                           ffn1_params + ffn2_params)

print(f"每个Transformer层的近似参数数量：{approx_params_per_layer:,}")

# 一个包含12个此类层的模型（如BERT-base）
num_layers = 12
# 添加嵌入参数（vocab_size * hidden_dim）——假设词汇表大小为3万
vocab_size = 30522
embedding_params = vocab_size * hidden_dim

total_params_estimate = ((num_layers * approx_params_per_layer) +\
                         embedding_params)
print(f"一个12层模型的总估算参数数量：{total_params_estimate:,}")

# 与一个更大的模型进行比较（例如，扩展 hidden_dim）
large_hidden_dim = 1280
large_ffn_dim = large_hidden_dim * 4
large_num_heads = 16

large_qkv = 3 * large_hidden_dim * large_hidden_dim
large_attn_out = large_hidden_dim * large_hidden_dim
large_ffn1 = large_hidden_dim * large_ffn_dim
large_ffn2 = large_ffn_dim * large_hidden_dim
large_layer_params = (large_qkv + large_attn_out +\
                      large_ffn1 + large_ffn2)
print(f"每个层的大致参数数量 "\
      f"（大型模型）：{large_layer_params:,}")
# 一个包含24个此类层的模型
large_num_layers = 24
large_embedding_params = vocab_size * large_hidden_dim
large_total_params = ((large_num_layers * large_layer_params) +\
                      large_embedding_params)
print(f"一个24层大型模型的总估算参数数量：{large_total_params:,}")
```

此代码片段展示了架构选择（如 `hidden_dim`、`ffn_dim`、`num_layers`）如何直接影响参数数量，而参数数量是衡量规模的主要指标。本课程中讨论的模型通常范围从数亿到数千亿甚至数万亿参数，需要相应增加数据和计算量。

总而言之，规模并非仅仅为了大而大。它根据缩放定律可预测地推动性能提升，使定性上新的涌现 (emergence)能力成为可能，并且需要在模型参数、数据集大小和计算预算之间取得仔细的平衡。理解规模的重要性对于应对构建和训练高效大型语言模型所涉及的工程难题非常重要。

## 参考资料

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  提出了语言模型性能与模型大小、数据集大小和计算量相关的经验性缩放定律。
- [Emergent Abilities of Large Language Models](https://jmlr.org/tmlr/papers/v1/22-132.html) — Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, David Dohan, Sharan Narang, Aakanksha Chowdhery, Dennis Anil, Aitor Lewkowycz, Erica Grant, Adam Roberts, Kevin Robinson, Brennan Saeta, Hyung Won Chung, Azalia Mirhoseini, Charles Sutton, Siva Reddy, P. J. Liu, William Fedus, Xiangru Tang, Michele Catasta, Xavier Garcia, Dan Garrette, Kevin Lacker, Srinivas Ramabhadran, Peter J. Liu, Adam Roberts, Jonathon Shlens, Noam Shazeer, Maithra Raghu, Jordan Hoffmann, Henryk Michalewski, Jeffrey Dean (2022)
  Journal: Transactions on Machine Learning Research; Publisher: JMLR.org; Volume: 1; Pages: 1-27; DOI: [10.48550/arXiv.2206.07682](https://doi.org/10.48550/arXiv.2206.07682)
  定义并阐述了大型语言模型中随规模增大而定性出现的涌现能力。
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre (2022)
  Journal: arXiv preprint arXiv:2203.15556; DOI: [10.48550/arXiv.2203.15556](https://doi.org/10.48550/arXiv.2203.15556)
  研究了在给定计算预算下，模型大小和训练数据之间的最佳平衡。
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf) — Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 33; Pages: 1877-1901
  通过扩展规模，展示了大型语言模型进行少样本学习和指令遵循的能力。

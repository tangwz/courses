---
course: "how-to-build-a-large-language-model"
chapter: "scaling-transformers-architectural-choices"
lesson: "scaling-laws-neural-language-models"
sourceId: 6001
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-11-scaling-transformers-architectural-choices/scaling-laws-neural-language-models"
title: "神经网络语言模型的缩放定律"
description: "回顾模型表现如何随大小、数据和计算量进行缩放的经验研究结果。"
order: 1
plots: ["plots/6001-0.json"]
sourceHash: "049cc6fc5ffc97bf825ad5e66df235dd73b880c81ac554392d35f83fbfc7b415"
sourceCorrections: []
---

在扩展Transformer模型时，一个自然而然的问题出现了：我们如何最佳地投入计算资源？我们应该优先增加层数、拓宽隐藏维度，还是在更多数据上训练？仅仅在没有明确策略的情况下增加参数 (parameter)可能导致收益递减或昂贵计算周期的低效利用。幸运的是，经验研究揭示了模型表现、模型大小、数据集大小以及训练中使用的计算量之间可预测的关系。这些关系通常被称为“缩放定律”。

理解这些缩放定律提供了一个有价值的框架，用于对模型架构和训练方案做出明智的决定。它们使我们能够估计扩大训练过程不同方面所带来的预期性能提升，并协助优化计算预算的分配。

### 缩放定律的经验基础

开创性工作，值得注意的是OpenAI的Kaplan等人（2020），证明了语言模型的表现（通常通过在保留测试集上的交叉熵损失来衡量），随着我们扩展模型大小（非嵌入 (embedding)参数 (parameter)数量，$N$）、数据集大小（训练标记 (token)数量，$D$）和计算量（总浮点运算，$FLOPs$, $C$）而可预测地提高。

核心发现是，当其他因素不是瓶颈时，测试损失$L$通常与$N$、$D$和$C$遵循幂律关系。对于模型大小$N$和数据集大小$D$，这种关系通常可以建模为：


$$
L(N, D) \approx L_{\infty} + \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}
$$


其中：

- $L(N, D)$是预测的交叉熵损失。
- $L_{\infty}$表示一个不可约的损失下限，可能与语言固有的熵或建模方法的局限性有关。
- $N$是模型中非嵌入参数的数量。
- $D$是训练数据集中标记的数量。
- $N_c$、$D_c$、$\alpha_N$和$\alpha_D$是模型架构和训练设置的特定常数，通过将公式拟合到不同小规模训练运行的结果中来凭经验确定。通常，$\alpha_N$和$\alpha_D$小于1，表示随着$N$或$D$的增加，收益递减。

这个公式表明，损失主要受对应于限制因素的项的影响：如果模型太小（$N \ll N_c$），增加$N$会显著有益；如果数据集太小（$D \ll D_c$），增加$D$更有益。

类似地，损失通常随计算预算$C$呈幂律关系：


$$
L(C) \approx L_{\infty} + \left(\frac{C_c}{C}\right)^{\alpha_C}
$$


其中$C_c$和$\alpha_C$也是凭经验确定的常数。这些关系通常在几个数量级内成立，使其对推断有用。



![缩放定律：损失与模型大小（对数-对数）](plots/6001-0.json)



> 一幅对数-对数图，展示测试损失如何随模型大小增加而降低，通常在几个数量级内遵循可预测的幂律。

### 计算最优缩放与Chinchilla研究结果

对缩放定律理解的一项重要改进来自于DeepMind的Hoffmann等人（2022），被称为“Chinchilla”研究。他们进行了仔细的分析，以确定固定计算预算$C$在模型大小$N$和数据大小$D$之间的最佳分配。

训练密集型Transformer模型的计算量、模型大小和数据大小之间的近似关系通常估算为：


$$
C \approx 6 \times N \times D
$$


这表明计算成本大致与参数 (parameter)数量乘以处理的标记 (token)数量成正比。鉴于固定的计算预算$C$，Chinchilla研究建议，为了获得最佳表现（最低损失），$N$和$D$都应该大致按照计算预算的平方根比例进行缩放。也就是说，如果计算预算翻倍，您应该尝试将*模型大小*和*数据集大小*都增加大约$\sqrt{2} \approx 1.4$倍。

这一发现很重要，因为它表明在这项研究之前训练的许多大型语言模型（如GPT-3或Gopher）相对于其训练数据可能“参数过多”了。对于所使用的计算预算，通过在更多数据上训练更小的模型，可能已经取得了更好的表现。Chinchilla模型本身，根据这些“计算最优”原则进行训练，以更少的参数但更多的训练数据，取得了比其一些大型同期模型更出色的成果。

让我们用一个简化的计算来说明。假设我们拟合了一个缩放定律，发现当$N \approx k \sqrt{C}$和$D \approx \frac{C}{6N} \approx \frac{C}{6k\sqrt{C}} = \frac{\sqrt{C}}{6k}$（其中$k$为某个常数）时，会获得最佳表现。如果我们有一个计算预算$C_1$并找到了最优的$N_1, D_1$，那么对于更大的预算$C_2 = 4 C_1$，最优参数将近似为$N_2 \approx k \sqrt{4C_1} = 2 N_1$和$D_2 \approx \frac{\sqrt{4C_1}}{6k} = 2 D_1$。我们将模型和数据都与$\sqrt{C}$成比例地缩放。

### 训练大型语言模型的实际意义

这些缩放定律对构建大型模型的工程师有几点直接影响：

1. **资源分配：** 它们为是否投资更多计算硬件（以增加$N$和$D$）、更多数据获取和清洗（以增加$D$），或者可能改变常数（$N_c, D_c, \alpha_N, \alpha_D$）的架构改进提供了量化 (quantization)指导。
2. **性能预测：** 通过运行小规模实验来估计缩放参数 (parameter)（$\alpha_N, \alpha_D$等），团队可以在投入完整训练运行所需的大量资源之前，预测更大模型的预期表现。这降低了风险并有助于证明计算请求的合理性。
3. **预算优化：** 对于固定预算$C$，这些定律有助于确定$N$和$D$之间的最佳平衡，以最大化预期表现。这避免了模型因其大小或训练数据而严重受限的情况。
4. **基准测试：** 缩放定律为评估新架构或训练技术提供了基准。如果一种新方法产生的表现显著优于标准Transformer的既定缩放定律所预测的，则表明其是真正的改进。

### 计算需求估计

我们可以使用$C \approx 6ND$公式来估计训练的计算成本。例如，在1万亿个标记 (token)（$D=1 \times 10^{12}$）上训练一个70亿参数 (parameter)模型（$N=7 \times 10^9$）大约需要：

```python
import math

# 近似参数数量（不包括嵌入，简化）
N = 7e9
# 训练标记数量
D = 1e12

# 使用6ND规则估算的FLOPs
C_flops = 6 * N * D

# 将FLOPs转换为Petaflop-天 (1 Petaflop = 1e15 FLOP/s)
# 一天中的秒数 = 86400
petaflops = 1e15
seconds_per_day = 86400

C_petaflop_days = C_flops / (petaflops * seconds_per_day)

print(f"估算计算量: {C_flops:.2e} FLOPs")
print(f"估算计算量: {C_petaflop_days:.2f} Petaflop-天")
# 输出：
# Estimated Compute: 4.20e+22 FLOPs
# Estimated Compute: 486.11 Petaflop-days
```

这一计算凸显了所涉及的巨大计算规模。一个能够持续10 Petaflops的集群，进行这样的运行大约需要49天，这还不考虑开销和潜在的低效率。这强调了为何由缩放定律指导的有效资源分配如此重要。

### 注意事项与考量

尽管它们功能强大，但仍需记住这些缩放定律的背景：

- **经验性：** 它们是观察到的趋势，而非普适定律。它们严重依赖于特定的架构（例如Transformer）、优化器、数据分布以及其他训练细节。从用于推导这些定律的实验范围之外进行过度推断会带来风险。
- **数据质量：** 这些定律主要建模数据的*数量*（$D$）。然而，数据质量、多样性和相关性也是影响最终模型表现的非常重要的因素，这些因素并未在这些简单公式中明确体现。在大量低质量数据上进行训练可能比在少量但更干净的数据上训练产生更差的结果。
- **不可约损失：** $L_{\infty}$项暗示了一个性能限制。随着模型变得非常大并在大量数据集上训练，当它们接近这个极限时，改进可能会放缓。
- **计算估算：** $6ND$公式是一个粗略的近似值。实际的FLOPs可能会因具体的实现、序列长度和硬件效率而异。更精确的计算可能会单独考虑前向和反向传播 (backpropagation)。

总而言之，缩放定律提供了一个宝贵的定量视角，用于审视构建更大、更强大语言模型 (LLM)的过程。它们将缩放从猜测游戏转变为更可预测的工程学科，从而更有效地利用计算资源，并提供了一个评估该领域进展的框架。在接下来的章节中，当我们讨论具体的架构选择时，请记住这些缩放原则，因为它们经常是大型Transformer设计决策的依据。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，为现代大语言模型奠定了基础。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  建立了语言模型经验性缩放定律的开创性论文，将性能与模型大小、数据集大小和计算量关联起来。
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre (2022)
  Journal: arXiv preprint arXiv:2203.15556; DOI: [10.48550/arXiv.2203.15556](https://doi.org/10.48550/arXiv.2203.15556)
  提出了Chinchilla研究，完善了缩放定律，以确定在给定计算预算下模型和数据大小的最佳分配。

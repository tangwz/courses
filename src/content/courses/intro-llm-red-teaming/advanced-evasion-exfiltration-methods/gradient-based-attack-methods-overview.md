---
course: "intro-llm-red-teaming"
chapter: "advanced-evasion-exfiltration-methods"
lesson: "gradient-based-attack-methods-overview"
sourceId: 6328
sourceUrl: "https://apxml.com/zh/courses/intro-llm-red-teaming/chapter-4-advanced-evasion-exfiltration-methods/gradient-based-attack-methods-overview"
title: "基于梯度的攻击方法：概述"
description: "基于梯度的攻击方法论概述及其在LLM上的适用性（白盒情境）。"
order: 1
plots: []
sourceHash: "998ebc21749b800b949aadafd3298cfb36d64752977e3f1d98ec007568def715"
sourceCorrections: []
---

当攻击者充分了解大型语言模型的架构及其参数 (parameter)时，一类被称为基于梯度的攻击的强大技术就变得可行。这些方法利用驱动LLM训练的相同数学机制，特别是梯度，但将其重新用于对抗目的。概括描述了这些攻击在通常被称为“白盒”情境下是如何运作的。

### 白盒假设

基于梯度的攻击通常假定为**白盒**设置。这意味着攻击者可以访问：

1. 模型的架构（例如，Transformer类型、层数、隐藏单元大小）。
2. 模型的参数 (parameter)或权重 (weight)（$\theta$），它们定义了已学习的函数 $M(x; \theta)$。
3. 计算某个函数（通常是损失函数 (loss function)）相对于模型输入的梯度的能力。

“虽然在许多LLM部署中（模型通常通过API访问），完全的白盒访问可能看起来是一个高要求，但理解这些攻击很重要。它们代表了在强假设下攻击者能力的上限，并构成了更复杂的黑盒 (black box)攻击的基础，例如我们稍后将讨论的迁移攻击。此外，如果专有模型被泄露，或者组织在进行内部红队测试并拥有完整的模型访问权限，这些技术可以直接应用。”

### 借助梯度生成对抗性输入

在标准模型训练中，损失函数 (loss function)相对于模型*权重 (weight)*（$\nabla_{\theta} J$）的梯度用于更新这些权重并提高性能。在基于梯度的攻击中，攻击者关注的是损失函数 $J$ 相对于模型*输入* $x$ 的梯度（即 $\nabla_x J$）。

这个输入梯度 $\nabla_x J(M(x; \theta), y_{target})$ 告诉我们如何改变输入 $x$ 的每个组成部分，以实现损失函数 $J$ 的最大增加（或减少）。攻击者根据其恶意目标定义 $J$ 和 $y_{target}$。例如：

- **规避/错误分类：** 如果LLM用于分类任务（例如，毒性检测），$y_{target}$ 可能是一个不正确的标签。攻击者想要修改 $x$ 为 $x_{adv}$，使得 $M(x_{adv}; \theta)$ 产生这个不正确的标签，通常通过最大化真实标签的损失或最小化目标不正确标签的损失来实现。
- **定向有害生成：** 攻击者可能希望LLM生成特定的有害或不期望的输出。此时，$J$ 可以定义为衡量LLM实际输出与攻击者期望的恶意输出之间的差异。然后，攻击者扰动输入 $x$ 以最小化这种差异。

核心想法是找到一个小的扰动 $\delta$，使得当LLM处理对抗性输入 $x_{adv} = x + \delta$ 时，$M(x_{adv}; \theta)$ 能够达到攻击者的目标。扰动 $\delta$ 是使用梯度 $\nabla_x J$ 精心构建的。

以下图表说明了通用过程：

> 该图表显示了如何使用从损失函数（反映攻击者目标）导出的梯度来修改原始输入，从而生成对抗性输入。然后将此对抗性输入提供给LLM。

### 一种示例方法：快速梯度符号法 (FGSM)

最早且最简单的基于梯度的攻击方法之一是快速梯度符号法 (FGSM)。这是理解输入梯度如何使用的一个良好起点。对于输入 $x$ 和损失函数 (loss function) $J(M(x; \theta), y)$（其中 $y$ 是真实标签，在分类情境下，或作为生成目标的占位符），FGSM 计算对抗性示例 $x_{adv}$ 如下：


$$
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(M(x; \theta), y))
$$


让我们来分解它：

- $\nabla_x J(M(x; \theta), y)$：这是损失函数相对于输入 $x$ 的梯度。它指出了输入特征应该朝哪个方向改变才能最大化损失。
- $\text{sign}(\cdot)$：符号函数获取梯度向量 (vector)中每个元素的符号（如果为正则为+1，如果为负则为-1，如果为零则为0）。这简化了梯度，本质上是说“每个输入特征应该朝哪个方向移动？”。
- $\epsilon$：这是一个小的标量值，它决定了扰动的大小。较大的 $\epsilon$ 意味着更强、更明显的扰动，但攻击可能更有效。攻击者选择 $\epsilon$ 足够小，使得 $x_{adv}$ 通常不易察觉或在语义上与 $x$ 相似，但又足够大以实现对抗目标。

FGSM中的“快速”来源于它只需要一次梯度计算。更高级的迭代方法会多次应用这一想法，在梯度方向上采取更小的步长，这通常会导致更有效的攻击，但计算成本也更高。示例包括投影梯度下降 (gradient descent) (PGD) 或基本迭代方法 (BIM)。

### 基于梯度的攻击为何有效？

基于梯度的攻击的有效性源于包括LLM在内的深度学习 (deep learning)模型学习方式的本质。它们创建复杂的、高维的决策边界或生成流形。梯度提供了一种直接的方法来找到穿过这些边界的路径，进入模型行为不正确或产生不期望内容的区域。

- **效率：** 梯度引导对有效对抗性示例的搜寻，而不是随机尝试扰动，这使得过程更加高效。
- **利用局部线性：** 许多模型，尽管它们整体上是非线性的，但在输入空间的局部区域表现出一定的线性。例如，FGSM明确利用了这种假定的局部线性。

尽管这些方法主要适用于白盒情境，但从中获得的理解具有多方面的重要性：

1. **鲁棒性基准测试：** 它们使开发人员能够测试其模型对某些类型扰动敏感性的理论极限。
2. **防御措施的制定：** 理解梯度如何被利用有助于防御措施的制定，例如对抗训练，这涉及使用这些技术生成的对抗性示例来训练模型。
3. **其他攻击的基础：** 如前所述，基于梯度的攻击是迁移攻击的组成部分，后者旨在使对抗性示例对黑盒 (black box)模型有效。

在接下来的章节中，我们将看到这些基础思想如何被扩展，以创建更复杂的规避和数据窃取技术，即使在无法获得完整白盒访问权限的情况下也是如此。

## 参考资料

- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) — Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (2014)
  Journal: CoRR; Volume: abs/1412.6572; DOI: [10.48550/arXiv.1412.6572](https://doi.org/10.48550/arXiv.1412.6572)
  这篇原始论文介绍了对抗性示例和快速梯度符号方法（FGSM），解释了微小、难以察觉的扰动如何导致深度神经网络的错误分类。
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://openreview.net/forum?id=rJzIBfZAb) — Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu (2018)
  Journal: 6th International Conference on Learning Representations, ICLR 2018 - Conference Track Proceedings; DOI: [10.48550/arXiv.1706.06083](https://doi.org/10.48550/arXiv.1706.06083)
  这篇论文提出了投影梯度下降（PGD）作为一种强大的、一阶对抗性攻击，并将其作为评估模型鲁棒性的基准，通常被认为是FGSM的改进迭代版本。

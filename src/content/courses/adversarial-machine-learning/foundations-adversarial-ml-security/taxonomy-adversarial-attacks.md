---
course: "adversarial-machine-learning"
chapter: "foundations-adversarial-ml-security"
lesson: "taxonomy-adversarial-attacks"
sourceId: 4120
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-1-foundations-adversarial-ml-security/taxonomy-adversarial-attacks"
title: "对抗性攻击的分类体系"
description: "根据攻击者目的、了解程度和扰动类型对对抗性攻击进行归类。"
order: 5
plots: []
sourceHash: "533dd17bb7486f1d98d93d3b87fb49c64487c3418f9ba83c8854634bb4707ad7"
sourceCorrections: []
---

了解对抗性攻击需要一种结构化的方式来归类它们。鉴于我们已经讨论了威胁模型以及训练与推理 (inference)期间可用的不同攻击面，我们现在可以建立一个分类体系。这种归类有助于分析薄弱点，比较攻击效果，并制定合适的防御策略。对抗性攻击通常根据几个主要方面进行归类：

### 归类维度

用于归类对抗性攻击的主要维度包括：

1. **攻击者目的：** 攻击者意图达成什么？
2. **攻击者了解程度：** 攻击者对目标模型掌握多少信息？
3. **攻击特定性：** 攻击是为了引起任意错误，还是预定的特定错误？
4. **攻击频率/时机：** 攻击在模型生命周期（训练或推理 (inference)）的哪个阶段发生？

让我们更详细地查看这些维度。

### 按攻击者目的归类

攻击目标对其设计有显著影响：

- **错误分类（完整性攻击）：** 最常见的目的是使模型对给定输入产生错误输出。对于分类任务，这意味着导致 $f(x_{adv}) \neq y_{true}$，其中 $x_{adv}$ 是对抗性输入，$f$ 是模型，$y_{true}$ 是正确标签。攻击者不关心错误的输出具体是什么，只要它不正确即可。
- **降低置信度：** 攻击者可能不直接造成完全错误分类，而是旨在降低模型对其特定输入预测的置信度。这可能使模型显得不可靠，或更容易在后续攻击中被愚弄。
- **源/目标错误分类（定向攻击）：** 一个更具体的目标是，攻击者希望模型将特定输入 $x$ (源) 分类为一个特定的错误目标类别 $t$。目标是 $f(x_{adv}) = t$，其中 $t \neq y_{true}$。这些通常比非定向错误分类更难实现。
- **可用性攻击：** 攻击者旨在降低模型的整体性能，通常通过增加其计算时间（例如，拒绝服务）或使其在大量输入上失效。这通常与训练期间的投毒攻击有关。

### 按攻击者了解程度归类

攻击者对目标模型的了解程度决定了可行的攻击策略：

- **白盒攻击：** 攻击者拥有模型的完整信息，包括其架构、参数 (parameter)（权重 (weight)和偏置 (bias)）、激活函数 (activation function)，以及可能的训练数据或其分布。这使得能够进行直接利用模型结构和梯度的有效攻击。例子包括快速梯度符号法（FGSM）、投影梯度下降 (gradient descent)（PGD）和Carlini & Wagner (C&W) 攻击，这些方法常利用梯度信息来优化扰动。
- **黑盒 (black box)攻击：** 攻击者对模型的内部运作了解有限或一无所知。交互通常仅限于向模型查询输入并观察输出。
  - **基于分数的攻击：** 攻击者可以获得与模型预测相关的置信分数、概率或逻辑值。这些部分信息可以用于估计梯度（例如，使用有限差分）或采用依赖分数反馈的优化技术。
  - **基于决策的攻击：** 攻击者每次查询只收到最终预测标签（硬分类输出）。这是最严格的设置，通常需要大量查询才能找到对抗样本，例如通过界定决策边界（例如，边界攻击）。
  - **迁移攻击：** 一种常见的黑盒策略是针对本地*替代*模型（该模型经过训练以模仿目标模型或标准预训练 (pre-training)模型）生成对抗样本，然后将这些样本迁移到目标黑盒模型。这表明对抗样本通常在不同架构之间表现出可迁移性。

### 按攻击特定性归类

这个维度与攻击者目的有重叠，但更侧重于错误分类的定向性：

- **非定向攻击：** 目的仅仅是使模型错误分类输入 $x_{adv}$，不关心产生的错误类别是什么。从数学角度看，攻击者试图找到 $x_{adv}$，使得 $f(x_{adv}) \neq y_{true}$，同时最小化扰动（例如，$||x_{adv} - x||_p$）。
- **定向攻击：** 目标是使模型将 $x_{adv}$ 分类为特定的目标类别 $t$，其中 $t \neq y_{true}$。目标是找到 $x_{adv}$ 使得 $f(x_{adv}) = t$，同时最小化扰动。定向攻击通常更具挑战性，但在某些情况下也更具恶意（例如，强制自动驾驶汽车的感知系统将停车标志分类为限速标志）。

### 按攻击频率/时机归类

这种归类与前面讨论的攻击面直接相关：

- **规避攻击（测试时/推理 (inference)时）：** 这些攻击发生在模型训练并部署之后。攻击者在推理时操作单个输入以引起错误分类。这是研究最广泛的攻击类型，并假设模型参数 (parameter)是固定的。我们将在第2章中详细讨论这些内容。
- **投毒攻击（训练时）：** 这些攻击发生在训练阶段。攻击者将精心制作的恶意数据点（毒药）注入训练数据集。目标是破坏学习到的模型参数，无论是为了降低其整体性能（可用性攻击），还是为了安装后门，使其在推理时对某些输入产生特定错误行为（完整性攻击）。这些将在第3章中进行介绍。
- **信息获取攻击：** 这些攻击不一定旨在立即引起错误分类，而是为了获取有关模型或其训练数据的信息。例子包括成员推断（判断特定数据点是否用于训练）、属性推断（从训练数据推断敏感属性）和模型窃取（复制专有模型的功能）。这些隐私和安全影响将在第4章中讨论。

"> 对抗性机器学习 (machine learning)攻击主要归类维度的可视化。攻击常常是这些类别（例如，白盒、定向规避攻击）的组合。"

理解此分类体系非常重要。它提供了一个分析机器学习系统所面临威胁的体系，并有助于选择或设计合适的对策。在评估模型的安全性或提议的防御措施时，重要的是要考虑正在处理哪些类型的攻击（由这些类别定义）。随后的章节将介绍具体攻击和防御，通常会参考此分类方案。

## 参考资料

- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) — Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6572](https://doi.org/10.48550/arXiv.1412.6572)
  提出了快速梯度符号法 (FGSM)，并讨论了对抗性样本的基本特性，为白盒规避攻击奠定了基础。
- [Towards Evaluating the Robustness of Neural Networks](https://doi.org/10.1109/SP.2017.49) — Nicholas Carlini and David A. Wagner (2017)
  Journal: IEEE Symposium on Security and Privacy (SP); Publisher: IEEE; Pages: 39-57; DOI: [10.1109/SP.2017.49](https://doi.org/10.1109/SP.2017.49)
  介绍了强大的基于优化的白盒对抗性攻击 (C&W 攻击)，并详细说明了生成有目标和无目标对抗性样本的方法。
- [Practical Black-Box Attacks against Machine Learning Systems using Adversarial Examples](https://dl.acm.org/doi/10.1145/3052973.3053009) — Nicolas Papernot, Patrick McDaniel, Arjund Sinha, Z. Berkay Celik (2017)
  Journal: ACM Asia Conference on Computer and Communications Security (ASIACCS); Publisher: ACM; Pages: 506-519; DOI: [10.1145/3052973.3053009](https://doi.org/10.1145/3052973.3053009)
  展示了对抗性样本的迁移性，并介绍了使用替代模型的黑盒攻击策略。
- [Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models](https://arxiv.org/abs/1712.04248) — Wieland Brendel, Jonas Rauber, and Matthias Bethge (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1712.04248](https://doi.org/10.48550/arXiv.1712.04248)
  介绍了边界攻击，这是一种高效的基于决策的黑盒攻击，只需访问模型的最终标签预测。

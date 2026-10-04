---
course: "adversarial-machine-learning"
chapter: "advanced-evasion-attacks"
lesson: "attacking-ensemble-models"
sourceId: 4134
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-2-advanced-evasion-attacks/attacking-ensemble-models"
title: "攻击集成模型"
description: "制定针对模型集成生成有效对抗性样本的策略。"
order: 6
plots: []
sourceHash: "58b0a366d76343115e32bf25100d629b6e36dc278161d5f2df1b3fce941215a0"
sourceCorrections: []
---

集成方法结合了多个独立模型的预测，常被用来不仅提高预测准确性，还作为对抗性攻击的一种启发式防御。其直观原因是，为欺骗某个特定模型而制作的对抗性样本可能对其他模型无效，特别是当集成模型的成员各不相同时（例如，架构不同，或在不同数据子集上训练）。然而，集成模型并非不受规避攻击影响。攻击它们需要考虑其聚合决策过程的特定策略。

我们来看一个由 $M$ 个独立模型 $f_1, f_2, \ldots, f_M$ 组成的集成 $F$。最终预测 $F(x)$ 通常通过组合各个预测 $f_i(x)$ 得出，常用方法包括多数投票（用于分类）或平均概率。


$$
F(x) = \text{聚合}(f_1(x), f_2(x), \ldots, f_M(x))
$$


攻击者的目标是找到一个扰动 $\delta$，使得 $F(x + \delta) \neq F(x)$，同时满足 $||\delta||_p \le \epsilon$。

### 攻击集成模型的策略

可以采用几种方法来生成对抗集成模型的对抗性样本：

1. **运用可迁移性：** 正如在可迁移性部分讨论的，为某个模型制作的对抗性样本通常对其他模型也有效，特别是那些在类似数据上训练或具有类似架构的模型。攻击者可以生成一个对抗性样本，以集成模型中某个单一的、可能较弱的成员 $f_i$ 为目标，并希望它能充分迁移以欺骗聚合决策 $F$。或者，攻击者可以训练自己的*替代模型*来模仿集成模型的行为（如果集成模型是黑盒 (black box)），或者使用集成模型中已知的、可访问的模型作为代理。然后在这个替代/代理模型上执行攻击。虽然这种方法更简单，但它可能不如直接针对集成模型的攻击有效。
2. **针对聚合输出的优化：** 一个更直接的方法是，将攻击表述为一个以集成模型的组合输出为目标的优化问题。

   - **平均集成模型：** 如果集成模型平均预测概率（或对数几率），攻击者可以目标是最大化在此平均输出上计算的损失函数 (loss function)。例如，如果 $p_i(x)$ 是模型 $f_i$ 输出的概率向量 (vector)，则集成模型输出可能是 $\bar{p}(x) = \frac{1}{M} \sum_{i=1}^M p_i(x)$。基于优化的攻击（如PGD或C&W）可以调整为在寻找 $\delta$ 时，使用损失函数关于 $\bar{p}(x + \delta)$ 的梯度。
   - **投票集成模型：** 攻击多数投票集成模型更棘手，因为聚合函数不可微分。然而，可以使用近似方法。一种常见策略是将*平均*概率/对数几率输出作为替代目标，假设将平均预测推向错误类别可能会改变多数投票结果。另一种方法是迭代攻击当前对正确共识做出贡献的单个模型，目标是逐个改变它们的投票，直到多数方发生转变。
3. **同时攻击所有成员：** 攻击者可以尝试找到一个单一扰动 $\delta$，它对*所有*或*大多数*独立模型 $f_i$ 都有效。这通常涉及修改基于优化攻击中的目标函数。例如，目标不再是最小化单个模型的损失，而是最小化所有集成成员损失的总和或最大值：

   
   $$
   \text{最大化 } \sum_{i=1}^M \mathcal{L}(f_i(x+\delta), y_{\text{目标}}) \quad \text{或} \quad \text{最大化 } \max_{i} \mathcal{L}(f_i(x+\delta), y_{\text{目标}})
   $$
   

   受限于 $||\delta||_p \le \epsilon$。这通常会大幅增加攻击的计算成本，因为在每一步中都需要为所有模型计算梯度。

> 攻击集成模型的过程。攻击者寻找一个扰动 $\delta$ 来创建对抗样本 $x_{adv}$，该样本被输入到多个模型（$f_1, f_2, f_3$）中。各个模型的输出由一个聚合机制组合，目标是使最终的集成模型预测 $F(x_{adv})$ 不正确。

### 挑战与考量

攻击集成模型通常比攻击单一模型在计算上成本更高，特别是在针对聚合输出进行优化或同时攻击所有成员时。攻击的有效性通常取决于集成成员的多样性。由高度相似模型组成的集成模型可能不会比单一模型提供更多的额外弹性。反之，高度多样化的集成模型则在攻击上更具挑战性，难以通过单一的小扰动成功攻破。

在评估集成防御的鲁棒性时，使用专门为集成模型设计的攻击很重要，而不是仅仅依赖于针对单个成员生成的攻击的可迁移性。正如我们将在第6章中看到的，评估防御需要针对所测试的特定防御机制进行调整的自适应攻击。

下一节将提供一个实践环节，您将在其中实现本章中讨论的一些规避攻击，可能包括基本的集成攻击思想。

## 参考资料

- [Practical Black-Box Attacks against Machine Learning Systems using Adversarial Examples](https://doi.org/10.1145/3052973.3053009) — Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z. Berkay Celik, and Ananthram Swami (2017)
  Journal: Proceedings of the 2017 ACM Asia Conference on Computer and Communications Security (ASIACCS); Publisher: Association for Computing Machinery, Inc.; Pages: 506-519; DOI: [10.1145/3052973.3053009](https://doi.org/10.1145/3052973.3053009)
  详细介绍了利用迁移性和代理模型进行黑盒对抗攻击的方法，这是在模型细节未知时攻击集成模型的主要策略。
- [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204) — Florian Tramèr, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, Patrick McDaniel (2017)
  Journal: arXiv preprint arXiv:1705.07204; DOI: [10.48550/arXiv.1705.07204](https://doi.org/10.48550/arXiv.1705.07204)
  介绍了集成对抗训练，并讨论了针对集成模型的白盒攻击策略，包括优化组合输出和同时攻击多个成员。
- [Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://proceedings.mlr.press/v80/athalye18a/athalye18a.pdf) — Anish Athalye, Nicholas Carlini, and David Wagner (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 80; Pages: 273-282; DOI: [10.55982/athalye18a](https://doi.org/10.55982/athalye18a)
  展示了依赖混淆梯度的防御如何被自适应攻击规避，强调了对对抗性防御进行严格评估的需求。

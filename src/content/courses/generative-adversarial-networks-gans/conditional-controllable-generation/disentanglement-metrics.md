---
course: "generative-adversarial-networks-gans"
chapter: "conditional-controllable-generation"
lesson: "disentanglement-metrics"
sourceId: 2723
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-4-conditional-controllable-generation/disentanglement-metrics"
title: "解耦度量与挑战"
description: "讨论与评估潜在空间解耦相关的度量方法及难题。"
order: 6
plots: []
sourceHash: "95fb40f11a8ca492044e28e91fa61ee28e80a5134d89fdc40a1f543672e291d5"
sourceCorrections: []
---

评估潜在表征在多大程度上实际 *解耦* 了数据中变化的潜在因素，是一项主要的任务。解耦的目标是学习一个潜在空间 $z$（或其内部的特定编码 $c$），使得单个维度或维度组与生成输出 $G(z, c)$ 的不同、可解释的语义特征相对应。例如，在一个人脸数据集中，理想情况下，一个潜在维度可能控制头发颜色，另一个可能控制姿态，第三个可能控制眼镜的有无，且这些都是独立的。可解释的潜在表征在各种生成模型中都备受关注。

实现这种解耦表征非常有价值。它显著提升了生成器的可控性，允许精确操作特定输出属性而不影响其他属性。它还可以提高可解释性，帮助我们了解模型学到了什么数据结构。此外，一些研究表明，解耦表征可能带来更好的泛化能力和样本质量，尽管这是当前仍在进行的研究方向。

然而，衡量解耦面临很大困难。尚无单一、普遍接受的解耦数学定义，导致出现了多种提出的度量方法，每种方法都有其自身的假设和局限性。

### 解耦的定量度量方法

"已经提出了若干度量方法，用以量化 (quantization)解耦程度。其中许多方法依赖于能够获取数据集的真实变化因素，这是一项显著的局限性，因为在实际情况中，此类标签通常不可得。"

以下是一些常见的度量方法:

1. **互信息间隙 (MIG):** 与InfoGAN一同被提出，MIG旨在衡量单个潜在维度 $z_i$ 包含关于单个真实因素 $v_j$ 的信息量。它计算所有 $(i, j)$ 对的互信息 $I(z_i; v_j)$，并且对于每个因素 $v_j$，找出具有最高互信息的潜在维度 $z_i$。“间隙”是指该因素的最高和次高互信息之间的归一化 (normalization)差值。较高的间隙表明因素 $v_j$ 主要由单个潜在维度捕获。

   - **直观理解:** 高MIG分数表示每个已知变化因素与 *一个* 特定潜在维度强相关，而与其它维度弱相关。
   - **局限性:** 需要真实因素。对互信息的估计敏感。无法捕获因素可能在多个潜在维度上线性编码的情况。
2. **FactorVAE分数:** 在FactorVAE论文中提出，该度量训练一个简单的分类器（通常是基于共享相同因素 $v_j$ 的样本的 $z_i$ 中位数进行多数投票的分类器），仅使用 *一个* 潜在维度 $z_i$（特别是对该因素方差最小的维度）来预测真实因素 $v_j$ 的值。该分类器的准确性作为分数。

   - **直观理解:** 如果单个潜在维度很好地编码了一个因素，那么仅使用该维度的一个简单分类器就应该能够准确预测该因素的值。
   - **局限性:** 需要真实因素。分类器的选择及其简单性可能无法捕获更复杂的编码。
3. **分离属性可预测性 (SAP) 分数:** 与MIG的精神相似，SAP分数也衡量从单个潜在维度 $z_i$ 预测真实因素 $v_j$ 的能力。它训练一个线性SVM或逻辑回归分类器，以从每个潜在维度 $z_i$ 预测每个因素 $v_j$。一个因素的SAP分数是使用最具预测性的潜在维度和次具预测性的潜在维度的预测准确性之间的差值。

   - **直观理解:** 高SAP分数表明每个因素都由一个特定潜在维度比其他维度更好地预测，侧重于线性可预测性。
   - **局限性:** 需要真实因素。侧重于线性可分性，可能遗漏非线性编码。
4. **解耦性、完整性和信息量 (DCI) 分数:** 该框架试图通过衡量三个方面来提供一种视角:

   - **解耦性:** 单个潜在维度在捕获少数因素时是否具有选择性？（通过从潜在维度预测因素的分类器所获得的特征重要性稀疏性来衡量）。
   - **完整性:** 每个因素是否主要映射到单个潜在维度？（通过每个因素的预测器熵来衡量）。
   - **信息量:** 从学习到的整体潜在表征中，真实因素能被预测得有多好？（整体预测准确性）。
   - **直观理解:** 提供了映射的多方面视角。
   - **局限性:** 需要真实因素。计算和解释可能更复杂。

### 解耦中的核心挑战

评估和实现解耦面临几个基本挑战:

1. **解耦的定义:** 缺乏形式化、一致认可的定义，使得评估本质上是主观的并依赖于度量方法。什么构成了一个“变化因素”？它们应该相互独立吗？线性可分吗？这些问题仍待解决。
2. **对真实值的依赖:** 大多数定量度量方法高度依赖于了解数据集中真实潜在的变化因素。这通常是不切实际的，因为数据集很少附带如此完美的标注。不要求真实值的度量方法存在，但通常被认为可靠性较低或更难解释。
3. **解耦与质量的权衡:** 研究表明，在实现高水平解耦（根据现有度量方法）和生成高保真样本之间，可能存在一个基本张力。追求极致解耦，例如使用高 $\beta$ 值的Beta-VAE等方法，有时会导致重建质量下降或生成的样本视觉吸引力较差。寻找正确的平衡通常取决于应用。
4. **度量方法的敏感性和可靠性:** 现有度量方法可能对超参数 (parameter) (hyperparameter)（例如，MI估计的bin数量、分类器选择）、随机种子和实现细节敏感。不同的度量方法有时会产生相互矛盾的模型排名，使得难以得出明确的结论。可复现性可能是一个问题。
   "5. **隐含假设:** 许多度量方法隐含假设变化因素应在潜在编码中被独立地、通常是线性地捕获。数据变化可能更复杂且相互关联，使得根据这些假设的完美解耦变得不可能，甚至是不理想的。"
5. **可扩展性:** 计算某些度量方法可能计算密集，特别是对于高维潜在空间或大型数据集。

总之，尽管解耦是构建更可控、可解释生成模型的一个非常有吸引力的目标，但准确可靠地衡量它仍然是一个重要的未解决问题。现有度量方法提供了有用的诊断信息，特别是在已知因素的受控环境中，但应谨慎解释它们，考虑到其固有的局限性以及关于何为真正解耦的持续争议。定性评估，涉及可视化遍历单个潜在维度的效果，仍然是一项重要的补充评估技术。

## 参考资料

- [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](http://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf) — Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel (2016)
  Journal: Advances in Neural Information Processing Systems 29; Publisher: Curran Associates, Inc.; Volume: 29; Pages: 2172--2180
  介绍了InfoGAN用于学习可解释表示，以及用于评估解耦的互信息间隙（MIG）度量。
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=H1xHAE9cK_) — Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Alexander Lerchner, and Andrea Banino (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1804.03599](https://doi.org/10.48550/arXiv.1804.03599)
  介绍了Beta-VAE，一种通过修改VAE目标来实现解耦表示的方法，并讨论了解耦与重建质量之间的权衡。
- [Disentangling by Factorizing](https://proceedings.mlr.press/v80/kim18a/kim18a.pdf) — Hyunjik Kim and Andriy Mnih (2018)
  Journal: International Conference on Machine Learning (ICML); Volume: 80; Pages: 2642-2651
  介绍了FactorVAE用于学习解耦表示，以及FactorVAE分数作为定量度量。
- [Towards a Better Understanding of Disentangling Representations](https://proceedings.mlr.press/v97/locatello19a/locatello19a.pdf) — Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schoelkopf, and Olivier Bachem (2019)
  Journal: International Conference on Machine Learning (ICML); Volume: 97; Pages: 4032-4043
  一项综合研究，评估了各种解耦方法和度量，强调了当前方法的局限性以及定义和衡量解耦的挑战。

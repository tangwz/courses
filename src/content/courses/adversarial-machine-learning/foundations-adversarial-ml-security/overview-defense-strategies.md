---
course: "adversarial-machine-learning"
chapter: "foundations-adversarial-ml-security"
lesson: "overview-defense-strategies"
sourceId: 4122
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-1-foundations-adversarial-ml-security/overview-defense-strategies"
title: "防御策略概览"
description: "介绍对抗性攻击防御机制的主要类别。"
order: 6
plots: []
sourceHash: "1e5b536942ed236711009a87e135558f1644d7a6f84c573291cb0c59d60308be"
sourceCorrections: []
---

在确定了漏洞的存在以及攻击者破坏机器学习 (machine learning)模型的各种方式之后，接下来的自然问题是：我们如何构建能够抵御这些威胁的系统？正如软件工程包含安全实践一样，机器学习也需要专门的防御策略来保护模型在训练和推理 (inference)过程中的安全。本节对防御对抗性攻击的主要方法提供一个整体概览，为第5章中更详细的讨论奠定基础。

开发有效的防御方法是一个活跃且富有挑战的研究方向。这些策略通常旨在阻止攻击成功，或对其进行检测和减轻。我们可以根据它们在机器学习流程中发挥作用的位置，对这些防御方法进行大致分类。

### 主要防御策略类别

1. **修改训练过程（提升内在鲁棒性）：**
   也许最直接的方法是在训练阶段使模型本身对对抗性扰动具有更强的抵抗力。目的是训练一个能学习到足够强大的特征以抵御微小输入变化的模型。

   - **对抗训练：** 这是目前对抗规避攻击最有效且被广泛研究的防御技术之一。其主要思想简单但有效：在训练数据集中添加即时生成的对抗样本。通过强制模型在训练期间正确分类这些对抗性输入，模型能学会对攻击者使用的扰动类型不那么敏感。我们将在第5章考察基于投影梯度下降 (gradient descent)（PGD）的对抗训练等变体。
   - **正则化 (regularization)技术：** 其他方法则修改训练目标或过程，以促使决策边界更平滑，或以间接提升鲁棒性的方式惩罚模型复杂度。
2. **修改输入（输入预处理）：**
   这些防御方法不改变模型或其训练，而是在将数据输入模型之前对其进行预处理。目的是去除或减少输入中存在的对抗性扰动。

   - **输入转换：** 空间平滑、JPEG压缩或特征挤压等技术作用于输入图像或数据点。其理念是对抗性扰动通常依赖于特定的高频模式，这些转换可能破坏这些模式，从而在分类前有效地“去噪”输入。虽然有时对特定攻击有效，但它们需要仔细评估，因为它们也可能轻微降低在正常输入上的性能。
3. **修改模型（架构或认证）：**
   这一类别涉及改变模型架构，或采用提供形式化保证的鲁棒性技术。

   - **架构变更：** 一些研究着眼于网络架构、激活函数 (activation function)的修改，或使用可能更有效的特定类型的层，尽管找到普遍有效的架构防御仍然困难。
   - **认证防御：** 这类防御方法特别引人注意，它们提供在特定扰动幅度内（例如，对于半径为 $\epsilon$ 的 $L_p$ 球体内的任何输入扰动）*可证明的*鲁棒性保证。随机平滑是我们将介绍的一个重要例子，它通过分析模型在随机噪声增强输入版本上的预测，提供概率性鲁棒性证明。
4. **使用外部模型（检测与拒绝）：**
   这些方法在主模型旁边增加独立的组件，以检测输入是否可能是对抗性的。

   - **对抗检测器：** 这些通常是辅助分类器，训练用于区分正常输入和对抗样本。如果一个输入被标记 (token)为对抗性，它可以被拒绝或以不同方式处理。设计自身能抵御自适应攻击的检测器是一个很大的挑战。

> 对抗对抗性攻击的常见防御策略的一个概括性分类。

### 严格评估的重要性

谨慎对待防御机制是很重要的。从历史上看，许多提出的防御措施很快就被略微修改或*自适应*攻击攻破，这些攻击是专门设计来规避防御的。一个常见问题是*梯度遮蔽*或*混淆*，即防御措施使得基于梯度的攻击（如FGSM或PGD）更难找到对抗样本，从而造成虚假的安全感。然而，更强的基于优化的攻击或不同的攻击策略仍然可能成功。因此，严格评估防御措施以对抗强大、自适应的攻击者是非常必要的，我们将在第6章专门讨论这个话题。

### 防御其他威胁

尽管许多防御侧重于推理 (inference)时的规避攻击，但也有减轻数据投毒、后门攻击（第3章）和推理攻击（第4章）的策略。防御投毒通常涉及数据净化、训练期间的异常检测或联邦学习中的聚合方法。防御与隐私相关的推理攻击常与差分隐私等技术相交叉。

本概览介绍了防御策略。没有一种完美的防御；通常，有效的保护涉及多种技术的结合，并伴随着权衡，这通常体现在性能与正常数据上的标准准确性之间，或计算成本的增加。后续章节将对具体的先进攻击以及旨在防御它们的机制进行更详细的考察。

## 参考资料

- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) — Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1706.06083](https://doi.org/10.48550/arXiv.1706.06083)
  引入了PGD对抗训练，这是一种用于提高模型鲁棒性的基础且广泛采用的防御技术。
- [Certified Adversarial Robustness via Randomized Smoothing](https://proceedings.mlr.press/v97/cohen19b.html) — Jeremy Cohen, Elan Rosenfeld, Zico Kolter (2019)
  Journal: Proceedings of the 36th International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 97; Pages: 1310-1320; DOI: [10.48550/arXiv.1902.02918](https://doi.org/10.48550/arXiv.1902.02918)
  介绍了随机平滑法，一种为深度学习模型提供可证明的对抗攻击鲁棒性保证的方法。
- [Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://proceedings.mlr.press/v80/athalye18a.html) — Anish Athalye, Nicholas Carlini, David Wagner (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning; Publisher: PMLR; Volume: 80; Pages: 274-283; DOI: [10.48550/arXiv.1802.00420](https://doi.org/10.48550/arXiv.1802.00420)
  严格评估了许多现有防御措施，强调了梯度掩蔽等问题以及对抗自适应攻击进行严谨评估的必要性。

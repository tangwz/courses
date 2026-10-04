---
course: "adversarial-machine-learning"
chapter: "evaluating-model-robustness"
lesson: "interpreting-robustness-results"
sourceId: 4184
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-6-evaluating-model-robustness/interpreting-robustness-results"
title: "解释鲁棒性评估结果"
description: "报告和解释对抗性鲁棒性评估结果的最佳实践。"
order: 5
plots: ["plots/4184-0.json"]
sourceHash: "4c56bde8190dcbb595d8bf470e0ee34f46cbbd6472fca60e58c4ff4329afae33"
sourceCorrections: []
---

评估模型鲁棒性会生成大量来自各种攻击和防御方法的数字、图表和日志。理解和解释这些结果是主要任务。仅仅报告“模型X在PGD-$L_\infty$攻击下，当$\epsilon=8/255$时，达到60%的准确率”是不够的。解释鲁棒性需要仔细考虑背景、所用指标以及将面临的问题。

### 结合背景理解鲁棒性数据

单一的鲁棒性指标，例如攻击下的准确率，只有置于恰当背景中才有意义。请自问以下问题：

- **具体的攻击配置是什么？** 请明确说明攻击算法（例如，PGD、C&W、AutoAttack）、扰动限制（$L_p$范数和幅度$\epsilon$）、攻击迭代次数、步长、随机初始化次数以及其他超参数 (parameter) (hyperparameter)。这些参数的微小变化可能会明显改变结果。
- **基线是什么？** 模型的准确率与在干净、未受扰动数据上的准确率相比如何？在相同攻击下，标准、未防御模型的准确率是多少？与这些基线进行比较可以显示防御的效果及其相关成本（通常是干净准确率的下降）。
- **评估代表什么样的威胁模型？** 评估是否假设攻击者是拥有完全知识的白盒攻击者，还是更受限的黑盒 (black box)场景？解释会根据假设的攻击者能力发生显著变化。对抗弱攻击的鲁棒性，在面对更强的对手时可能作用甚微。
- **评估是否是自适应的？** 如前所述，防御有时可以通过掩盖梯度或其他破坏特定攻击算法的技术来获得表观鲁棒性。评估是否采用了专门设计用于规避防御机制的自适应攻击？仅针对标准、非自适应攻击所显示的鲁棒性应持怀疑态度看待。

### 不再局限于简单准确率

尽管特定攻击下的准确率是一个常见指标，但它并不能说明全部情况。全面的解释应考虑多个方面：

- **准确率与扰动预算（$\epsilon$）的关系：** 除了报告单一$\epsilon$值下的准确率外，还要分析趋势。随着允许的扰动幅度增加，准确率下降的速度有多快？绘制准确率随$\epsilon$变化的曲线，能更全面地反映模型的鲁棒能力。鲁棒模型通常与标准模型相比，准确率下降更为平缓。



![准确率与扰动预算的关系](plots/4184-0.json)



> 标准模型与鲁棒模型在不断增强的$L_\infty$ PGD攻击（$\epsilon$）下的准确率下降情况。

- **鲁棒性-准确率权衡：** 许多有效的防御方法，特别是对抗训练，通常会导致在干净、未受扰动数据上的准确率下降。请量化 (quantization)这种权衡。对于目标应用而言，鲁棒性的提升是否值得以标准性能的潜在下降为代价？
- **不同$L_p$范数：** 评估$L_\infty$、$L_2$以及可能的$L_0$或$L_1$等约束。在一个范数下的表现不自动意味着在其他范数下的表现。例如，一个对微小、广泛变化（$L_\infty$）鲁棒的模型，仍然可能容易受到影响较少像素的较大变化（$L_2$或$L_0$）的攻击。相关的范数通常取决于应用领域和预期的扰动类型。
- **最小扰动：** 除了固定$\epsilon$并测量准确率外，你还可以测量导致错误分类所需的*平均最小扰动*。平均最小$\epsilon$值越高，表明鲁棒性越强。
- **攻击可迁移性：** 如果评估黑盒 (black box)鲁棒性，请考虑针对不同代理模型生成的攻击的可迁移性。高可迁移性表明存在不特定于某个模型架构或权重 (weight)的漏洞。
- **计算成本：** 对抗训练或可证明防御等防御方法可能会明显增加训练时间或推理 (inference)延迟。这种实际成本必须与安全收益进行权衡。

### 识别问题

警惕那些好得令人难以置信的结果。一个常见问题是**梯度混淆**，即防御机制使得*特定*基于梯度的攻击更难找到对抗样本，而非真正地提升模型的鲁棒性。梯度混淆的迹象包括：

1. 对强梯度攻击（如PGD）表现出高鲁棒性，但易受更简单的无梯度方法（如SPSA）或基于分数/决策的攻击影响。
2. 对迭代次数较少的攻击表现出高鲁棒性，但当攻击步数大幅增加时，准确率急剧下降。
3. 对白盒攻击表现强劲，但对黑盒 (black box)迁移攻击表现不佳。

始终使用多种攻击算法进行健全性检查，如果评估新型防御，还要包括自适应攻击。仅仅依赖标准PGD或FGSM来评估新型防御是不够的。

### 报告与比较的挑战

比较不同研究中的鲁棒性结果可能具有挑战性，原因在于：

- 数据集（MNIST、CIFAR-10、ImageNet具有不同的固有难度）。
- 模型架构。
- 评估框架（例如，ART、CleverHans、Foolbox）内的攻击实现和超参数 (parameter) (hyperparameter)。
- 攻击的计算预算。

在报告自己的结果时，请明确说明所有这些细节，以便进行公平比较。在理解他人的结果时，也要查找这些信息。像RobustBench这样的资源旨在标准化特定基准上的评估，从而提供更可靠的比较基础，但即便如此，它们在所考虑的攻击范围方面也存在局限性。

归根结底，解释鲁棒性评估结果并非为了找到一个单一数字，而是为了在特定、明确定义的假设下，对模型的安全态势建立全面的理解。这包括分析趋势、考虑权衡、与适当的基线进行比较，并警惕梯度混淆等评估错误。这种理解对于在安全敏感应用中部署机器学习 (machine learning)模型并做出明智决策非常重要。

## 参考资料

- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) — Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1706.06083](https://doi.org/10.48550/arXiv.1706.06083)
  这篇基础性论文介绍了PGD对抗训练（一种基准防御），并详细阐述了PGD攻击，为评估模型对抗强大白盒对抗者的鲁棒性设定了早期标准。
- [Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://arxiv.org/abs/1802.00420) — Anish Athalye, Nicholas Carlini, David Wagner (2018)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1802.00420](https://doi.org/10.48550/arXiv.1802.00420)
  这是一篇重要论文，揭露了梯度模糊现象，即防御措施在标准攻击下看似稳健，但在自适应评估下则失效。它为正确的鲁棒性评估提供了指导。
- [Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Attacks](https://arxiv.org/abs/2003.01690) — Francesco Croce, Matthias Hein (2020)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.2003.01690](https://doi.org/10.48550/arXiv.2003.01690)
  这篇论文介绍了AutoAttack，一个全面的评估框架，它使用多种攻击组合来提供更可靠的对抗鲁棒性衡量，超越了单一攻击评估。

---
course: "adversarial-machine-learning"
chapter: "foundations-adversarial-ml-security"
lesson: "attack-surfaces-training-inference"
sourceId: 4115
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-1-foundations-adversarial-ml-security/attack-surfaces-training-inference"
title: "攻击面：训练与推理"
description: "分析机器学习模型的训练阶段与推理阶段的不同攻击途径。"
order: 3
plots: []
sourceHash: "1caede7ec2beb1974943c7fc09a18818e025bc2034ce482b58aa0ccebe23bc56"
sourceCorrections: []
---

了解机器学习 (machine learning)系统的安全状况，需要分析攻击者何时以及如何与系统交互。机器学习模型的生命周期通常分为两个主要阶段：训练和推理 (inference)。每个阶段都为攻击者提供了不同的机会和限制，从而形成不同的攻击面。

## 训练阶段攻击面

训练阶段是模型从数据集中学习模式和关联的过程。主要目的是优化模型参数 (parameter)，以使训练数据上的损失函数 (loss function)最小化。此阶段的攻击旨在破坏学习过程本身，最终影响最终训练模型的完整性或可用性。

**攻击者在训练阶段的目标：**

- **模型性能下降（可用性攻击）：** 降低模型在合法输入上的整体性能。
- **定向错误行为（完整性攻击）：** 导致模型错误分类特定输入或表现出偏颇行为。
- **后门植入：** 在模型中植入隐藏的恶意功能，这些功能在推理 (inference)时遇到特定触发器时被激活。

**攻击者能力：**

攻击者影响训练过程的能力决定了这些攻击的可行性。常见情况包括：

- **数据篡改：** 最常分析的途径是修改训练数据集。这可能意味着注入恶意样本（数据投毒）或微小地改变现有样本。
- **算法操纵：** 如果攻击者能够控制学习算法（例如，影响超参数 (hyperparameter)、损失函数或优化过程），他们可以直接引导模型进入受损状态。这在典型部署场景中不太常见，但在联邦学习或机器学习 (machine learning)即服务（MLaaS）平台中，如果安全措施不当，则有可能发生。
- **基础设施控制：** 破坏底层训练基础设施（硬件、软件环境）可提供广泛的控制，但这更多属于传统网络安全范畴。

**训练阶段攻击示例：**

- **数据投毒：** 涉及向训练集中注入精心构造的恶意数据点。这些点可以设计成移动决策边界、降低整体准确性或制造定向弱点。例如，攻击者可能会添加错误标签的图像以降低分类器的性能，或引入微小的伪影，旨在稍后导致特定输入的错误分类。
- **后门（特洛伊木马）攻击：** 一种更隐蔽的投毒形式，其中注入的数据训练模型对典型输入正常响应，但在输入中存在攻击者定义的特定触发模式时表现出恶意行为（例如，错误分类）。此触发器可以是图像中的一小块补丁、文本中的特定短语等。模型在标准验证期间看似正常，但包含一个隐藏的缺陷。

训练期间的攻击从根本上改变了模型 $\theta$ 的学习参数 $f_\theta(x)$。由此产生的模型本身就存在缺陷，即使在推理时遇到任何潜在的恶意输入之前也是如此。

> 机器学习生命周期包含不同的训练和推理阶段，每个阶段都呈现出各自的攻击面。训练阶段攻击（红色）针对学习过程，而推理阶段攻击（蓝色）则利用已部署的模型。

## 推理 (inference)阶段攻击面

推理（或测试/部署）阶段是指已训练模型用于对新的、以前未见过的数据进行预测。此阶段的攻击假设模型参数 (parameter) $\theta$ 是固定的，并且旨在在不直接修改模型的情况下，利用已部署的模型 $f_\theta(x)$。

**攻击者在推理阶段的目标：**

- **规避：** 构造恶意输入（对抗样本）$x'$，使其与合法输入 $x$ 相似（例如，小的 $L_p$ 距离，$||x' - x||_p \leq \epsilon$），但导致模型产生错误的输出 $f_\theta(x') \neq f_\theta(x)$。
- **模型窃取（提取）：** 通过查询黑盒 (black box)模型并观察输入-输出对，复制其功能，从而构建一个替代模型。
- **信息提取（推理攻击）：** 通过与模型互动，推断有关训练数据或模型属性的敏感信息。示例包括成员推断（判断特定数据点是否用于训练）和属性推断（判断训练数据的敏感属性）。

**攻击者能力：**

- **输入操纵：** 攻击者可以修改输入模型的输入。修改的程度通常受限（例如，通过 $L_p$ 范数），以保持与合法数据的相似性或确保感知上的相像。
- **查询访问：** 攻击者可以使用选定的输入查询模型并查看其输出。访问可能是黑盒（仅输入-输出对可用）、灰盒（输出包含置信度分数）或白盒（完全访问模型架构和参数）。
- **查看：** 攻击者查看模型的预测，可能包括置信度分数或其他元数据。

**推理阶段攻击示例：**

- **规避攻击（对抗样本）：** 这些是研究最广泛的推理时攻击。像快速梯度符号法（FGSM）、投影梯度下降 (gradient descent)（PGD）以及Carlini & Wagner（C&W）攻击等技术会计算对输入的微小扰动，旨在导致错误分类。这些攻击通常利用梯度信息（白盒），但也可以通过迁移性或基于查询的方法进行黑盒访问。
- **成员推断攻击：** 通过查看模型对特定数据点的输出（例如，置信度分数），攻击者试图判断这些点是否属于模型的训练集。这引发了隐私问题，特别是对于在敏感数据上训练的模型。
- **模型窃取：** 攻击者使用不同的输入查询目标模型（通常是API），并利用这些响应来训练自己的模型副本。这可能损害原始模型中投入的知识产权。

推理阶段攻击针对模型的*应用*，通过其学习到的决策边界或其预测泄漏信息。模型本身在这些攻击期间保持不变。

## 攻击面的对比

训练和推理 (inference)攻击面之间的区别对于理解和减轻威胁很重要：

- **影响：** 训练攻击从一开始就损害模型的*完整性*。推理攻击则损害其对特定输入或查询的预测的*可靠性*或*保密性*。
- **时机：** 训练攻击发生在部署之前，而推理攻击发生在部署之后。
- **攻击者互动方式：** 训练攻击通常需要影响数据供应链或训练过程。推理攻击通常涉及与已部署的模型端点进行互动。
- **防御：** 防御通常是针对特定阶段的。数据清洗和训练程序（如对抗训练）主要解决训练阶段的威胁。输入验证、梯度遮蔽和差分隐私更适用于推理阶段的威胁。

根据攻击者在给定威胁模型中的目标和能力，了解攻击者可能瞄准哪个攻击面，对于为您的机器学习 (machine learning)系统设计合适的安全措施非常重要。接下来的章节将详细说明在这些阶段中执行攻击的具体技术。

## 参考资料

- [Poisoning Attacks against Support Vector Machines](https://arxiv.org/abs/1206.6389) — Battista Biggio, Blaine Nelson, and Pavel Laskov (2012)
  Journal: Proceedings of the 29th International Conference on Machine Learning (ICML); Publisher: Omnipress; Pages: 1807-1814; DOI: [arXiv:1206.6389](https://doi.org/arXiv%3A1206.6389)
  本文介绍了早期针对支持向量机的投毒攻击，展示攻击者如何操纵训练数据来降低模型性能。
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) — Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1706.06083](https://doi.org/10.48550/arXiv.1706.06083)
  本文提出投影梯度下降 (PGD) 作为一种强大的对抗性攻击和一种鲁棒的对抗性训练方法，以提高模型的弹性。
- [Membership Inference Attacks Against Machine Learning Models](https://ieeexplore.ieee.org/document/8014035) — Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov (2017)
  Journal: IEEE Symposium on Security and Privacy (S&P); Publisher: IEEE; Pages: 372-387; DOI: [10.1109/SP.2017.37](https://doi.org/10.1109/SP.2017.37)
  本文展示了一种实用的成员推断攻击，允许攻击者判断某个数据点是否属于模型的训练集。

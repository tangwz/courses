---
course: "adversarial-machine-learning"
chapter: "foundations-adversarial-ml-security"
lesson: "math-formulation-adversarial-examples"
sourceId: 4117
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-1-foundations-adversarial-ml-security/math-formulation-adversarial-examples"
title: "对抗样本的数学表述"
description: "了解对抗样本的数学基础以及扰动约束。"
order: 4
plots: []
sourceHash: "fa3fa30bbd189d32dcd25cb7b1758b665c5cfa1b1b187e5b5069b6ae367ee124"
sourceCorrections: []
---

在明确机器学习 (machine learning)模型可能存在弱点，且攻击者具有不同目标和能力后，我们来对*对抗样本*这一核心思想进行形式化定义。对抗样本本质上是一种经过精心制作的输入，它看起来几乎与合法输入相同，但却能使模型产生错误的输出。

设有一个机器学习模型，用函数 $f$ 表示，它将来自某个域（例如图像或文本）的输入 $\boldsymbol{x}$ 映射到输出 $y$（例如类别标签）。设 $y = f(\boldsymbol{x})$ 是给定输入 $\boldsymbol{x}$ 的正确输出。

一个无目标对抗样本 $\boldsymbol{x}'$ 是满足以下两个条件的修改后输入：

1. **误分类：** 模型对 $\boldsymbol{x}'$ 输出错误的标签。
   
   $$
   f(\boldsymbol{x}') \neq y
   $$
   
2. **接近性：** 修改后的输入 $\boldsymbol{x}'$ 与原始输入 $\boldsymbol{x}$ “接近”。

在目标攻击中，目标更具体：使模型输出特定的错误标签 $y_{target}$（其中 $y_{target} \neq y$）。第一个条件变为：


$$
f(\boldsymbol{x}') = y_{target}
$$


### 定义扰动

原始输入与对抗样本之间的差异是*扰动*，表示为 $\boldsymbol{\delta}$：


$$
\boldsymbol{\delta} = \boldsymbol{x}' - \boldsymbol{x}
$$


因此，我们可以将对抗样本写为 $\boldsymbol{x}' = \boldsymbol{x} + \boldsymbol{\delta}$。“接近性”条件表示扰动 $\boldsymbol{\delta}$ 必须根据某种度量是小的。

### 度量扰动大小：$L_p$ 范数

我们如何从数学上量化 (quantization)“小”？对抗机器学习 (machine learning)中的标准方法是使用 $L_p$ 范数来度量扰动向量 (vector) $\boldsymbol{\delta}$ 的大小。范数的选择反映了关于何种变化是“不可察觉”或“允许”的不同假设。

设 $\boldsymbol{\delta}$ 是一个维度为 $d$ 的向量（例如，图像中的 $d$ 个像素）。常用范数包括：

- **$L_\infty$ 范数（最大变化量）：** 度量输入中任意单个元素的最大绝对变化。定义为：

  
  $$
  \|\boldsymbol{\delta}\|_\infty = \max_{i=1, \dots, d} |\delta_i|
  $$
  

  一个 $L_\infty$ 约束 $\|\boldsymbol{\delta}\|_\infty \le \epsilon$ 表示任何单个输入特征（例如像素值）的变化不超过 $\epsilon$。这在图像攻击中广泛使用，因为小的均匀变化通常难以察觉。对于像素值归一化 (normalization)到 $[0, 1]$ 的图像，一个常用的 $\epsilon$ 值是 $8/255$。
- **$L_2$ 范数（欧几里得距离）：** 度量 $\boldsymbol{x}$ 和 $\boldsymbol{x}'$ 之间的标准欧几里得距离。

  
  $$
  \|\boldsymbol{\delta}\|_2 = \sqrt{\sum_{i=1}^{d} \delta_i^2}
  $$
  

  一个 $L_2$ 约束 $\|\boldsymbol{\delta}\|_2 \le \epsilon$ 限制了变化向量的整体大小。变化可能集中在少数特征上，或者分散地薄弱分布。
- **$L_1$ 范数（绝对变化和）：** 度量所有元素的绝对变化之和。

  
  $$
  \|\boldsymbol{\delta}\|_1 = \sum_{i=1}^{d} |\delta_i|
  $$
  

  一个 $L_1$ 约束 $\|\boldsymbol{\delta}\|_1 \le \epsilon$ 促使稀疏性，表示扰动可能涉及较大的变化，但只影响极少数特征（适用于文本特征等高维稀疏数据）。
- **$L_0$ 范数（变化元素数量）：** 计算 $\boldsymbol{\delta}$ 中非零元素的数量。

  
  $$
  \|\boldsymbol{\delta}\|_0 = \sum_{i=1}^{d} \mathbb{I}(\delta_i \neq 0)
  $$
  

  其中 $\mathbb{I}(\cdot)$ 是指示函数（如果为真则为 1，否则为 0）。一个 $L_0$ 约束 $\|\boldsymbol{\delta}\|_0 \le k$ 表示至多 $k$ 个特征（例如像素）可以被修改。这在计算上更难处理，但直接模拟了对有限数量输入组件的修改。

$L_p$ 范数和扰动预算 $\epsilon$（或 $L_0$ 的 $k$ 值）的选择是威胁模型的重要组成部分，定义了攻击者的能力。

### 作为优化问题的对抗样本生成

找到对抗样本通常可以被视为一个优化问题。有两种常见表述：

1. **最小化扰动：** 寻找能导致误分类的最小扰动 $\boldsymbol{\delta}$（由某个 $L_p$ 范数度量）。

   
   $$
   \min_{\boldsymbol{\delta}} \|\boldsymbol{\delta}\|_p \quad \text{使得} \quad f(\boldsymbol{x} + \boldsymbol{\delta}) \neq y
   $$
   

   我们通常还需要确保生成的 $\boldsymbol{x}' = \boldsymbol{x} + \boldsymbol{\delta}$ 仍然是一个有效输入（例如，像素值保持在允许的 $[0, 1]$ 范围内）。
2. **最大化损失（在预算内）：** 寻找扰动 $\boldsymbol{\delta}$，使其在预设扰动预算 $\epsilon$ 内最大化模型的预测误差（损失）。

   
   $$
   \max_{\boldsymbol{\delta} \text{ 满足 } \|\boldsymbol{\delta}\|_p \le \epsilon} \mathcal{L}(f(\boldsymbol{x} + \boldsymbol{\delta}), y)
   $$
   

   这里，$\mathcal{L}$ 是一个损失函数 (loss function)（例如交叉熵），它度量模型对扰动输入 $f(\boldsymbol{x} + \boldsymbol{\delta})$ 的预测与原始正确标签 $y$ 之间的差异。对于目标攻击，我们会最大化相对于原始标签的损失，或最小化相对于目标标签 $y_{target}$ 的损失。这种表述直接引出了基于梯度的攻击方法，我们将在下一章中进行考察。

### 几何视角

设想输入所处的高维空间 (high-dimensional space)。模型 $f$ 将这个空间划分为对应不同类别的区域，由决策边界分隔开。原始输入 $\boldsymbol{x}$ 位于其正确类别 $y$ 的区域内。对抗样本 $\boldsymbol{x}'$ 的找到方式是：将 $\boldsymbol{x}$ 稍微移动（在半径为 $\epsilon$ 的 $L_p$ 球内），使其穿过决策边界进入对应不同类别 $y'$ 的区域。

> 对抗样本的示意图。原始输入 $\boldsymbol{x}$（蓝色点）被正确分类。添加了一个小扰动 $\boldsymbol{\delta}$，得到 $\boldsymbol{x}'$（红色点），它恰好位于决策边界的另一侧并被误分类。扰动的大小受到限制，通常通过 $L_p$ 范数（$\|\boldsymbol{\delta}\|_p \le \epsilon$）来限制。

这种数学框架使我们能够精确定义对抗样本及其生成所受的约束。它为开发特定攻击算法（例如基于梯度或优化的算法）以及设计和评估防御方法奠定了基础，这些是后续章节的主题。理解这种表述对于分析机器学习 (machine learning)模型的安全属性非常重要。

## 参考资料

- [Intriguing properties of neural networks](https://doi.org/10.48550/arXiv.1312.6199) — Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, Rob Fergus (2013)
  Journal: arXiv preprint arXiv:1312.6199; DOI: [10.48550/arXiv.1312.6199](https://doi.org/10.48550/arXiv.1312.6199)
  这篇开创性论文介绍了对抗性样本的概念，展示了微小、难以察觉的扰动如何导致深度神经网络错误分类输入。
- [Explaining and Harnessing Adversarial Examples](https://doi.org/10.48550/arXiv.1412.6572) — Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (2014)
  Journal: arXiv preprint arXiv:1412.6572; DOI: [10.48550/arXiv.1412.6572](https://doi.org/10.48550/arXiv.1412.6572)
  本文解释了对抗性样本产生的原因，将其与深度神经网络的线性特性联系起来，并介绍了快速梯度符号法（FGSM）来生成它们。
- [Adversarial Machine Learning: A Survey of Recent Advances](https://doi.org/10.1145/3313936) — Xiaoyong Yuan, Pan He, Qiming Zhu, Xin Li, Shouhuai Xu (2019)
  Journal: ACM Computing Surveys (CSUR); Publisher: Association for Computing Machinery (ACM); Volume: 52; Pages: 1-36; DOI: [10.1145/3313936](https://doi.org/10.1145/3313936)
  这篇全面的综述概述了对抗性机器学习，包括对抗性样本的形式化定义、各种攻击方法（例如使用 $L_p$ 范数的方法）和防御策略。

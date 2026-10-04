---
course: "adversarial-machine-learning"
chapter: "model-inference-privacy-attacks"
lesson: "attribute-inference-techniques"
sourceId: 4155
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-4-model-inference-privacy-attacks/attribute-inference-techniques"
title: "属性推断技术"
description: "了解攻击者如何从模型预测中推断出关于训练数据的敏感属性。"
order: 2
plots: ["plots/4155-0.json"]
sourceHash: "54cf0a0a871c86d9c0fd72d5e85e4e151a72ca4b6eeb7cea443e667a8374cacb"
sourceCorrections: []
---

在我们对推断攻击有了认识后，我们现在关注旨在显露模型训练数据中特定（通常是敏感）特征的技术。成员推断是询问某个记录*是否*在训练集中，而属性推断则是尝试确定训练记录中特定属性的*值*，即使该属性在推断过程中并未直接用作输入。

设想一个机器学习 (machine learning)模型，它根据各种申请人特征训练来预测贷款资格。攻击者可能拥有关于申请人的部分信息（例如，年龄、位置、请求的贷款金额），并查询该模型。根据模型的预测（或其置信分数），攻击者可能会尝试推断一个最初未知的敏感属性，例如申请人的种族或婚姻状况，特别是如果这些属性是原始训练数据的一部分，并且与模型的输出存在关联。

## 属性推断的目标

正式来说，假设训练集 $D_{train}$ 中的一个数据记录 $x$。该记录通常可以分为公共或已知特征 $x_{pub}$ 以及一个或多个敏感属性 $x_{sens}$，即 $x = (x_{pub}, x_{sens})$。攻击者的目标是，在可以访问模型 $f$ 以及已知属于训练集的目标个体的公共特征 $x_{pub}$ 的情况下，确定 $x_{sens}$ 的值。

这种攻击造成了重大的隐私风险，因为它使得攻击者可以知晓那些数据被用于训练的个体的潜在隐私信息，凭借模型作为信息泄露的途径。

## 常见属性推断策略

已发展出多种方法来执行属性推断，这些方法通常取决于攻击者的已知信息以及他们对目标模型的访问类型。

### 1. 基于置信度的攻击

一种比较直观的方法依赖于模型产生的置信分数。其核心思想是，即使敏感属性在预测期间并未直接输入模型，模型在其预测中也可能表现出不同的置信水平，这取决于该敏感属性的值。

假设攻击者知道目标记 (token)录 $x$ 的 $x_{pub}$。他们想确定 $x_{sens}$ 等于值 $v_1$ 还是值 $v_2$。攻击者可以构造两个输入：$x'_1 = (x_{pub}, x_{sens}=v_1)$ 和 $x'_2 = (x_{pub}, x_{sens}=v_2)$。尽管攻击者可能无法直接用 $x_{sens}$ 进行查询，但他们可以观察模型输出 $f(x)$（其中 $x$ 是带有未知 $x_{sens}$ 的*实际*记录）或类似已知记录的相关输出。

攻击者假设，如果真实敏感属性是 $v_1$，则模型输出 $f(x)$ 可能与输出 $f(x'_1)$ 相比更“接近”（某种程度上，或许是预测置信度）而非与 $f(x'_2)$ 接近。这通常涉及观察到，当模型遇到训练数据中特定子群体的输入特征时，有时会表现出更高的置信度或犯特定类型的错误。

例如，攻击者可能会观察人脸识别模型识别一个人的置信分数。如果攻击者*假设*该人属于某个特定人口群体（根据辅助信息或统计可能性），并且模型表现出显著更高的置信度，则他们可能会推断该人很可能属于该群体。

### 2. 似然比检验 (LRT)

一种更正式的统计方法是使用似然比。攻击者比较在敏感属性的不同可能值下，观察到模型输出（例如，预测概率）的可能性。

设 $O$ 是当用与目标记 (token)录相关的信息查询模型 $f$ 时（通常使用 $x_{pub}$）观测到的输出。攻击者想在两个假设之间做出判断：$H_1: x_{sens} = v_1$ 和 $H_2: x_{sens} = v_2$。似然比为：


$$
LR = \frac{P(O | H_1)}{P(O | H_2)}
$$


如果 $LR > 1$，则在 $x_{sens} = v_1$ 的假设下，观测到的输出更有可能出现。如果 $LR < 1$，则 $x_{sens} = v_2$ 的可能性更大。计算这些条件概率 $P(O | H_i)$ 通常需要攻击者具备一些背景信息，或者构建一个模型来描述 $f$ 的输出与 $x_{sens}$ 之间的关联。

### 3. 使用攻击模型

类似于成员推断中的影子训练，攻击者可以专门为属性推断训练一个*攻击模型*。

此过程通常包含以下步骤：

1. **获取数据：** 攻击者需要数据点 $(x'_{pub}, f(x'), x'_{sens})$，其中 $x'_{pub}$ 是公共特征，$f(x')$ 是目标模型对记录 $x'$ 的输出（例如，预测向量 (vector)），$x'_{sens}$ 是该记录的已知敏感属性。这些数据可能来源于公共数据集、辅助来源，或者如果攻击者对训练分布有部分了解，则通过合成查询获得。
2. **训练攻击模型：** 训练一个分类器，即攻击模型 $g_{attack}$，它以公共特征 $x'_{pub}$ 和目标模型的输出 $f(x')$ 作为输入，并预测敏感属性 $x'_{sens}$。
   
   $$
   g_{attack}(x'_{pub}, f(x')) \approx x'_{sens}
   $$
   
3. **推断属性：** 对于一个已知 $x_{pub}$ 并观测到目标模型输出 $f(x)$ 的目标记 (token)录 $x$，攻击者使用他们训练的模型来预测敏感属性：$\hat{x}_{sens} = g_{attack}(x_{pub}, f(x))$。

这种方法的成功很大程度上取决于用于训练攻击模型的数据的质量和相关性，以及目标模型输出与敏感属性之间的关联程度。

> 攻击者使用已知公共特征和目标模型的相应输出来训练攻击模型，从而预测新目标记录的未知敏感属性。

## 影响属性推断成功的因素

属性推断攻击的有效性取决于多个因素：

- **关联程度：** $x_{sens}$ 与模型所用特征或模型输出 $f(x)$ 之间的关联程度越强，推断就越容易。如果某个属性对模型的预测影响很小，推断它就会困难得多。
- **模型过拟合 (overfitting)：** 对训练数据过拟合的模型可能会不经意间记住与敏感属性相关的特定关联，从而使其更易受攻击。
- **攻击者的已知信息：** 攻击者可获得的辅助信息的数量和质量会显著影响他们构建有效攻击模型或计算似然比的能力。
- **输出粒度：** 获取置信分数或概率向量 (vector)等详细输出，通常比仅获取最终类别标签能提供更多推断信息。

## 属性推断示例：推断位置

假设有一个模型，它根据用户帖子训练以分类情感（积极/消极）。训练数据包括帖子文本（用作输入）和如用户位置（潜在敏感）之类的元数据。攻击者可能可以访问用户的帖子 ($x_{pub}$) 和模型的情感预测 $f(x)$。如果来自某些位置的用户倾向于使用特定措辞或讨论影响情感预测的本地话题，攻击者就可能训练一个攻击模型 $g_{attack}(\text{posts}, f(\text{posts}))$ 来推断用户的地理位置 ($x_{sens}$)。



![置信度分布与敏感属性](plots/4155-0.json)



> 示例分布显示了模型置信度如何根据潜在的敏感属性而有所不同。当属性为 B 时，高置信度预测更频繁，这可能使得攻击者通过观察置信分数来推断该属性。

## 与隐私及防御的关系

属性推断本质上是一种隐私攻击。它表明，即使模型不直接请求或输出敏感数据，模型的行为仍然可能泄露这些信息。差分隐私等技术旨在通过确保模型输出不会因训练集中是否包含任何单个个体（及其属性）而发生显著变化，从而提供对抗此类泄露的形式保障。减少过拟合 (overfitting)的正则化 (regularization)方法也可以间接帮助减轻属性推断，通过阻止模型记住与敏感属性相关的虚假关联。我们将在第 5 章中更详细地讨论包括差分隐私在内的推断攻击防御措施。

“了解属性推断对评估部署机器学习 (machine learning)模型（特别是那些用敏感个人数据训练的模型）的隐私影响而言非常必要。它促使我们不仅要考虑模型准确性，还要考虑通过模型交互导致信息泄露的可能性。”

## 参考资料

- [Model Inversion Attacks That Exploit Confidence Information and Other Properties of Machine Learning Models](https://dl.acm.org/doi/10.1145/2810103.2813733) — Matthew Fredrikson, Somesh Jha, and Thomas Ristenpart (2015)
  Journal: Proceedings of the 2015 ACM SIGSAC Conference on Computer and Communications Security (CCS); Publisher: ACM; Pages: 603-615; DOI: [10.1145/2810103.2813733](https://doi.org/10.1145/2810103.2813733)
  一篇基础论文，展示了属性推断（模型反演）攻击，明确讨论了如何利用置信度分数和模型输出来推断敏感属性。
- [Deep Learning with Differential Privacy](https://doi.org/10.1145/2976749.2978318) — Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang (2016)
  Journal: Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security; Publisher: ACM; Pages: 308-318; DOI: [10.1145/2976749.2978318](https://doi.org/10.1145/2976749.2978318)
  一篇里程碑式的论文，提出了一种使用差分隐私训练深度神经网络的实用方法，直接解决了包括推断攻击在内的隐私泄露问题。
- [Exploiting the learning process of deep neural networks for attribute inference attacks](https://link.springer.com/article/10.1007/s00521-020-05597-2) — Ziqi Liu, Mingxuan Liu, Xiaojie Cui, Peng He, Jianmin Wang, Mingming Fan, Jing Li (2021)
  Journal: Neural Computing and Applications; Publisher: Springer; Volume: 33; Pages: 17329-17342; DOI: [10.1007/s00521-020-05597-2](https://doi.org/10.1007/s00521-020-05597-2)
  本文探讨了如何利用深度神经网络的内部学习过程进行属性推断，提供了对漏洞的更深入理解。

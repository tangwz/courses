---
course: "federated-learning"
chapter: "enhancing-privacy-federated-learning"
lesson: "privacy-attacks-fl"
sourceId: 3565
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-3-enhancing-privacy-federated-learning/privacy-attacks-fl"
title: "隐私攻击：推断与重建"
description: "考察联邦学习系统中的常见隐私攻击，例如成员推断和数据重建。"
order: 7
plots: []
sourceHash: "92e5f4f875cf89b602fbe9ac117d1f6e4493b2e3a1374382974bba0a67ec4fae"
sourceCorrections: []
---

尽管联邦学习避免了原始数据的集中，但客户端和服务器之间交换的模型更新本身不具备隐私性。这些更新，无论是梯度还是模型参数 (parameter)，都直接源自客户端的本地数据。一个有足够动机的攻击者，可能是中央服务器本身，甚至是恶意的参与客户端，可能会尝试利用这种信息泄露来获取关于私有数据集的敏感信息。了解这些潜在攻击，有助于我们认识到隐私增强技术（如差分隐私（DP）、安全多方计算（SMC）和同态加密（HE））的必要性。

我们将这些攻击大致分为两类：推断攻击，旨在推断数据的属性或成员身份；和重建攻击，试图恢复原始训练样本。

### 推断攻击

推断攻击试图提取关于客户端数据集的特定信息，而非重建数据点本身。

#### 成员推断

成员推断攻击的目标是确定一个特定数据点是否属于某个客户端的训练数据集。设想一个场景，一个联邦学习模型在多个医院的医疗数据上进行训练。攻击者可能想知道某个特定患者的记录（他们可能从外部获得）是否包含在A医院的训练集中。

这种攻击是如何进行的？模型对于其训练过的数据和未见过的数据通常表现出不同的行为。例如，模型可能会对训练过的数据输出更高置信度的预测，或者在训练样本上计算的损失可能低于在类似但未见过样本上的损失。攻击者通过观察模型更新或查询最终模型，可能会利用这些差异。

- **梯度分析：** 训练期间计算的梯度有时会泄露成员身份。例如，如果一个客户端发送了异常大的梯度更新，这可能对应于模型认为特别令人惊讶或难以处理的数据点，进而可能与特定训练样本的属性相关联。
- **输出分析：** 攻击者可能会使用目标数据点查询最终的联邦模型。如果模型的置信度异常高，这可能表明该数据点是训练集的一部分。

成功的成员推断攻击会侵犯个人隐私，泄露对潜在敏感数据集的参与情况。

#### 属性推断

属性推断攻击旨在推断客户端私有数据集的聚合特征或属性，即使不识别单个成员。示例包括：

- 推断某一类别的比例（例如，客户端数据集中包含猫的图像百分比）。
- 确定人口统计分布（例如，来自特定区域的贡献更新用户的近似年龄分布）。
- 识别客户端数据中存在的偏见。

这些攻击通常通过分析模型更新随时间的方向和幅度来起作用。如果客户端的更新持续以特定方式推动模型的决策边界，这可能会揭示其数据分布的潜在属性。例如，如果一个客户端持续提供更新，提高了模型在分类特定少数群体上的准确性，这可能表明该群体在该客户端的本地数据中得到了很好的代表（或者可能被过度代表）。尽管不泄露单个数据点，但属性推断仍然可以泄露关于贡献数据的群体的敏感信息。

### 重建攻击

重建攻击通常更具威力，旨在恢复客户端使用的实际训练数据样本。

#### 基于梯度的重建

这或许是研究最多的一种重建威胁，特别是在“梯度深度泄露”等研究中得到了强调。核心思想是，针对数据样本 $(x_i, y_i)$，损失函数 (loss function) $L$ 对模型参数 (parameter) $w$ 计算得到的梯度 $\nabla L(w, x_i, y_i)$ 本身包含了关于该样本的大量信息。

设想一个攻击者（通常是服务器）从客户端接收到一个梯度更新 $g_i = \nabla L(w, x_i, y_i)$。如果攻击者知道模型架构、用于计算梯度的参数 $w$ 以及损失函数，他们就可以尝试重建产生这个精确梯度的输入 $x_i$ 和标签 $y_i$。

这种攻击通常迭代进行：

1. 初始化随机的伪数据输入 ($x'$) 和标签 ($y'$)。
2. 使用模型 $w$ 在伪数据上计算梯度 $g'$：$g' = \nabla L(w, x', y')$。
3. 计算接收到的梯度 $g_i$ 和伪梯度 $g'$ 之间的差异（例如，余弦相似度或L2距离）。
4. 使用梯度下降 (gradient descent)更新伪数据 $x'$ 和 $y'$，以最小化此差异。
5. 重复步骤2-4，直到差异最小化。

如果成功，优化后的 $x'$ 和 $y'$ 将与原始的 $x_i$ 和 $y_i$ 非常相似。

> 一个迭代过程，攻击者通过优化伪数据以匹配接收到的客户端梯度，从而可能重建原始客户端数据样本。

梯度重建的成功取决于几个因素：

- **批大小：** 当在非常小的批次（理想情况下，批大小为1）上计算梯度时，攻击会显著更容易。在更大的批次上平均梯度本身会模糊单个样本的信息。
- **模型架构：** 某些层（如直接应用于输入的全连接层）可能比其他层泄露更多信息。
- **数据类型：** 图像等视觉可解释的数据通常更容易评估重建质量。
- **激活函数 (activation function)：** 某些激活函数可能会在梯度中保留更多输入信息。

#### 从模型更新中重建

直接从聚合的模型*参数*更新（如传统FedAvg中共享的）重建数据通常比从原始梯度重建要困难得多。聚合过程 $\Delta w = \frac{1}{N} \sum_{i=1}^{N} \Delta w_i$ 混合了来自多个客户端的信息，模糊了单个贡献。然而，这并非不可能，特别是如果：

- 一轮中只有少数客户端参与。
- 攻击者对潜在数据有很强的先验知识。
- 攻击者可以在多轮中观察更新。

### 攻击与防御的关联

这些推断和重建攻击的存在强调了为什么简单的联邦学习不足以提供强大的隐私保护。这正是本章讨论的技术变得不可或缺的地方：

- **差分隐私 (DP)：** 通过向梯度或模型更新（$g_i + \text{噪声}$ 或 $\Delta w + \text{噪声}$）添加校准噪声，DP掩盖了任何单个数据点的精确贡献。这直接阻碍了成员推断，并显著降低了基于梯度的重建质量，使其通常变得不切实际。隐私预算（$\epsilon, \delta$）量化 (quantization)了对此类攻击的保护程度。
- **安全多方计算 (SMC)：** SMC协议允许服务器计算更新的总和（$\sum g_i$ 或 $\sum \Delta w_i$），而无需看到单独的 $g_i$ 或 $\Delta w_i$。这直接阻止了服务器进行梯度重建或分析单个客户端更新以进行推断，前提是协议本身是安全的。
- **同态加密 (HE)：** 类似于SMC，HE允许服务器在加密更新上计算总和（$\text{Enc}(g_1) + \dots + \text{Enc}(g_N) = \text{Enc}(\sum g_i)$）。服务器永远不会访问明文更新，从而挫败了依赖于观察单个客户端贡献的攻击。

值得注意的是，即使是标准的FedAvg，与发送原始梯度相比，仅仅由于平均步骤，也对最直接的梯度重建形式提供*一些*实际保护。然而，这不像DP、SMC或HE那样提供正式保证。

理解这些攻击途径对于设计和部署联邦学习系统而言极为重要。隐私增强技术的选择取决于具体的威胁模型（例如，好奇的服务器与恶意客户端）、所需的隐私级别以及计算和通信方面的可接受开销。

## 参考资料

- [Inverting Gradients-How much information do gradients reveal?](https://proceedings.neurips.cc/paper_files/paper/2020/file/2065848f68311d95738b581e17af2b25-Paper.pdf) — Jonas Geiping, Hartmut Bauermeister, Hannah Dröge, and Michael Rath (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 33; Publisher: NeurIPS; Pages: 16110-16120; DOI: [10.5591/00021](https://doi.org/10.5591/00021)
  这项研究提出了更先进、更稳健的从梯度重建训练数据的方法，进一步强调了梯度共享中的隐私风险。
- [Advances and Open Problems in Federated Learning](https://doi.org/10.1561/2200000083) — Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, Rafael G. L. D’Oliveira, Hubert Eichner, Salim El Rouayheb, David Evans, Josh Gardner, Zachary Garrett, Adrià Gascón, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaid Harchaoui, Chaoyang He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara Javidi, Gauri Joshi, Mikhail Khodak, Jakub Konecný, Aleksandra Korolova, Farinaz Koushanfar, Sanmi Koyejo, Tancrède Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock, Ayfer Özgür, Rasmus Pagh, Hang Qi, Daniel Ramage, Ramesh Raskar, Mariana Raykova, Dawn Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tramèr, Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu, and Sen Zhao (2021)
  Journal: Foundations and Trends® in Machine Learning; Publisher: Now Publishers; Volume: 14; Pages: 1-210; DOI: [10.1561/2200000083](https://doi.org/10.1561/2200000083)
  联邦学习的全面综述，其中有大量篇幅介绍隐私攻击、防御措施（差分隐私、安全多方计算、同态加密）以及该领域的开放性问题。

---
course: "adversarial-machine-learning"
chapter: "advanced-evasion-attacks"
lesson: "decision-based-attack-techniques"
sourceId: 4130
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-2-advanced-evasion-attacks/decision-based-attack-techniques"
title: "基于判定的攻击方法"
description: "了解仅需模型最终判定标签的攻击手段（边界攻击）。"
order: 4
plots: ["plots/4130-0.json"]
sourceHash: "5f29249bd17d7c53f37390e50a7f0656714d8045c66e893e5d49f7a1d6894413"
sourceCorrections: []
---

梯度攻击和优化攻击运用模型的详细信息（梯度），分数攻击则使用输出概率，但存在一种更受限的情况：如果攻击者只从模型中获取最终预测标签（例如“猫”或“狗”）怎么办？这被称为**硬标签黑盒 (black box)设置**。基于判定的攻击正是为此类复杂情况设计的，它们适用于攻击那些不提供输出分数或分数被量化 (quantization)的系统。

其根本难题在于，在没有梯度或分数信息指引下，找到修改输入 $x$ 使其具备对抗性的方向。基于判定的方法本质上执行一种巧妙的查找，通常需要对模型进行多次查询，以推断判定边界的位置信息。

### 边界攻击

最主要的基于判定方法之一是**边界攻击**。它基于一个简单但有效的设计思路运行：

1. 从一个已具备对抗性的输入开始。这个初始对抗性样本 $x_{adv}^{(0)}$ 不需要与原始输入 $x$ 相近；它甚至可以是恰好被误分类为目标类别的随机噪声，或是通过其他方式（如迁移攻击）生成的对抗性样本。
2. 迭代地调整对抗性样本 $x_{adv}^{(t)}$，使其更接近原始输入 $x$，同时确保它保持对抗性。

主要思想是在原始类别和目标对抗类别之间的判定边界上“移动”。在每一步 $t$，算法尝试通过从 $x_{adv}^{(t)}$ 迈出小步来寻找新的点 $x_{adv}^{(t+1)}$。这一步经过慎重选择：

- **方向**：这一步通常沿随机方向进行，但受限于大致与边界相切。目标是*沿着*边界向 $x$ 移动，而不是直接朝向它（这可能使其回到原始类别）或直接远离它（这会增加扰动）。
- **大小**：步长会被调整。如果提议的步进使样本被正确分类，则减小步长以保持更接近边界。
- **判定标准**：如果新候选样本 $x'_{adv}$ 满足两个条件，则接受该步：
  - 它保持对抗性：$f(x'_{adv}) \neq f(x)$。
  - 它减小了到原始输入的距离：$||x'_{adv} - x||_p < ||x_{adv}^{(t)} - x||_p$，其中 $p$ 是选定的范数（通常 $p=2$）。

这种迭代过程逐渐减小扰动大小 $||x_{adv}^{(t)} - x||_p$，找出与原始输入逐步接近的对抗性样本。



![边界攻击路径](plots/4130-0.json)



> 边界攻击的二维示意图。从一个初始对抗点（红色）开始，攻击沿着判定边界（虚线）迭代地向原始输入（蓝色）移动，在保持对抗性分类的同时减小扰动距离。

### 查询复杂度及其他方法

基于判定攻击的一个重要性质是它们的**查询复杂度**。由于它们缺少直接的梯度或分数信息，它们必须多次查询模型（通常是数千甚至数百万次）才能有所进展。每次查询都涉及向模型发送候选输入并观察输出标签。这使得它们可能很慢且成本高昂，特别是当模型存在速率限制或查询费用时。

尽管边界攻击是此类攻击中的基本方法，但也存在其他基于判定的方法，例如：

- **HopSkipJumpAttack**：建立在边界查找思路之上，与原始边界攻击相比，通常能实现更好的查询效率或找到更小的扰动。
- **Sign-OPT**：运用优化技巧，这些技巧由仅使用判定结果获得的梯度方向估算值引导。

这些方法通常试图改进查找策略，以减少所需的查询次数。

### 优点与缺点

**优点：**

- **所需信息极少**：即使只有最终预测标签可用，它们也能发挥作用，使其适用于高度受限的黑盒 (black box)系统。
- **适用范围广**：它们不依赖于模型的可微分性或特定架构。

**缺点：**

"\* **高查询成本**：通常需要非常大量的模型查询，这使得它们在许多有查询限制或成本的场景中不切实际。"

- **可能产生更大扰动**：与PGD或C&W等白盒攻击相比，基于判定的攻击有时可能会找到具有更大$L_p$失真的对抗性样本，尽管它们的目标是尽量减少这一点。
- **收敛速度较慢**：由于其迭代特性和高查询次数，找到高质量的对抗性样本可能需要相当长的时间。

基于判定的攻击代表了规避攻击中攻击者知识层面的极端情况。理解这些方法对于评估模型内部和输出受到严密保护的场景中的安全性很重要。它们表明，即使反馈极少，漏洞仍然可以被加以利用，尽管这通常会带来更高的计算成本。

## 参考资料

- [Decision-Based Adversarial Attacks: Reliable Attacks on Black-Box Machine Learning Models](https://arxiv.org/abs/1712.04248) — Wieland Brendel, Jonas Rauber, Matthias Bethge (2018)
  Journal: Sixth International Conference on Learning Representations (ICLR 2018); DOI: [10.48550/arXiv.1712.04248](https://doi.org/10.48550/arXiv.1712.04248)
  这篇开创性论文介绍了边界攻击，一种用于硬标签黑盒对抗性示例的基础性基于决策的方法。
- [HopSkipJumpAttack: A Query-Efficient Decision-Based Attack](https://arxiv.org/pdf/1904.02144) — Pinyu Chen, Huan Zhang, Yuan Sinha, Ruoxi Jia, Gauthier Gidel, and Cho-Jui Hsieh (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Curran Associates, Inc.; Volume: 32; DOI: [10.48550/arXiv.1904.02144](https://doi.org/10.48550/arXiv.1904.02144)
  介绍了HopSkipJumpAttack，这是一种改进的、查询高效的基于决策的攻击，它建立在边界探索技术之上。
- [Sign-OPT: A Query-Efficient Hard-label Adversarial Attack](https://arxiv.org/abs/1909.04018) — Minhao Cheng, Simranjit Singh, Shiyu Chang, Jonghyun Choi, Pengtao Xie, Anna Choromanska, and Ahmed Elgammal (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1909.04018](https://doi.org/10.48550/arXiv.1909.04018)
  提出了Sign-OPT，另一种查询高效的硬标签对抗性攻击，它通过估计梯度符号来指导其搜索。
- [A Survey on Black-Box Adversarial Attacks and Defenses](https://doi.org/10.1109/TPAMI.2020.3039148) — Xiaofei Wang, Pengfei Li, Jianxin Li, Jun Liu, Bo Li, and Yu-Gang Jiang (2021)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 43; Pages: 3711-3736; DOI: [10.1109/TPAMI.2020.3039148](https://doi.org/10.1109/TPAMI.2020.3039148)
  全面概述了黑盒对抗性攻击和防御，为本节讨论的基于决策的方法提供了背景。

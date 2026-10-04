---
course: "adversarial-machine-learning"
chapter: "foundations-adversarial-ml-security"
lesson: "review-ml-security-vulnerabilities"
sourceId: 4111
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-1-foundations-adversarial-ml-security/review-ml-security-vulnerabilities"
title: "机器学习安全弱点概述"
description: "回顾机器学习流程和模型部署中的常见安全弱点。"
order: 1
plots: []
sourceHash: "4a99b6b4a1c2ab647b8f269b22cd74d97ea76a2c4c8446359fb0812cd5f2c0d5"
sourceCorrections: []
---

机器学习 (machine learning)系统虽然能够实现显著成就，但本身并非免疫于安全故障。它们在复杂的生态系统中运行，并拥有独有特点，会引入特定的脆弱点。识别这些不足是首要步骤，在此之后我们才能有效分析潜在攻击或制定防御措施。与传统软件安全问题（例如缓冲区溢出或SQL注入）不同，机器学习安全通常侧重于数据、模型学习过程及其预测方式。

这些弱点可以出现在机器学习生命周期的不同阶段。我们可以大致将它们分为与数据依赖性、模型本身的固有属性以及用于部署的基础设施相关的类别。

### 数据相关弱点

“垃圾进，垃圾出”这句格言在机器学习 (machine learning)中有了新的安全意义。模型根本上由训练数据决定，使得数据本身成为安全问题的主要途径。

- **数据投毒易受攻击性：** 收集和准备训练数据的过程通常涉及从多个来源提取数据的复杂流程。这为攻击者注入恶意数据创造了机会。此类*数据投毒*可以旨在降低模型的整体性能（一种*可用性*攻击），或者更隐蔽地，针对特定输入导致特定、有目标的错误分类（一种*完整性*攻击）。例如，攻击者可能会引入细微改动的图像，导致人脸识别系统错误识别特定个人，或插入数据点以创建后门，该后门由正常操作中不可见的特定模式触发。第三章将介绍数据投毒技术和防御措施。
  "\* **训练数据偏差与偏见：** 在未准确代表数据分布或包含社会偏见的数据上训练的模型，表现出一种脆弱性。虽然这并非总是直接攻击的结果，但这种偏差可能导致性能不佳、对某些子群体产生不公平结果以及普遍缺乏可靠性。了解这些偏见的攻击者可能会利用它们构造更容易被错误分类的输入，或有意引发不公平预测。此外，输入空间某些区域的数据稀疏性可以使模型更容易受到针对性攻击。"

### 模型固有弱点

机器学习 (machine learning)模型的学习参数 (parameter)和决策机制，特别是深度神经网络 (neural network)等复杂模型，会产生独有的脆弱性。

- **对输入扰动的敏感性（规避攻击）：** 广为人知的弱点是许多模型易受*对抗性样本*影响。这些输入（$x$）通过添加精心构造、通常人眼难以察觉的扰动（$\delta$）进行了细微修改，从而产生了新输入 $x' = x + \delta$。尽管 $x'$ 在人类看来与 $x$ 几乎相同，但它会导致模型产生错误输出（例如，错误分类）。扰动通常受到限制，常使用 $L_p$ 范数，使得 $\||\delta||_p \le \epsilon$ 对于某个小的 $\epsilon$。这种现象的产生是因为模型学习到的高维决策边界可能出乎意料地脆弱，并且与合法数据点距离近。规避攻击发生在推理 (inference)时，是对抗性机器学习研究的一个主要关注点。第二章将介绍高级规避技术。
- **通过模型输出进行信息泄露：** 已部署的模型可能通过它们提供的预测和置信度分数，无意中泄露有关其训练数据或内部结构的信息。
  - **模型提取（窃取）：** 通过反复查询模型（即使通过黑盒 (black box)API），并使用选择的输入和观察输出，攻击者可以训练一个功能副本，或*替代模型*。这会损害知识产权，并促成其他攻击，例如构造可迁移到原始模型的规避攻击。
  - **成员推断：** 攻击者可能通过分析模型对特定数据记录的响应，判断该记录是否包含在模型的训练集中。这构成直接的隐私风险，特别是对于在医疗或金融数据等敏感信息上训练的模型。
  - **属性推断：** 攻击者可能会推断训练数据主体的敏感属性（例如，人口统计信息、个人偏好），即使模型并未明确设计用于预测这些属性。
  - **模型反演：** 这些攻击旨在重建训练数据中有代表性的样本或原型，有时甚至恢复特定的训练示例，仅通过访问模型。第四章将详细介绍这些推断和隐私攻击。

### 部署和基础设施弱点

机器学习 (machine learning)模型通常集成到更大的软件系统中，并通过基础设施部署，这引入了其自身的一系列安全考量。

- **不安全的服务端点：** 以服务形式暴露的模型（例如，通过REST API）容易受到传统网络安全威胁，如认证不足、授权缺陷或拒绝服务（DoS）攻击。针对机器学习而言，如果预测是资源密集型的，过度查询可用于模型提取或经济型DoS攻击。限速、强认证和输入验证等标准安全实践是必要的初步措施。
- **受损的机器学习流程：** 支持机器学习模型的整个工作流程，包括数据摄取、预处理脚本、特征提取代码、训练环境、模型序列化和部署自动化，构成一个攻击面。任何阶段的泄露都可能允许攻击者污染上游数据、篡改模型的权重 (weight)、在再训练期间引入后门，或窃取模型或敏感数据。这强调了保护整个机器学习运维环境（MLOps）的重要性。
- **监控和日志记录不足：** 如果没有适当的监控，输入数据分布的变化、预测行为的突然改变，或预示着攻击的异常查询模式可能不被察觉。全面日志记录对于检测攻击和在发生泄露时执行取证分析也必不可少。

### 弱点可视化

下图演示了这些弱点通常出现在典型机器学习 (machine learning)系统生命周期中的位置：

> 机器学习工作流程概述，标示出不同类别的安全弱点（彩色平行四边形）的潜在进入点。

找出这些潜在弱点为理解机器学习系统面临的威胁提供了必要的基础。它强调了为什么安全不能是事后考虑，而必须贯穿整个生命周期。这些弱点直接促使了对系统性威胁建模的需求，下一节将介绍这一点，并为分析本课程其余部分涵盖的特定攻击技术和防御策略做好准备。

## 参考资料

- [Adversarial Machine Learning: A Survey of the State-of-the-Art](https://doi.org/10.1016/j.patcog.2018.05.013) — Battista Biggio and Fabio Roli (2018)
  Journal: Pattern Recognition; Publisher: Elsevier; Volume: 84; Pages: 317-332; DOI: [10.1016/j.patcog.2018.05.013](https://doi.org/10.1016/j.patcog.2018.05.013)
  该综述文章概述了针对机器学习的攻击，包括数据投毒、对抗性样本和隐私推断攻击。
- [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199) — Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus (2013)
  Journal: arXiv preprint arXiv:1312.6199; DOI: [10.48550/arXiv.1312.6199](https://doi.org/10.48550/arXiv.1312.6199)
  这篇开创性论文提出了对抗性样本的概念，表明微小、难以察觉的扰动可以导致深度神经网络错误分类。
- [Poisoning Attacks Against Support Vector Machines](https://arxiv.org/pdf/1206.6389) — Battista Biggio, Blaine Nelson, and Pavel Laskov (2012)
  Journal: Proceedings of the 29th International Conference on Machine Learning (ICML '12); Publisher: JMLR, Inc.; Pages: 1467-1474
  这是一项关于数据投毒攻击的基础性工作，详细描述了如何通过操纵训练数据来降低模型性能，特别是针对支持向量机。
- [Membership Inference Attacks Against Machine Learning Models](https://ieeexplore.ieee.org/document/7958568) — Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov (2017)
  Journal: 2017 IEEE Symposium on Security and Privacy (SP); Publisher: IEEE; Pages: 3-18; DOI: [10.1109/SP.2017.41](https://doi.org/10.1109/SP.2017.41)
  该论文首次提出了实际的成员推断攻击，展示了攻击者如何确定特定数据记录是否是模型训练集的一部分。

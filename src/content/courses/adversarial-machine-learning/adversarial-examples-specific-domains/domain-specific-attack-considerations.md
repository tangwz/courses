---
course: "adversarial-machine-learning"
chapter: "adversarial-examples-specific-domains"
lesson: "domain-specific-attack-considerations"
sourceId: 4190
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-7-adversarial-examples-specific-domains/domain-specific-attack-considerations"
title: "特定应用场景的攻击考量"
description: "讨论在为特定应用场景设计攻击时的独特方面和约束。"
order: 5
plots: ["plots/4190-0.json"]
sourceHash: "5a091794bb0a0acd2791d97f4e940a30ea2a52ab7a7f54ac4fe142903c5e732a"
sourceCorrections: []
---

尽管对抗攻击和防御的基本原理广泛适用，但它们的实际操作和有效性很大程度上受机器学习 (machine learning)模型运行的特定应用场景影响。数据特性、模型任务以及有意义扰动的定义在计算机视觉、自然语言处理和强化学习 (reinforcement learning)等不同范畴之间存在很大差异。了解这些特定场景的细节是制定相关攻击策略和构建强固防御的必要条件。

### 计算机视觉（CV）考量

在计算机视觉中，模型通常处理连续、高维度的数据，例如像素值。

- **扰动约束：** 计算机视觉中扰动的最常见数学约束是$L_p$范数，特别是$L_\infty$和$L_2$范数。目标通常是在给定范数下找到导致错误分类的最小扰动。然而，最终约束通常是*感知相似性*。微小的$L_\infty$扰动在视觉上仍然可能被察觉，尤其是在平滑的图像区域，而较大的扰动可能隐藏在纹理区域。结构相似性指数（SSIM）等度量标准有时比简单的$L_p$范数更能捕捉人类感知。构建难以察觉的攻击通常需要仔细考虑人类视觉系统。
- **攻击目标：** 攻击目标可能不是简单地导致整个图像错误分类，而是针对物体检测器（导致物体被漏检或检测到虚假物体）、语义分割模型（破坏像素级标签）或人脸识别系统。“成功攻击”的定义很大程度上取决于具体的计算机视觉任务。
  "\* **物理世界攻击：** 创建打印后并经相机捕获仍有效的对抗样本，会带来很大的挑战。攻击必须能承受光照、角度、距离、相机传感器噪声以及打印/显示过程本身的变化。这通常需要专门的优化技术，例如变换期望（EOT），以创建能承受变化的攻击。"

### 自然语言处理（NLP）考量

自然语言处理模型处理离散数据（字符、单词、标记 (token)），这从根本上改变了攻击的构建方式。

- **输入空间：** 文本的离散性意味着为连续输入开发的基于梯度的方法不能直接应用于修改标记。扰动涉及改变离散单元：字符、单词或句子。
- **扰动约束：** 主要约束是*语义保持*。理想情况下，对抗性文本样本应在欺骗模型的同时保持原始含义和语法结构。仅仅根据嵌入 (embedding)相似性交换单词可能导致无意义或语法不正确的文本。技术包括：
  - **同义词替换：** 用同义词替换单词，使含义改变最小化。
  - **意译：** 重写句子或短语。
  - **字符级编辑：** 引入拼写错误，添加/删除空格（通常对基于字符的模型有效）。
  - **单词插入/删除：** 仔细添加或删除单词以改变模型预测。
- **攻击目标：** 常见目标包括反转情感分析预测、改变主题分类、修改机器翻译输出、导致聊天机器人生成不良内容或绕过内容过滤器。
- **扰动度量：** 扰动大小通常不是用$L_p$范数衡量，而是通过编辑距离（字符或单词级别）或从语言模型或嵌入比较中得出的语义相似性分数来衡量。改变的单词数量是一个简单、易于理解的度量标准。



![不同应用场景下扰动度量标准的适用性](plots/4190-0.json)



> 常见扰动度量标准的适用性在计算机视觉和自然语言处理任务之间差异很大。

### 强化学习 (reinforcement learning)（RL）考量

强化学习中的对抗攻击目标是智能体的决策过程，该过程在一个环境循环中运行。

- **攻击面：** 攻击可以针对智能体的*观察*（如果输入是视觉的，则与计算机视觉类似）、*奖励信号*（误导智能体关于任务成功的信息），或者在攻击者能够控制时针对*环境动态*本身（在标准威胁模型中较不常见）。
- **攻击目标：** 目标通常是降低智能体性能，导致其学习到次优策略、进入不良状态（例如，不安全条件），或未能实现其目标。攻击可能是有目标的（强制采取特定错误动作）或无目标的（仅仅降低总体奖励）。
- **序贯特性：** 攻击可能需要在多个时间步上持续施加，以显著影响智能体的行为或学习路径。单一扰动观察的效果可能很小，但累积扰动会使策略偏离正轨。
- **挑战：** 由于交互循环，评估很复杂。攻击的长期后果难以预测。防御通常侧重于策略优化或检测观察或奖励中的异常。

### 一般应用背景

数据模态和具体的应用背景决定了对抗性考量：

- **模型架构：** 不同架构（例如，CNN、RNN、Transformer）对对抗性扰动表现出不同的敏感度。理解架构属性有助于指导攻击设计。
- **数据预处理：** 归一化 (normalization)、标记 (token)化、特征缩放以及其他预处理步骤会影响攻击的有效性以及扰动需要如何构建。攻击可能需要在预处理后的空间中进行，或者被设计成能通过预处理流程。
- **影响：** 鲁棒性的可接受程度与应用场景紧密相关。导致电影推荐系统错误分类的攻击，其风险远低于影响自动驾驶车辆感知系统或医疗诊断工具的攻击。安全要求必须与潜在危害相符。

设计有效的对抗攻击或防御，要求仔细考量数据、任务、模型和应用场景的独特特性。通用方法不足以应对不同领域的挑战。例如，对图像分类器进行的成功$L_\infty$攻击，对于如何构建文本摘要模型上的意义保持攻击，几乎没有直接的启发。因此，针对特定背景调整策略，是理解和减轻已部署机器学习 (machine learning)系统中对抗风险的根本。

## 参考资料

- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) — Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (2014)
  Journal: arXiv preprint arXiv:1412.6572; Publisher: arXiv; DOI: [10.48550/arXiv.1412.6572](https://doi.org/10.48550/arXiv.1412.6572)
  介绍了生成对抗样本的快速梯度符号法（FGSM）。
- [Adversarial Examples in Natural Language Processing: A Survey](https://www.jair.org/index.php/jair/article/view/12181) — Yi-Ting Zhang, Bing-Yi Lu, Chao-Kai Xu, Jian-Xin Li (2020)
  Journal: Journal of Artificial Intelligence Research; Publisher: AI Access Foundation; Volume: 69; Pages: 1199-1234; DOI: [10.1613/jair.12181](https://doi.org/10.1613/jair.12181)
  提供了自然语言处理中对抗样本方法与挑战的概述。
- [Adversarial Attacks on Neural Network Policies](https://doi.org/10.48550/arXiv.1702.02284) — Sandy Huang, Nicolas Papernot, Ian Goodfellow, Yan Duan, Pieter Abbeel (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1702.02284](https://doi.org/10.48550/arXiv.1702.02284)
  研究了针对深度强化学习智能体的对抗攻击方法。

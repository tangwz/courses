---
course: "adversarial-machine-learning"
chapter: "data-poisoning-backdoor-attacks"
lesson: "targeted-data-poisoning"
sourceId: 4141
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-3-data-poisoning-backdoor-attacks/targeted-data-poisoning"
title: "定向数据投毒技术"
description: "了解制作投毒数据以导致训练模型出现特定错误行为的方法。"
order: 2
plots: []
sourceHash: "e03cf74e59e9759bdef8bc207204620820dff91b11fe7ae1d5950399ac19fe89"
sourceCorrections: []
---

可用性攻击就像旨在降低模型整体性能的粗糙工具，而定向数据投毒则是一种精准工具。其目的不仅仅是干扰模型，而是根据攻击者的计划，使其能够可靠地错误分类特定、预设的输入，或者具有某些共同特征的输入。这允许进行更不显眼且可能损害更大的操作，模型在大部分时间看起来运行正常，但在攻击者选定的目标上会可预测地失败。

实现这一点，需要精心制作恶意数据点（毒药），这些数据点在加入训练集后，会不显眼地扭曲学习到的决策边界，正是攻击者需要其失败的位置。通常，这必须使用相对于干净训练数据量而言少量的毒药样本来完成，以保持不显眼并避免通过数据清理方法轻易被发现。

### 基于优化的投毒

许多高级定向投毒技术将毒药数据的创建视为一个优化问题。主要思想是生成毒药样本 $(x_{poison}, y_{poison})$，当这些样本加入到干净训练集 $D_{clean}$ 后，训练出的模型 $f_{	heta^*}$ 会在特定目标输入 $x_{target}$ 上展现出期望的恶意行为。例如，攻击者可能希望将 $x_{target}$（其真实标签为 $y_{true}$）错误分类为 $y_{poison}$。

攻击者通过最小化一个编码其目标的客观函数，来寻找最优的毒药数据 $x_{poison}$。这通常转化为一个复杂的双层优化问题：

1. **内部优化（模型训练）：** 这一层模拟使用组合数据集 $D_{clean} \cup \{ (x_{poison}, y_{poison}) \}$ 的标准模型训练过程。结果是投毒模型的参数 (parameter)集 $\theta^*$。
   
   $$
   \theta^* = \arg \min_\theta \sum_{(x,y) \in D_{clean} \cup \{ (x_{poison}, y_{poison}) \}} \mathcal{L}(f_\theta(x), y)
   $$
   
2. **外部优化（毒药制作）：** 这一层调整毒药数据 $x_{poison}$（有时也调整其标签 $y_{poison}$），以使用从内层循环获得的参数 $\theta^*$ 来最大化攻击者的目标。一个常见的目标是最大化目标样本 $x_{target}$ 相对于期望的恶意标签 $y_{poison}$ 的损失，同时可能限制毒药点看起来有些自然，或确保模型在干净数据上的准确率不会下降太多（为了隐蔽性）。
   
   $$
   \max_{x_{poison}} \mathcal{L}(f_{\theta^*} (x_{target}), y_{poison}) \quad \text{受限于 } x_{poison} 	ext{ 的约束}
   $$
   

直接解决这种双层优化通常是不可行的。实际方法通常采用近似处理。一种常见方法是在毒药数据上使用梯度上升。攻击者计算与目标样本错误分类相关的损失梯度，相对于毒药数据的参数 ($x_{poison}$)，并迭代更新毒药数据以接近恶意目标。这需要估算 $x_{poison}$ 的变化如何影响最终训练的参数 $\theta^*$，通常涉及近似训练动态或通过优化步骤（例如SGD更新）进行微分等技术。

### 特征碰撞攻击

一种更直观的定向投毒机制涉及制造“特征碰撞”。在这种情况下，攻击者不一定解决复杂的优化问题，而是侧重于操纵模型的内部特征表示。

该策略是制作一个毒药样本 $x_{poison}$（分配恶意标签 $y_{poison}$），使其在模型的一个或多个中间层中的表示被设计得与目标样本 $x_{target}$ 的特征表示非常接近。

在毒药数据集上训练期间，模型学习将特征空间中包含 $x_{target}$ 和 $x_{poison}$ 表示的区域与毒药标签 $y_{poison}$ 关联起来。因此，当在推理 (inference)时将干净的 $x_{target}$ 提供给训练好的模型时，其特征会激活特征空间中的“投毒”区域，导致模型输出攻击者期望的标签 $y_{poison}$。

> 特征碰撞的示例。左侧：在原始模型的特征空间中，目标属于A类。右侧：一个被标记 (token)为B类的毒药点（菱形）被精心制作，使其特征接近目标。模型学习到一个新的边界（红色虚线），将该区域与B类关联，导致目标被错误分类。

### 简单方法与挑战

存在一些不那么复杂的方法，例如简单的**标签翻转**。这包括找到与目标 $x_{target}$ 相似的现有训练样本，并简单地将它们的标签更改为期望的 $y_{poison}$。尽管简单，但与基于优化的方法相比，标签翻转通常需要改变更多的样本才能达到可靠的定向效果。这种增加的修改量会使攻击更不隐蔽，并且更容易受到数据过滤或异常值去除防御的影响。

制作有效的定向毒药会带来一些重要的障碍：

- **隐蔽性：** 毒药样本理想情况下应类似于良性数据点，以逃避检测。基于优化的方法有时可以在客观函数中包含项，以强制与干净数据分布的相似性。
- **有效性：** 少量的毒药点理想情况下应能可靠地引发期望的错误分类。所需的投毒比例（毒药数据的百分比）是一个重要因素。
- **训练动态：** 毒药数据的精确影响可能对训练过程的具体细节很敏感，包括所使用的优化器（例如Adam、SGD）、学习率调度、批量大小以及随机梯度下降 (gradient descent)固有的随机性。为一种设置优化的攻击可能在另一种设置中失败。

这些定向投毒技术主要旨在通过操纵训练数据分布，破坏模型在特定、通常是干净输入上的行为。这种侧重使它们与后门攻击有所区别，我们接下来会分析后门攻击。后门攻击也实现定向错误分类，但通常依赖于嵌入 (embedding)一个独特的人工触发模式，以激活隐藏的恶意功能。

## 参考资料

- [Manipulating the Learning Process of Deep Neural Networks Using Data Poisoning Attacks](https://ieeexplore.ieee.org/document/8418653) — Małgorzata Jagielski, Artem Oprea, Battista Biggio, Chang Liu, Alina Oprea, Cristina Nita-Rotaru, and Bo Li (2018)
  Journal: 2018 IEEE Symposium on Security and Privacy (SP); Publisher: IEEE; Pages: 19-35; DOI: [10.1109/SP.2018.00030](https://doi.org/10.1109/SP.2018.00030)
  一篇关于针对深度神经网络的投毒攻击的论文，详细介绍了基于优化的技术。

---
course: "cnns-for-computer-vision"
chapter: "advanced-transfer-learning-domain-adaptation"
lesson: "few-shot-learning-cnns"
sourceId: 2607
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-6-advanced-transfer-learning-domain-adaptation/few-shot-learning-cnns"
title: "基于CNN的小样本学习"
description: "介绍使CNN能从极少量标注数据中有效学习的技术。"
order: 5
plots: []
sourceHash: "cc3712755482b3fb8256c546dcb73d9b2657383081d4632eab106dfadbbc6df6"
sourceCorrections: []
---

常见的迁移学习 (transfer learning)，通常涉及微调 (fine-tuning)预训练 (pre-training)模型，一般需要目标任务有足够数量的标注数据。然而，在许多实际情况中，由于成本、时间或某些类别固有的稀有性，获取大量标注数据集是不可行的。想象一下，您需要仅凭几张照片识别一种新发现的鸟类，或者调整医疗影像系统，以便用少量患者样本识别罕见病症。在这种情况下，小样本学习 (FSL) 就变得非常重要了。

FSL 解决的问题是如何在仅有极少量标注样本（通常每个类别只有一到五个）的情况下学习识别新类别。正式来说，这通常被构建为一个 $N$ 类 $K$ 样本分类问题：模型会获得 $N$ 个新类别中每个类别的 $K$ 个标注样本（“支持集”），这些类别在初始训练时未曾出现；它的目标是正确分类属于这 $N$ 个类别之一的新未标注样本（“查询集”）。当 $K=1$ 时，它被称为单样本学习。

直接使用每个类别仅 $K$ 个样本来微调 ResNet 或 EfficientNet 等大型 CNN，通常会导致严重的过拟合 (overfitting)。模型的高容量使其能够简单地记住少量支持样本，而未学习到新类别的通用特征。因此，需要专门的方法。FSL 方法通常分为几类，它们通常基于在大型相关数据集（如 ImageNet）上预训练学到的强大特征表示。

### 度量学习方法

FSL 度量学习背后的主要思想是学习一个嵌入 (embedding)函数，该函数将图像映射到特征空间中，使同一类别的图像彼此靠近，不同类别的图像彼此远离。然后可以通过比较查询图像的嵌入与支持样本的嵌入来进行分类。

#### 原型网络

原型网络提供了一种直观且高效的度量学习方法。在训练和测试期间，它们以“情景”的形式运行，这些情景旨在模拟小样本场景。

1. **嵌入：** CNN 主干网络（通常是预训练 (pre-training)的）处理支持图像和查询图像，以提取特征嵌入。令 $f_\phi$ 为由 $\phi$ 参数 (parameter)化的嵌入函数。
2. **原型计算：** 对于支持集 $S = \{(x_1, y_1), ..., (x_{NK}, y_{NK})\}$ 中的 $N$ 个类别，其中 $x_i$ 是图像，$y_i$ 是其类别标签，计算一个单一的原型向量 (vector) $c_n$。这通常是属于类别 $n$ 的 $K$ 个支持样本嵌入的平均值：
   
   $$
   c_n = \frac{1}{K} \sum_{(x_i, y_i) \in S_n} f_\phi(x_i)
   $$
   
   其中 $S_n$ 是类别 $n$ 的支持样本子集。
3. **分类：** 查询图像 $x_q$ 根据其在嵌入空间中到类别原型的距离进行分类。通过对负距离（或相似度）应用 softmax 来计算类别上的分布。使用平方欧几里得距离 $d$， $x_q$ 属于类别 $n$ 的概率为：
   
   $$
   p(y=n | x_q) = \frac{\exp(-d(f_\phi(x_q), c_n))}{\sum_{n'=1}^N \exp(-d(f_\phi(x_q), c_{n'}))}
   $$
   
4. **训练：** 通过最小化在多个采样情景中查询样本真实类别的负对数似然来学习网络参数 $\phi$。

情景式训练使嵌入函数 $f_\phi$ 能够生成对新类别具有良好泛化能力的表示，因为它必须在每个情景之前，为那些未曾专门训练过的类别持续形成簇并分离原型。

#### 孪生网络

另一种基础的度量学习技术是孪生网络。这些网络使用相同的 CNN（共享权重 (weight) $\phi$）处理图像对。网络输出两幅图像的嵌入，并使用距离函数（如欧几里得距离或余弦相似度）比较这些嵌入。网络通过最小化同类图像对之间的距离，并最大化不同类图像对之间的距离进行训练，通常使用对比损失或三元组损失函数 (loss function)。对于小样本分类，查询图像的嵌入可以与所有支持图像的嵌入进行比较，分类通常基于最近的支持样本的类别。

### 基于优化的方法（元学习）

基于优化的方法不学习固定的嵌入 (embedding)空间，而是专注于学习一种算法或模型初始化方法，这种方法仅需少量样本即可快速适应新任务。这通常被称为“学会学习”或元学习。

#### 模型无关元学习 (MAML)

MAML 是一种流行且多功能的元学习算法。它的目标是找到一组初始模型参数 (parameter) $\theta$，使得将这些参数适应新任务只需使用该任务的小型支持集进行几次梯度更新，就能在该任务的查询集上取得良好表现。

过程涉及两个优化循环：

1. **内循环（任务适应）：** 对于一个特定的采样任务（FSL 情境中的一个情景），模型参数 $\theta$ 会使用支持集 $S_{task}$ 临时更新一次或几次梯度步长。这会得到任务特定的参数 $\theta'_{task}$。对于学习率为 $\alpha$ 的一步更新：
   
   $$
   \theta'_{task} = \theta - \alpha \nabla_\theta \mathcal{L}_{S_{task}}(\theta)
   $$
   
   其中 $\mathcal{L}_{S_{task}}$ 是在当前任务的支持集上计算的损失。
2. **外循环（元优化）：** 初始参数 $\theta$ 根据适应后的参数 $\theta'_{task}$ 在同一任务的*查询*集 $Q_{task}$ 上的表现进行更新。损失使用适应后的参数计算，但梯度是相对于原始参数 $\theta$ 计算的。
   
   $$
   \theta \leftarrow \theta - \beta \nabla_\theta \sum_{task \sim p(task)} \mathcal{L}_{Q_{task}}(\theta'_{task})
   $$
   
   其中 $\beta$ 是元学习率。

这个外循环更新涉及对内循环的梯度更新进行微分，通常需要二阶导数（尽管一阶近似很常见）。MAML 旨在找到一个在参数空间中具有战略位置的初始化 $\theta$，使其对从任务分布 $p(task)$ 中抽取的各种小样本任务高度敏感且易于适应。

### 情景式训练说明

许多 FSL 方法，特别是原型网络等度量学习方法，都非常依赖情景式训练。这种策略在训练阶段直接模拟小样本问题。

> 情景式训练过程从一个更大的基础数据集中抽取小的 $N$ 类 $K$ 样本任务（情景）。模型的训练目标是仅根据每个情景中提供的支持样本，使其在该情景的查询样本上表现良好。

在每次训练迭代中：

1. 从基础训练数据集中随机选择 $N$ 个类别。
2. 为每个 $N$ 个类别随机采样 $K$ 个支持样本。
3. 从这 $N$ 个类别中剩余的样本里随机采样 $Q$ 个查询样本。
4. 模型处理支持集（例如，计算原型）。
5. 模型对查询集进行预测。
6. 根据查询集预测结果计算损失。
7. 通过反向传播 (backpropagation)更新模型参数 (parameter)。

通过反复训练这些多样化、随机生成的小样本任务，模型能学习到有效的表示或适应策略，即使是对于在元测试期间遇到的全新类别也能适用，前提是这些新类别来自相似的分布。

### 与迁移学习 (transfer learning)的关联

小样本学习与迁移学习有内在联系。大多数成功的 FSL 方法并非仅使用情景式程序从头开始训练整个 CNN。相反，它们通常使用在大型数据集（如 ImageNet）上预训练 (pre-training)的 CNN 主干网络作为强大的特征提取器。然后，FSL 技术（度量学习、MAML 等）主要应用于最终层，或在该预训练主干网络生成的特征空间内运行。预训练提供了通用视觉特征的坚实根基，FSL 方法随后专门调整或使用这些特征，以完成基于少量样本区分新类别的任务。

### 实际考量

- **基础数据集：** 用于预训练 (pre-training)或情景采样的基础数据集的选择和大小会显著影响性能。通常，与目标小样本任务相关的基础数据集更有益。
- **主干网络架构：** CNN 主干网络架构（例如，ResNet 变体）的容量和适用性很重要。
- **评估：** FSL 模型是在从训练期间完全未见的类别中抽取出的独立*元测试*情景上进行评估的。性能通常在许多此类测试情景上取平均值。
- **方法选择：** 原型网络等度量学习方法通常比 MAML 等基于优化的方法更易于实现，计算需求也更低，但最佳选择可能取决于具体任务。

小样本学习提供了一套强大的工具，用于在数据稀缺的环境中调整 CNN，将计算机视觉的应用范围扩展到无法获取大量标注数据集的场景。它代表了一种复杂的模型适应形式，在显著数据限制下提升学习能力。

## 参考资料

- [Few-Shot Learning: A Survey](https://doi.org/10.1145/3386252) — Yaqing Wang, Quanming Yao, James Tin-Yau Kwok, Lionel Ming-shuan Ni (2020)
  Journal: ACM Computing Surveys (CSUR); Publisher: ACM; Volume: 53; Pages: 1-34; DOI: [10.1145/3386252](https://doi.org/10.1145/3386252)
  提供少样本学习的全面概述，包括问题定义、常见范式以及度量学习和元学习等各种算法方法。
- [Prototypical Networks for Few-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2017/file/cb8a6747970d375666f507b233a70050-Paper.pdf) — Snell, Jake and Swersky, Kevin and Zemel, Richard S. (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Neural Information Processing Systems Foundation; Volume: 30; Pages: 4077-4087; DOI: [10.5555/3157382.3157545](https://doi.org/10.5555/3157382.3157545)
  介绍了原型网络，这是一种用于少样本分类的关键度量学习方法，通过计算平均嵌入作为类别原型。
- [Siamese Neural Networks for One-shot Image Recognition](https://proceedings.mlr.press/v37/koch15.pdf) — Koch, Gregory and Zemel, Richard S. and Salakhutdinov, Ruslan R. (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Publisher: JMLR: W&CP; Volume: 37; Pages: 1721-1729
  介绍了Siamese网络在计算机视觉单样本学习问题上的最早且有影响力的应用之一。
- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: ICML 2017; Volume: 70; Pages: 1126-1135; DOI: [10.48550/arXiv.1703.03400](https://doi.org/10.48550/arXiv.1703.03400)
  提出了模型不可知元学习（MAML），这是一种通用元学习算法，用于训练模型通过少量梯度步骤快速适应新任务。

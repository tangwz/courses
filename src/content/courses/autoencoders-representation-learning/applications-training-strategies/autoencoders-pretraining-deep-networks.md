---
course: "autoencoders-representation-learning"
chapter: "applications-training-strategies"
lesson: "autoencoders-pretraining-deep-networks"
sourceId: 2922
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-7-applications-training-strategies/autoencoders-pretraining-deep-networks"
title: "自动编码器用于深度网络预训练"
description: "使用自动编码器对深度监督模型中的层进行无监督预训练。"
order: 3
plots: []
sourceHash: "38aa62cc2f562f8ec34cb8f2da2b36bddd12d9d146bdeed3c664c06afdae8cb0"
sourceCorrections: []
---

在大量标注数据集以及 He 或 Xavier 初始化、批量归一化 (normalization)和残差连接等先进方法普及之前，训练非常深的神经网络 (neural network)是出了名的困难。随机初始化的权重 (weight)经常导致梯度消失或梯度爆炸，使得收敛缓慢或不可能。无监督预训练 (pre-training)，尤其是使用自动编码器，成为一种重要方法来缓解这些问题。其核心理念是首先以无监督方式学习输入数据的有用表示，然后将这种学习到的、嵌入 (embedding)在网络权重中的表示，作为后续监督任务的更优起点。

### 预训练 (pre-training)的理由

为什么学习重建输入数据会帮助完成分类或回归等不同任务？这个假设在其鼎盛时期在实践中得到很大验证，即：将数据压缩到较低维度潜在空间（编码器），然后重建它（解码器），这个过程迫使编码器捕获数据中最显著和统计上重要的变动。这些习得的特征通常形成一个层次化表示：早期层捕获简单模式（边缘、纹理），而深层则捕获更复杂的结构或语义。

这种无监督的特征学习提供了几个潜在优势：

1. **更好的初始化：** 相较于使用随机权重 (weight)开始监督训练阶段，网络从一开始就具备了已调整好的权重，能够从输入范围中提取有意义的特征。这能将模型置于参数 (parameter)空间中一个更易于为监督任务优化的区域。
2. **正则化 (regularization)作用：** 在大型未标注数据集上进行预训练可以作为一种正则化形式，尤其当目标任务的标注数据集较小时，它能有效防止监督模型过拟合 (overfitting)。学习到的特征对数据分布具有普遍性，而不仅仅针对特定的标注样本。
3. **充分利用未标注数据：** 在许多情况下，未标注数据丰富且获取成本低廉，而标注数据稀缺且获取昂贵。预训练能让你充分利用未标注数据中包含的丰富信息。

### 预训练 (pre-training)机制

自动编码器主要通过两种方式用于预训练：

1. **贪婪逐层预训练（历史方法）：** 这是最初受到重视的方法。

   - 在原始输入数据上训练一个单层自动编码器（或浅层自动编码器），以学习目标深度网络的第一个隐藏层的特征。
   - 冻结第一层的权重 (weight)。使用该层的输出（编码表示）作为输入，训练*第二个*自动编码器，学习第二个隐藏层的特征。
   - 逐层重复此过程，堆叠训练好的编码器。
   - 所有层预训练完成后，添加任务专用输出层（例如，用于分类的softmax层）。
   - 最后，使用标注数据对整个网络执行监督微调 (fine-tuning)，调整所有权重（通常最初对预训练层使用较小的学习率）。

   这种贪婪方法将深度网络的复杂优化问题分解为一系列较浅的问题。尽管具有基础意义，但由于现代端到端训练方法的有效性，此方法今天已很少使用。

> 贪婪逐层预训练后进行微调的流程。每个自动编码器都在前一个学习到的表示上进行训练。

2. **端到端自动编码器预训练：** 一种更现代的方法，涉及在未标注数据上训练一个完整的、可能是深层的自动编码器。

   - 定义并训练一个适合数据的自动编码器（编码器 + 解码器）架构（例如，用于图像的卷积自动编码器）。目标是最小化重建损失 $L(\mathbf{x}, \text{解码器}(\text{编码器}(\mathbf{x})))$。
   - 一旦训练收敛，丢弃解码器部分。
   - 用*训练好的编码器*的权重初始化目标监督网络的第一层。
   - 在预训练编码器之上添加必要的任务专用层。
   - 在标注数据集上对整个网络进行微调。

   这种方法比逐层方法实现起来更简单，并且利用了编码器在无监督重建任务中学习到的特征提取能力。去噪自动编码器（DAE）常为此方法所青睐，因为噪声注入鼓励学习出对输入扰动更具抗性的特征。

   > 使用端到端自动编码器预训练的两阶段过程。训练好的编码器被重用于监督任务。

### 现代视角与自监督学习 (supervised learning) (self-supervised learning)的关系

尽管上述专门的自动编码器预训练 (pre-training)在当今标准监督任务中已较不常见（归功于更好的架构、归一化 (normalization)和初始化），但其核心原则仍然高度相关。它已演变为自监督学习（**SSL**）这一更宽广的范畴。

许多现代SSL方法可视为自动编码的精巧形式。例如：

- **去噪自动编码器：** 直接符合从损坏版本预测原始数据的场景。
- **遮蔽自动编码器（MAE）：** 如第五章中简要提及，这些模型（通常基于Transformer）通过重建随机遮蔽的输入片段（如图像）来学习表示，这迫使模型理解上下文 (context)。这是一种卓有成效的预训练策略。
- **对比学习（例如 SimCLR, MoCo）：** 尽管不严格属于自动编码器，这些方法通过将同一样本的增强视图在嵌入 (embedding)空间中拉近，同时推开不同样本的视图来学习表示。它们与自动编码器共享从未标注数据中学习有意义特征的目标。

因此，理解自动编码器预训练为更好地理解当前最先进的自监督方法提供了有价值的背景。其核心理念依然存在：充分利用未标注数据以学习有效的特征表示，这些表示随后能提升下游任务的性能，尤其当标注数据有限时。在缺乏大量标注数据但拥有充足未标注数据的专业场景下，例如某些类型的医学影像或工业传感器读数，这仍然是一种可行策略。

## 参考资料

- [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  介绍了深度自编码器和无监督预训练在降维和初始化深度网络方面有效性的基础论文。
- [Extracting and Composing Robust Features with Denoising Autoencoders](https://dl.acm.org/doi/10.1145/1390156.1390294) — Pascal Vincent, Hugo Larochelle, Yoshua Bengio, Pierre-Antoine Manzagol (2008)
  Journal: Proceedings of the 25th International Conference on Machine Learning (ICML); Publisher: ACM Press; Pages: 1096-1103; DOI: [10.1145/1390156.1390294](https://doi.org/10.1145/1390156.1390294)
  介绍了去噪自编码器（DAE），这是一种学习鲁棒表示的特殊自编码器，常用于端到端预训练。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 14
  一本涵盖深度学习理论基础和实际应用的综合教材，包含自编码器及其在预训练中历史作用的详细章节。
- [Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) — Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick (2022)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 16000-16009
  介绍了掩码自编码器（MAE），这是自监督学习的一个重要进展，它利用自编码方法通过重建被掩码的图像块来学习强大的视觉表示。

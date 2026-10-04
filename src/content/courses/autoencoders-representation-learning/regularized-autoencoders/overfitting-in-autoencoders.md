---
course: "autoencoders-representation-learning"
chapter: "regularized-autoencoders"
lesson: "overfitting-in-autoencoders"
sourceId: 2823
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-3-regularized-autoencoders/overfitting-in-autoencoders"
title: "处理自编码器中的过拟合"
description: "明确自编码器中过拟合的原因以及正则化的必要性。"
order: 1
plots: ["plots/2823-0.json"]
sourceHash: "4e93852aef46850a32bdc9ad1f3a3a41543b4ea8ad05e073971925ede50a7090"
sourceCorrections: []
---

自编码器是一种神经网络 (neural network)，其基本架构旨在通过最小化重构误差来学习数据的压缩表示。基本的自编码器目标是使输出 $x'$ 尽可能接近输入 $x$，形式上写作 $x' = g(f(x)) \approx x$；这里 $f$ 是编码器，$g$ 是解码器。虽然功能强大，但这个简单目标可能导致一个重要问题：过拟合 (overfitting)。

当模型过度学习训练数据，捕捉到噪声和特定细节而非数据中固有的规律时，就会发生过拟合。在自编码器的背景下，这常常表现为网络学习到一个近似的*恒等函数*。如果自编码器，特别是编码器 $f$ 和解码器 $g$，相对于数据复杂性拥有足够的能力（例如，通过深层或宽层拥有大量参数 (parameter)），它就可以简单地学习将输入以最小的信息损失通过瓶颈层，有效地记忆训练样本。

考虑一个瓶颈维度等于或大于输入维度的自编码器。在没有任何限制的情况下，网络理论上可以学习将输入直接复制到输出，从而在训练数据上实现近乎完美的重构。即使是欠完备瓶颈（潜在维度 < 输入维度），如果网络能力很高，它仍可能找到复杂的映射，准确重现训练数据，但无法泛化到未见过的新样本。所得到的潜在表示，虽然可以很好地重构已知的输入，但可能无法捕捉到数据分布中本质的、可泛化的特性。相反，它可能会嵌入 (embedding)噪声或训练集特有的特征。

这会导致在新的、未见过的数据上表现不佳。在验证集或测试集上的重构误差可能明显高于训练集，或者学习到的潜在特征对于分类或聚类等下游任务而言可能变得无用。



![自编码器过拟合：训练损失与验证损失对比](plots/2823-0.json)



> 自编码器训练期间的过拟合说明。当训练损失持续下降时，验证损失在某个点后开始增加，这表明模型正在记忆训练数据而不是进行泛化。

为了防止自编码器仅仅学习恒等函数，并鼓励找到更有意义的表示，我们需要在学习过程中引入额外的约束或惩罚。这就是**正则化 (regularization)**背后的核心思想。正则化技术修改自编码器的目标函数或训练过程，以阻止过于复杂的解决方案，并促进学习到的潜在编码 $z = f(x)$ 中期望的属性。这些属性可能包括稀疏性（只激活少数神经元）、对噪声的抗性，或平滑性（输入中的微小变化导致表示中的微小变化）。

通过应用正则化，我们引导自编码器学习的表示不仅能很好地重构输入，而且能以一种更具泛化能力的方式捕捉数据中固有的结构。接下来的部分将审视具体的正则化策略，从稀疏自编码器、去噪自编码器和收缩自编码器开始，每种都旨在对抗过拟合并提升学习到的特征的质量。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍了深度学习，包括专门讨论正则化技术以对抗过拟合的章节，以及自编码器的架构和应用。
- [Extracting and Composing Robust Features with Denoising Autoencoders](https://doi.org/10.1145/1390156.1390294) — Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, Pierre-Antoine Manzagol (2008)
  Journal: Proceedings of the 25th International Conference on Machine Learning (ICML); Publisher: ACM; Pages: 1096-1103; DOI: [10.1145/1390156.1390294](https://doi.org/10.1145/1390156.1390294)
  介绍了去噪自编码器，这是一种正则化技术，旨在通过训练自编码器从损坏的输入中重建原始输入来学习鲁棒的特征表示。
- [UFLDL Tutorial: Sparse Autoencoders](http://ufldl.stanford.edu/wiki/index.php/Autoencoders) — Andrew Ng (2011)
  Publisher: Stanford University
  解释了稀疏自编码器的概念和实现，这是一种正则化方法，通过在潜在层鼓励稀疏表示来防止过拟合并学习更有意义的特征。
- [Contractive Auto-Encoders: Explicit Invariance Through Penalizing Local Contractions](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEVpvzZNx7fcu7z8GcjxpazF8B6Wy-aFkYljW1Hw1aUjd0FUDWVvHbpS0EPWkje1hbN57GNjB0CjcKDtrAJYOLMvtfewpQw9db7u-fKo1MPLcbcf5obszDRWUtjnorLfynXdjauy8Q=) — Salah Rifai, Pascal Vincent, Xavier Muller, Xavier Glorot, Yoshua Bengio (2011)
  Journal: Proceedings of the 28th International Conference on Machine Learning (ICML); Publisher: JMLR Workshop and Conference Proceedings; Pages: 833-840
  介绍了收缩自编码器，这是一种正则化技术，通过惩罚编码函数的大导数，鼓励学习到的表示对输入的微小扰动保持鲁棒性。

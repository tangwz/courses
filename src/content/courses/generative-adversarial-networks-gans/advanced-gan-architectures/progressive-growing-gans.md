---
course: "generative-adversarial-networks-gans"
chapter: "advanced-gan-architectures"
lesson: "progressive-growing-gans"
sourceId: 2666
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-2-advanced-gan-architectures/progressive-growing-gans"
title: "渐进式生成对抗网络 (ProGAN)"
description: "了解ProGANs如何通过逐步增加网络深度来实现高分辨率图像合成。"
order: 1
plots: []
sourceHash: "baede181ff6cfe492c63122839efdf324a4de91f761937eb84409c3e2eac7000"
sourceCorrections: []
---

直接在高分辨率图像上训练生成对抗网络 (GAN)会面临重大挑战。大型网络难以优化，梯度可能消失或爆炸，并且生成器和判别器难以有效协调它们的学习过程，尤其是在早期阶段，生成的图像与目标分布几乎没有相似之处。在学习图像的粗略结构的同时，从零开始生成精细细节，要求很高。

为解决直接训练高分辨率图像生成对抗网络所面临的挑战，2017年由Karras等人（NVIDIA）提出了渐进式生成对抗网络（ProGAN）这一巧妙方法。ProGAN不是从一开始就为目标高分辨率训练一个单一的大型网络，而是从非常低分辨率的图像（例如4x4像素）开始，并逐步为生成器（G）和判别器（D）添加层，以处理逐步提升的分辨率（8x8、16x16，…，直到1024x1024或更高）。

### 核心思想：逐步增长

基本原则是首先训练网络以在低分辨率下理解图像分布的粗略结构。一旦这个初始阶段收敛得相对好，就会在G和D中添加新层，以使空间分辨率翻倍。先前训练的层提供了一个稳定的基础，新层则侧重于学习与增加的分辨率相关的更精细细节。这个过程重复进行，直到达到期望的输出分辨率。

### 通过层淡入稳定增长

突然引入新层可能会冲击系统并使训练不稳定。ProGAN通过平滑地融入新层来解决这个问题。当从分辨率 $R \times R$ 过渡到 $2R \times 2R$ 时，会添加新层，但其影响通过参数 (parameter) $\alpha$ 逐渐增加，该参数在多次迭代中从0递增到1。

以生成器为例：

1. 现有层以 $R \times R$ 分辨率生成图像。此输出被上采样到 $2R \times 2R$。
2. 新添加的层也处理前一阶段的特征图，并生成在新的 $2R \times 2R$ 分辨率下操作的图像块。
3. 最终输出是这两个路径的加权组合：
   $\text{输出}_{2R \times 2R} = (1 - \alpha) \times (\text{上采样输出}_{R \times R}) + \alpha \times (\text{新层输出}_{2R \times 2R})$

判别器中也发生类似的淡入过程，但方向相反： $2R \times 2R$ 的输入图像由新层处理，而一个下采样版本（$R \times R$）则绕过它们进入网络的旧部分。判别器的判断基于这两个路径输出的凸组合（由 $\alpha$ 控制）。

> 渐进式增长阶段过渡。添加新层（G中蓝色，D中红色）以处理 $2R \times 2R$ 分辨率。它们的输出与先前 $R \times R$ 阶段的输出（G中上采样，D中下采样）结合，使用一个从0增加到1的参数 $\alpha$，确保平滑过渡。

这种逐步适应让网络能够融入新的细节处理能力，而不会干扰从较低分辨率学到的已稳定特征。

### 渐进式训练的优点

1. **训练稳定性：** 通过首先侧重于更简单的低分辨率结构，优化问题在初始阶段会更容易。网络在处理复杂的高频细节之前建立起扎实的基础。这大幅降低了训练大型GANs时常出现的灾难性模式崩溃或发散训练动态的可能性。
2. **训练更快：** 尽管训练涉及多个阶段，但每个阶段的训练速度都快于从一开始就尝试训练完整的高分辨率网络。早期阶段在低分辨率数据上快速收敛。
3. **高分辨率合成：** ProGAN在生成1024x1024等分辨率的高质量、连贯图像方面具有里程碑意义，这在当时是一大进步。

### 架构考量及支持技术

虽然渐进式增长是核心思想，但ProGAN的成功也依赖于在每个阶段应用的其他几种架构选择和训练技术：

- **等效学习率：** ProGAN不是依赖仔细的权重 (weight)初始化，而是使用运行时机制来动态缩放每层中的权重。具体来说，权重 $w_i$ 通过一个源自He初始化器的逐层常数 $c$ 进行缩放：$\hat{w}_i = w_i / c$。这确保了输出的方差在各层和量级之间保持一致，提高了稳定性。
- **逐像素特征向量 (vector)归一化 (normalization)：** 在生成器中每个卷积层之后应用，此技术将每个像素的特征向量归一化为单位长度。这有助于防止信号幅度不断增大，在生成器网络中特别有用。
- **小批量标准差：** 为了鼓励生成器生成更多样化的样本并防止模式崩溃，判别器末端附近添加了一层。此层计算每个空间位置在小批量样本中特征的标准差，计算所有特征和位置的平均标准差，并将此标量值作为附加特征图附加到判别器最后一层的输入。这为判别器提供了关于批次统计信息的信号，隐式地鼓励生成器创建具有与真实数据相似统计信息的批次。

### ProGAN的影响

渐进式增长展示了一种用于训练高分辨率GANs的强大方法。它强调了课程学习原则（从简单开始，逐步增加复杂度）在生成模型中的重要性。尽管StyleGAN等架构在此基础上进行改进，但ProGAN引入的渐进式分辨率提升的核心思想仍然是GAN实践者工具包中的一项重要技术，显示了周密的架构设计如何解决基本的训练难题。

## 参考资料

- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) — Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.10196](https://doi.org/10.48550/arXiv.1710.10196)
  首次提出渐进式增长方法，用于稳定、高分辨率GAN训练的原始论文。
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) — Tero Karras, Samuli Laine, Timo Aila (2019)
  Journal: CVPR 2019; Pages: 4401-4410; DOI: [10.48550/arXiv.1812.04948](https://doi.org/10.48550/arXiv.1812.04948)
  本文直接基于ProGAN，提出了StyleGAN架构，改进了图像合成的控制。该文最初于2018年在arXiv上发布。
- [Generative Adversarial Networks: A Survey and Taxonomy](https://doi.org/10.1613/jair.12711) — Zekun Pan, Weike Yan (2021)
  Journal: Journal of Artificial Intelligence Research; Publisher: AI Access Foundation; Volume: 71; Pages: 257-302; DOI: [10.1613/jair.12711](https://doi.org/10.1613/jair.12711)
  提供各种GAN架构和训练技术背景的综合性综述，包括ProGAN。

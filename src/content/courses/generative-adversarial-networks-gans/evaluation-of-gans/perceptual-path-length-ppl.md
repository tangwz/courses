---
course: "generative-adversarial-networks-gans"
chapter: "evaluation-of-gans"
lesson: "perceptual-path-length-ppl"
sourceId: 2743
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans/perceptual-path-length-ppl"
title: "感知路径长度 (PPL)"
description: "通过测量插值过程中的感知变化，评估潜在空间的平滑度。"
order: 7
plots: []
sourceHash: "508cf2094ee1e066f304700bdf578f5b29af803382ba55d67a7bd1216a7bb4cd"
sourceCorrections: []
---

尽管像FID和IS这样的指标能提供关于整体分布相似性和样本质量的重要信息，但它们并未明确评估生成器学得的潜在空间的*结构*。一个性能良好的生成器，理想情况下应将潜在空间中相邻的点映射到感知上相似的图像。潜在代码中微小步长导致生成图像的突然变化，常表示潜在空间结构不佳或“纠缠不清”，这会影响平滑图像插值或属性调整等任务。

感知路径长度（PPL）指标专门用于量化 (quantization)潜在空间平滑度的这一理念。它平均测量在潜在空间中沿着插值路径迈出小步时，生成图像的感知变化。主要思想是，如果潜在空间结构良好，在两个潜在代码之间进行插值应导致生成图像中相应平滑的感知过渡。

### PPL工作原理

计算PPL涉及以下几个步骤：

1. **潜在空间采样：** 从生成器的输入分布（例如，标准高斯分布 $P(z)$）中随机抽取成对的潜在向量 (vector) $z_1$ 和 $z_2$。
2. **插值：** 定义连接这两点的插值路径。线性插值（lerp）是常用方法：
   $z(t) = (1-t)z_1 + t z_2$
   其中 $t$ 的范围是0到1。
3. **步进与生成：** 沿着这条路径迈出非常小的步长。对于给定的 $t$ 和小步长 $\epsilon$（例如，$\epsilon = 10^{-4}$），生成路径上无限接近的两个点对应的图像：$G(z(t))$ 和 $G(z(t+\epsilon))$。
4. **感知距离：** 计算这两个生成图像之间的感知距离。这需要一个能捕捉人类感知相似性的预训练 (pre-training)网络。一个常用选择是学习感知图像块相似度（LPIPS）指标，它通常基于从VGG或AlexNet等网络提取的特征。令 $d(\cdot, \cdot)$ 表示此感知距离函数。
5. **求平均：** 计算路径上许多小步长 $t$ 对应的距离 $d(G(z(t)), G(z(t+\epsilon)))$。对这些距离求平均。重要的是，原始公式将此距离乘以 $1/\epsilon^2$ 以近似导数，但在实践中，常报告的指标是乘以 $1/\epsilon$ 的平均距离。最终PPL是许多随机抽样的对 $(z_1, z_2)$ 的这些平均路径距离的期望值。

从数学角度看，单条路径的核心计算是对许多点 $t$ 处的标度感知距离求平均：
$\text{PPL}_{\text{路径}} = \text{均值}_{t} \left( \frac{1}{\epsilon} d(G(\text{lerp}(z_1, z_2, t)), G(\text{lerp}(z_1, z_2, t+\epsilon))) \right)$
最终PPL得分是许多初始对 $(z_1, z_2)$ 的 $\text{PPL}_{\text{路径}}$ 的平均值。

### 插值空间：Z与W

对于像StyleGAN这样利用映射网络将初始噪声向量 (vector) $z$ 转换为中间潜在代码 $w$（通常位于表示为 $W$ 的空间中）的生成器，插值可以在初始 $Z$ 空间或中间 $W$ 空间中进行。

- **Z空间中的PPL ($PPL_z$)：** 直接在 $z_1$ 和 $z_2$ 之间进行插值，所得的 $z(t)$ 向量通过映射网络，然后输入到合成网络。
- **W空间中的PPL ($PPL_w$)：** 首先，将 $z_1$ 和 $z_2$ 映射到 $w_1 = \text{Mapping}(z_1)$ 和 $w_2 = \text{Mapping}(z_2)$。然后，在 $W$ 空间中进行插值：$w(t) = \text{lerp}(w_1, w_2, t)$。这些插值后的 $w(t)$ 向量随后输入到合成网络 $G_{\text{synth}}(w(t))$。

已经观察到，特别是对于StyleGAN架构，$W$ 空间通常比 $Z$ 空间更“解耦”。在 $W$ 中插值倾向于在输出图像中产生感知上更线性的过渡。因此，$PPL_w$（在 $W$ 中插值计算的PPL）通常得到比 $PPL_z$ 更低（更好）的分数，并被认为是衡量合成网络自身实现的感知平滑度更具代表性的指标。

> 感知路径长度计算流程。插值在潜在空间进行，生成图像使用感知距离指标进行比较，结果进行平均。

### PPL分数的解读

- **分数越低越好：** 较低的PPL分数表示潜在空间中的小步长对应生成图像中小的感知变化。这表明潜在表示更平滑，可能更解耦，这通常是人们希望的。
- **分数越高越差：** 高PPL分数意味着潜在代码中的小扰动可能导致大而突然的视觉变化。这通常与视觉伪影相关，或表明潜在空间结构与感知变化对应不佳。

PPL对于比较旨在改善潜在空间属性的不同生成器架构或正则化 (regularization)方法特别有用。例如，StyleGAN论文广泛使用PPL来展示映射网络和使用 $W$ 空间进行风格调制的好处。

### 局限性

尽管有这些方面，PPL仍有一些局限性：

- **计算成本：** 它需要生成大量图像并进行成对感知距离计算，使其计算成本比FID或IS显著更高。
- **对感知指标的依赖：** 结果本质上取决于所用的特定感知距离函数 ($d$)（例如，使用VGG骨干网络的LPIPS）。不同的指标可能产生不同的分数或排名。
- **侧重平滑度：** PPL主要衡量插值路径上的局部平滑度。它不像FID或IS那样直接量化 (quantization)整体样本保真度（真实性）或多样性，尽管极端的PPL值通常与质量问题相关。
- **对 $\epsilon$ 的敏感性：** 步长 $\epsilon$ 的选择会影响结果，尽管通常使用非常小的值。

尽管有这些方面，PPL通过直接评估生成器学得的潜在空间的结构和感知一致性，为GAN性能评估提供了一个有价值的视角，补充了侧重生成分布全局属性的指标。

## 参考资料

- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) — Tero Karras, Samuli Laine, Timo Aila (2019)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 4401-4410; DOI: [10.48550/arXiv.1812.04948](https://doi.org/10.48550/arXiv.1812.04948)
  介绍了StyleGAN架构和感知路径长度（PPL）指标，用于评估潜在空间的平滑度和解缠结性。
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) — Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang (2018)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 586-595; DOI: [10.48550/arXiv.1801.03924](https://doi.org/10.48550/arXiv.1801.03924)
  介绍了学习感知图像块相似度（LPIPS）指标，该指标常被用作PPL计算中的感知距离函数。

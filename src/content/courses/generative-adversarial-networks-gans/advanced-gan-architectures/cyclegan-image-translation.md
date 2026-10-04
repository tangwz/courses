---
course: "generative-adversarial-networks-gans"
chapter: "advanced-gan-architectures"
lesson: "cyclegan-image-translation"
sourceId: 2680
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-2-advanced-gan-architectures/cyclegan-image-translation"
title: "非配对图像到图像转换 (CycleGAN)"
description: "学习在没有配对示例的情况下，用于域间转换的循环一致性损失。"
order: 6
plots: []
sourceHash: "2a5eec07fe3118f261a2fcd9e1c3590f72302230ee2404b734a89133193eff05"
sourceCorrections: []
---

"许多图像生成任务都包含将图像从源域 $X$ 转换到目标域 $Y$。例如，将照片转为绘画、更改图像中的季节，或者将马变为斑马。当存在配对训练数据时（例如，建筑草图与对应照片的配对），像pix2pix这样的模型可以使用监督技术有效地学习这种映射。然而，获取此类配对数据集通常成本高昂、困难，或者根本不可能。仅有 $X$ 域的图像集合和来自 $Y$ 域的**独立**、**不相关**图像集合时，我们如何学习在 $X$ 和 $Y$ 域之间进行转换呢？"

CycleGAN 为**非配对图像到图像转换**这一问题提供了一个巧妙的解决办法。它在不需要两个域中个体图像之间有任何直接对应关系的情况下，学习这种映射。

### 核心思想：循环一致性

想象一下，您想将马的照片（域 $X$）转换成类似斑马的图像（域 $Y$）。CycleGAN 采用两个生成网络：

- $G$: 学习映射 $G: X \to Y$（例如，马到斑马）。
- $F$: 学习逆映射 $F: Y \to X$（例如，斑马到马）。

它还使用两个判别网络：

- $D_Y$: 旨在区分域 $Y$ 中的真实图像（真实斑马）和生成的图像 $G(x)$（由马的照片 $x$ 生成的假斑马）。
- $D_X$: 旨在区分域 $X$ 中的真实图像（真实马）和生成的图像 $F(y)$（由斑马的照片 $y$ 生成的假马）。

标准对抗损失促使 $G$ 生成看起来属于域 $Y$ 的输出 $G(x)$，并促使 $F$ 生成看起来属于域 $X$ 的输出 $F(y)$。

然而，仅有对抗损失是不够的。生成器可能学会将一个域的所有输入映射到另一个域中一个单一的、看起来真实的图像，这虽然满足了判别器，但未能捕捉到所需的输入-输出关系。例如，$G$ 可能会学会为**任何**输入马照片生成一张**合理**的斑马图像。

为了解决这个问题，CycleGAN 引入了**循环一致性损失**。其理念很简单：如果您将一张图像从域 $X$ 转换到域 $Y$，然后再将其转换回域 $X$，您应该得到一个与原始图像非常接近的结果。从域 $Y$ 开始也适用同样的道理。

在数学上，这个限制是使用损失函数 (loss function)（通常是L1距离）来执行的，如本章引言中所述：


$$
L_{cyc}(G, F) = \mathbb{E}_{x \sim p_{data}(x)}[||F(G(x)) - x||_1] + \mathbb{E}_{y \sim p_{data}(y)}[||G(F(y)) - y||_1]
$$


这种损失惩罚了原始图像（$x$ 或 $y$）与其经过正向和反向转换后重建版本（$F(G(x))$ 或 $G(F(y))$）之间的差异。L1 范数 ($|| \cdot ||_1$) 通常优于 L2 范数 ($|| \cdot ||_2^2$)，因为它在图像生成任务中倾向于产生较少模糊的结果。

### 完整目标函数

CycleGAN 的完整目标函数结合了两个映射方向的对抗损失和循环一致性损失：


$$
L_{total}(G, F, D_X, D_Y) = L_{GAN}(G, D_Y, X, Y) + L_{GAN}(F, D_X, Y, X) + \lambda L_{cyc}(G, F)
$$


其中：

- $L_{GAN}(G, D_Y, X, Y)$ 是 $X \to Y$ 映射的对抗损失（生成器 $G$ 试图欺骗判别器 $D_Y$）。
- $L_{GAN}(F, D_X, Y, X)$ 是 $Y \to X$ 映射的对抗损失（生成器 $F$ 试图欺骗判别器 $D_X$）。
- $L_{cyc}(G, F)$ 是前面定义的循环一致性损失。
- $\lambda$ 是一个超参数 (parameter) (hyperparameter)，控制对抗损失与循环一致性损失的相对重要性。原始论文中使用的典型值为 $\lambda = 10$。

目标是找到生成器 $G^*$ 和 $F^*$，使它们在最小化循环一致性损失的同时，最大化对抗各自判别器的损失；这些判别器也同时训练以区分真实图像和生成图像：


$$
G^*, F^* = \arg \min_{G, F} \max_{D_X, D_Y} L_{total}(G, F, D_X, D_Y)
$$


### 架构与训练细节

尽管循环一致性损失是主要贡献，CycleGAN 的实现常包含已知能提升 GAN 稳定性和质量的架构和技术：

- **生成器：** 通常使用带有残差块的卷积架构，这受到风格迁移中成功模型的启发。它们通常由下采样层、多个残差块，然后是上采样层组成。
- **判别器：** 常用 PatchGAN 分类器。PatchGAN 不会为整个图像是真实还是伪造输出单一概率，而是输出一个概率网格，其中每个值对应于重叠图像块的“真实性”。这有助于图像生成更清晰的细节。
- **损失函数 (loss function)：** 原始论文使用最小二乘 GAN (LSGAN) 损失代替对抗项 ($L_{GAN}$) 的标准 minimax 对数损失，发现这带来了更稳定的训练。
- **图像缓冲区：** 为避免判别器过快地适应最新生成的图像，CycleGAN 使用一个缓冲区来存储过去生成的图像历史。判别器通过混合使用真实图像和从该缓冲区中采样的图像进行训练。

> 图解 CycleGAN 框架。它包含两个生成器（$G$, $F$）、两个判别器（$D_X$, $D_Y$）、用于确保生成图像与目标域分布匹配的对抗损失，以及用于强制输入图像与重建图像之间结构相似的循环一致性损失。

### 优点与局限

CycleGAN 的主要优点是它无需配对数据即可执行图像转换，这使得许多以前无法实现的应用成为可能。它在风格迁移（照片到莫奈画风、梵高画风等）、物体变形（马到斑马、苹果到橙子）和域适应（合成图像到真实图像）等任务中展现了显著效果。

然而，它也有局限性：

- **几何变化：** CycleGAN 难以处理需要显著几何变换的任务（例如，猫狗图像之间的转换，它们有不同的典型姿势和形状）。循环一致性损失隐含地假定底层结构在转换和重建过程中可以在很大程度上得以保留。
- **内容保留：** 尽管在改变纹理和风格方面很有效，但如果对抗损失占主导地位或任务本身模糊，它有时可能无法保留输入图像的精确内容细节。
- **模式崩溃：** 像其他 GAN 一样，CycleGAN 也会受到模式崩溃的影响，尽管与更简单的对抗设置相比，循环一致性损失通常有助于减轻这种情况。
- **资源密集：** 训练涉及四个网络（两个生成器，两个判别器），这使得它比标准 GAN 或配对转换模型在计算上要求更高。

## 参考资料

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros (2017)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Pages: 222-230; DOI: [10.1109/ICCV.2017.27](https://doi.org/10.1109/ICCV.2017.27)
  介绍了CycleGAN框架、循环一致性损失及其在无配对图像到图像翻译中的应用。
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros (2017)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 1125-1134; DOI: [10.48550/arXiv.1611.07004](https://doi.org/10.48550/arXiv.1611.07004)
  提出了用于监督式图像到图像翻译的pix2pix模型，为理解CycleGAN所扩展的配对方法提供了基础。
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2016)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 770-778; DOI: [10.48550/arXiv.1512.03385](https://doi.org/10.48550/arXiv.1512.03385)
  介绍了残差块，这是CycleGAN生成器和许多其他深度学习模型中使用的基本架构组件。

---
course: "synthetic-data-gans-diffusion"
chapter: "advanced-gan-architectures-techniques"
lesson: "cyclegan-unpaired-translation"
sourceId: 5651
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-2-advanced-gan-architectures-techniques/cyclegan-unpaired-translation"
title: "非配对图像到图像转换 (CycleGAN)"
description: "为域适应实现CycleGAN，无需配对数据。"
order: 3
plots: []
sourceHash: "3e0c3e7848637b9dced0bdc1577a6ea84f314cc29357c58477abc05caeffbb62"
sourceCorrections: []
---

图像到图像转换涉及学习输入图像域与输出图像域之间的映射关系。尽管像pix2pix这样的监督学习 (supervised learning)方法取得了令人赞叹的成果，但它们需要大量的*配对*图像数据集，每张输入图像都对应一个目标输出。获得此类配对数据对于许多任务来说通常很困难、昂贵，甚至不可能，例如艺术风格转换（莫奈到梵高）或转换自然界中不存在精确配对的对象（马到斑马）。

CycleGAN通过使用**非配对**训练数据进行图像到图像转换来解决这一重大局限。非配对数据的主要难点在于如何约束转换：我们如何确保生成的图像反映输入图像的内容，而不是仅仅是目标域中的任意样本？单独的标准GAN损失是不够的，因为生成器可能学会忽略输入并生成欺骗判别器的输出，但这些输出与源图像缺乏对应关系（这是一种与模式崩溃相关的现象）。

### 循环一致性约束

CycleGAN的核心创新是引入了**循环一致性损失**。其思路很直接：如果我们将一张图像从域A转换到域B，然后再将所得图像从域B转换回域A，我们应该能得到与原始图像非常接近的结果。这在域之间强制建立了结构和内容上的对应关系，即使没有直接的配对数据。

为此，CycleGAN使用了两个生成器和两个判别器：

1. **生成器 $G_{A \to B}$（或简称为 $G$）：** 将图像从域A转换到域B。$G: A \to B$。
2. **生成器 $G_{B \to A}$（或简称为 $F$）：** 将图像从域B转换到域A。$F: B \to A$。
3. **判别器 $D_B$：** 区分来自域B的真实图像（$y$）和生成的图像 $G(x)$，其中 $x$ 来自域A。
4. **判别器 $D_A$：** 区分来自域A的真实图像（$x$）和生成的图像 $F(y)$，其中 $y$ 来自域B。

循环一致性损失通过数学方式强化了上述思路。它衡量了原始图像与其经过一次正向和一次反向转换后的重建图像之间的差异（通常使用L1范数以获得比L2更清晰的结果）：


$$
\mathcal{L}_{\text{循环一致}}(G, F) = \mathbb{E}_{x \sim p_{\text{数据}}(A)}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y \sim p_{\text{数据}}(B)}[\|G(F(y)) - y\|_1]
$$


这里，$p_{\text{数据}}(A)$ 和 $p_{	ext{数据}}(B)$ 分别表示域A和域B的数据分布。第一项惩罚从 $A \to B \to A$ 转换时的偏差，第二项惩罚从 $B \to A \to B$ 转换时的偏差。

### 整体目标函数

CycleGAN的完整目标函数结合了每个生成器-判别器对的标准对抗性损失与循环一致性损失：

1. **$G$ 和 $D_B$ 的对抗性损失**：鼓励 $G$ 生成看起来属于域B的图像 $G(x)$。

   
   $$
   \mathcal{L}_{\text{GAN}}(G, D_B, A, B) = \mathbb{E}_{y \sim p_{\text{数据}}(B)}[\log D_B(y)] + \mathbb{E}_{x \sim p_{\text{数据}}(A)}[\log(1 - D_B(G(x)))]
   $$
   
2. **$F$ 和 $D_A$ 的对抗性损失**：鼓励 $F$ 生成看起来属于域A的图像 $F(y)$。

   
   $$
   \mathcal{L}_{\text{GAN}}(F, D_A, B, A) = \mathbb{E}_{x \sim p_{\text{数据}}(A)}[\log D_A(x)] + \mathbb{E}_{y \sim p_{\text{数据}}(B)}[\log(1 - D_A(F(y)))]
   $$
   

要优化的完整目标函数是：


$$
\mathcal{L}(G, F, D_A, D_B) = \mathcal{L}_{\text{GAN}}(G, D_B, A, B) + \mathcal{L}_{\text{GAN}}(F, D_A, B, A) + \lambda \mathcal{L}_{\text{循环一致}}(G, F)
$$


超参数 (parameter) (hyperparameter) $\lambda$ 控制着对抗性损失与循环一致性损失的相对重要性。原始论文中使用的典型值为 $\lambda = 10$。目标是找到生成器 $G$ 和 $F$，它们在判别器 $D_A$ 和 $D_B$ 试图最大化损失的情况下，最小化这种组合损失。

> 该图呈现了CycleGAN框架。它显示了由循环一致性损失强制执行的两个转换循环（$A \to B \to A$ 和 $B \to A \to B$），以及由判别器 $D_A$ 和 $D_B$ 评估的对抗性损失。

### 架构与实现细节

CycleGAN中的生成器（$G$和$F$）通常采用从神经风格迁移和超分辨率任务中修改而来的架构。常见的选择包含：

- 带有卷积层的编码器阶段，用于对输入进行下采样。
- 使用多个残差块（ResNet块）的转换器阶段，用于处理较低空间分辨率的特征。
- 使用转置卷积（或上采样后接卷积）的解码器阶段，用于将特征上采样回原始图像大小。

通常使用实例归一化 (normalization)而非批量归一化，因为转换应独立于批次中的其他图像。

判别器（$D_A$和$D_B$）常采用类似于pix2pix的PatchGAN架构。PatchGAN不将整个图像分类为真实或虚假，而是输出一个预测网格，其中每个预测对应于输入图像的一个图像块。这比单一分类输出更能有效促使图像清晰和局部真实。

为了提升训练稳定性，CycleGAN的实现通常在训练判别器时，使用之前生成的图像缓冲区而非最新图像。这可以防止判别器过快地适应当前生成器的输出。此外，对抗性损失的目标函数有时会用最小二乘GAN (LSGAN) 目标替代标准的负对数似然目标，因为这有助于训练更稳定。

### 应用和用途

CycleGAN已在缺乏配对数据的领域获得广泛应用：

- **风格迁移**：在不同画家（例如，莫奈 $\leftrightarrow$ 梵高）之间或照片与绘画之间进行艺术风格的转换。
- **对象变形**：改变对象的样式（例如，马 $\leftrightarrow$ 斑马，苹果 $\leftrightarrow$ 橙子）。
  "\* **域适应**：将图像从一个域（例如，合成渲染图）适应到另一个域（例如，照片），以提高主要在合成数据上训练的模型的性能。"
- **季节转换**：改变照片中描绘的季节（例如，夏季 $\leftrightarrow$ 冬季）。
- **照片增强**：例如增强弱光照片或为黑白图像上色等任务，尽管如果有配对数据，配对数据方法通常表现更好。

### 局限性

尽管强大，CycleGAN仍有局限性：

- **几何变化**：它难以处理需要显著几何变化的转换（例如，将猫转换为狗时，通常会保留猫的姿态）。循环一致性损失主要强制颜色和纹理的变化。
- **模式崩溃**：尽管循环一致性有帮助，但模式崩溃仍可能发生，特别是当 $\lambda$ 过小时。
- **伪影**：生成的图像有时会包含视觉伪影，尤其是在域差异很大时。
- **身份保留**：对于照片增强等任务，可能需要额外的身份损失，以确保输出在内容上与输入非常相似，并结合循环一致性。

尽管存在这些局限性，CycleGAN仍是生成建模方面的一大进步，表明了无需配对数据集即可进行高质量图像到图像转换的可行性。它通过引入循环一致性原则，巧妙地解决了非配对转换的约束不足问题，使其成为先进生成建模工具包中的重要工具。

## 参考资料

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros (2017)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 2242-2251; DOI: [10.48550/arXiv.1703.10593](https://doi.org/10.48550/arXiv.1703.10593)
  介绍CycleGAN的开创性论文，详细阐述了循环一致性损失及其在非配对图像到图像翻译中的应用。
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems 27 (NIPS 2014); Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  引入生成对抗网络（GANs）的基础论文，为所有基于GAN的架构提供了理论基础。
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros (2017)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 1125-1134; DOI: [10.48550/arXiv.1611.07004](https://doi.org/10.48550/arXiv.1611.07004)
  介绍了pix2pix，一种有监督的图像到图像翻译方法，对CycleGAN的架构和PatchGAN概念产生了深远影响。

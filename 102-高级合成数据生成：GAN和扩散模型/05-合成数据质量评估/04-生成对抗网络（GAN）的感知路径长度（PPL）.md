# 生成对抗网络（GAN）的感知路径长度（PPL）

来源：[原文](https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-5-evaluating-synthetic-data-quality/perceptual-path-length-ppl)

[返回章节目录](README.md) · [返回课程目录](../README.md)

弗雷歇特起始距离（FID）和起始分数（IS）等指标能够提供关于生成样本相对于真实数据集的整体质量和多样性的有益理解。然而，这些指标未能全面描述生成器的行为。具体来说，它们不直接评估GAN潜在空间的*平滑性*或*可解释性*。设想您想在两张生成的面孔之间进行插值，或编辑某个特定属性；一个结构良好的潜在空间应允许您在潜在空间中移动时，输出图像发生平滑、逐渐的变化。突然、不合理的改变表明潜在空间结构不佳或存在“纠缠”。正是在此背景下，感知路径长度（PPL）指标应运而生，它提供了一种量化 (quantization)这种平滑性的方法。它随StyleGAN一起被提出，但也适用于其他GAN架构。

PPL衡量的是在潜在空间中迈出的一小步，导致生成图像感知上发生多大程度的变化。主要想法是，如果潜在空间结构良好且已解缠，则对潜在向量 (vector)的微小扰动应仅导致相应输出图像中发生微小、语义上有意义的变化。相反，如果在潜在空间中微小的一步导致巨大、刺眼的视觉变化，则表明潜在空间未能有效学会将输入噪声中的连续变化映射到输出图像感知特征中的连续变化。

### 计算感知路径长度

为了计算PPL，我们模拟在潜在空间中的移动，并测量沿该路径无限接近的点之间生成的图像的感知距离。以下是流程的步骤细述：

1. **采样潜在向量 (vector)：** 从GAN的先验分布$P(z)$（通常是标准高斯分布）中随机采样两个潜在向量，$z_1$和$z_2$。
2. **插值：** 定义这些点之间的一条路径。虽然可以使用线性插值（$lerp$），但通常更偏好球面线性插值（$slerp$），特别是当潜在向量已归一化 (normalization)时：
   
   $$
   z(t) = slerp(z_1, z_2, t) \quad \text{对于} \quad t \in [0, 1]
   $$
   
   这里，$t$ 参数 (parameter)化了沿路径的位置。
3. **在接近点生成图像：** 为该路径上无限接近的两个点生成图像，例如在$t$和$t + \epsilon$处，其中$\epsilon$是非常小的步长。将这些图像记为$G(z(t))$和$G(z(t + \epsilon))$。
4. **测量感知距离：** 使用感知距离指标$d(\cdot, \cdot)$计算这两张图像之间的距离。该指标应与人类对图像相似性的感知保持一致。常见选择包括基于深度网络特征的指标，例如LPIPS（学习感知图像块相似性），它通常使用VGG-16或AlexNet等网络中的激活。简单的像素级距离（如L1或L2）通常不足，因为它们不能很好地捕捉感知相似性。
5. **在路径段上求平均：** 在位置$t$处，步长为$\epsilon$的瞬时感知变化近似为$d(G(z(t)), G(z(t + \epsilon)))$。PPL旨在找出这种变化的期望值，并按步长归一化。更正式地，它被定义为对所有可能的端点（$z_1, z_2$）和沿路径的所有位置（$t$）的期望：
   
   $$
   \text{PPL} = \mathbb{E}_{z_1, z_2 \sim P(z), t \sim U(0, 1)} \left[ \frac{1}{\epsilon^2} d(G(slerp(z_1, z_2, t)), G(slerp(z_1, z_2, t + \epsilon))) \right]
   $$
   
   这种除以 $\epsilon^2$ 的操作源于原始论文中涉及雅可比计算的定义，但在实践中，计算的是小固定步长 $\epsilon$ 上的平均距离。

在实践中，期望通过对许多随机采样的成对（$z_1, z_2$）求平均，并将它们之间的路径分成大小为$\epsilon$的小固定步长来近似。感知距离$d$在连续步长生成的图像之间计算，并将这些距离求平均。

### PPL得分的解释

- **低PPL：** 表示潜在空间更平滑。潜在空间中的小步长持续导致生成图像中发生小的感知变化。这是合乎要求的，表明更好的解缠，并使模型更适合通过潜在空间操纵进行图像编辑或风格混合等任务。
- **高PPL：** 表明潜在空间“不连贯”或纠缠。微小的移动可能导致图像中出现大的、通常是非语义的偏移，使得平滑插值变得困难。

### Z空间与W空间中的PPL（StyleGAN）

对于像StyleGAN这样使用映射网络将初始潜在编码$z$（来自先验$P(z)$）转换为中间潜在编码$w \in W$的架构，PPL可以在任一空间中计算：

- **$PPL_Z$：** 插值和步长在初始$Z$空间中执行。
- **$PPL_W$：** 插值和步长在中间$W$空间中执行（在映射网络之后）。

StyleGAN中的$W$空间被有意设计得比$Z$空间更解缠。因此，$W$中的插值通常会产生更平滑的视觉过渡。因此，$PPL_W$（使用$W$空间中路径计算的PPL）通常比$PPL_Z$给出明显更低（更好）的分数，并且是评估映射网络有效性和所学合成过程整体平滑性的常用报告指标。

> 该图对比了低PPL和高PPL。低PPL表示，在潜在空间（Z或W）中沿路径移动一小段距离，会导致生成图像感知特征中相应的小而平滑的变化。高PPL则表明，类似的微小步长可能导致感知外观中出现大的、不连续的跳跃。

### 实际考量

计算PPL在计算上比FID或IS要求更高。它涉及沿多条插值路径生成大量图像，并重复使用另一个深度网络计算成对感知距离。感知距离函数的选择（例如，VGG或LPIPS变体中的特定层）和采样参数 (parameter)（路径数量、步长$\epsilon$）会影响PPL的绝对分数，因此在比较不同模型时，这些设置的一致性很重要。

PPL作为FID和IS的补充指标。FID评估真实图像和生成图像分布之间的总体匹配度，而PPL专门检验生成器潜在空间的内部结构和连续性。一个GAN可能通过捕捉目标数据的多样性而获得良好的FID分数，但如果其潜在空间缺乏平滑过渡，仍可能具有高PPL，从而阻碍其在精细控制和编辑方面的可用性。因此，将PPL与其他指标一同考量，可以更全面地反映GAN的性能，特别是其潜在空间的质量和可用性。

## 参考资料

- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) — Tero Karras, Samuli Laine, and Timo Aila (2019)
  Journal: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); DOI: [10.48550/arXiv.1812.04948](https://doi.org/10.48550/arXiv.1812.04948)
  介绍了StyleGAN架构和感知路径长度（PPL）指标，用于量化潜在空间的平滑性和解缠结性。
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) — Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang (2018)
  Journal: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); DOI: [10.48550/arXiv.1801.03924](https://doi.org/10.48550/arXiv.1801.03924)
  提出了LPIPS，一种学习型感知度量，常用于PPL计算，相比传统像素级差异更能与人类感知对齐。

---

[上一节](03-%E5%88%86%E5%B8%83%E5%BA%A6%E9%87%8F%EF%BC%9A%E6%A0%B8Inception%E8%B7%9D%E7%A6%BB%20%28KID%29.md) · [下一节](05-%E5%AE%9A%E6%80%A7%E8%AF%84%E4%BC%B0%E6%96%B9%E6%B3%95.md)

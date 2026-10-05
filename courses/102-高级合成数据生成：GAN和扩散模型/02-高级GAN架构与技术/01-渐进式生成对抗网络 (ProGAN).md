# 渐进式生成对抗网络 (ProGAN)

来源：[原文](https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-2-advanced-gan-architectures-techniques/progressive-growing-gans)

[返回章节目录](README.md) · [返回课程目录](../README.md)

直接使用标准生成对抗网络 (GAN) 生成 1024x1024 像素等高分辨率图像，会带来显著的训练难题。随着网络深度和图像尺寸的增加，梯度可能变得不稳定，导致发散或收敛缓慢。此外，生成器可能难以从一开始就同时学习粗略结构和精细细节。由 Karras 等人（NVIDIA）提出的渐进式生成对抗网络 (ProGAN) 方法，通过在训练过程中逐步提高任务复杂度，提供了一种巧妙的解决方案。

ProGAN 的主要思路是，首先使用极低分辨率图像（例如 4x4 像素）开始训练生成器 (G) 和判别器 (D)，然后逐渐向两个网络添加新层，以处理更高分辨率（8x8、16x16，…，直到 1024x1024 或更高）。这种递增的方法使网络能够首先在低分辨率下学习图像分布的大尺度结构，随后将重心转向更精细的细节，随着分辨率的提高而调整。

### 增长过程

训练从一个简单的生成器和判别器对开始，它们处理低分辨率图像（例如 4x4）。一旦这对初始网络显示出收敛和稳定的迹象，就会为 G 和 D 添加新层，以使图像分辨率翻倍（例如，从 4x4 变为 8x8）。

重要的是，这种转换并非突然。新添加的层会在一段训练迭代期内平滑地“淡入”。这是通过使用参数 (parameter) $\alpha$ 实现的，其范围从 0 到 1。当添加新的层块（处理更高分辨率）时：

1. **在生成器中：** 先前分辨率块的输出进行上采样，并计算新块的输出（该输出也以上采样特征作为输入）。最终输出是加权组合：$(\text{新输出}) \times \alpha + (\text{上采样旧输出}) \times (1 - \alpha)$
2. **在判别器中：** 输入图像由新层处理，结果被下采样。输入图像也会直接下采样并由先前分辨率块处理。判别器使用这两条路径的加权组合：$(\text{来自新层}) \times \alpha + (\text{来自下采样后旧层}) \times (1 - \alpha)$

参数 $\alpha$ 逐渐从 0 增加到 1。最初（$\alpha=0$），新层没有影响，网络像在先前分辨率下一样运行。随着 $\alpha$ 的增加，新层的贡献增加，使网络能够平稳适应更高分辨率任务。一旦 $\alpha=1$，旧的连接路径被有效地移除，网络完全以新的、更高分辨率运行。此过程会随着后续分辨率的提高而重复。

> 图示 ProGAN 训练阶段。最初（阶段 1），G 和 D 在低分辨率（4x4）下运行。在阶段 2，添加用于 8x8 分辨率的新层，并使用参数 $\alpha$ 淡入。在阶段 3，淡入完成后（$\alpha=1$），8x8 网络得到稳定训练。此过程会随着更高分辨率的提升而重复。

### 支持稳定性和质量的技术

ProGAN 采用了一些额外技术，以进一步提升训练稳定性和图像质量，尤其是在更高分辨率下：

- **小批量标准差：** 为了应对模式崩溃（即生成器只生成种类有限的样本），通常在判别器末端添加一个小批量标准差层。该层计算当前小批量中所有空间位置和样本的特征标准差。然后将此统计数据平均并复制到一个额外的特征图中，该特征图与原始特征拼接。通过向判别器提供关于批量统计的信息，如果生成器产生变化异常低的批量，判别器可以隐式地对其进行惩罚，从而鼓励样本多样性。
- **均衡学习率：** 标准的权重 (weight)初始化方案（如 Xavier 或 He）并非总能阻止梯度在非常深的神经网络 (neural network)中爆炸或消失，尤其是在激活函数 (activation function)或架构不同时。ProGAN 采用*均衡学习率*，这是一种动态权重缩放形式。在运行时，在每次通过卷积层或全连接层的前向或反向传播 (backpropagation)之前，权重 $w_i$ 会按因子 $c$ 进行缩放：
  $\hat{w}_i = w_i / c$
  这里，$c$ 是一个逐层归一化 (normalization)常数，通常根据 He 初始化器的原理计算：$c = \sqrt{\frac{2}{\text{扇入}}}$，其中 $\text{扇入}$ 是层的输入连接数。这种缩放确保了输出（和梯度）的方差在各层之间大致保持不变，无论连接数或参数 (parameter)尺度如何，有效地均衡了所有权重的学习速度。
- **像素归一化：** 应用于生成器中每个卷积层之后（激活函数之前）。此技术将每个像素 $(x, y)$ 处的特征向量 (vector)归一化为单位长度：
  $b_{x,y} = \frac{a_{x,y}}{\sqrt{\frac{1}{N} \sum_{j=0}^{N-1} (a_{x,y}^j)^2 + \epsilon}}$
  这里，$N$ 是特征通道数，$a_{x,y}$ 是像素 $(x, y)$ 处的原始特征向量，$b_{x,y}$ 是归一化向量，$\epsilon$（例如 $10^{-8}$）可防止除以零。这种局部响应归一化与批量归一化类似，但不依赖于批量统计。它能防止生成器内部的信号幅度因生成器和判别器之间可能的竞争而失控，极大促进了训练稳定性。

### 优点与考量

渐进式增长的方法带来几项优点：

- **稳定的高分辨率合成：** 它使高分辨率（1024x1024）的训练过程稳定性大幅提高，这在以前非常困难。
- **更快的初始训练：** 早期阶段侧重于低分辨率图像，计算需求较低，使网络能够快速习得粗略特征。
- **提升的样本质量：** 逐步细化过程通常会产生比从头训练大型网络更高保真度的生成图像。

但也有一些需要考量之处：

- **训练时间增加：** 尽管单个阶段初始可能较快，但所有阶段的总训练时间可能相当可观。
- **架构复杂性：** 与静态架构相比，设计和实现平滑淡入的层增加了复杂性。
- **超参数 (parameter) (hyperparameter)敏感性：** 淡入时间表（$\alpha$ 进展）和分辨率转换的时机需要仔细调整。

ProGAN 在生成高分辨率、高质量图像方面迈出了重要一步。它的渐进式训练和专业归一化 (normalization)技术的核心思路影响了后续的先进模型，包括我们将接下来进行研究的 StyleGAN 系列。

## 参考资料

- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) — Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.10196](https://doi.org/10.48550/arXiv.1710.10196)
  介绍渐进式GAN架构的原始论文，详细阐述了逐步提高分辨率和稳定性技术。
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  介绍生成对抗网络概念的奠基性论文，ProGAN在此基础上发展。
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) — Tero Karras, Samuli Laine, Timo Aila (2019)
  Journal: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); DOI: [10.48550/arXiv.1812.04948](https://doi.org/10.48550/arXiv.1812.04948)
  一项后续工作，通过引入基于风格的生成器来扩展ProGAN的思想，特别是其渐进式训练，以提高控制能力和图像质量。

---

[上一节](../01-%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E5%9B%9E%E9%A1%BE/05-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E5%8E%9F%E7%90%86%E4%BB%8B%E7%BB%8D.md) · [下一节](02-%E5%9F%BA%E4%BA%8E%E9%A3%8E%E6%A0%BC%E7%9A%84%E7%94%9F%E6%88%90%E5%99%A8%EF%BC%88StyleGAN%E5%8F%98%E4%BD%93%EF%BC%89.md)

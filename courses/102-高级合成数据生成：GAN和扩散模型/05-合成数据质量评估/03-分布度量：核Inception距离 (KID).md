# 分布度量：核Inception距离 (KID)

来源：[原文](https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-5-evaluating-synthetic-data-quality/distributional-metrics-kid)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管Inception分数 (IS) 和Fréchet Inception距离 (FID) 等度量通过比较从预训练 (pre-training)网络中提取的特征来提供信息，但它们也有局限性。例如，FID将提取的特征（通常来自Inception网络）建模为多元高斯分布，并比较它们的均值和协方差。这种假设可能并非总是成立，FID的估计值可能存在偏差，尤其是在小样本集上计算时。

为解决这些问题，核Inception距离 (KID) 提供了一种比较真实数据和生成数据特征分布的替代方法。KID不假设高斯性，而是采用最大均值差异 (MMD)，这是一种非参数 (parameter)统计检验，用于判断两组样本是否源自同一分布。

### 最大均值差异 (MMD)

本质上，MMD衡量的是两个分布在被称为再生核希尔伯特空间 (RKHS) 的高维特征空间中其均值嵌入 (embedding)之间的距离。直观理解是，如果两个分布相同，它们在该空间中的均值表示也将相同。均值嵌入的距离越远，MMD越大，这表示分布之间的差异越大。

两个分布$\mathbb{P}_r$（真实）和$\mathbb{P}_g$（生成）之间使用核函数$k$的MMD平方定义为：


$$
\text{MMD}^2(\mathbb{P}_r, \mathbb{P}_g) = \mathbb{E}_{x, x' \sim \mathbb{P}_r}[k(x, x')] - 2 \mathbb{E}_{x \sim \mathbb{P}_r, y \sim \mathbb{P}_g}[k(x, y)] + \mathbb{E}_{y, y' \sim \mathbb{P}_g}[k(y, y')]
$$


这里，$x$和$x'$是来自真实分布的样本，$y$和$y'$是来自生成分布的样本，$k(a, b)$是评估样本$a$和$b$之间相似性的核函数。

### 计算KID

KID专门使用MMD计算从Inception网络中提取的特征，这与FID的运行方式相似。典型步骤如下：

1. **特征提取：** 通过预训练 (pre-training)的Inception网络（通常到特定层，例如最终平均池化层）处理批次的真实图像 ($X$) 和生成图像 ($Y$)，以获得每张图像的特征向量 (vector)。
2. **MMD估计：** 使用提取的特征向量计算MMD平方的经验估计。对于批次$X = \{x_1, ..., x_n\}$和$Y = \{y_1, ..., y_m\}$，通常使用无偏经验估计。
3. **核选择：** KID通常采用多项式核，例如$k(a, b) = (\frac{1}{d} a^T b + 1)^3$，其中$d$是特征向量的维度。该核函数考虑了FID所使用的高阶统计量。
4. **平均：** 计算通常在数据的多个随机子集（分割）上重复进行，然后将结果平均以获得更稳定的KID估计值。

最终的KID值通常报告为MMD平方估计值，有时会乘以一个比例因子（例如100）。与FID一样，较低的KID值表示生成图像特征的分布更接近真实图像特征的分布，这表明更高的质量和多样性。

### KID的优点

1. **无偏估计器：** MMD提供分布距离的无偏估计，这在处理较小样本量时特别有利，因为此时FID估计可能存在噪声和偏差。
2. **无高斯假设：** KID不假设Inception特征服从高斯分布，这使得当该假设被违反时，它可能更可靠。
3. **灵敏性：** 由于使用多项式核等核函数，它能够捕捉高阶统计量中的差异。

### 实际考量

- **计算成本：** 计算KID可能比FID的计算量更大，特别是MMD部分，它涉及成对核评估（对$n$个样本而言，朴素的复杂度为$O(n^2)$，尽管存在线性时间近似）。
- **实现：** 需要仔细实现MMD估计器并获取Inception特征。标准实现可在诸如`torch_fidelity`之类的库中获得，也可以使用科学计算库构建。
- **解释：** 与FID一样，KID是一个相对度量。其绝对值不如其在不同模型或训练检查点之间的比较具有信息量。使用不同特征提取器或核函数获得的KID值进行比较通常没有意义。

总之，KID作为一种评估生成模型的有力分布度量，尤其在数据有限时提供鲁棒性，并避免了FID中固有的高斯假设。它为真实数据和生成数据特征分布之间的相似性提供了一个补充视角。

## 参考资料

- [A Kernel Two-Sample Test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) — Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alex Smola (2012)
  Journal: Journal of Machine Learning Research; Publisher: JMLR.org; Volume: 13; Pages: 723-773; DOI: [10.5555/2246738.2246797](https://doi.org/10.5555/2246738.2246797)
  介绍了最大均值差异（MMD），这是一种用于比较两个分布的非参数检验，是KID的核心组成部分。

---

[上一节](02-%E5%AE%9A%E9%87%8F%E6%8C%87%E6%A0%87%EF%BC%9AIS%E3%80%81FID%E3%80%81%E7%B2%BE%E7%A1%AE%E7%8E%87%E3%80%81%E5%8F%AC%E5%9B%9E%E7%8E%87.md) · [下一节](04-%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%EF%BC%88GAN%EF%BC%89%E7%9A%84%E6%84%9F%E7%9F%A5%E8%B7%AF%E5%BE%84%E9%95%BF%E5%BA%A6%EF%BC%88PPL%EF%BC%89.md)

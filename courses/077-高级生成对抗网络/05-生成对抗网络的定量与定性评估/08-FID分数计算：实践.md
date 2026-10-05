# FID分数计算：实践

来源：[原文](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans/calculating-fid-score-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Fréchet Inception Distance (FID) 的计算是评估GANs的一个主要实践方面。虽然FID的概念在于比较特征空间中的分布，但实际计算需要涉及预训练 (pre-training)模型和统计计算的具体步骤。这里提供计算FID分数的实践指南。

### FID计算流程

FID分数的计算涉及比较Inception V3模型为真实图像集和生成图像集生成的激活值统计量。核心步骤如下：

1. **准备数据集：** 从目标领域收集具有代表性的一组真实图像，并使用训练好的生成器生成一组生成图像。常见做法是每组至少使用10,000张图像，尽管通常更倾向于使用50,000张以获得更稳定的分数。确保两组图像都进行相同的预处理。
2. **加载预训练 (pre-training)的Inception V3：** 使用在ImageNet数据集上预训练的Inception V3模型。具体来说，您需要使用模型到某个激活层，通常是最终的平均池化层（输出2048个特征）。
3. **提取激活值：** 将真实图像和生成图像都输入到（截断的）Inception V3模型中。收集所选层为两组中每张图像生成的2048维激活向量 (vector)。
4. **计算统计量：** 计算真实图像激活值（$\mu_r, \Sigma_r$）和生成图像激活值（$\mu_g, \Sigma_g$）的均值向量（$\mu$）和协方差矩阵（$\Sigma$）。
5. **计算FID：** 使用计算出的统计量应用Fréchet距离公式。

FID分数是使用以下公式计算的：


$$
FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$


其中：

- $\mu_r$ 和 $\mu_g$ 分别是真实图像和生成图像激活值的均值向量。
- $\Sigma_r$ 和 $\Sigma_g$ 是真实图像和生成图像激活值的协方差矩阵。
- $||\cdot||^2$ 表示平方欧几里得范数（平方差之和）。
- $Tr(\cdot)$ 是矩阵的迹（对角线元素之和）。
- $(\Sigma_r \Sigma_g)^{1/2}$ 是协方差矩阵乘积的矩阵平方根。

> Fréchet Inception Distance (FID) 的计算流程图。

### 实现细节

虽然您可以从头开始使用深度学习 (deep learning)框架（如PyTorch或TensorFlow）加载Inception V3并使用NumPy/SciPy进行统计计算来实现此过程，但许多维护良好的库大大简化了这一点。诸如`pytorch-fid`或`tensorflow-gan` (TFGAN) 之类的库通常提供方便的命令行工具或函数。

让我们概述使用NumPy和SciPy的核心计算步骤，假设您已经获得了`act_r`（真实图像的）和`act_g`（生成图像的）激活数组，每个数组的形状均为`(N, 2048)`，`N代表图像数量`。

```python
import numpy as np
from scipy.linalg import sqrtm

# 假设 act_r 和 act_g 是形状为 (N, 2048) 的 NumPy 数组
# 包含真实图像和生成图像的 Inception 激活值。

# 1. 计算均值和协方差
mu_r = np.mean(act_r, axis=0)
mu_g = np.mean(act_g, axis=0)

sigma_r = np.cov(act_r, rowvar=False)
sigma_g = np.cov(act_g, rowvar=False)

# 2. 计算均值之间的平方欧几里得距离
diff_mean_sq = np.sum((mu_r - mu_g)**2)

# 3. 计算乘积的矩阵平方根
# 如果需要，添加小的单位矩阵以提高数值稳定性
epsilon = 1e-6 
covmean_sqrt, _ = sqrtm(sigma_r.dot(sigma_g) + epsilon * np.eye(sigma_r.shape[0]), disp=False)

# 检查是否存在复数（表示数值不稳定）
if np.iscomplexobj(covmean_sqrt):
    print("警告：矩阵平方根中生成了复数。使用实部。")
    covmean_sqrt = covmean_sqrt.real

# 4. 计算迹项
trace_term = np.trace(sigma_r + sigma_g - 2.0 * covmean_sqrt)

# 5. 计算FID
fid_score = diff_mean_sq + trace_term

print(f"计算出的FID分数: {fid_score}")
```

### 实际考量

- **样本数量：** 使用过少的样本（例如，少于10,000个）会导致FID分数不稳定和不可靠。特别是协方差矩阵的估计，需要足够的数据。
- **预处理：** 务必确保真实图像和生成图像在输入到Inception V3之前进行 *相同* 的预处理。这通常包括将图像大小调整为299x299像素，并根据预训练 (pre-training)Inception模型的要求归一化 (normalization)像素值（通常缩放到`[-1, 1]`或`[0, 1]`，并可能减去ImageNet均值）。任何差异都会使比较失效。
- **预计算统计量：** 对于CIFAR-10、CelebA或LSUN等标准数据集，真实图像的均值和协方差统计量（$\mu_r, \Sigma_r$）通常会预先计算并提供。使用这些标准统计量可以确保不同研究论文之间的可比性，前提是您使用相同的Inception V3检查点和预处理。
- **数值稳定性：** 计算矩阵平方根$(\Sigma_r \Sigma_g)^{1/2}$有时可能数值不稳定，特别是当协方差矩阵是病态的（例如，接近奇异，这可能在样本不足或特征方差低时发生）。在计算平方根之前添加一个小的单位矩阵倍数（$epsilon * I$），如示例代码所示，可以帮助缓解此问题。始终检查结果是否包含复数，这提示着可能存在问题。
- **库的使用：** 通常建议使用已有的库。它们处理细节，例如正确加载特定的Inception V3权重 (weight)、执行适当的截断、处理批处理以提高内存效率，以及纳入数值稳定性修复。

遵循这些步骤和考量，您可以可靠地计算FID分数，这提供了一个有价值的定量指标，用于比较您的GAN与基线的表现或追踪训练过程中的改进。请记住，FID主要衡量生成分布在Inception特征空间中与真实分布的相似度，包含样本逼真度和多样性两方面。

## 参考资料

- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://doi.org/10.48550/arXiv.1706.08500) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Volume: 30; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  提出了Fréchet Inception Distance（FID）作为评估生成图像质量的通用指标。
- [pytorch-fid: FID Score for PyTorch](https://github.com/mseitzer/pytorch-fid) — Maximilian Seitzer (2020)
  使用PyTorch计算FID分数的官方实现和代码库，在实践中被广泛采用。
- [Functions of Matrices: Theory and Computation](https://doi.org/10.1137/1.9780898717778) — Nicholas J. Higham (2008)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); DOI: [10.1137/1.9780898717778](https://doi.org/10.1137/1.9780898717778)
  一本关于矩阵函数的参考书，为理解FID计算中矩阵平方根的数值方面提供了扎实的数学基础。

---

[上一节](07-%E6%84%9F%E7%9F%A5%E8%B7%AF%E5%BE%84%E9%95%BF%E5%BA%A6%20%28PPL%29.md) · [下一节](../06-GANs%EF%BC%9A%E4%B8%8D%E5%8F%AA%E7%94%9F%E6%88%90%E5%9B%BE%E5%83%8F/01-%E7%A6%BB%E6%95%A3%E6%95%B0%E6%8D%AE%E5%B8%A6%E6%9D%A5%E7%9A%84%E9%9A%BE%E9%A2%98%EF%BC%9A%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90.md)

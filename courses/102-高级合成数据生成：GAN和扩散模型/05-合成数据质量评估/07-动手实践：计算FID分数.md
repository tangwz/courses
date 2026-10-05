# 动手实践：计算FID分数

来源：[原文](https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-5-evaluating-synthetic-data-quality/practice-calculating-fid)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Fréchet Inception Distance (FID) 是评估生成模型时广为采用的指标之一。FID通过比较从真实图像中提取的特征分布与从合成图像中提取的特征分布，来评估生成图像的质量和多样性。较低的FID分数表示生成图像的分布更接近真实图像的分布，说明其具有更高的保真度和多样性。

此计算依赖于使用预训练 (pre-training)的Inception-v3网络提取的特征表示。我们将使用PyTorch和`torchvision`进行本次实践。

### FID计算的主要步骤

计算FID分数涉及以下主要步骤：

1. **特征提取：** 将一组真实图像和一组生成图像都通过预训练 (pre-training)的Inception-v3网络进行处理（通常处理到倒数第二层，即最终分类层之前）。这会为每张图像获得高维特征向量 (vector)。
2. **分布建模：** 假定真实数据集 ($X$) 和生成数据集 ($G$) 的提取特征均服从多元高斯分布。计算每个特征集的均值向量 ($\mu_x$，$\mu_g$) 和协方差矩阵 ($\Sigma_x$，$\Sigma_g$)。
3. **Fréchet距离计算：** 计算这两个高斯分布 ($N(\mu_x, \Sigma_x)$ 和 $N(\mu_g, \Sigma_g)$) 之间的Fréchet距离。

FID公式如下所示：


$$
FID(x, g) = ||\mu_x - \mu_g||^2_2 + \text{迹}(\Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2})
$$


这里，$||\mu_x - \mu_g||^2_2$ 是均值向量之间的欧几里得距离的平方，而 $\text{迹}$ 表示矩阵的迹。项 $(\Sigma_x \Sigma_g)^{1/2}$ 代表协方差矩阵乘积的矩阵平方根。

### 使用PyTorch实现

我们来逐步实现FID计算。请确保您已安装 `torch`、`torchvision`、`numpy` 和 `scipy`。

#### 1. 加载Inception-v3模型

我们需要在ImageNet上预训练 (pre-training)的Inception-v3模型。`torchvision` 轻松提供了这一点。我们将对其进行微小修改，使其从中间层输出特征，而不是分类logit。`FIDInceptionA` 块（输出维度2048）常被使用。

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from scipy.linalg import sqrtm

# 加载预训练的Inception v3模型
inception_model = models.inception_v3(pretrained=True, transform_input=False)

# 修改模型，使其从所需层输出特征
# 我们将使用最后一个池化层（FIDInceptionA）的输出
class InceptionV3FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.inception_v3(pretrained=True, aux_logits=False)
        # 移除最终的全连接层
        self.model.fc = torch.nn.Identity() 
        # 确保输入转换符合Inception V3的要求
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        # 应用转换
        # 注意：如果输入已是张量批次，请单独应用转换
        # 或确保DataLoader处理转换。
        # 为简单起见，假设x是PIL图像的批次或需要转换。
        # 如果x已经是张量批次[N, C, H, W]，如果之前已完成，则在此处跳过转换。

        # 处理可能的输入类型（PIL图像与张量）
        if not isinstance(x, torch.Tensor):
             # 假设x是PIL图像的列表/批次
             x = torch.stack([self.transform(img) for img in x])
        elif x.shape[2] != 299 or x.shape[3] != 299:
             # 如果是张量但尺寸不正确，应用转换（根据需要调整）
             # 这部分可能需要根据输入管道进行仔细处理
             x = torch.stack([transforms.ToPILImage()(img) for img in x]) # 转换为PIL
             x = torch.stack([self.transform(img) for img in x])      # 应用完整转换

        # 将输入通过Inception模型
        # 确保模型处于评估模式
        self.model.eval() 
        with torch.no_grad():
            features = self.model(x)
            # 输出可能需要根据具体的InceptionV3用法进行重塑
            # 原始FID实现可能使用特定的池化层。
            # 对于`self.model.fc = torch.nn.Identity()`，输出应直接是[N, 2048]。
            # 如果在最终池化之前使用特征，可能需要自适应池化：
            # features = adaptive_avg_pool2d(features, (1, 1))
            # features = features.view(features.shape[0], -1) 
        return features

# 实例化特征提取器
fid_extractor = InceptionV3FeatureExtractor()
# 如果GPU可用，移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fid_extractor.to(device)
fid_extractor.eval() # 设置为评估模式

print("InceptionV3模型已加载用于特征提取。")
```

*重要提示：* Inception-v3模型要求输入图像大小为 $299 \times 299$，并进行特定归一化 (normalization)。请确保您的数据加载管道正确预处理真实图像和生成图像。示例代码包含一个基本转换，但您可能需要根据数据存储和加载方式（例如，使用PyTorch `DataLoader`）对其进行调整。

#### 2. 提取特征

现在，编写一个函数来遍历您的数据集（真实和生成），使用准备好的模型提取特征，并将其收集起来。

```python
def get_features(dataloader, model, device, max_samples=None):
    features = []
    count = 0
    for batch in dataloader:
        # 假设dataloader生成图像张量批次
        # 将批次移动到合适的设备
        if isinstance(batch, (list, tuple)): # 处理 (图像, 标签) 等情况
            images = batch[0].to(device)
        else:
            images = batch.to(device) # 假设批次仅包含图像

        # 如果DataLoader中未进行转换，则在此处应用
        # 确保图像已为模型正确格式化（例如，大小299x299，已归一化）
        # 上述InceptionV3FeatureExtractor包含一个基本转换，
        # 但为了效率，通常最好在DataLoader中处理。
        # 如果在模型内部使用转换，请传入PIL图像或原始张量。

        batch_features = model(images).detach().cpu().numpy()
        features.append(batch_features)

        count += images.shape[0]
        if max_samples is not None and count >= max_samples:
            break

    features = np.concatenate(features, axis=0)
    if max_samples is not None:
        features = features[:max_samples]
    return features

# 示例用法（请替换为您的实际DataLoader）
# 假设real_dataloader和fake_dataloader是PyTorch DataLoader
# 它们生成批次，包含正确预处理的尺寸为(N, 3, 299, 299)的图像张量

# print("正在提取真实图像的特征...")
# real_features = get_features(real_dataloader, fid_extractor, device, max_samples=10000)
# print(f"已提取 {real_features.shape[0]} 个真实特征。")

# print("正在提取生成图像的特征...")
# fake_features = get_features(fake_dataloader, fid_extractor, device, max_samples=10000)
# print(f"已提取 {fake_features.shape[0]} 个生成特征。")

# 为演示目的，我们创建虚拟特征：
feature_dim = 2048
num_samples = 10000 
print(f"正在生成虚拟特征（{num_samples}个样本，维度={feature_dim}）。..")
real_features = np.random.rand(num_samples, feature_dim) 
# 使生成特征略有不同，以便FID不为零
fake_features = np.random.rand(num_samples, feature_dim) * 1.1 + 0.1 
print("虚拟特征已生成。")
```

为了获得可靠的FID分数，建议从真实分布和生成分布中都使用大量的样本，通常是10,000或50,000个。

#### 3. 计算统计数据和FID

提取特征后，计算均值和协方差矩阵，然后将它们代入FID公式。

```python
def calculate_fid(features1, features2):
    # 计算均值和协方差统计量
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    # 计算均值之间的平方差
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # 计算协方差乘积的平方根
    # 如有需要，添加一个小的epsilon以提高数值稳定性
    eps = 1e-6
    covmean_sqrt, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    # 检查并修正矩阵平方根中的虚数
    if np.iscomplexobj(covmean_sqrt):
        # print("警告：sqrtm中遇到复数。使用实部。")
        covmean_sqrt = covmean_sqrt.real

    # 计算分数
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean_sqrt)

    # 处理由于数值不稳定导致的潜在负值
    if fid < 0:
        # print(f"警告：检测到负FID（{fid}）。截断为0。")
        fid = 0.0

    return fid

# 使用提取的（或虚拟的）特征计算FID
print("正在计算FID分数...")
fid_score = calculate_fid(real_features, fake_features)

print(f"计算得到的FID分数：{fid_score:.4f}") 
# 虚拟FID示例：会根据随机数而变化，可能较大。
# 实际的FID计算可能得到5.0、10.0、50.0等值。越低越好。
```

`scipy.linalg.sqrtm` 函数计算矩阵的平方根。请注意处理因数值精度问题可能出现的复数；在这种情况下，我们取实部。由于浮点误差，特别是当分布非常接近时，也可能出现小的负FID值；这些值通常被截断为零。

### FID分数解读

- **越低越好：** 完美的分数为0，表示真实和生成特征的分布完全相同。较低的分数表明对齐 (alignment)效果更好。
- **相对比较：** FID最适用于在*相同*的真实数据集上比较不同的生成模型或同一模型的不同训练阶段。绝对FID值在很大程度上取决于数据集和具体的实现细节。
- **敏感性：** FID对所使用的样本数量敏感。为了公平比较，始终使用相同数量的真实和生成样本，并在分数旁边报告此数量（例如，FID-10k，FID-50k）。
- **局限性：** 尽管功能强劲，但FID未能涵盖图像质量的所有方面。它主要基于Inception特征来衡量分布相似性。模型若对训练集过拟合 (overfitting)，或生成未大幅改变特征统计量的伪影，则可能使FID失效。务必将FID与定性目视检查和可能存在的其他指标结合使用。

本次实践提供了计算您自己的生成模型的FID分数的工具。使用FID等指标进行一致且严谨的评估，对于跟踪进度和比较不同GAN与扩散模型体系结构及训练策略的性能来说非常重要。

## 参考资料

- [mseitzer/pytorch-fid: FID Score for PyTorch](https://github.com/mseitzer/pytorch-fid) — Markus Seitzer and contributors (2024)
  一个官方且广泛使用的PyTorch FID指标实现，作为进行可靠计算的实践指南和参考，包括Inception-v3集成的最佳实践。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zack C. Lipton, Mu Li, Alex Smola (2024)
  Publisher: Cambridge University Press
  一本内容全面的在线教科书，涵盖了GAN等生成模型，并提供了各种评估指标（包括FID）的背景信息，以便理解合成数据生成的理论和实践方面。

---

[上一节](06-%E8%AF%84%E4%BC%B0%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B.md) · [下一节](../06-%E9%AB%98%E7%BA%A7%E5%BA%94%E7%94%A8%E4%B8%8E%E6%95%B4%E5%90%88/01-%E9%AB%98%E5%88%86%E8%BE%A8%E7%8E%87%E7%94%9F%E6%88%90%E7%AD%96%E7%95%A5.md)

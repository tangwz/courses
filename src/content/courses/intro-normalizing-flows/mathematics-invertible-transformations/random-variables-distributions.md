---
course: "intro-normalizing-flows"
chapter: "mathematics-invertible-transformations"
lesson: "random-variables-distributions"
sourceId: 8253
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-1-mathematics-invertible-transformations/random-variables-distributions"
title: "随机变量与分布"
description: "理解生成建模中连续随机变量和基概率分布的基础知识。"
order: 1
plots: ["plots/8253-0.json"]
sourceHash: "d957d98cb08243463120702b046b1999374a22c522b53c2398ed467fab99561f"
sourceCorrections: []
---

生成建模需要扎实的数学基础来表示数据。在构建标准化流时，我们将数据集视为从底层概率分布中提取的样本集合。为了处理这些分布并构建能够生成新数据的模型，我们必须首先确定连续随机变量的规则。

随机变量为随机过程的结果分配一个数值。在本课程中，我们专注于连续随机变量。与取特定孤立值的离散随机变量不同，连续变量在给定范围内可以取无限数量的可能值。由于可能性空间是无限的，连续随机变量取一个精确且高度特定的值的概率实际上为零。

我们不使用单个点的绝对概率，而是使用概率密度函数（PDF）来描述连续变量。设连续随机变量为 $x$。其概率密度函数写为 $p(x)$。一个有效的 PDF 必须满足两个特定的数学条件。首先，它必须在任何地方都是非负的，即 $p(x) \ge 0$。其次，密度曲线下的总面积在整个定义域内的积分必须恰好为 1：

$\int_{-\infty}^{\infty} p(x) dx = 1$

将 $p(x)$ 误认为直接概率是一个常见的错误。概率密度函数在特定点的值完全可以超过 1。密度仅表示值落在 $x$ 周围微小区域内的相对可能性。

标准化流通过在两个不同的数学空间之间移动数据来运行。第一个是隐空间（latent space），由基分布（base distribution）填充。我们将这个隐空间中的随机变量表示为 $z$。基分布（数学上表示为 $p_Z(z)$）被特意选择为简单的分布。我们需要一个采样成本低且具有已知概率密度函数（我们可以精确计算）的分布。

在几乎所有的标准化流实现中，标准多元高斯分布都充当基分布。标准高斯分布具有零均值向量 (vector)和单位协方差矩阵。我们使用它是因为它的性质高度可预测，即使在数千个维度中也可以高效地计算其密度。

第二个空间是数据空间。这个空间包含我们希望模型生成的实际可观测数据，例如图像的像素值或分子的空间坐标。我们将数据空间中的随机变量表示为 $x$，其对应的目标分布表示为 $p_X(x)$。

目标分布本质上是复杂的、高度相关的且具有多峰性。像人类语音波形这样的数据集的底层结构无法用简单的线性高斯方程来描述。标准化流模型的目标是学习一种可逆的数学变换，将简单的基分布 $p_Z(z)$ 重塑为复杂的目标分布 $p_X(x)$。



![简单基分布与复杂目标分布对比](plots/8253-0.json)



> 左图显示了在隐空间中评估的简单单峰基分布。右图代表了数据空间中的复杂目标分布，其特征是存在多个峰值和谷值。

要在代码中实现基分布，我们依赖标准的机器学习 (machine learning)框架。PyTorch 在 `torch.distributions` 模块中提供了一套完整的统计分布。由于大多数真实数据集具有多个维度，我们首先定义一个连续多元正态分布。

```python
import torch
from torch.distributions import MultivariateNormal

# 定义隐空间的维度
dimensions = 2

# 均值向量（零）和协方差矩阵（单位阵）
loc = torch.zeros(dimensions)
covariance_matrix = torch.eye(dimensions)

# 初始化基分布 p_Z(z)
base_distribution = MultivariateNormal(loc, covariance_matrix)

# 采样一组包含 5 个隐变量的批次
z_samples = base_distribution.sample((5,))
print("样本 z:\n", z_samples)

# 计算样本的对数概率密度
log_probs = base_distribution.log_prob(z_samples)
print("对数概率:\n", log_probs)
```

请注意，我们使用 `log_prob` 而不是计算原始概率密度。在生成建模中，将许多小的概率值相乘会迅速导致数值下溢，即计算机将极小的浮点数四舍五入为零。通过在对数空间中操作，我们将乘法转换为加法，这为训练深度神经网络 (neural network)提供了所需的数值稳定性。此方法返回的值正是 $\log p_Z(z)$，这是我们在通过最大似然估计优化流模型时会大量使用的一个指标。

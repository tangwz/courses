---
course: "cnns-for-computer-vision"
chapter: "advanced-training-optimization"
lesson: "batch-norm-alternatives"
sourceId: 2540
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-2-advanced-training-optimization/batch-norm-alternatives"
title: "批量归一化内部运作及替代方案"
description: "详细解析批量归一化的工作原理，并查看层归一化和组归一化等替代方案。"
order: 4
plots: []
sourceHash: "39d118b2eeaacde9af8ae47bcc44fb46874b7217f11538f8989f038c178aa70a"
sourceCorrections: []
---

先进且复杂的卷积神经网络 (neural network) (CNN)的开发，使得确保训练稳定高效变得日益重要。一个主要难题是这一现象有时被称为*内部协变量偏移*，即在训练期间，随着前一层参数 (parameter)的更新，中间层激活的分布会发生变化。尽管内部协变量偏移稳定化对其成功的具体贡献存在争议，但批量归一化 (normalization)（BN）作为一种高效技术出现，可加速训练，允许更高的学习率，并提供轻微的正则化 (regularization)效果。然而，BN并非万能药。了解其内部运作方式以及分析层归一化、实例归一化和组归一化等替代方案，对于处理各种训练情况来说很必要。

### 批量归一化 (normalization)机制

其核心在于，批量归一化通过减去mini-batch均值并除以mini-batch标准差来归一化前一个激活层的输出。这对于每个特征通道独立进行。重要的一点是，它会应用可学习的缩放（$\gamma$）和偏移（$\beta$）参数 (parameter)，允许网络在有利于表示学习的情况下恢复原始激活。

我们来看看大小为$m$的mini-batch $B = \{x_1, ..., x_m\}$ 的激活值$x$。对于特定的特征通道，训练期间的BN转换包含以下步骤：

1. **计算Mini-Batch均值：**
   $\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$
2. **计算Mini-Batch方差：**
   $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$
3. **归一化：**
   $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
   这里，$\epsilon$是一个为数值稳定性而添加的小常数，以防止方差非常小导致除零。
4. **缩放和偏移：**
   $y_i = \gamma \hat{x}_i + \beta$
   参数$\gamma$（缩放）和$\beta$（偏移）通过反向传播 (backpropagation)与网络权重 (weight)一同学习。它们使网络能够控制归一化激活的均值和方差。如果网络学习到$\gamma = \sqrt{\sigma_B^2 + \epsilon}$和$\beta = \mu_B$，它就可以有效地恢复原始激活，确保BN不会不必要地限制网络的表示能力。

**训练与推理 (inference)：**

一个重要细节是训练和推理之间的差异。在训练期间，$\mu_B$和$\sigma_B^2$是按mini-batch计算的。然而，在推理期间，我们可能只处理单个样本，使得mini-batch统计数据失去意义或不可用。此外，我们希望模型在推理时对给定输入产生确定性输出。

为解决此问题，BN层在训练期间维护总体均值（$\mu_{pop}$）和方差（$\sigma_{pop}^2$）的运行估计值，通常使用指数移动平均：

$\mu_{pop} \leftarrow \text{动量} \times \mu_{pop} + (1 - \text{动量}) \times \mu_B$
$\sigma_{pop}^2 \leftarrow \text{动量} \times \sigma_{pop}^2 + (1 - \text{动量}) \times \sigma_B^2$

推理时，这些固定的总体统计数据$\mu_{pop}$和$\sigma_{pop}^2$用于归一化步骤，而不是mini-batch统计数据：

$\hat{x}_{inference} = \frac{x - \mu_{pop}}{\sqrt{\sigma_{pop}^2 + \epsilon}}$
$y_{inference} = \gamma \hat{x}_{inference} + \beta$

使用这些聚合统计数据可确保推理期间输出一致且具有确定性。

### 批量归一化 (normalization)的优点与局限性

BN具有多项优点：

- **改进梯度流：** 通过将激活保持在更稳定的范围内，BN有助于缓解梯度消失或爆炸问题，从而支持更深的网络。
- **更高的学习率：** 归一化效果使损失函数 (loss function)看起来更平滑，从而允许更大的学习率和更快的收敛。
- **正则化 (regularization)：** 使用mini-batch统计数据引入的噪声作为一种正则化形式，有时减少了对Dropout等其他技术的需求。
- **降低初始化敏感性：** 带有BN的网络对权重 (weight)初始化方法的选择不那么敏感。

然而，BN也有局限性：

- **批次大小依赖性：** 其有效性依赖于对mini-batch激活统计数据的准确估计。当批次大小非常小（例如，< 8）时，性能会显著下降，因为mini-batch统计数据变得嘈杂且不可靠地估计总体统计数据。这对于内存密集型任务（如训练大型分割模型或某些生成模型）来说是个问题。
- **训练/推理 (inference)差异：** 训练和推理期间使用不同统计数据有时会导致性能的细微差异。
- **计算开销：** BN每层引入额外的计算和参数 (parameter)（$\gamma$, $\beta$）。

### 批量归一化 (normalization)的替代方案

当BN的局限性变得难以承受时，有几种替代方案提供了不同的归一化策略：

#### 层归一化（LN）

层归一化不是对批次维度进行归一化，而是对批次中*每个单独样本*的特征（通道）维度进行归一化。

对于单个样本的输入特征向量 (vector)$x$（该样本的所有$C$个通道），LN计算如下：

1. **计算特征均值：**
   $\mu = \frac{1}{C} \sum_{i=1}^{C} x_i$
2. **计算特征方差：**
   $\sigma^2 = \frac{1}{C} \sum_{i=1}^{C} (x_i - \mu)^2$
3. **归一化：**
   $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
4. **缩放和偏移（可学习的$\gamma, \beta$）：**
   $y_i = \gamma \hat{x}_i + \beta$

LN与批次大小无关，使其适用于小批次和循环神经网络 (neural network)（RNN）的情况，因为在时间维度上应用BN会很麻烦。

#### 实例归一化（IN）

实例归一化在卷积层中将LN进一步发展。它独立地对批次中*每个通道*和*每个样本*的空间维度（高度$H$，宽度$W$）进行归一化。

对于单个样本$n$和单个通道$c$，IN计算如下：

1. **计算空间均值：**
   $\mu_{n,c} = \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n,c,h,w}$
2. **计算空间方差：**
   $\sigma_{n,c}^2 = \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{n,c,h,w} - \mu_{n,c})^2$
3. **归一化：**
   $\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}}$
4. **缩放和偏移（每个通道可学习的$\gamma_c, \beta_c$）：**
   $y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w} + \beta_c$

IN也与批次大小无关。它在风格迁移任务中特别有效，因为它从特征图中移除了特定实例的对比信息，专注于风格元素。

#### 组归一化（GN）

组归一化是LN和IN之间的一种折衷方案。它将通道分为预定义数量的组（$G$），并在每个组内对每个样本的空间维度执行归一化。

对于单个样本$n$和特定组$g$（包含$C/G$个通道）：

1. **计算组均值：**
   $\mu_{n,g} = \frac{1}{(C/G)HW} \sum_{c \in g} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n,c,h,w}$
2. **计算组方差：**
   $\sigma_{n,g}^2 = \frac{1}{(C/G)HW} \sum_{c \in g} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{n,c,h,w} - \mu_{n,g})^2$
3. **归一化（针对组$g$中的通道$c$）：**
   $\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,g}}{\sqrt{\sigma_{n,g}^2 + \epsilon}}$
4. **缩放和偏移（每个组可学习的$\gamma_g, \beta_g$，通常简化为每个通道）：**
   $y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w} + \beta_c$

GN与批次大小无关，通常能提供很好的平衡，即使在小批次下也能在许多视觉任务上实现接近BN的性能。其主要超参数 (parameter) (hyperparameter)是组数$G$。常见设置为$G=32$。如果$G=C$，GN就变成了IN。如果$G=1$，GN就变成了LN（假设归一化是在通道维度之后应用的）。

> 归一化技术比较，基于其对输入张量(N, C, H, W)的统计数据计算维度。BN依赖于批次大小(N)，而LN、IN和GN则按每个样本计算统计数据，因此它们与批次大小无关。

### 选择合适的归一化 (normalization)方法

BN与其替代方案之间的选择很大程度上取决于具体的应用和约束：

- **批次大时可用：** 当有足够的批次大小（例如，>= 16 或 32）可行时，BN通常仍然是许多标准CNN分类或检测任务的默认选择。
- **批次小时必要：** 当内存限制迫使使用小批次时，GN经常是BN的有力替代。LN也可以考虑。
- **循环网络：** LN通常更适合RNN。
- **生成模型（风格迁移）：** IN因其能够归一化特定实例的对比度而常被使用。
- **性能敏感性：** 通常需要进行实验。虽然GN在BN难以处理小批次时表现良好，但最佳选择（包括GN的组数）可能因数据集和架构而异。

了解这些归一化技术在设计和训练深度学习 (deep learning)模型时提供了更大的灵活性，尤其是在面临资源限制时或处理标准批量归一化可能失效的架构时。

## 参考资料

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://doi.org/10.5555/3045118.3045167) — Sergey Ioffe, Christian Szegedy (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 37; Pages: 448-456; DOI: [10.5555/3045118.3045167](https://doi.org/10.5555/3045118.3045167)
  介绍批量归一化及其机制，以及对深度网络训练的益处。
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (2016)
  Journal: ArXiv; Publisher: arXiv.org; Volume: abs/1607.06450; Pages: 3132-3140; DOI: [1607.06450](https://doi.org/1607.06450)
  提出层归一化作为批量归一化的替代方案，对循环神经网络尤其有效。
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022) — Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky (2017)
  Journal: arXiv preprint; Publisher: arXiv.org; Volume: 9906; Pages: 921-937; DOI: [10.1007/978-3-319-46475-6_57](https://doi.org/10.1007/978-3-319-46475-6_57)
  提出实例归一化并展示其在艺术风格迁移方面的效果。
- [Group Normalization](https://arxiv.org/abs/1803.08494) — Yuxin Wu, Kaiming He (2018)
  Journal: European Conference on Computer Vision (ECCV); Pages: 3-19; DOI: [10.48550/arXiv.1803.08494](https://doi.org/10.48550/arXiv.1803.08494)
  提出组归一化，解决批量归一化对小批量大小的敏感性问题。

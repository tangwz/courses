---
course: "advanced-diffusion-architectures"
chapter: "advanced-unet-architectures"
lesson: "unet-normalization-techniques"
sourceId: 5394
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-2-advanced-unet-architectures/unet-normalization-techniques"
title: "归一化方法（GroupNorm, AdaLN）"
description: "比较组归一化和自适应层归一化等方法，以实现稳定训练。"
order: 6
plots: []
sourceHash: "2eb4e3199f8798743bd104f50a174329223280b80dab12cfd8b14f59ff9847c2"
sourceCorrections: []
---

适当的归一化 (normalization)对于深度神经网络 (neural network)的稳定高效训练很重要，扩散模型也是如此。尽管批归一化（BatchNorm）在许多视觉任务中普遍使用，但它对批次统计数据的依赖可能给扩散模型带来问题，原因如下：

1. **小批次大小：** 训练大型扩散模型由于内存限制，常需要小批次大小，这使得批次统计数据噪声多且不可靠。
2. **变化的统计数据：** 在扩散过程中，不同时间步（$t$）的激活统计数据可能大幅变化，这可能违反BatchNorm的假设。

因此，在用于扩散的U-Net架构中，更倾向于使用独立于当前批次的替代归一化方法。两种主要方法是组归一化和自适应层归一化。

### 组归一化 (normalization)（GroupNorm）

组归一化（GroupNorm）介于层归一化（跨所有通道进行归一化）和实例归一化（独立地对每个通道进行归一化）之间。它通过将通道分成预定义数量的组，并在每组内计算均值和方差来进行归一化。

设$x$是形状为$(N, C, H, W)$的输入特征图，$N$代表批次大小，$C$是通道数，$H, W$则是空间维度。GroupNorm将$C$个通道分成$G$个组，每组包含$C/G$个通道（假设$C$能被$G$整除）。然后，对每个组沿着$(C/G, H, W)$维度独立进行归一化。

均值 $\mu_g$ 和标准差 $\sigma_g$ 对于组 $g$ 的计算如下：


$$
\mu_g = \frac{1}{(C/G)HW} \sum_{k \in S_g} \sum_{h=1}^H \sum_{w=1}^W x_{n,k,h,w}
$$


$$
\sigma_g = \sqrt{\frac{1}{(C/G)HW} \sum_{k \in S_g} \sum_{h=1}^H \sum_{w=1}^W (x_{n,k,h,w} - \mu_g)^2 + \epsilon}
$$


$S_g$ 是属于组 $g$ 的通道索引集，而 $\epsilon$ 是一个用于数值稳定的小常数。归一化后的输出 $\hat{x}$ 为：


$$
\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_g}{\sigma_g}
$$


当 $c$ 属于组 $g$ 时。最后，可学习的缩放（$\gamma$）和平移（$\beta$）参数 (parameter)被应用于每个通道：


$$
y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w} + \beta_c
$$


GroupNorm的主要优点是它独立于批次大小$N$。它的统计数据是独立地为每个样本在通道组和空间维度上计算的，使其在扩散模型训练中常见的小批次或波动批次大小下表现良好。它通常插入到残差块内，常在非线性激活函数 (activation function)之前。组数$G$是一个超参数 (hyperparameter)，通常设置为32等值。

> 组归一化的数据流，突出显示了在计算统计数据之前对通道进行分组。

### 自适应层归一化 (normalization)（AdaLN）

尽管GroupNorm提供了稳定的归一化，但扩散模型常受益于*条件*归一化，即归一化过程本身由条件信息调节，最重要的是扩散时间步$t$。自适应层归一化（AdaLN），常以AdaLN-Zero的形式使用，实现了这一点。

AdaLN建立在层归一化（LayerNorm）之上，后者对给定样本的所有通道的特征进行归一化。其核心思想是使可学习的缩放（$\gamma$）和平移（$\beta$）参数 (parameter)成为条件输入的函数，例如时间步嵌入 (embedding)$\mathbf{e}_t$。

标准层归一化应用：


$$
y = \gamma \frac{x - \mu}{\sigma} + \beta
$$


$\mu$ 和 $\sigma$ 是在所有通道 $C$ 上计算的（如果应用于特征图，也包括空间维度）。

在AdaLN中，缩放和平移参数是从条件嵌入中预测的：

1. 对输入特征$x$计算标准层归一化（或有时是组归一化）。
2. 处理条件嵌入$\mathbf{e}_t$（常在通过SiLU等非线性函数之后），使用线性层（或小型MLP）来预测调制参数，通常是独立的缩放和平移分量。我们称它们为$\gamma_t$和$\beta_t$。
3. 将这些预测的参数应用于归一化后的特征$\hat{x}$：
   
   $$
   y = \gamma_t \hat{x} + \beta_t
   $$
   

一种常见变体，尤其是在扩散模型中，是**AdaLN-Zero**。它初始化调制参数，使其在开始时没有效果，促进训练早期的稳定。应用调制的输出块常构建为：


$$
y = \hat{x} * (1 + \gamma_t) + \beta_t
$$


在这里，预测$\gamma_t$和$\beta_t$的线性层被初始化为输出零。因此，一开始$y = \hat{x}$，网络会随着训练的进行逐渐学习调制特征。

AdaLN（或AdaLN-Zero）在扩散U-Net中特别有效，因为它允许网络根据噪声水平（时间步$t$）调整其特征处理。它常用于残差块内，通常在主卷积和归一化（如GroupNorm）*之后*应用，专门控制残差分支的输出，然后将其加回到恒等连接。

> 自适应层归一化（AdaLN-Zero变体）的数据流。条件嵌入用于预测调制归一化特征的缩放和平移参数。

### 归一化 (normalization)的选择与组合

在高级扩散U-Net架构中：

- **GroupNorm** 常作为卷积块中的主要归一化层，因为它独立于批次并在视觉任务中表现良好。它在非线性层或注意力层之前稳定激活。
- **AdaLN（或AdaLN-Zero）** 经常专门用于引入条件信息（例如时间步$t$或类别嵌入 (embedding)）。它通常调制残差块的输出或需要条件控制的特定特征图。它有时可以替代GroupNorm或LayerNorm的标准可学习缩放/平移参数 (parameter)，或作为补充使用。

结合使用GroupNorm进行通用稳定性和AdaLN进行条件调制，为构建高性能扩散模型提供了一个强大的组合。具体的位置和配置（GroupNorm的组数、预测AdaLN参数的MLP架构）是可能影响模型性能和训练动态的设计选择。

## 参考资料

- [Cascaded Diffusion Models for High Fidelity Image Generation](https://jmlr.org/papers/v23/21-0524a.html) — Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, Tim Salimans (2022)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 23; Pages: 1-33
  这篇论文引入了AdaLN-Zero，一种条件归一化技术，显著提高了扩散模型的稳定性和性能，尤其是在级联架构中。
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (2016)
  Journal: arXiv preprint arXiv:1607.06450; DOI: [10.48550/arXiv.1607.06450](https://doi.org/10.48550/arXiv.1607.06450)
  引入层归一化的基础论文，该技术为每个样本独立计算统计数据，使其对不同批量大小具有鲁棒性，也是AdaLN等自适应归一化技术的前身。

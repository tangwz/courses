---
course: "intro-diffusion-models"
chapter: "conditional-generation-diffusion"
lesson: "classifier-guidance"
sourceId: 5438
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-6-conditional-generation-diffusion/classifier-guidance"
title: "分类器引导"
description: "使用单独的分类器模型，将扩散过程引导至所需类别。"
order: 2
plots: []
sourceHash: "f0e2867c33e361661f5e035110abe6842f66ba6081933bc9dcabaec6d49414a7"
sourceCorrections: []
---

无条件生成会生成代表整体训练数据的样本。然而，我们常常需要更具体的输出。设想一下，如果只想生成猫的图像，或者对应数字‘8’的数字图像，我们如何控制已经学习了分布 $p(x)$ 的扩散模型，使其从*条件*分布 $p(x|y)$ 生成样本呢？$y$ 代表所需条件（例如 $y$= '猫' 或 $y$= '8'）。

实现此目的的首批成功方法之一是**分类器引导**。其主要思路是借助一个单独的、预训练 (pre-training)的分类器模型，我们称之为 $p_\phi(y|x)$，其中 $\phi$ 代表分类器的参数 (parameter)。该分类器经过训练，能够根据输入 $x$ 预测类别标签 $y$。

然而，在扩散模型的逆向采样过程中，我们处理的不是干净数据 $x_0$，而是不同时间步 $t$ 的含噪中间样本 $x_t$。因此，为了使分类器引导有效，分类器 $p_\phi(y|x_t)$ 必须经过训练，即使从含噪输入 $x_t$ 中也能识别类别 $y$。这意味着分类器不仅要在原始数据集（如干净图像）上进行训练，还要在数据的含噪版本上进行训练，以与扩散过程中遇到的噪声水平保持一致。

这个分类器如何引导生成呢？回想一下，逆向过程旨在逼近 $p(x_{t-1}|x_t)$。我们希望修改此步骤，从 $p(x_{t-1}|x_t, y)$ 中进行采样。使用贝叶斯定理，我们可以写出：


$$
p(x_t | y) = \frac{p(y | x_t) p(x_t)}{p(y)}
$$


对 $x_t$ 取对数再求梯度：


$$
\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(y | x_t) + \nabla_{x_t} \log p(x_t)
$$


项 $\nabla_{x_t} \log p(x_t)$ 是时间 $t$ 处边缘数据分布的得分函数。扩散模型的噪声预测网络 $\epsilon_\theta(x_t, t)$ 经过训练，旨在逼近 $-\sigma_t \nabla_{x_t} \log p(x_t)$（按噪声水平缩放）。项 $\nabla_{x_t} \log p(y | x_t)$ 是根据分类器，所需类别 $y$ 对数似然的梯度，在当前含噪样本 $x_t$ 处评估。该梯度指出输入空间 ($x_t$) 中使样本在分类器看来更像类别 $y$ 的方向。

分类器引导通过加入该分类器梯度来修改采样步骤。具体而言，在计算逆向步骤 $p_\theta(x_{t-1}|x_t)$ 的均值 $\mu_\theta(x_t, t)$ 时，我们使用分类器的梯度对其进行扰动。DDPM 采样步骤中均值的更新规则可以调整如下：

原始预测均值源自预测噪声 $\epsilon_\theta$：


$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$


引导均值 $\hat{\mu}_\theta(x_t, t, y)$ 变为：


$$
\hat{\mu}_\theta(x_t, t, y) = \mu_\theta(x_t, t) + s \cdot \Sigma_t \nabla_{x_t} \log p_\phi(y|x_t)
$$


此处：

- $\Sigma_t$ 是逆向步骤 $p_\theta(x_{t-1}|x_t)$ 的协方差矩阵（方差），通常与 $\beta_t$ 相关联。
- $s$ 是**引导强度**，一个控制分类器影响强度的超参数 (hyperparameter)。较高的 $s$ 值会更强地推动生成朝向目标类别 $y$。
- $\nabla_{x_t} \log p_\phi(y|x_t)$ 是由分类器 $p_\phi$ 计算的梯度。

采样过程随后使用此调整后的均值 $\hat{\mu}_\theta$ 进行：

1. 使用 U-Net 预测噪声 $\epsilon_\theta(x_t, t)$。
2. 计算原始均值 $\mu_\theta(x_t, t)$。
3. 计算所需类别 $y$ 的分类器梯度 $\nabla_{x_t} \log p_\phi(y|x_t)$。
4. 使用引导强度 $s$ 计算引导均值 $\hat{\mu}_\theta(x_t, t, y)$。
5. 从 $\mathcal{N}(x_{t-1}; \hat{\mu}_\theta(x_t, t, y), \Sigma_t)$ 中采样 $x_{t-1}$。

此过程从 $t=T$ 到 $t=1$ 重复进行。

> 该图说明了单个逆向步骤中的分类器引导机制。U-Net 预测噪声，而一个单独的分类器则根据目标类别 $y$ 提供梯度。这些被组合起来，并按 $s$ 缩放，以生成用于采样 $x_{t-1}$ 的引导均值。

**分类器引导的优点：**

- **明确控制：** 提供了一种直接方式，可将生成引导至分类器训练的特定属性。
- **使用现有分类器：** 理论上可以使用现成的、训练有素的分类器（尽管它们需要具备噪声鲁棒性）。

**分类器引导的缺点：**

- **需要单独的分类器：** 需要训练并维护一个额外的模型 $p_\phi(y|x_t)$。
- **分类器训练：** 分类器必须能够应对扩散过程中遇到的噪声水平，增加了训练的复杂性。
- **计算成本：** 需要在每个采样步骤中对扩散模型和分类器都运行推理 (inference)。
- **引导强度调整：** 找到合适的引导强度 $s$ 通常需要进行实验。过低时，引导效果不佳；过高时，样本可能变得不真实，或受到分类器梯度利用的对抗性影响。

尽管有效，对一个单独的、噪声感知的分类器的需求促使研究人员开发出在没有这种外部依赖的情况下实现类似引导的方法。这引出了我们即将讨论的下一项技术：无分类器引导（CFG）。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) — Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 34; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  本文介绍了去噪扩散概率模型（DDPM），并提出了用于条件生成的分类器指导方法，详细阐述了其数学公式和实现。
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600.pdf) — Yang Song and Stefano Ermon (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: NeurIPS Foundation; Volume: 32; DOI: [10.48550/arXiv.1907.05600](https://doi.org/10.48550/arXiv.1907.05600)
  这项工作提出了一个基于估计数据分布梯度（分数函数）的生成建模框架，为扩散模型中的基于梯度的指导提供了理论基础。
- [Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/pdf/2303.07222.pdf) — Zefan Yang, Haobo Shu, Chengyue Shang, Binjie Wang, Zhiyuan Li, Chenyang Xie, and Xuming He (2023)
  Journal: arXiv preprint arXiv:2303.07222; DOI: [10.48550/arXiv.2303.07222](https://doi.org/10.48550/arXiv.2303.07222)
  这项综述文章提供了扩散模型的广泛概述，包括分类器指导等多种条件生成策略，将其置于领域背景中。

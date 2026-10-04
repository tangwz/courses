---
course: "synthetic-data-gans-diffusion"
chapter: "diffusion-models-theory-implementation"
lesson: "diffusion-sde-foundations"
sourceId: 5688
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-4-diffusion-models-theory-implementation/diffusion-sde-foundations"
title: "数学基本原理：随机微分方程"
description: "理解扩散模型的连续时间表述。"
order: 1
plots: []
sourceHash: "947be207d82a8e94adaccb87d6ab028fe9193ec64a22e6c2eab9b13ca12ec39e"
sourceCorrections: []
---

扩散模型通过逐步向数据添加噪声，然后学习反向过程来运行。虽然像DDPM这样的模型通常使用离散时间步长表示，但当我们考虑这种加噪过程的连续时间极限时，会得到一个更普遍且强大的数学体系。这使我们接触到随机微分方程（SDE）的表达方式。

理解SDE的表述方式有助于更好地明白扩散模型的工作原理，并将各种离散时间扩散方法统一在单一的数学结构下。它还促成了更灵活的噪声调度和采样技术。

### 从离散步长到连续过程

回想DDPM中的离散正向过程，噪声在每个时间步长$t$逐步添加。如果考虑无限小的时间步长，这种变换序列将收敛于一个连续的随机过程。SDE描述了变量在连续时间上的演变，它既包含确定性变化（漂移），也包含随机波动（扩散）。

一个一般的Itô SDE形式如下：


$$
d\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t) dt + g(t) d\mathbf{w}_t
$$


此处：

- $\mathbf{x}_t$ 表示$t$时刻的状态（例如，图像）。
- $d\mathbf{x}_t$ 是$\mathbf{x}_t$的无穷小变化。
- $\mathbf{f}(\mathbf{x}_t, t)$ 是漂移函数，描述变化的确定性部分。
- $g(t)$ 是扩散系数，衡量随机噪声的大小。
- $d\mathbf{w}_t$ 表示标准维纳过程（布朗运动）的无穷小步长，模拟随机波动。它本质上是按$\sqrt{dt}$缩放的高斯噪声。

### 正向SDE：数据到噪声

在扩散模型中，正向过程将复杂数据$\mathbf{x}_0$转换为简单的噪声分布（通常是高斯分布），随着时间$t$从$0$推进到$T$。这种“信息破坏”过程可以通过特定的SDE来模拟。一个常见选择，常与DDPM关联的方差保持（VP）SDE为：


$$
d\mathbf{x}_t = -\frac{1}{2} \beta(t) \mathbf{x}_t dt + \sqrt{\beta(t)} d\mathbf{w}_t
$$


此处，$\beta(t)$是一个正的、随时间变化的函数，常被称为噪声调度。

- 漂移项$-\frac{1}{2} \beta(t) \mathbf{x}_t dt$将状态推向原点（均值为零）。
- 扩散项$\sqrt{\beta(t)} d\mathbf{w}_t$持续添加高斯噪声，其方差由$\beta(t)$控制。

随着$t$从$0$增加到$T$，初始数据$\mathbf{x}_0$的影响减小，$\mathbf{x}_T$趋近于标准高斯分布，与$\mathbf{x}_0$无关。

### 反向SDE：噪声到数据

扩散模型的生成能力来自反转此过程。我们从简单的噪声分布中抽取样本$\mathbf{x}_T$，并将其在时间上从$T$反向演化到$0$，以生成数据样本$\mathbf{x}_0$。随机微积分的一个显著成果（Anderson，1982）指出，如果已知边缘分布$p_t(\mathbf{x}_t)$的得分函数，则由正向SDE定义的扩散过程的反向轨迹也遵循SDE。

与上述正向过程对应的反向SDE如下：


$$
d\mathbf{x}_t = \left[ -\frac{1}{2} \beta(t) \mathbf{x}_t - \beta(t) \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \right] dt + \sqrt{\beta(t)} d\bar{\mathbf{w}}_t
$$


此处：

- $d\bar{\mathbf{w}}_t$ 表示一个在时间上反向运行的标准维纳过程。
- $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$ 是$t$时刻数据分布的得分函数。此项表示对数据$\mathbf{x}_t$的对数概率密度的梯度。直观地看，它指向数据密度增长最快的方向。

这个反向SDE告诉我们如何对当前状态$\mathbf{x}_t$进行无穷小的调整，使其在数据分布$p_t$下稍微更可能。漂移项现在包含了得分函数，有效地引导过程远离噪声，并趋向合理的数据结构。

### 得分匹配的作用

使用反向SDE进行生成的主要问题是，我们不知道中间分布$p_t$的真实得分函数$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$。这时神经网络 (neural network)就派上用场了。我们训练一个时间相关的神经网络，通常表示为$\mathbf{s}_\theta(\mathbf{x}_t, t)$，来近似真实的得分函数：


$$
\mathbf{s}_\theta(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)
$$


这个网络通常通过得分匹配或与DDPM中使用的目标（如章节介绍中提到的$L_{simple}$目标，它隐含地学习得分）等效的目标进行训练。一旦训练完成，$\mathbf{s}_\theta(\mathbf{x}_t, t)$就可以代入反向SDE中：


$$
d\mathbf{x}_t = \left[ -\frac{1}{2} \beta(t) \mathbf{x}_t - \beta(t) \mathbf{s}_\theta(\mathbf{x}_t, t) \right] dt + \sqrt{\beta(t)} d\bar{\mathbf{w}}_t
$$


从$\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$开始，在时间上反向模拟此SDE，我们可以生成新的数据样本$\mathbf{x}_0$。

> 图示正向SDE如何随时间破坏数据结构，以及学习到的反向SDE如何通过遵循估计的得分函数从噪声中重建数据。

### SDE体系的意义

从SDE的角度审视扩散模型具有以下几个优点：

1. **统一性：** 它提供了一个共同的数学体系，涵盖了各种离散时间模型，如DDPM以及从SDE的不同离散化方法中得到的关联方法。
2. **灵活性：** 它自然地处理连续时间，允许更灵活的噪声调度$\beta(t)$和固定离散步长的采样策略。
3. **理论依据：** 它将扩散模型与已有的随机过程和得分匹配理论联系起来，有助于进行更透彻的分析。
4. **高级采样：** SDE求解器（用于模拟SDE的数值方法）可以用于比离散模型的简单反向步进更快或更准确的生成。

这种连续时间视角为理解基于得分的生成建模以及DDIM等高级技术奠定了基础，后者凭借底层SDE的特性实现高效采样。我们将在后续章节中考察扩散模型的实现细节和改进时，以此基本原理为依据。

## 参考资料

- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
  这篇基础论文在随机微分方程的连续时间框架下，统一了基于分数的生成模型和扩散概率模型。
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) — Yang Song and Stefano Ermon (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1907.05600](https://doi.org/10.48550/arXiv.1907.05600)
  介绍了基于分数的生成模型概念，这些模型直接学习数据分布的分数函数，这是扩散模型中逆向随机微分方程的关键组成部分。

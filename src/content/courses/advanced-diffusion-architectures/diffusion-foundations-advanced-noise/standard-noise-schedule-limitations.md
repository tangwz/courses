---
course: "advanced-diffusion-architectures"
chapter: "diffusion-foundations-advanced-noise"
lesson: "standard-noise-schedule-limitations"
sourceId: 5369
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-1-diffusion-foundations-advanced-noise/standard-noise-schedule-limitations"
title: "标准噪声调度法的局限性"
description: "分析线性调度和余弦调度在特定生成任务中的不足。"
order: 4
plots: ["plots/5369-0.json"]
sourceHash: "a59238f055c0b1633b1c895e8ab26069745369ee0749ad3f83cabc31d739fafd"
sourceCorrections: []
---

噪声调度（通常由方差参数 (parameter)序列 $\beta_1, \dots, \beta_T$ 定义）控制着正向扩散过程 $q(x_t|x_{t-1})$ 中噪声添加的速率。因此，它显著影响模型 $p_\theta(x_{t-1}|x_t)$ 在逆向过程中必须学习的去噪任务的特性。虽然线性调度和余弦调度等标准调度法提供了合理的起点并已成功应用，但它们具有固有的局限性，会影响性能，特别是对于复杂数据集或追求顶尖结果时。了解这些不足之处促使人们寻求更精密的调度方法。

### 线性调度的问题

线性调度，即 $\beta_t$ 在 $T$ 个时间步长内从一个较小值 $\beta_{\text{start}}$（例如 $10^{-4}$）线性增长到一个较大值 $\beta_{\text{end}}$（例如 $0.02$），也许是最简单的表示形式。其主要特点是每一步添加的噪声方差以恒定增量增加。


$$
\beta_t = \beta_{\text{start}} + \frac{t-1}{T-1} (\beta_{\text{end}} - \beta_{\text{start}})
$$


然而，这种简单性也伴随着代价：

1. **极端情况下的低效：** 在正向过程开始时（$t$ 较小），添加的噪声 $\beta_t$ 非常小。这意味着 $x_t$ 与 $x_{t-1}$ 和 $x_0$ 保持高度关联。尽管这保留了信息，但也意味着逆向过程中的许多初始步骤都用于进行非常精细的调整，可能降低有效采样的速度。相反，在正向过程接近尾声时（$t$ 较大），$\beta_t$ 变得相对较大，导致信噪比 (SNR) 迅速下降。分布 $q(x_T|x_0)$ 有效地接近标准高斯分布，几乎丢失了关于 $x_0$ 的所有信息。这使得逆向过程的初始步骤（从 $t=T$ 开始）非常具有挑战性，因为模型必须从几乎纯粹的噪声中恢复重要的结构。
2. **信噪比衰减：** 累积噪声水平（通常通过 $\bar{\alpha}_t = \prod_{i=1}^t (1-\beta_i)$ 来追踪）决定了在步骤 $t$ 时的总体信噪比。线性调度会导致 $\bar{\alpha}_t$ 衰减的特定模式，这可能不是最佳的。信噪比在初始阶段下降非常缓慢，然后趋向于快速下降。调度与不同噪声水平下去噪的潜在复杂性之间的这种不匹配会影响样本质量。
3. **对高分辨率图像不理想：** 对于高分辨率图像，细节通常由高频分量表示。线性调度在后期时间步长上迅速破坏信息，会使得在逆向过程中重建这些细节变得困难，有时会导致结果略显模糊或细节不足，相比那些更仔细地管理信噪比衰减的调度法。

### 余弦调度的问题

余弦调度被提出是为了解决线性调度的一些局限，尤其是 $t=T$ 附近信噪比的快速下降。它直接使用余弦函数定义累积乘积 $\bar{\alpha}_t$，旨在实现时间步长之间更平滑的过渡：


$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad \text{其中} \quad f(t) = \cos^2\left(\left(\frac{t/T + s}{1+s}\right) \frac{\pi}{2}\right)
$$


此处，$s$ 是一个小的偏移量（例如 $0.008$），以防止 $\beta_t$ 在 $t=0$ 附近过小。从 $\bar{\alpha}_t$ 中，可以导出各个方差：$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$。

这种调度法通常比线性调度产生更好的结果，并已成为流行的默认选项。然而，它并非没有潜在的不足：

1. **仍是启发式的：** 尽管其动机是改善信噪比特性，余弦调度仍然是一个固定、预设的函数。它不适应所建模数据的具体属性。有可能，而且通常是这样，添加噪声的最佳方式取决于数据集的固有结构，而余弦调度没有考虑到这一点。
2. **缓慢启动：** 偏移量 $s$ 防止 $\beta_1$ 为零，但在最开始阶段（$t \ll T$）添加的噪声与后续步骤相比仍然很小。与线性调度相似，这可能意味着早期的逆向步骤对整体生成质量的贡献不那么显著，如果噪声处理方式不同，可能需要比严格必要更多的步骤。
3. **对 $T$ 的依赖：** 余弦调度的形状本质上与总时间步长 $T$ 相关联。如果 $T$ 显著减少（例如，为了更快采样），形状会受到压缩，可能以损害性能的方式改变噪声注入的特性，相比其最初调整的范围（通常是 $T=1000$ 或 $T=4000$）。

### 可视化调度差异

通过绘制 $\bar{\alpha}_t$，可以直观展现这些调度法如何随时间影响信号水平的差异。较低的 $\bar{\alpha}_t$ 表示添加了更多噪声，破坏了原始信号 $x_0$。



![噪声调度对比 (alpha_bar_t)](plots/5369-0.json)



> 线性调度和余弦调度在 $T=1000$ 时间步长下 $\bar{\alpha}_t$ 的对比。余弦调度在更长时间内保持更高的信号水平，然后才在末端更急剧地衰减，相比之下，线性调度在初期衰减更平缓，但下降速度更快。请注意，具体值取决于 $\beta_{\text{start}}$、$\beta_{\text{end}}$ 和 $s$。

### 后果及改进的动机

这些固定标准调度的局限性通常表现为：

- **不理想的样本质量：** 生成的样本可能缺乏细节，显得略微模糊，或未能捕捉训练数据的全部多样性。
- **低效采样：** 实现高质量可能需要大量的采样步骤（$T$），增加了推理 (inference)过程中的计算成本和延迟。使用标准调度减少 $T$ 通常会导致质量的明显下降。
- **训练挑战：** 在某些情况下，不理想的调度法可能导致训练不稳定或收敛速度变慢。

这些不足之处突出了对噪声调度策略的需求，它们应更精心设计或甚至从数据中学习。通过调整噪声调度，我们有望提高样本保真度，加快采样速度，并提升训练稳定性。这促使我们在以下章节中讨论相关技术，例如，基于信噪比分析设计自定义调度，或实现能够学习方差调度的模型。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  这篇基础论文介绍了去噪扩散概率模型（DDPM）以及用于前向扩散过程的初始线性噪声调度。
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — Alex Nichol and Prafulla Dhariwal (2021)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2102.09672](https://doi.org/10.48550/arXiv.2102.09672)
  本文提出了余弦噪声调度，作为线性调度的改进，提升了扩散模型的样本质量和训练稳定性。

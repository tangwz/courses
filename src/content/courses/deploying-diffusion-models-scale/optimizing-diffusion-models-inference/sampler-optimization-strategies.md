---
course: "deploying-diffusion-models-scale"
chapter: "optimizing-diffusion-models-inference"
lesson: "sampler-optimization-strategies"
sourceId: 5387
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-2-optimizing-diffusion-models-inference/sampler-optimization-strategies"
title: "采样器优化策略"
description: "减少采样器所需步数的方法（例如DDIM、PNDM）。"
order: 4
plots: ["plots/5387-0.json"]
sourceHash: "adf6ae0a036495500e514e38cf59faa88ff0fc83f40e270af514fe7f4d88eb74"
sourceCorrections: []
---

扩散模型推理 (inference)过程中的主要计算负担来自迭代采样过程。每一步通常都涉及通过大型U-Net模型的前向传播，这会导致显著的延迟，特别是在生成高质量内容时需要数百个步骤（$T$）的情况下。因此，一个主要优化目标是减少函数评估次数（NFE），这直接对应于采样步数，同时不明显降低生成内容的质量。这种减少直接转化为更低的延迟、更高的吞吐量 (throughput)以及每次生成更低的计算成本。

扩散模型中的采样过程涉及求解反向随机微分方程（SDE）或常微分方程（ODE）。传统采样器，例如去噪扩散概率模型（DDPM）中使用的，通常需要许多步骤（$T \approx 1000$），因为它们模拟的是具有小增量的随机过程。早期的工作侧重于开发能够以显著更少步骤达到同等质量的求解器。

### 从随机到确定性路径：DDIM

去噪扩散隐式模型（DDIM）带来了重要进步。DDIM将生成过程重新表述为求解一个ODE，这在给定起始噪声向量 (vector)$x_T$的情况下，会得到一条确定性采样路径。这种确定性很重要，因为它允许与随机采样器相比使用更大的步长。DDIM引入了一个参数 (parameter)$\eta$来控制随机性；将$\eta=0$设定会得到确定性ODE变体。

DDIM更快采样的核心思想是它定义了一个非马尔可夫前向过程，这使得在反向推理 (inference)过程中可以跳过步骤。虽然原始DDPM步骤仅依赖于$x_t$，但DDIM采样器可以有效跳跃，例如直接从$t=500$跳到$t=400$，通过一次计算近似路径段。这使得NFE可以从$\approx 1000$减少到$\approx 50$甚至$\approx 20$步，尽管与完整的DDPM过程相比，通常会在样本多样性或细节方面有所取舍。

> 图示说明了传统随机采样器（DDPM）与更快的确定性采样器（DDIM）在步数上的区别。DDIM采取更大的跳跃，减少了总NFE。

### 高阶数值求解器

DDIM本质上是一个一阶ODE求解器。就像在经典数值方法中一样，我们可以使用高阶求解器，以便每一步可能获得更好的精度，从而在目标质量水平下整体上需要更少的步骤。从数值积分技术调整而来的几种方法已成功应用于扩散模型：

1. **扩散模型的伪数值方法 (PNDM)：** PNDM使用线性多步法，利用先前步骤的信息（例如$x_{t+1}$、$x_{t+2}$）来改进$x_t$的估计。这可以比DDIM更快地收敛。
2. **DPM-求解器（扩散概率模型求解器）：** DPM-求解器及其变体（如DPM-求解器++）是专门为扩散ODE设计的高阶求解器。它们通过在中间步骤中解析求解ODE，即使在非常少的NFE（$<20$）下也能得到高精度的解。DPM-求解器++进一步提升了稳定性和准确性，通常在10-15步内就能获得出色的结果。其核心思路是在一个步长区间内，使用多个点的预测导数（噪声估计）来构建解路径的更精确多项式近似，从而允许更大、更准确的步长。

使用这些高阶求解器可以大幅缩短推理 (inference)时间。例如，在相同硬件上，DDIM在50步时生成一张图像可能需要5-10秒，但DPM-求解器++在15步时仅需1-2秒，前提是NFE是主要影响因素。

### 自适应步长与调度

大多数采样器使用预定义、固定的时间步长$t$调度（例如，线性间隔、二次方间隔）。然而，数据流形的变化率在整个扩散过程中可能不是均匀的。自适应步长求解器尝试根据估计的局部误差或轨迹曲率动态调整步长$\Delta t$。虽然实现和调优可能更复杂，但它们有望更有效地分配计算资源，仅在必要时使用较小的步长。

此外，时间步长（即“调度”）的*选择*会对固定NFE下的质量产生影响。关于最佳调度策略的研究仍在进行，有时发现将步骤集中在扩散过程的特定部分（例如，中间或后期阶段）可以为某些模型或数据集带来更好的结果。

### 评估采样器权衡

选择优化后的采样器需要平衡速度（NFE）与生成质量。质量通常使用Fréchet起始距离（FID）等指标来衡量，该指标比较生成图像与真实图像的分布，或者使用CLIP分数，该分数衡量生成图像与文本提示之间的一致性。

对您的特定模型和目标应用来说，对不同采样器和步数进行基准测试十分重要。一种常见方法是将FID或其他质量指标与NFE或实际推理 (inference)时间绘制成图表。



![采样器性能权衡](plots/5387-0.json)



> 图比较了不同采样器，显示了FID（越低越好）和推理时间与函数评估次数（NFE）的关系。注意，与DDIM相比，DPM-求解器++等高阶求解器能在明显更少的步骤（从而更短的时间）下实现更低的FID。实际值很大程度上取决于模型、硬件和具体实现。

### 实际应用

像Hugging Face的`diffusers`这样的现代库提供了方便使用各种预实现采样器（在它们的术语中称为调度器）的途径。在初始化推理 (inference)管道时，切换采样器通常只需更改几行代码。

```python
# 使用Hugging Face diffusers的示例
from diffusers import DiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# 使用50步的DDIM
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
images_ddim = pipeline(prompt="An astronaut riding a horse", num_inference_steps=50).images

# 使用20步的DPM-求解器++
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
images_dpm = pipeline(prompt="An astronaut riding a horse", num_inference_steps=20).images
```

在部署扩散模型时，选择合适的采样器和步数是一个重要的优化手段。这直接影响用户感知到的延迟以及运行推理服务的总成本。进行全面的基准测试对于确定在满足应用质量要求的同时最小化NFE的最佳操作点是必要的。这通常包括测试一系列步数（例如10、15、20、30、50）下的多个备选采样器（例如DDIM、DPM-求解器++），并对生成的图像进行定量（FID、CLIP）和定性评估。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍了去噪扩散概率模型的原始论文，描述了迭代采样过程的基础。
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Jiaming Song, Chenlin Meng, Stefano Ermon (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.02502](https://doi.org/10.48550/arXiv.2010.02502)
  通过将生成过程重构为求解ODE，引入了一种确定性采样方法，实现了更少步骤的快速推理。

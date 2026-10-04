---
course: "advanced-diffusion-architectures"
chapter: "advanced-sampling-optimization"
lesson: "higher-order-solvers"
sourceId: 5464
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-6-advanced-sampling-optimization/higher-order-solvers"
title: "高阶求解器 (DPM-求解器, UniPC)"
description: "实现与理解DPM-求解器和UniPC等高级ODE求解器，以实现更快、更准确的采样。"
order: 1
plots: ["plots/5464-0.json"]
sourceHash: "e066113070774c1d2a47f4a638ab84c31ef81cc3714ee0f7f39b9514c866c92d"
sourceCorrections: []
---

DDIM通过使用确定性ODE近似，相比DDPM显著提速，但它仍需要相对较多的步骤（通常为50-200步）才能获得高精度结果。这是因为DDIM本质上是一种一阶数值求解器（类似于欧拉方法），用于控制逆向扩散过程的潜在概率流ODE。逆向过程可由一个常微分方程（ODE）描述：


$$
dx = f(x_t, t) dt
$$


这里 $f(x_t, t)$ 取决于得分函数 $\nabla_{x_t} \log p_t(x_t)$ 或噪声预测 $\epsilon_\theta(x_t, t)$。欧拉（以及DDIM）等一阶方法，仅根据当前状态的导数，通过小步进方式来近似求解。此类方法要达到高精度，需要许多小步进，导致推理 (inference)速度较慢。

为克服此限制，我们可采用更复杂的、专为求解ODE设计的数值方法。这些*高阶求解器*运用多个先前步骤或步骤内的中间点的信息，以更精确地近似ODE轨迹。每步精度提升，允许更大的步长，显著减少生成样本所需的函数评估总数 (NFE)，通常能在远少于DDIM的步数（例如10-25步）内达到与其相当甚至更好的质量。

DPM-求解器和UniPC是适用于扩散模型的两大类高阶求解器。

### DPM-求解器 (扩散概率模型求解器)

DPM-求解器是一类专门针对扩散ODE结构设计的求解器。它基于以下见解：扩散ODE通常具有半线性形式，从而能够使用指数积分器等技术进行高效精确求解。

核心理念是在更大的时间间隔 $\Delta t$ 上更精确地近似ODE解。DPM-求解器有不同阶数（例如DPM-求解器-2、DPM-求解器-3），其中高阶数运用更多信息，可能获得更高精度，但有时会牺牲稳定性。

一种尤其有效的变体是**DPM-求解器++**。它常将指数积分器方法与数据预测项（预测 $x_0$ 并以此指导步进）结合，即使步数很少（通常少于20步），也能获得非常稳定和高质量的结果。

**优势：**

- **速度：** 与DDIM相比，在相似质量下，显著减少所需步数。
- **质量：** 通常可在15-25步内达到高精度生成。
- **专用性：** 专门为扩散ODE的结构设计。

**考量：**

- 最佳阶数和特定变体（例如DPM-求解器++ SDE）可能取决于具体的模型和数据集。
- 尽管库封装了复杂性，但了解基本原理有助于进行参数 (parameter)调整。

### UniPC (统一预测-校正器)

UniPC提供了一种基于经典预测-校正方法求解ODE的替代方法。它旨在将DDIM（预测器）和Analytic-DPM（校正器）等先前的方法统一到一个框架中。

预测-校正方法每步分两个阶段运行：

1. **预测器：** 对下一时间步的解进行初始估计（类似于DDIM或DPM-求解器的一个步进）。
2. **校正器：** 运用附加信息修正此估计，通常涉及另一次模型评估或运用预测得分。

UniPC巧妙地调整了这种方案以适用于扩散模型。通过应用一个或多个校正步，它能提升第一阶段估计的精度，相比仅有预测器的方法，在相同步数下，能有效实现更大步长或更好质量。

**优势：**

- **速度：** 在质量与步数关系方面常达到当前最优结果，有时仅需10-15步。
- **灵活性：** 提供统一框架，可能适用于不同的扩散形式。
- **稳定性：** 校正步有时有助于稳定积分过程。

**考量：**

- 如果使用多次校正评估，每步计算量可能略有增加，但总步数的减少通常会带来整体加速。
- 具有一套需要调整的超参数 (parameter) (hyperparameter)（例如，变体、阶数）。

### 比较与实际应用

DPM-求解器++和UniPC都代表着DDIM在快速采样方面的显著进步。它们通常能在10-25步内实现高质量图像生成，相比DDIM常需要的50-200步，这是一个显著的减少。



![质量与采样步数对比](plots/5464-0.json)



> 对比显示，DPM-求解器++和UniPC等高阶求解器比DDIM更快（更少步数）地达到高质量（低FID分数）。实际表现因模型和任务而异。

“最佳”选择通常取决于具体的扩散模型架构、其训练数据集以及速度与绝对最高质量之间的期望权衡。DPM-求解器++是一个非常强大且受欢迎的基准，而UniPC常在最少步数方面取得突破。

幸好，将这些求解器整合到实际工作流程中通常很简单，这得益于Hugging Face `diffusers`等库。切换调度器通常只需修改几行初始化调度器的代码。

```python
# 运用Hugging Face Diffusers的示例（说明性）
from diffusers import DiffusionPipeline
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1-base" # 或您的自定义模型

# 加载管线（Stable Diffusion示例）
pipe = DiffusionPipeline.from_pretrained(model_id)

# --- 选择您的调度器 ---
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# --- ---

prompt = "A photograph of an astronaut riding a horse on the moon"
num_inference_steps = 20 # 使用高级求解器可减少步数！

image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
# image.save("generated_image.png")
```

在优化扩散模型推理 (inference)速度时，强烈建议试验这些高级求解器。通过用DPM-求解器++或UniPC替换DDIM，您通常可大幅减少生成时间，同时不显著影响输出质量。这使得复杂的扩散模型在实时应用和资源受限环境中更具实用性。请记住也要试验`num_inference_steps`参数 (parameter)，因为这些求解器设计用于比DDIM低得多的值时也能良好运行。

## 参考资料

- [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling in Diffusion Models](https://arxiv.org/abs/2302.04867) — Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu (2023)
  Journal: NeurIPS 2023; DOI: [10.48550/arXiv.2302.04867](https://doi.org/10.48550/arXiv.2302.04867)
  提出了UniPC，一个预测-校正框架，它统一并改进了现有的扩散模型快速采样方法。
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Jiaming Song, Chenlin Meng, Stefano Ermon (2020)
  Journal: International Conference on Learning Representations (ICLR 2021); DOI: [10.48550/arXiv.2010.02502](https://doi.org/10.48550/arXiv.2010.02502)
  介绍了DDIM，一种用于扩散模型的确定性采样方法，它是逆向过程的基础一阶ODE求解器。

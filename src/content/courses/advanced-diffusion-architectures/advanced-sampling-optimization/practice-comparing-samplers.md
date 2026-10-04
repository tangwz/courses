---
course: "advanced-diffusion-architectures"
chapter: "advanced-sampling-optimization"
lesson: "practice-comparing-samplers"
sourceId: 5478
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-6-advanced-sampling-optimization/practice-comparing-samplers"
title: "动手实践：比较高级采样器"
description: "实现并比较给定模型上不同高级采样器（DDIM、DPM-Solver）的速度与质量权衡。"
order: 8
plots: ["plots/5478-0.json"]
sourceHash: "2f394c6d4324054b9a5a4798bf4863d13195a2a0494d3af671ee3e1ab718c65d"
sourceCorrections: []
---

高级采样算法旨在解决基本扩散方法的速度限制。在此，我们将实现并比较几种常用采样器。目的是观察生成速度（主要受函数评估次数，即NFE，与推理 (inference)步数相关的影响）与所得样本质量之间的权衡。

我们将使用预训练 (pre-training)扩散模型，并比较标准DDIM采样器与DPM-Solver++和UniPC等更高级求解器的性能。本次练习将为您为特定需求选择合适的采样器提供实用见解。

### 设置

首先，请确保已安装必要的库。我们将主要使用Hugging Face `diffusers` 库，因为它提供了便捷的管线和调度器实现，以及PyTorch。

```python
# pip install diffusers transformers accelerate torch
import torch
from diffusers import DiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler
import time
import matplotlib.pyplot as plt
from PIL import Image
import math

# 检查GPU可用性
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载预训练管线（例如，Stable Diffusion 或更小的模型）
# 使用分辨率降低的小型模型，如 runwayml/stable-diffusion-v1-5
# 或 google/ddpm-cifar10-32 可能有助于更快地进行实验。
# 这里以 stable-diffusion-v1-5 为例。请根据您的硬件进行调整。
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipeline = pipeline.to(device)

# 定义一个用于比较的通用提示词
prompt = "A photo of an astronaut riding a horse on the moon"
```

如果需要，请务必登录 Hugging Face (`huggingface-cli login`) 以下载 Stable Diffusion 等模型。根据您可用的硬件和所需的模型调整 `model_id` 和 `torch_dtype`。在兼容的GPU上使用 `torch.float16` 可显著加快生成速度。

### 定义采样器

`diffusers` 库可以轻松切换调度器（采样器）。我们将实例化要比较的采样器：DDIM、DPM-Solver++ 和 UniPC。

```python
# 实例化我们要比较的调度器
scheduler_ddim = DDIMScheduler.from_config(pipeline.scheduler.config)
scheduler_dpm = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
scheduler_unipc = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

# 将它们存储起来以便于访问
samplers = {
    "DDIM": scheduler_ddim,
    "DPM-Solver++": scheduler_dpm,
    "UniPC": scheduler_unipc
}
```

### 运行比较

现在，我们编写一个函数来生成图像，使用给定的采样器和推理 (inference)步数，并测量所需时间。我们将为每个采样器迭代不同的步数。

```python
def generate_and_time(pipe, sampler_name, sampler, prompt_text, num_steps, num_images=1):
    """使用指定采样器生成图像并测量耗时。"""
    pipe.scheduler = sampler  # 设置管线的调度器

    start_time = time.time()
    # 如需复现性，可使用固定生成器
    generator = torch.Generator(device=device).manual_seed(42) 

    images = pipe(
        prompt=prompt_text, 
        num_inference_steps=num_steps, 
        generator=generator,
        num_images_per_prompt=num_images
    ).images

    end_time = time.time()
    generation_time = end_time - start_time

    print(f"Sampler: {sampler_name}, Steps: {num_steps}, Time: {generation_time:.2f}s")

    return images, generation_time

# 定义要测试的步数
step_counts = [10, 20, 30, 50] 
num_samples_per_setting = 1 # 每个设置生成一张图像，以便快速进行视觉比较

results = {} # 存储图像和时间

# 为每个采样器和步数运行生成
for name, sampler_instance in samplers.items():
    results[name] = {"images": [], "times": [], "steps": []}
    for steps in step_counts:
        # 如果使用 torch.compile（可选优化），确保模型只编译一次
        # pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True) # 示例

        generated_images, gen_time = generate_and_time(
            pipeline, name, sampler_instance, prompt, steps, num_samples_per_setting
        )
        results[name]["images"].extend(generated_images) # 存储第一张图像
        results[name]["times"].append(gen_time)
        results[name]["steps"].append(steps)
```

*注意：可选的 `torch.compile` 行可以在较新的PyTorch版本和兼容硬件上额外加速生成，但首次运行时可能会增加开销。*

### 定性分析：视觉比较

比较采样器最直接的方法，特别是关于伪影或细节水平，是进行视觉检查。让我们整理生成的图像以便于比较。

```python
def plot_comparison_grid(results_dict, steps_list):
    """绘制生成的图像网格进行比较。"""
    num_samplers = len(results_dict)
    num_steps_options = len(steps_list)

    fig, axes = plt.subplots(num_samplers, num_steps_options, figsize=(num_steps_options * 3, num_samplers * 3.5))
    fig.suptitle("采样器比较：质量与步数", fontsize=16)

    for i, (sampler_name, data) in enumerate(results_dict.items()):
        for j, steps in enumerate(steps_list):
            img_index = j # 因为每个设置我们生成了一张图像
            if img_index < len(data["images"]):
                ax = axes[i, j]
                ax.imshow(data["images"][img_index])
                ax.set_title(f"{sampler_name}\n{steps} Steps ({data['times'][j]:.1f}s)")
                ax.axis('off')
            else:
                axes[i, j].axis('off') # 处理生成失败的潜在错误

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以避免标题重叠
    plt.show()

plot_comparison_grid(results, step_counts)
```

仔细查看输出网格：

- **低步数（例如，10-20）：** 高级采样器（DPM-Solver++、UniPC）是否生成明显更好的结果？观察其连贯性、细节和更少的伪影。通常，高阶求解器在较少步数下也能保持更佳质量。
- **高步数（例如，30-50）：** 图像是否收敛到相似的质量水平？是否有任何采样器即使在较高步数下仍引入特定的伪影？DDIM 可能需要更多步数才能达到其他采样器更早达到的质量。
- **一致性：** 采样器是否持续生成合理的结果，还是即使有足够步数也会偶尔出现失败？

您可能会观察到，与DDIM相比，DPM-Solver++和UniPC可以在更少步数下生成合理的图像。然而，最佳步数和输出风格的细微差异可能因模型和提示词 (prompt)而异。

### 定量分析：速度

我们已经记录了生成时间。让我们将时间与每个采样器的步数进行可视化。



![生成时间与步数的关系](plots/5478-0.json)



> 不同采样器在不同推理 (inference)步数下的生成时间比较。注意：实际时间很大程度上取决于硬件、模型大小、图像分辨率和软件优化。此处显示的数据仅供参考。

此图通常显示，对于大多数采样器而言，生成时间与步数大致呈线性关系。尽管高级求解器每一步的计算开销可能略有不同，但其主要优势在于用*更少*的步数达到良好的质量，从而减少总时间。在此示例数据中，DPM-Solver++和UniPC在相同步数下显示出略低的耗时，但其主要益处在于所需步数更少（例如，在20步时达到良好质量，而DDIM可能需要30-50步）。

### 讨论与要点

本次实践展示了使用DPM-Solver++和UniPC等高级采样算法的明显好处。

- **速度与质量的权衡：** 高级求解器通常可以实现更快的推理 (inference)，在显著更少的步数下达到与DDIM相当或更好的质量。对于需要近乎实时生成的应用，通常更偏爱使用DPM-Solver++或UniPC等求解器，并设定较低的步数（例如15-25）。
- **采样器选择：** “最佳”采样器可能取决于具体任务。虽然DPM-Solver++和UniPC是出色的通用选择，但DDIM对于特定的复现性需求，或者当需要其特有的平滑效果时，可能仍有用。进行实验非常重要。
- **超参数 (parameter) (hyperparameter)调整：** 推理步数是一个重要的超参数。本次实践展示了结果会随之发生巨大变化。其他参数，如引导尺度（CFG），也与采样器和步数相互作用，需要仔细调整以获得最佳结果。

通过亲自运行这些比较，可能使用不同的模型或提示词 (prompt)，您可以为有效选择和配置采样器获得宝贵经验，从而在您自己的项目中平衡生成速度和输出保真度的要求。还要记住考虑后续会介绍的优化方法，例如量化 (quantization)或模型编译，它们与采样器选择结合，可额外提升性能。

## 参考资料

- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Jiaming Song, Chenlin Meng, Stefano Ermon (2020)
  Journal: ICLR 2021; DOI: [10.48550/arXiv.2010.02502](https://doi.org/10.48550/arXiv.2010.02502)
  本文介绍了去噪扩散隐式模型（DDIM），一种快速稳定的扩散模型采样方法。
- [UniPC: A Unified Predictor-Corrector Framework for Fast Diffusion Sampling](https://arxiv.org/abs/2302.04867) — Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu (2023)
  Journal: NeurIPS 2023; DOI: [10.48550/arXiv.2302.04867](https://doi.org/10.48550/arXiv.2302.04867)
  本文介绍了UniPC，一个用于扩散采样的统一预测-校正框架，能在更少步数下实现高质量生成。
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Diffusers库的官方文档，提供实现扩散模型和采样器的指南及API参考。

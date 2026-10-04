---
course: "how-to-build-a-large-language-model"
chapter: "data-sampling-strategies-training"
lesson: "data-pacing-annealing-schedules"
sourceId: 5994
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-9-data-sampling-strategies-training/data-pacing-annealing-schedules"
title: "数据步进与退火调度"
description: "在训练过程中动态改变数据混合或难度。"
order: 5
plots: ["plots/5994-0.json"]
sourceHash: "b9a61b8ccde95cc70fc2ad11a1d3da17d683be6c3b3019be19b48b4b46a352ae"
sourceCorrections: []
---

虽然静态数据混合为训练提供了一个固定的方案，但大型语言模型可能会从更具动态性的数据呈现方式中获益，这就像课程指导学习一样。我们不是在整个训练过程中都以相同比例馈送数据源，而是可以随着时间动态调整混合比例或采样过程本身。这就是数据步进和退火调度的主要思想。这些方法旨在通过控制数据集不同部分的*何时*以及*如何*被着重处理来改善学习过程。

### 数据步进：控制接触数据的速度

数据步进是指在训练过程中控制模型接触不同子集或类型数据的速度。可以将其视为根据模型的学习进展来管理信息流。例如，你可能主要从高质量、精心整理的语料库开始训练，以打下扎实的语言理解基础。随着训练的进行，模型能力增强，你可以逐渐增加来自噪声较大的数据源（如网页抓取数据）或特定类别数据（如代码库）的比例。

步进可以基于多种标准：

1. **数据源质量：** 从干净数据开始，逐渐引入噪声更大或更复杂的数据。
2. **类别特定性：** 首先使用通用文本，然后根据需要逐步加入特定类别文本。
3. **数据难度：** 如果能够估计数据难度（例如，根据文本复杂性或想法的稀有程度），可以逐步引入从简单到困难的例子，尽管在大规模情况下界定难度是一项困难。

实施数据步进通常需要创建一个调度函数，该函数根据当前的训练步骤或周期修改与不同数据源相关的采样权重 (weight)或概率。

```python
import numpy as np

# 示例数据源和初始权重
data_sources = {
    "curated_corpus": {"weight": 0.7, "path": "/path/to/curated"},
    "web_crawl": {"weight": 0.3, "path": "/path/to/web"},
    # 根据需要添加更多数据源
}

# 总训练步数
total_steps = 1_000_000

def get_pacing_weights(current_step, total_steps, sources):
    """根据训练进度计算动态权重。"""
    progress = current_step / total_steps
    new_weights = {}

    # 示例：线性降低整理数据的权重，增加网页抓取数据的权重
    # 确保权重总和仍为1（或之后归一化）
    curated_weight = max(0.1, 0.7 - 0.6 * progress) # 不要低于10%
    web_weight = 1.0 - curated_weight # 为简化起见，假设只有两个数据源

    new_weights["curated_corpus"] = curated_weight
    new_weights["web_crawl"] = web_weight
    # 如果存在其他数据源，在此添加逻辑

    # 归一化权重以确保它们总和为1（很重要！）
    total_weight = sum(new_weights.values())
    normalized_weights = {
        k: v / total_weight
        for k, v in new_weights.items()
    }

    return normalized_weights

# --- 在训练循环内部 ---
# current_training_step = ... 获取当前步数 ...
# current_weights = get_pacing_weights(
#     current_training_step, total_steps, data_sources)
# 使用这些新权重重新配置数据采样器/加载器
# sampler.set_weights(current_weights)
# batch = next(data_loader)
# ... 训练步骤的其余部分 ...
```

这段代码展示了一个简单的线性步进调度。根据期望的学习进程，可以设计更复杂的函数（例如，指数型、阶梯型）。重要的一点是，随着训练的进行，动态调整从每个数据源采样的可能性。

### 数据采样的退火调度

退火，在数据采样的语境下，通常指逐渐改变一个参数 (parameter)，该参数控制数据源采样分布的*形态*或*随机性*。这与之前讨论的基于温度的采样紧密联系。

回顾一下，温度 $T$ 修改了从权重 (weight) $w_i$ 推导出的概率 $p_i$：


$$
p_i = \frac{\exp(w_i / T)}{\sum_j \exp(w_j / T)}
$$


退火调度可能涉及从较高的温度 $T > 1$ 开始，并在训练过程中逐渐将其降低至 $T = 1$（甚至略低）。

- **高温（训练早期）：** 当 $T > 1$ 时，概率 $p_i$ 变得更均匀。这促进了多样性采样，意味着模型在不同数据源之间进行更均匀的采样，即使是那些初始权重较低的数据源。这在早期可能有利，可以防止模型过快地专注于主要的训练数据。
- **低温（训练后期）：** 随着 $T$ 趋向于 1 降低，采样概率变得更接近由原始权重 $w_i$ 定义的分布。如果 $T < 1$，分布会变得更尖锐，进一步突出高权重的数据源。这使得模型在训练结束时，能将其学习成果集中于被认为更重要或具有代表性的数据上。

退火调度可以应用于温度参数，也可以直接应用于权重本身（例如，逐渐增加高权重和低权重之间的差异）。常见的调度包括温度参数的线性衰减、余弦衰减或指数衰减，或者应用于权重的类似调制因子。



![温度退火调度示例](plots/5994-0.json)



> 示例温度退火调度，涵盖 100 万训练步，从 T=2.0 开始，使用线性和余弦函数衰减至 T=1.0。

### 步进与退火的结合

数据步进和退火并非互斥。你可以将它们结合起来，创建更巧妙的数据馈送方法。例如：

- 使用**步进调度**随时间逐渐改变不同数据源的*目标权重 (weight)*（例如，增加特定类别数据的权重）。
- 同时，在采样温度上使用**退火调度**，以控制基于这些目标权重的采样*随机性*（例如，从高温开始以促进多样性采样，后期降低温度以集中处理高权重数据源）。

这种组合可以精细地控制模型在不同训练阶段看到*什么*数据以及它*多严格地*遵循预设的混合比例。

### 实现考量

实施动态采样调度会增加训练流程的复杂性。

- **进度跟踪：** 训练循环需要准确跟踪当前步骤或周期，以计算调度的当前状态。
- **采样器/加载器配合：** 数据加载机制必须足够灵活，能够动态更新其采样概率或源权重 (weight)，可能在每一步或每几千步进行。这可能涉及自定义采样器逻辑。像 `torch.utils.data.WeightedRandomSampler` 这样的框架可以进行调整，或者可能需要管理多个数据集的自定义迭代器。
- **监控：** 使用动态调度时，密切监控训练稳定性（损失曲线、梯度范数）和评估指标。数据分布的突然变化有时会导致暂时的不稳定。
- **调整：** 找到最佳的步进或退火调度通常需要通过实验。理想的调度形状（线性、余弦、阶梯）及其持续时间很大程度上取决于数据集、模型大小和训练目标。

虽然比静态采样更复杂，但数据步进和退火调度为大型语言模型的学习过程提供了有效的工具。通过随时间精心控制数据输入，你有可能加快收敛速度，增强健壮性，并更好地塑造模型的最终能力以满足特定需求。

## 参考资料

- [Curriculum Learning](https://doi.org/10.1145/1553374.1553380) — Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston (2009)
  Journal: Proceedings of the 26th Annual International Conference on Machine Learning (ICML); Publisher: ACM Press; Pages: 41-48; DOI: [10.1145/1553374.1553380](https://doi.org/10.1145/1553374.1553380)
  介绍了课程学习的概念，即模型从简单的例子开始学习，然后逐步过渡到更具挑战性的例子。
- [Categorical Reparameterization with Gumbel-Softmax](https://doi.org/10.48550/arXiv.1611.01144) — Eric Jang, Shixiang Gu, and Ben Poole (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1611.01144](https://doi.org/10.48550/arXiv.1611.01144)
  介绍了Gumbel-Softmax技巧，它使用温度参数来调整类别采样的分布，为退火调度提供了基础。

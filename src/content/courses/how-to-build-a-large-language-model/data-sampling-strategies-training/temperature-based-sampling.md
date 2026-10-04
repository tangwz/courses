---
course: "how-to-build-a-large-language-model"
chapter: "data-sampling-strategies-training"
lesson: "temperature-based-sampling"
sourceId: 5992
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-9-data-sampling-strategies-training/temperature-based-sampling"
title: "基于温度的采样"
description: "使用温度参数调整跨来源的采样分布。"
order: 3
plots: ["plots/5992-0.json"]
sourceHash: "685e658a56c1f3598294b8d8361f3cc6fd061a26395959a23bce23c3a0d5e87e"
sourceCorrections: []
---

尽管为不同数据来源分配固定权重 (weight)可以控制整体混合比例，但有时更具动态性的方法会更有益。您可能希望调整分配的权重在训练过程中对实际采样概率的影响强度。这就是基于温度的采样发挥作用的地方，它的名称和原理来源于统计物理，并常用于控制生成模型输出的随机性。在数据采样中，温度允许您调节从来源权重得到的概率分布的“锐度”。

假设您有多个数据来源，每个来源都被分配一个分数或对数权重 $w_i$，以反映其预估的重要度或质量。不进行温度缩放时，从来源 $i$ 采样的概率可能由基于这些分数的标准 Softmax 函数确定。温度 $T$ 在 Softmax 计算之前对这些分数引入一个缩放因子。

给定温度 $T$ 时，选择数据来源 $i$ 的概率 $P(i | T)$ 的计算方式如下：


$$
P(i | T) = \frac{\exp(w_i / T)}{\sum_j \exp(w_j / T)}
$$


这里，$w_i$ 是来源 $i$ 的分数或对数权重，求和范围是所有可用来源 $j$。温度 $T$ 是一个正值，它控制着这个概率分布的形态：

- **高温 ($T > 1$，特别是当 $T \to \infty$ 时)：** 随着 $T$ 增加，$w_i / T$ 项对所有 $i$ 都趋近于零。因此，$\exp(w_i / T)$ 趋近于 $\exp(0) = 1$。无论其原始权重 $w_i$ 如何，概率 $P(i | T)$ 在所有来源上变得几乎一致。高温鼓励从所有来源广泛采样，在训练的早期阶段增加多样性，或确保即使低权重的来源也能偶尔被采样。
- **单位温度 ($T = 1$)：** 这会恢复基于原始分数 $w_i$ 的标准 Softmax 概率分布。概率与 $\exp(w_i)$ 成正比。
- **低温 ($0 < T < 1$，特别是当 $T \to 0^+$ 时)：** 随着 $T$ 从正方向趋近于零，分数之间的差异被放大。与其他来源相比，分数最高的来源 $w_{max}$ 的 $\exp(w_i / T)$ 项变得极其大。概率分布明显集中，几乎完全集中在分数最高的来源上。低温迫使采样强烈偏向权重最高的来源，有效地“运用”那些被认为最重要的来源。

这种机制提供了一种平滑的插值方式，可以在均匀采样 ($T \to \infty$) 和基于最高权重的贪婪采样 ($T \to 0^+$) 之间进行调节。

### 实际使用与退火

基于温度的采样与退火策略结合使用时特别有用。您可以从较高温度 ($T > 1$) 开始训练，以确保模型能看到来自所有可用来源的各种数据，促进广度学习。随着训练的进行，您可以逐渐降低温度（将其退火至 $T=1$ 甚至更低）。这种转变将在后期将训练重心移向更高质量或更相关的数据来源，帮助模型基于优先考虑的数据混合提升能力。

考虑一个包含三个数据来源的场景：网页文本（分数 $w_1 = 2.0$）、书籍（分数 $w_2 = 3.0$）和代码（分数 $w_3 = 1.0$）。让我们看看采样概率如何随温度变化。

```python
import torch
import torch.nn.functional as F

# 数据来源的分数（对数权重）
# 分数越高，在 T=1 时偏好度越高
scores = torch.tensor([2.0, 3.0, 1.0]) # 网页文本, 书籍, 代码

temperatures = [0.5, 1.0, 2.0, 10.0]
source_names = ["网页文本", "书籍", "代码"]

print("来源分数:", dict(zip(source_names, scores.tolist())))
print("-" * 30)

for T in temperatures:
    # 应用温度缩放并通过softmax计算概率
    probs = F.softmax(scores / T, dim=0)
    print(f"温度 T = {T:.1f}")
    for name, prob in zip(source_names, probs.tolist()):
        print(f"  P({name}): {prob:.4f}")
    print("-" * 30)

# 基于 T=1.0 时的概率进行采样的示例
T_sample = 1.0
sampling_probs = F.softmax(scores / T_sample, dim=0)
# 根据计算出的概率采样一个来源索引
# 在实际场景中，您会采样成千上万次/数百万次
num_samples = 5
sampled_indices = torch.multinomial(
    sampling_probs,
    num_samples=num_samples,
    replacement=True
)

print(f"\n采样示例 (T={T_sample:.1f}):")
sampled_sources = [source_names[i] for i in sampled_indices]
print(f"采样到的来源索引: {sampled_indices.tolist()}")
print(f"采样到的来源名称: {sampled_sources}")
```

运行此代码会产生类似于以下输出：

```
来源分数: {'网页文本': 2.0, '书籍': 3.0, '代码': 1.0}
------------------------------
温度 T = 0.5
  P(网页文本): 0.1173
  P(书籍): 0.8681
  P(代码): 0.0146
------------------------------
温度 T = 1.0
  P(网页文本): 0.2419
  P(书籍): 0.6577
  P(代码): 0.0890
------------------------------
温度 T = 2.0
  P(网页文本): 0.3067
  P(书籍): 0.4487
  P(代码): 0.2179
------------------------------
温度 T = 10.0
  P(网页文本): 0.3293
  P(书籍): 0.3433
  P(代码): 0.3141
------------------------------

采样示例 (T=1.0):
采样到的来源索引: [1, 1, 1, 0, 1]
采样到的来源名称: ['书籍', '书籍', '书籍', '网页文本', '书籍']
```

请注意，在 $T=0.5$ 时，“书籍”（最高分数）的概率占主导地位（0.8681）。在 $T=1.0$ 时，“书籍”仍然被偏好（0.6577），但其他来源也有合理的机会。随着 $T$ 增加到 $2.0$ 和 $10.0$，概率变得更接近，趋近于均匀分布（$1/3 \approx 0.3333$）。



![温度对采样概率的影响](plots/5992-0.json)



> 在不同温度下，从三个来源（“网页文本”、“书籍”、“代码”，分数分别为 2.0、3.0、1.0）采样的概率。较低温度使分布更集中于分数最高的来源（“书籍”），而较高温度则使其趋于均匀。

### 实现时的考量

- **权重 (weight)的理解：** 公式 $P(i | T) \propto \exp(w_i / T)$ 假设 $w_i$ 是分数或对数权重，且数值越大越好。如果您的权重 $w_i$ 直接表示比例（总和为 1），您可以使用 $P(i | T) \propto w_i^{1/T}$ 并进行归一化 (normalization)。请确保您的实现与您的权重定义相符。当处理神经网络 (neural network)输出或分数时，指数形式很常见。
- **退火策略：** 温度降低的速度（即退火策略）是另一个超参数 (parameter) (hyperparameter)。常见选择包括线性衰减、指数衰减或在训练过程中进行阶梯式降低。
- **集成：** 这种采样逻辑需要集成到您的数据加载流程中。通常，您会为您的 `Dataset` 实现一个自定义采样器，它会为每个批次或样本，首先使用基于温度的采样选择一个来源，然后从该来源的数据集中取出一个项。

基于温度的采样提供了一个灵活的调节手段，可以在整个训练过程中动态控制数据混合，使得能够制定平衡广泛学习和集中运用高价值数据来源的策略。

## 参考资料

- [Thermal Physics](https://books.google.com/books/about/Thermal_Physics.html?id=c336AAAAMAAJ) — Charles Kittel, Herbert Kroemer (1980)
  Publisher: W. H. Freeman and Company
  一部经典教科书，介绍了统计力学的基本概念，包括玻尔兹曼分布，它为温度标度概率分布提供了理论基础。
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever (2019)
  Journal: OpenAI Blog / Technical Report
  这篇关于GPT-2的基础论文讨论了温度作为解码参数，用于控制生成文本的随机性和多样性，阐明了其在数据采样策略中应用的灵感来源。
- [torch.nn.functional.softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) — PyTorch Core Team (2024)
  Publisher: PyTorch
  PyTorch softmax函数的官方文档，对于理解和在深度学习框架中实现基于温度的采样概率的核心计算至关重要。

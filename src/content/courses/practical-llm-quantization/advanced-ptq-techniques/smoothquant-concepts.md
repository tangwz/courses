---
course: "practical-llm-quantization"
chapter: "advanced-ptq-techniques"
lesson: "smoothquant-concepts"
sourceId: 4629
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques/smoothquant-concepts"
title: "SmoothQuant：减轻激活离群值"
description: "了解SmoothQuant如何应对量化具有大动态范围的激活值所面临的挑战。"
order: 4
plots: ["plots/4629-0.json"]
sourceHash: "58084e2de6ddd8e5f7753c03939da221d5904897dcac684660ca1533c3271e87"
sourceCorrections: []
---

基本的训练后量化 (quantization)（PTQ）方法在严格应用时可能面临挑战，特别是对于INT4甚至INT8这样的低比特宽度，无论是权重 (weight)还是激活值。这种困难的主要原因在于模型内部值的分布，特别是激活张量中存在大量级的值，通常被称为*离群值*。

大型语言模型经常出现激活张量中少数值远大于其余值的情况。这在注意力机制 (attention mechanism)层或多层感知机（MLP）块的中间输出中很常见。对这些激活值进行量化时，这些离群值会极大地扩大用于确定量化比例因子和零点的所需范围 $[\min(X), \max(X)]$。如果采用按张量量化方案，比例因子会被拉伸以适应这少数几个大值。这会将大多数较小、更常见的激活值压缩到非常少的量化区间内，导致精度显著下降，并最终降低模型准确性。

想象一下，在一个房间里，既有普通人，也有身材高大的篮球运动员，你尝试用一根只标有英尺刻度的尺子来测量他们的身高。这位篮球运动员迫使你使用一个缺乏更精细分辨率（英寸）的刻度，而这种分辨率对于准确表示普通身材个体之间的身高差异是必需的。

SmoothQuant 为此问题提供了一个巧妙的解决办法。SmoothQuant 不直接尝试量化具有困难分布的激活值，而是旨在将量化难度从激活值（由于离群值而难以量化）*转移*到权重（通常更容易量化）。它通过在量化*之前*应用一个数学上等效的变换来实现这一点。

### 平滑变换

考虑LLM层中典型的矩阵乘法：
$Y = XW$
这里，$X$ 表示输入激活张量，$W$ 表示权重 (weight)张量。SmoothQuant 引入了一个对角缩放矩阵 $S$（意味着所有非对角元素均为零），并将方程变换如下：
$Y = X S^{-1} S W = (X S^{-1}) (S W)$
我们定义变换后的激活值和权重：
$\hat{X} = X S^{-1}$
$\hat{W} = S W$
计算在数学上保持一致：$Y = \hat{X} \hat{W}$。然而，主要思路是仔细选择 $S$ 中的缩放因子。目标是使变换后的激活值 $\hat{X}$ "更平滑" —— 意味着它们具有更小的动态范围（即 $\max(|\hat{X}|)$ 相较于 $\max(|X|)$ 减小），因此更容易准确量化 (quantization)。

应用于激活值 ($S^{-1}$) 的缩放因子通过将逆缩放 ($S$) 应用于权重来抵消。这意味着 $\hat{W}$ 中的值通常会比 $W$ 中的值更大。然而，权重的分布通常比带有离群值的激活值更规则，这使得它们在量化过程中对这种幅度的增加更具适应性。我们已有效地将缩放挑战从有问题的激活值转移到更容易处理的权重。

### 确定平滑因子

缩放因子 $s_j$（$S$ 的对角元素）通常按通道计算。该计算旨在平衡每个输出通道 $j$ 中激活值和权重 (weight)之间的动态范围。确定 $s_j$ 的一个常见方法是：
$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$
这里：

- $X_j$ 表示对应于第 $j$ 个输入通道的激活值。
- $W_j$ 表示对应于第 $j$ 个输入通道的权重值。
- $\max(|\cdot|)$ 表示最大绝对值。
- $\alpha$ 是一个迁移强度超参数 (parameter) (hyperparameter)，通常设为 $0.5$ 左右，它控制从激活值转移到权重的难度量。$0.5$ 的值旨在使平滑后的激活值和权重之间的最大范围相等。

使用 2 的幂作为缩放因子有时对硬件效率有益，尽管其原理保持不变。



![SmoothQuant对值范围的影响](plots/4629-0.json)



> 上图说明了 SmoothQuant 如何减小激活值的最大绝对值（动态范围），同时可能增加权重的范围，使激活值更易于量化 (quantization)。

### 优点与考虑事项

SmoothQuant 具有多项优点：

- **提高准确性：** 它显著减轻了由激活离群值引起的精度损失，通常能在基本PTQ可能失效的情况下，为权重 (weight)和激活值实现准确的INT8量化 (quantization)。
- **无需重新训练：** 这是一种在量化前作为离线变换应用的PTQ技术，无需模型微调 (fine-tuning)。
- **普适性：** 它可应用于各种Transformer架构。

然而，也有一些需要考虑的地方：

- **超参数 (parameter) (hyperparameter)调整：** 迁移强度 $\alpha$ 可能需要针对特定模型或任务进行一些调整，以获得最佳结果。
- **预处理步骤：** 它在量化流程中增加了一个额外步骤，需要对模型权重进行变换和保存。
- **权重幅度增加：** 尽管通常可控，但应考虑权重幅度的增加，尽管这通常比直接量化离群激活值问题小。

与其他高级PTQ方法相比，SmoothQuant 特别关注激活值和权重分布之间的相互影响。虽然 GPTQ 等方法侧重于精密的权重量化，AWQ 基于激活值幅度保护显著权重，但 SmoothQuant 执行一个前置的数学平滑操作，使*激活值和权重*在之后都更适合标准的量化技术。它是一种有用的工具，用于实现精确的低精度量化，尤其是在激活离群值是主要问题时。

## 参考资料

- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: MLSys 2024; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  提出了一种考虑激活特征的权重量化方法，可与SmoothQuant进行比较。

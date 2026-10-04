# 变分自编码器（VAE）的优化进阶策略

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-7-advanced-vae-topics-extensions/advanced-optimization-vaes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练变分自编码器（VAE），特别是更复杂的结构和混合模型，通常会带来特殊的优化难题。在证据下界（ELBO）中，重建准确度与KL散度正则化 (regularization)项之间的精细平衡可能难以获得。常见问题包括后验坍塌（即KL项消失，潜在变量被忽视）或过强的正则化抑制重建。介绍了高级优化策略，以帮助稳定训练、提升模型性能，并有效地处理VAE目标函数。

### KL散度退火

稳定VAE训练的一种广泛采用的技术是KL退火，当退火系数表示为$\beta$时，也称为$\beta$-退火。其主要思想是在训练初期逐步增加ELBO中KL散度项的权重 (weight)。修改后的目标函数如下：
$L_{ELBO} = E_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta_t D_{KL}(q_\phi(z|x) || p(z))$
此处，$\beta_t$是训练步骤$t$时的退火系数。初始阶段，$\beta_t$通常设置为一个小值（通常为0），这让模型通过优先考虑第一项来专注于学习良好的重建。随着训练的进行，$\beta_t$逐步增加到1（如果为解耦使用$\beta$-VAE，则增加到目标值，如第3章所述）。这种缓慢引入正则化 (regularization)压力的方式有助于防止KL项过早地压倒重建损失，从而避免编码器在解码器学会有效使用潜在编码之前，将后验分布$q_\phi(z|x)$坍塌到先验分布$p(z)$（即后验坍塌）。

$\beta_t$的常见退火计划包括：

1. **线性退火：** $\beta_t = \min(1.0, \frac{\text{当前步骤}}{T_{warmup}})$，其中$T_{warmup}$是$\beta_t$从0增加到1所需的热身步骤数。
2. **周期性退火：** $\beta_t$遵循周期性模式，例如余弦波或三角波，通常在0到1之间。这有时能帮助模型摆脱不佳的局部最小值，或重新激活可能已坍塌的潜在维度。例如，一个周期的余弦计划可以是$\beta_t = 0.5 \times (1 + \cos(\pi \times (\frac{\text{当前步骤} \mod T_{cycle}}{T_{cycle}} - 1)))$，假设它从0开始，在$T_{cycle}/2$处达到1，并在$T_{cycle}$处回到0。更常见的是，它会逐步上升到1并保持，或在小值和1之间周期性波动。



[交互图表：β的KL退火计划](https://apxml.com/zh/courses/vae-representation-learning/chapter-7-advanced-vae-topics-extensions/advanced-optimization-vaes#plot-rjvtb2)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "β的KL退火计划",
    "xaxis": {
      "title": "训练迭代次数"
    },
    "yaxis": {
      "title": "β值",
      "range": [
        0,
        1.1
      ]
    },
    "height": 350,
    "legend": {
      "traceorder": "normal"
    }
  },
  "data": [
    {
      "name": "线性退火 (Twarmup=5000)",
      "x": [
        0,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000
      ],
      "y": [
        0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#228be6"
      }
    },
    {
      "name": "余弦周期 (Tcycle=10000, 演示用一个0-1-0周期)",
      "x": [
        0,
        1250,
        2500,
        3750,
        5000,
        6250,
        7500,
        8750,
        10000
      ],
      "y": [
        0,
        0.146,
        0.5,
        0.853,
        1.0,
        0.853,
        0.5,
        0.146,
        0
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#fa5252"
      }
    }
  ]
}
```

</details>



> KL退火计划示例。线性退火逐步将$\beta$增加到1并保持，而余弦计划（此处显示为一个周期用于说明）可以周期性地改变$\beta$。常见的余弦退火会逐步上升到1并保持，或在低值和1之间波动。

$T_{warmup}$或周期参数 (parameter)的选择是一个超参数 (hyperparameter)，通常需要根据数据集和模型复杂性进行一些调整。

### 自适应学习率策略

标准学习率计划，例如步长衰减或指数衰减，可以有效，但更具适应性的策略有时能带来更好的结果或更快的收敛，尤其是在损失曲面复杂的情况下。

**周期性学习率（CLR）**，由Leslie N. Smith提出，涉及在下限（base\_lr）和上限（max\_lr）之间周期性地改变学习率。其想法是，周期性地增加学习率可以帮助模型穿过鞍点或摆脱尖锐的局部最小值，而降低学习率则允许模型稳定到更宽泛、更具泛化能力的最小值。常见的周期性模式包括三角波、线性波或余弦波。



[交互图表：周期性学习率（三角计划）](https://apxml.com/zh/courses/vae-representation-learning/chapter-7-advanced-vae-topics-extensions/advanced-optimization-vaes#plot-x6e7eg)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "周期性学习率（三角计划）",
    "xaxis": {
      "title": "训练迭代次数"
    },
    "yaxis": {
      "title": "学习率"
    },
    "height": 350
  },
  "data": [
    {
      "name": "学习率",
      "x": [
        0,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000
      ],
      "y": [
        0.001,
        0.0055,
        0.01,
        0.0055,
        0.001,
        0.0055,
        0.01,
        0.0055,
        0.001
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#7048e8"
      }
    }
  ]
}
```

</details>



> 一个三角形周期性学习率计划。学习率在基准值和最大值之间波动。这里的`step_size`是2000次迭代（半个周期）。

**单周期策略**是一种特定的CLR计划，它涉及一个周期：从低学习率开始，逐步增加到最大值，然后逐步降低到非常低的速率，这通常在固定数量的训练轮次内完成。这种方法能带来更快的训练和更好的正则化 (regularization)效果。

确定CLR合适的`base_lr`和`max_lr`值的实际方法是**学习率范围测试**。这需要训练模型几个训练轮次，同时将学习率从一个很小的值线性增加到一个较大的值，并观察损失。`base_lr`可以选择损失开始下降的点，而`max_lr`可以选择损失开始爆炸或显著恶化的点。

### 优化器选择与改进

虽然Adam是一种常用优化器，但某些改进或替代方案可能对VAE有利：

- **AdamW：** 带有解耦权重 (weight)衰减的Adam。Adam中的标准L2正则化 (regularization)通常通过将权重的L2范数添加到损失中实现，这会与自适应学习率相互影响。AdamW直接将权重衰减应用于权重更新规则，这能带来更好的泛化能力，并防止在自适应学习率较大时权重衰减过快。在使用Adam时，这通常优于标准L2正则化。
  AdamW中，权重$w$在梯度$g$和学习率$\alpha$下的更新规则为：
  $w_t = w_{t-1} - \alpha (m_t / (\sqrt{v_t} + \epsilon) + \lambda w_{t-1})$
  其中$m_t$和$v_t$是偏置 (bias)的一阶和二阶矩估计，$\lambda$是权重衰减率。
- **RAdam（修正Adam）：** 解决了Adam训练早期自适应学习率存在大方差的问题，这有时会导致次优收敛。RAdam根据二阶矩估计的方差动态调整自适应学习率，有效地为学习率适应提供了一个热身阶段。

在类似Adam的优化器中，仔细调整$\beta_1$、$\beta_2$和$\epsilon$等优化器超参数 (parameter) (hyperparameter)也能影响性能，尽管默认值通常效果不错。

### 梯度管理与潜在空间使用

大梯度会使训练不稳定，而未充分使用的潜在维度则限制了VAE的表达能力。

- **梯度裁剪：** 这种技术通过将梯度限制在特定阈值（通过值或范数）来防止梯度过大。如果梯度的L2范数超过阈值，梯度将被按比例缩小。这在深度网络或带有循环组件的模型中特别有用，因为这些情况下可能发生梯度爆炸。
  示例：PyTorch中的`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`。
- **自由位（Free Bits）：** 为了对抗后验坍塌并确保潜在维度被积极使用，可以采用“自由位”策略。这种技术修改了ELBO中的KL散度项。对于每个潜在维度$z_i$，会鼓励其具有最小信息量（例如，$\lambda_i$纳特）。维度$i$的KL散度仅在其超过此阈值时才受到惩罚：
  $L_{KL,i} = \max(\lambda_i, D_{KL}(q_\phi(z_i|x) || p(z_i)))$
  总KL散度项为$\sum_i L_{KL,i}$。通过设置$\lambda_i > 0$，我们确保优化过程不会将该维度上的$D_{KL}(q_\phi(z_i|x) || p(z_i))$推向零，除非它确实不包含信息。这鼓励VAE使用更多的潜在能力。

### 批量大小及其影响

批量大小的选择影响VAE的训练动态：

- **小批量大小：** 在梯度估计中引入更多噪声。这种噪声有时能帮助优化器摆脱尖锐的局部最小值，并找到更平坦、更具泛化能力的解。然而，非常小的批量可能会减慢训练速度并导致不稳定收敛。
- **大批量大小：** 提供更准确的梯度估计，可能带来每训练轮次更快的收敛。然而，它们需要更多内存，并且有时会收敛到泛化能力较差的尖锐最小值。也有迹象表明，如果不是伴随学习率缩放等其他调整，非常大的批量大小可能会增加“泛化差距”（训练集和测试集之间的性能差异）。

改变批量大小时，通常需要调整其他超参数 (parameter) (hyperparameter)，最显著的是学习率。一个常见的启发式方法是“线性缩放规则”：如果将批量大小乘以$k$，则学习率也乘以$k$（尽管此规则有其局限性，并且对于较大的学习率可能需要一个热身期）。

### 复杂设置中的优化

当处理像VAE-GANs这样的混合模型（本章前面讨论过）时，你通常需要优化多个互相作用的网络（例如，VAE的编码器/解码器和GAN的判别器）。这通常需要：

- **独立的优化器：** 每个网络组件可能有自己的优化器。
- **差异化学习率：** 模型的不同部分可能从不同的学习率中受益。
- **仔细平衡更新：** 每个组件的更新频率（例如，比GAN中的生成器更频繁地更新判别器）需要仔细调整以保持稳定性。优化变成一个多目标问题，其中仔细平衡是必不可少的。

### 迭代改进与监控

没有一种优化策略适用于所有VAE和所有数据集。有效的优化通常是一个迭代过程，涉及实验和仔细监控。密切关注以下各项：

- **ELBO组成部分：** 分别追踪重建损失和KL散度项。
- **重建质量：** 直观检查样本，如果可用，使用量化 (quantization)指标。
- **生成样本的质量和多样性：** 对于生成任务，评估从$p(z) \rightarrow p_\theta(x|z)$生成的样本的新颖性和真实性。
- **潜在空间特性：** 监控“活跃单元”数量（具有不可忽略KL散度的潜在维度）等指标，或潜在空间的可视化。
- **验证指标：** 始终在保留的验证集上评估性能，以指导超参数 (parameter) (hyperparameter)调整并防止过拟合 (overfitting)。

从简单的优化方案开始，并根据需要逐步引入更高级的技术，是一种合理的方法。此处讨论的策略提供了一套工具，用于应对训练高级VAE模型时遇到的优化挑战。

## 参考资料

- [Auto-Encoding Variational Bayes](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG9cyrrD_WT2ilYPZGYZXQ10Gc9j3kdKyOhyA2OSvsA46xibEjEVH0fXmZA_66EMkEQpC9QshUtdz6fEKT5cjnpK3yE8dQLeQEsCd3ibfowjva0tN8J8-Gk-s-IVazkbzP27PtSXZcij1uCaQ==) — Diederik P. Kingma, Max Welling (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器（VAEs）和证据下界（ELBO）目标函数的奠基性论文。
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2o-yNYC) — Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner (2017)
  Journal: International Conference on Learning Representations
  介绍了 β-VAE，它使用加权的KL散度项，是VAE中KL退火和解耦的重要概念。
- [Cyclical Learning Rates for Training Neural Networks](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEUlaNLenX23c3HunVPFJ8EGur85ulXzS2zY_EAhd4ODJq0U2sq8FxxEGsiFSCEpHcDwi0H7n6MUj498Pp_X4MlmsKL7trHSgG93Rt--Y_jayEq_qOEBflYtaulmA5_kwBEKdv1j2Y=) — Leslie N. Smith (2017)
  Journal: IEEE Winter Conference on Applications of Computer Vision (WACV); DOI: [10.1109/WACV.2017.58](https://doi.org/10.1109/WACV.2017.58)
  提出了循环学习率（CLR）作为超参数调整和加速训练收敛的有效策略。
- [Decoupled Weight Decay Regularization](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEXw4pRVXVRWR4Gbf5xaTJEatcHB499XDDHNPVgPeuspBz0PyQGh1y5N42Le1_Em6TnVqns_QC7Po5NqtwbIbdhZYzSrHh6LuqWV3ywPkN_AHgQ1934wTaesaT7EbIs5IXzgA3_QyaFw_1YFek5DxeiiNIuVtEdKoUH5eg2Og0MLMjua5X87c0r5l6ENk-roKczmteFL-ADw9TssD-rLER2uYDZDJVRd0kctWx9mtqFjsqdAoiwrCPa0kmJ) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  介绍了AdamW，是Adam优化器的一种改进，它将权重衰减与梯度更新解耦，从而改善了泛化能力。
- [On the Variance of the Adaptive Learning Rate and Beyond](https://iclr.cc/) — Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, Jiawei Han (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1908.03265](https://doi.org/10.48550/arXiv.1908.03265)
  提出了纠正型Adam（RAdam），以解决自适应学习率在训练早期阶段的方差问题。

---

[上一节](06-%E5%8E%BB%E5%99%AAVAE%E4%B8%8E%E8%BE%93%E5%85%A5%E6%89%B0%E5%8A%A8%E9%B2%81%E6%A3%92%E6%80%A7.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E6%96%BD%20VAE-GAN%20%E6%B7%B7%E5%90%88%E6%9E%B6%E6%9E%84.md)

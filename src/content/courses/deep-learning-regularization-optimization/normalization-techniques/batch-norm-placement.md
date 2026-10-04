---
course: "deep-learning-regularization-optimization"
chapter: "normalization-techniques"
lesson: "batch-norm-placement"
sourceId: 4962
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-4-normalization-techniques/batch-norm-placement"
title: "网络中的考量与放置"
description: "讨论实际考量，例如在网络中放置批归一化层的位置。"
order: 7
plots: []
sourceHash: "e53c7f90bfc58a50bf78b4d7b8920e61a1232ae90e6dd81f7fe0970265f1c9b8"
sourceCorrections: []
---

批归一化 (normalization) (BN) 是一种为训练的稳定性和速度带来诸多益处的神经网络 (neural network)技术。然而，简单地将其加入网络并不总是最好的。明白在哪里以及如何应用它，需要考量其与其他网络部件的配合，以及您的训练设置的具体情况。

### 批归一化 (normalization)层的放置

一个主要问题是，BN 层应该相对于主层（如全连接层或卷积层）及其随后的非线性激活函数 (activation function)（如 ReLU、Sigmoid 或 Tanh）放置在哪里。

原始的批归一化论文提出在激活函数*之前*应用 BN。原因很简单：BN 旨在归一化激活层的输入，确保它们落入激活函数有响应且梯度能有效流动的范围。如果在激活*之后*进行归一化，您会失去对激活函数输入分布的这种直接控制。

考虑一个典型的序列：

1. **线性变换：** 输入 $x$ 经过一个线性或卷积层：$z = Wx + b$。
2. **批归一化：** 归一化结果 $z$：$\hat{z} = BN(z)$。这一步取代了对偏置 (bias)项 $b$ 的需要，因为它的影响会被 BN 中的均值减法抵消。BN 中可学习的 $\beta$ 参数 (parameter)实际上充当一个新的、自适应的偏置。
3. **激活：** 应用非线性激活函数：$a = g(\hat{z})$。

这个序列确保了 $g$ 的输入分布一致，减轻了梯度消失或激活饱和等问题。虽然存在一些变体，但将 BN 紧随线性/卷积层之后、非线性激活之前放置，是最常见且普遍推荐的做法。

> 一个典型的网络块，展示了批归一化在线性变换之后、激活函数之前应用。

### 与激活函数 (activation function)的配合

- **ReLU 及其变体：** 将 BN 放在 ReLU ($g(x) = max(0, x)$) 之前非常有效。通过在应用 ReLU 之前将输入分布中心移至零，BN 有助于确保更大比例的激活是非零的，可能缓解神经元变得不活跃的“ReLU 死亡”问题。
- **Sigmoid 和 Tanh：** 对于 Sigmoid ($\sigma(x) = 1 / (1 + e^{-x})$) 和 Tanh ($tanh(x)$) 等饱和激活函数，BN 尤其适用。它使输入 $\hat{z}$ 远离饱和区域（梯度接近零的地方），从而改善反向传播 (backpropagation)过程中的梯度流动并加速学习。

### 对批大小的依赖

批归一化 (normalization)使用当前小批量计算均值 $\mu$ 和方差 $\sigma^2$。这使其有效性在某种程度上依赖于批大小：

- **大批大小：** 提供对整个数据集上层激活的真实均值和方差更准确的估计。
- **小批大小：** 导致 $\mu$ 和 $\sigma^2$ 的估计更具噪声。这种噪声有时会产生轻微的正则化 (regularization)作用（如下文所述），但如果批大小*太*小（例如，小于 8 或 16，尽管确切数字取决于具体问题），估计就会变得不可靠。不可靠的估计会使训练不稳定，抵消 BN 的好处，甚至阻碍性能。

如果您因内存限制而不得不使用非常小的批大小，像层归一化或组归一化这样的技术可能是更合适的替代方案，因为它们计算统计量的方式不同，且对批大小的依赖较小。

### 正则化 (regularization)作用

使用小批量统计量而非整个数据集统计量引入的噪声，起到了一种正则化作用。每个小批量提供略有不同的归一化 (normalization)变换，阻止网络过于依赖批内任何特定的激活模式。这种作用通常是温和的，但有时可以减少对 Dropout 等其他正则化技术的需求。如果您同时使用 BN 和 Dropout，可能需要调整 Dropout 率，因为它们的正则化作用会叠加。我们将在第 8 章更详细地讨论组合技术。

### 在卷积网络 (CNN) 中的应用

在 CNN 中，BN 通常在卷积层之后、激活函数 (activation function)之前应用。一个主要区别是统计量如何计算。对于生成尺寸为（批大小、通道、高度、宽度）特征图的卷积层，BN *按每个通道*计算均值和方差，聚合批维度*和*空间维度（高度、宽度）上的统计量。这意味着同一特征图内的所有空间位置在小批量归一化 (normalization)中共享相同的均值和方差。可学习参数 (parameter) $\gamma$ 和 $\beta$ 也按每个通道应用，使网络能够独立地学习每个特征图的最佳缩放和偏移。

### 在循环网络 (RNN) 中的应用

将标准 BN 直接应用于 RNN（如 LSTM 或 GRU）的循环连接是有难度的。在每个时间步独立地跨批维度归一化 (normalization)会破坏 RNN 试图学习的时间动态，因为统计量会从一个时间步到下一个时间步显著波动。尽管存在适用于 RNN 的 BN 变体，但层归一化通常在循环架构中更有效且更容易应用，因为它在单个时间步和实例*内*跨特征进行归一化，独立于批。

### 实现细节

在 PyTorch 或 TensorFlow 等框架中使用 BN 层时，您通常会发现它们是预构建模块（例如 `torch.nn.BatchNorm1d`、`torch.nn.BatchNorm2d`）。

```python
import torch
import torch.nn as nn

# 2D 卷积层示例
conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False) # 使用 BN 时，偏置通常设为 False
bn = nn.BatchNorm2d(num_features=32) # num_features 与卷积层的输出通道数匹配
relu = nn.ReLU()

# 典型序列
# 假设 'input_tensor' 的形状为 (N, 16, H, W)
x = conv(input_tensor)
x = bn(x)
output_tensor = relu(x) # 形状 (N, 32, H, W)

# 线性层示例
linear = nn.Linear(in_features=128, out_features=64, bias=False)
bn1d = nn.BatchNorm1d(num_features=64) # num_features 与输出特征数匹配
relu_linear = nn.ReLU()

# 典型序列
# 假设 'input_vec' 的形状为 (N, 128)
y = linear(input_vec)
y = bn1d(y)
output_vec = relu_linear(y) # 形状 (N, 64)
```

您在 BN 层中可能会遇到的一些常见参数 (parameter)包括：

- `num_features`：输入的特征或通道数（例如，CNN 的通道数，线性层的输出维度）。
- `eps`：添加到分母 ($\sqrt{\sigma^2 + \epsilon}$) 的一个很小的值 ($\epsilon$)，用于数值稳定性，防止除以零。通常默认为 `1e-5` 等小值。
- `momentum`：控制推理 (inference)期间使用的均值和方差运行估计的更新规则。典型值为 `0.1`，这意味着运行估计的更新方式为 $\hat{x}_{new} = (1 - momentum) \times \hat{x} + momentum \times x_{batch}$。
- `affine`：一个布尔值，指示层是否应学习仿射变换参数 $\gamma$ 和 $\beta$（默认为 True）。将其设置为 `False` 将执行不带缩放和偏移的归一化 (normalization)。
- `track_running_stats`：一个布尔值，指示层是否应维护均值和方差的运行估计，供评估/推理期间使用（默认为 True）。

了解这些考量有助于您将批归一化有效地融入网络架构中，从而实现更稳定的训练，并可能获得更好的模型性能。请记住，像深度学习 (deep learning)中的许多技术一样，在您的特定任务和数据集上进行经验验证对于找到最佳配置很重要。

## 参考资料

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://proceedings.mlr.press/v37/ioffe15.html) — Sergey Ioffe, Christian Szegedy (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 37; Pages: 448-456; DOI: [10.48550/arXiv.1502.03167](https://doi.org/10.48550/arXiv.1502.03167)
  介绍批归一化，详细说明其机制、益处以及在神经网络中的最佳位置。
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (2016)
  Journal: arXiv preprint arXiv:1607.06450; DOI: [10.48550/arXiv.1607.06450](https://doi.org/10.48550/arXiv.1607.06450)
  提出层归一化作为批归一化的替代方案，尤其适用于循环神经网络和小批量训练。
- [Group Normalization](https://arxiv.org/abs/1803.08494) — Yuxin Wu, Kaiming He (2018)
  Journal: European Conference on Computer Vision (ECCV); Volume: 11211; Pages: 3-19; DOI: [10.48550/arXiv.1803.08494](https://doi.org/10.48550/arXiv.1803.08494)
  介绍组归一化，该方法对不同批次大小具有鲁棒性，在许多情况下性能与批归一化相当。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，全面涵盖深度学习主题，包括归一化技术和最佳实践。
- [torch.nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) — PyTorch Documentation (2024)
  PyTorch框架中BatchNorm2d的官方文档，说明其用法、参数和行为。

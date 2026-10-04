---
course: "deep-learning-regularization-optimization"
chapter: "normalization-techniques"
lesson: "batch-norm-test-time"
sourceId: 4959
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-4-normalization-techniques/batch-norm-test-time"
title: "测试时的批量归一化"
description: "阐述批量归一化在推理时如何使用总体统计量。"
order: 6
plots: []
sourceHash: "d6f8a1ef3bbe662486a9019b94232b91ebc567f7107cf442d90636542b139b59"
sourceCorrections: []
---

在训练期间，批量归一化 (normalization)通过使用从*当前小批量*计算的均值和方差来标准化层输入，从而发挥作用。这种方法有助于稳定训练过程，如前所述。然而，一个问题自然而然地出现：当我们想要使用训练好的模型对新数据进行预测（推理 (inference)或测试）时，会发生什么？

在推理时，有几个问题使得使用小批量统计数据变得不切实际或不合需求：

1. **单个实例：** 通常，您可能希望对单个数据点进行预测。从一个实例的“小批量”中计算有意义的均值和方差是不可行的。
2. **确定性：** 在推理期间，我们需要模型的输出对于给定输入保持一致和确定。在测试时依赖可能不同的小批量统计数据会给预测带来随机性，这通常是不希望的。相同输入图像的输出应该始终相同。
3. **数据批量处理：** 在推理期间，我们可能不会批量处理数据，或者批量大小可能显著不同。

### 推理 (inference)时使用总体统计量

标准方法是使用表示*整个训练数据集*（或其合理估计值）的统计数据，而不是仅仅使用当前小批量的数据。这些常被称为*总体统计量*。具体来说，在推理期间，批量归一化 (normalization)使用从训练数据中得出的均值和方差的固定估计值来标准化输入。

全局统计量对于在测试阶段标准化输入是必要的。这些统计量通常在训练过程中通过指数移动平均来估计。当模型经过多个小批量数据训练时，框架会跟踪每个被归一化特征维度的均值 ($\mu_{pop}$) 和方差 ($\sigma^2_{pop}$) 的运行估计值。

在训练期间，对于每个小批量 $\mathcal{B}$，框架会计算批量均值 $\mu_{\mathcal{B}}$ 和方差 $\sigma^2_{\mathcal{B}}$。然后，这些值用于更新运行估计值，通常会使用一个动量项（我们称之为`momentum`，通常接近0.1）：


$$
\mu_{pop} \leftarrow (1 - \text{动量}) \times \mu_{pop} + \text{动量} \times \mu_{\mathcal{B}}
$$


$$
\sigma^2_{pop} \leftarrow (1 - \text{动量}) \times \sigma^2_{pop} + \text{动量} \times \sigma^2_{\mathcal{B}}
$$


这些更新发生在每个训练步骤中，逐渐修正 $\mu_{pop}$ 和 $\sigma^2_{pop}$ 的估计值，以体现训练期间观测到的激活值的总体统计情况。

### 推理 (inference)时的计算

因此，当模型设置为评估模式时（例如，在PyTorch中使用`model.eval()`），批量归一化 (normalization)层会改变其行为。它不再从输入中计算 $\mu_{\mathcal{B}}$ 和 $\sigma^2_{\mathcal{B}}$，而是使用预先计算好的运行估计值 $\mu_{pop}$ 和 $\sigma^2_{pop}$。

推理期间的归一化计算变为：


$$
\hat{x}_i = \frac{x_i - \mu_{pop}}{\sqrt{\sigma^2_{pop} + \epsilon}}
$$


此处，$x_i$ 是输入特征，$\mu_{pop}$ 和 $\sigma^2_{pop}$ 分别是估计的总体均值和方差，而 $\epsilon$ 是为数值稳定性而添加的小常数。

请记住，学习到的缩放参数 (parameter) $\gamma$ 和平移参数 $\beta$ 在此归一化之后仍会应用，就像在训练期间一样：


$$
y_i = \gamma \hat{x}_i + \beta
$$


这些 $\gamma$ 和 $\beta$ 参数是模型学习到的权重 (weight)的一部分，并在训练后保持固定。

### 框架处理

幸运的是，PyTorch和TensorFlow等深度学习 (deep learning)框架会自动处理训练和推理 (inference)行为之间的这种切换。当您定义一个批量归一化 (normalization)层时，它会在内部维护这些运行统计数据。

```python
import torch
import torch.nn as nn

# 示例：一个包含 Conv -> BN -> ReLU 的简单模块
conv_block = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
    # 64个通道的批量归一化层
    nn.BatchNorm2d(num_features=64), # 跟踪 running_mean 和 running_var
    nn.ReLU()
)

# --- 训练期间 ---
# 将模型设置为训练模式
conv_block.train()
# 生成一些虚拟输入数据 (批量大小=4, 通道数=3, 高度=32, 宽度=32)
input_data_train = torch.randn(4, 3, 32, 32)
# 训练期间的前向传播：BN 使用小批量统计数据并更新运行统计数据
output_train = conv_block(input_data_train)

# --- 推理期间 ---
# 将模型设置为评估模式
conv_block.eval()
# 生成一些虚拟测试数据 (批量大小=1)
input_data_test = torch.randn(1, 3, 32, 32)
# 推理期间的前向传播：BN 使用存储的 running_mean 和 running_var
output_test = conv_block(input_data_test)

# 打印 BN 层跟踪的运行均值
# 注意：这些值在训练阶段被填充
print("运行均值形状:", conv_block[1].running_mean.shape)
print("运行方差形状:", conv_block[1].running_var.shape)
```

在上面的PyTorch示例中，调用`conv_block.train()`会将`Sequential`块内的模块（包括`BatchNorm2d`）设置为训练模式。在此模式下，`BatchNorm2d`会从输入批次中计算统计数据并更新其内部的`running_mean`和`running_var`。调用`conv_block.eval()`会将模块切换到评估模式。此时，`BatchNorm2d`不再使用当前输入批次的统计数据，而是使用在训练期间估计的`running_mean`和`running_var`。这确保了推理期间输出的一致性和确定性。

大多数批量归一化实现都有一个参数 (parameter)，例如`track_running_stats`（通常默认为`True`）。当设置为`True`时，该层在训练期间估计总体统计数据并在评估期间使用它们。如果设置为`False`，它将*总是*使用批量统计数据，这对于标准推理来说通常是不希望的，除非您有特殊原因。

总之，批量归一化在训练和测试之间调整其行为。它在训练期间使用动态的小批量统计数据来稳定学习，并通过移动平均估计总体统计数据。在测试时，它使用这些固定的总体统计数据以及学习到的缩放和平移参数，以确保输入的确定性与一致性归一化。

## 参考资料

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) — Sergey Ioffe and Christian Szegedy (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Pages: 448-456; DOI: [10.48550/arXiv.1502.03167](https://doi.org/10.48550/arXiv.1502.03167)
  介绍批量归一化的原始论文，阐述其训练方法和推理时总体统计量的估计。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8.7.1
  一本权威教材，详细讨论了批量归一化，包括训练和推理时操作的区别。
- [torch.nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) — PyTorch Developers (2024)
  PyTorch 2D批量归一化层的官方文档，详细说明了 `running_mean`、`running_var` 和 `track_running_stats` 在训练和评估模式下的使用。

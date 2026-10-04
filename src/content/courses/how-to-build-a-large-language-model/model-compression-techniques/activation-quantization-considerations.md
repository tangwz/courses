---
course: "how-to-build-a-large-language-model"
chapter: "model-compression-techniques"
lesson: "activation-quantization-considerations"
sourceId: 6093
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-27-model-compression-techniques/activation-quantization-considerations"
title: "激活量化考量"
description: "讨论推理期间量化模型激活的难题与技术。"
order: 3
plots: []
sourceHash: "62a7b6b8de8e7f5f8f345bf5d2250fa9fdeda12f7c0e48b8aa96e094422c3694"
sourceCorrections: []
---

如前所述，对模型权重 (weight)进行量化 (quantization)可以显著节省内存，但对激活进行量化会带来它自己的一系列困难和考量。激活，即神经元层的输出作为输入传递到下一层，本质上是动态的。与训练后固定的权重不同，激活值会随模型处理的每个输入样本而变动。这种动态特性使其量化更复杂但也同样重要，能实现推理 (inference)速度的最大提升和内存带宽的减少，尤其是在有专用低精度计算单元的硬件上。

### 难题：动态范围和敏感性

量化 (quantization)激活的主要难题源于其可能很宽且不可预测的动态范围。考虑ReLU或GeLU等激活函数 (activation function)的输出。这些非线性特性可以产生跨越多个数量级的值，并且与通常遵循某种可预测分布（例如，围绕零居中）的权重 (weight)不同，激活分布可以根据输入数据和网络中特定层而明显不同。

> 激活范围在Transformer块内不同层之间明显不同。前馈网络（FFN）中的ReLU或GeLU等非线性函数尤其可以扩展动态范围。

使用低精度格式（如INT8）量化具有如此宽范围的张量迫使做出权衡。如果按比例量化以适应极端最大值和最小值（异常值），那么表示大多数值（可能聚集在小得多的范围里）的精度会变得非常粗糙。这种精度损失，称为量化误差，会显著降低模型精度。反之，如果您为值的密集聚类优化尺度，异常值将被截断，可能丢失重要信息。激活，尤其是在注意力机制 (attention mechanism)或中间FFN层内，可能对此截断或精度损失特别敏感。

### 校准：确定量化 (quantization)参数 (parameter)

为了有效映射激活的浮点范围到低精度整数范围，我们需要确定适当的量化参数：一个尺度因子 ($s$) 和一个零点 ($z$)。一般映射是：


$$
\text{量化值} = \text{取整}(\frac{\text{浮点值}}{s}) + z
$$


找到最优 $s$ 和 $z$ 值的过程称为**校准**。它通常涉及通过模型输入一个代表性数据集（训练或验证数据的一个子集，通常是几百到几千个样本），并观察网络中不同点激活值的范围。存在几种校准方法：

1. **最小-最大校准：** 这是最简单的方法。记录校准期间观察到的最小 ($x_{min}$) 和最大 ($x_{max}$) 激活值。然后计算尺度 $s$ 和零点 $z$，将这个观察到的范围 $[x_{min}, x_{max}]$ 映射到目标整数范围（例如，有符号INT8的 $[-128, 127]$）。

   - *优点:* 实现简单。
   - *缺点:* 对异常值高度敏感。一个极端值会显著恶化典型值的量化精度。
2. **均方误差（MSE）校准：** 此方法迭代观察到的最小-最大范围内的不同潜在截断范围（阈值）。对于每个阈值，它计算量化参数并测量原始浮点激活与其量化-反量化等效值之间的平均平方误差。选择使MSE最小的阈值。

   - *优点:* 与简单最小-最大方法相比，通常能在截断异常值和保持大部分分布的精度之间找到更好的平衡。
   - *缺点:* 由于需要搜索阈值，计算量比最小-最大方法大。
3. **熵（KL散度）校准：** 此方法旨在最小化量化过程中的信息损失。它选择量化参数 ($s$ 和 $z$)，使得原始浮点激活分布与量化-反量化激活分布之间的Kullback-Leibler（KL）散度最小化。

   - *优点:* 通常被认为是最有效的方法，特别是对于非均匀分布，因为它直接尝试保持原始激活分布的形状。
   - *缺点:* 可能是计算最复杂的校准方法。

以下是PyTorch风格的示例，说明了在校准特定激活张量期间如何使用观察器：

```python
import torch

# Assume 'activation_tensor' holds the activations from a specific layer
# during a forward pass with calibration data.
# 假设 'activation_tensor' 在使用校准数据进行前向传播期间，保存了来自特定层的激活值。

# --- Calibration Phase ---
# --- 校准阶段 ---
# Observer object tracks statistics
# 观察器对象跟踪统计信息
# Example: MinMaxObserver
# 示例：MinMaxObserver
class MinMaxObserver:
    def __init__(self):
        self.min_val = torch.tensor(float('inf'))
        self.max_val = torch.tensor(float('-inf'))

    def forward(self, x):
        # Detach tensor to avoid tracking
        # gradients during observation
        # 分离张量以避免在观察期间追踪梯度
        x_detached = x.detach()
        self.min_val = torch.min(x_detached, self.min_val)
        self.max_val = torch.max(x_detached, self.max_val)
        return x # Pass input through unmodified
                 # during calibration
                 # 在校准期间，不修改地通过输入

    def calculate_qparams(self, dtype=torch.qint8):
        # Determine scale and zero_point based on
        # observed min/max
        # 根据观察到的最小值/最大值确定尺度和零点
        qmin = torch.iinfo(dtype).min
        qmax = torch.iinfo(dtype).max

        scale = (self.max_val - self.min_val) / float(qmax - qmin)

        # Ensure scale is not zero
        if scale == 0.0:
            scale = torch.tensor(1e-8) # Small epsilon or
                                       # handle appropriately
                                       # 确保尺度不为零
                                       # 很小的epsilon值，或适当地处理

        zero_point = qmin - torch.round(self.min_val / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)
        zero_point = zero_point.to(torch.int) # Clamp to valid
                                              # range
                                              # 限制在有效范围

        return scale, zero_point

# In your model's forward pass during calibration:
# 在模型校准阶段的前向传播中：
# observer = MinMaxObserver() # Or other observer type
                             # (MSE, Entropy based)
                             # 观察器 = MinMaxObserver() （或其他观察器类型，如基于MSE、基于熵的）
# ... layer computation ...
# ... 层计算 ...
# activations = some_layer(input)
# activations = observer(activations) # Observer watches
                                    # the activations
                                    # 激活值 = 某个层(输入)
                                    # 激活值 = 观察器(激活值) （观察器观察激活值）
# ... rest of forward pass ...
# ... 前向传播的其余部分 ...

# After running calibration data through:
# 运行校准数据后：
# scale, zero_point = observer.calculate_qparams()
# print(f"Calculated Scale: {scale}, Zero-Point: {zero_point}")
# print(f"计算出的尺度: {scale}, 零点: {zero_point}")

# --- Inference Phase ---
# --- 推理阶段 ---
# Use the calculated scale and zero_point
# to quantize activations
# 使用计算出的尺度和零点来量化激活值
# quantized_activations = torch.quantize_per_tensor(
#     activation_tensor, scale, zero_point, torch.qint8
# )
```

### 每张量与更细粒度的量化 (quantization)

就像权重 (weight)一样，激活可以使用不同粒度进行量化：

- **每张量：** 对层生成的整个激活张量使用单一的尺度因子 $s$ 和零点 $z$。这是最简单的方法，开销最低。
- **每通道/组：** 对于可能具有不同激活统计数据的卷积层或线性层，为每个通道或每组通道使用独立的量化参数 (parameter)可以提高精度。
- **每token（Transformer）：** 在Transformer中，激活值通常具有对应于序列长度和隐藏大小的维度。按每个token进行量化（为序列中每个token的隐藏维度向量 (vector)独立计算 $s$ 和 $z$）有时能更准确地捕获沿序列长度维度的变化，尽管会增加开销。

选择取决于特定的层类型、观察到的激活分布以及可接受的性能开销。每张量量化因其简单性和效率而普遍使用，但更细粒度的量化可能需要在敏感层中恢复精度。

### 校准期间的异常值处理

如前所述，异常值严重影响最小-最大校准，也可能对MSE/熵方法产生负面影响。一种常见的缓解方法是**截断**。在计算量化 (quantization)参数 (parameter)之前，观察到的激活范围根据百分位数进行截断。例如，不是使用绝对最小值和最大值，可以使用第1和第99百分位数，或第0.1和第99.9百分位数。

> 截断在计算量化参数前移除极端异常值，可能提高大多数值的精度，代价是截断的异常值被饱和。

截断有助于将量化范围集中在数据主体上，提高典型值的精度。然而，它对截断的异常值引入了饱和，如果这些异常值携带重要信息，可能有害。选择正确的截断阈值通常需要经验调整。

### 静态量化 (quantization)与动态量化

激活量化参数 (parameter)可以离线（静态）或即时（动态）确定：

- **静态量化（训练后静态量化 - PTSQ）：** 这是性能敏感应用中最常见的方法。每个激活张量的尺度 $s$ 和零点 $z$ 使用上述校准过程*一次性*确定。这些固定参数随后存储并在推理 (inference)期间使用。这使得量化和与前序操作的潜在融合能够高效进行，而无需范围计算的运行时开销。
- **动态量化（训练后动态量化 - PTDQ）：** 在此方法中，激活张量的最小/最大范围（以及 $s$ 和 $z$）在推理期间对每个输入*动态*计算。这避免了对校准数据集的需求。虽然应用起来更简单，但计算范围的运行时成本通常会抵消使用低精度算术带来的任何计算加速。其主要好处通常是减少内存占用和带宽，因为激活值仍可以低精度格式存储和传输，即使计算涉及动态反量化。对于延迟很重要的LLM，激活的动态量化使用频率低于静态量化。

### 在量化 (quantization)感知训练 (QAT) 中的作用

量化感知训练直接将量化效果的模拟（包括权重 (weight)*和激活*）集成到训练循环中。它使用“伪量化”节点来模拟在前向传播期间量化和反量化激活的过程，同时允许梯度在反向传播 (backpropagation)期间通过。

```python
import torch
import torch.nn as nn
import torch.ao.quantization as quant

class QuantizableLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        # QAT uses observers during training which double as fake quantizers
        # QAT 在训练期间使用观察器，这些观察器也兼作伪量化器
        # Placeholder for where activation quantization simulation happens
        # 激活量化模拟发生处的占位符
        self.activation_quant_stub = quant.QuantStub()
        self.activation_dequant_stub = quant.DeQuantStub()

    def forward(self, x):
        # Simulate quantization of input activation to this layer
        # 模拟输入到该层的激活量化
        x = self.activation_quant_stub(x)

        x = self.linear(x)
        x = self.relu(x)

        # Dequantize output activation before passing to next
        # (potentially float) layer
        # 在传递到下一个（可能是浮点）层之前，反量化输出激活
        x = self.activation_dequant_stub(x)
        return x

# During QAT, these stubs (along with weight fake quantization)
# simulate quantization errors, allowing the model to adapt.
# 在 QAT 期间，这些桩（以及权重伪量化）
# 模拟量化误差，使模型能够适应。
# After QAT, the model can be converted to a true quantized model
# using the statistics gathered by the observers within the stubs.
# QAT 结束后，模型可以使用桩内观察器收集到的统计数据转换为真正的量化模型。
```

通过在训练期间让模型接触量化噪声，QAT 允许网络调整其权重和激活分布，变得对量化误差更具弹性。这通常会产生明显更高的精度，尤其是在目标位宽非常低（如 INT4）时，或处理高度敏感的激活分布时。

### 总结

重要考量包括：

- **动态范围：** 激活随输入而变动，可以有宽广、不可预测的范围，使其对量化 (quantization)误差敏感。
- **校准：** 使用代表性数据选择适当的量化参数 (parameter) ($s, z$) 非常重要。最小-最大、MSE和熵等方法在简单性和稳定性之间提供不同的权衡。
- **异常值处理：** 截断等技术通常是必要的，以防止极端值主导量化范围。
- **粒度：** 每张量量化很常见，但更细粒度（每通道、每token）的量化可能为了精度所需。
- **静态与动态：** 静态量化（校准后参数固定）通常在性能方面更受青睐；动态量化避免了校准，但增加了运行时开销。
- **QAT：** 使用模拟量化训练模型通常能带来最佳精度，尤其是在激进的量化方案中。

成功应对这些考量可以显著减少内存带宽并潜在提升计算速度，使大型模型更适合部署。

## 参考资料

- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html) — PyTorch Developers (2019)
  Publisher: PyTorch Foundation
  这份全面的文档详细阐述了激活量化的实际操作，包括不同观察器类型（例如，用于KL散度的HistogramObserver、MovingAverageMinMaxObserver）、静态与动态量化，以及QAT如何在框架中实现。

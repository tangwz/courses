# 量化感知训练 (QAT)

来源：[原文](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-2-advanced-quantization-techniques/quantization-aware-training-qat)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然训练后量化 (quantization) (PTQ) 提供了一种计算成本较低的模型量化方式，但其效果可能会明显下降，尤其是在目标位宽较低（如 INT4 或更低）时。PTQ 是在模型训练*之后*根据少量数据集校准量化参数 (parameter)的，但模型本身并未学习如何弥补量化引入的噪声。这可能导致对精度有较高要求的 LLM 出现不可接受的精度下降。

量化感知训练 (QAT) 通过在模型训练或微调 (fine-tuning)过程中模拟量化效果来解决这一局限。其主要思路是让模型感知即将进行的量化步骤，从而在训练期间调整其权重 (weight)，以减少量化可能导致的精度损失。与 PTQ 相比，这通常会带来更高的模型保真度，尤其是在较低精度下，但代价是需要访问训练流程和代表性数据。

### 模拟量化 (quantization)：伪量化的前向传播

QAT 的工作原理是在计算图中插入节点，这些节点模拟量化和反量化的效果。这些节点通常被称为“伪”量化节点或量化-反量化 (QDQ) 节点。

在 QAT 的前向传播过程中：

1. 一个浮点权重 (weight)或激活张量进入 QDQ 节点。
2. 张量使用估算或学习到的尺度因子和零点，量化为目标低精度格式（例如 INT8、INT4）。这包括缩放、舍入、截断和移位。
   
   $$
   x_{quant} = \text{截断}(\text{舍入}(x_{float} / \text{尺度} + \text{零点}))
   $$
   
3. 重要的是，这个量化后的整数值会立即使用相同的尺度因子和零点反量化回浮点值。
   
   $$
   x_{dequant} = (x_{quant} - \text{零点}) \times \text{尺度}
   $$
   
4. 这个反量化后的张量（现在包含模拟量化引入的“误差”或“噪声”）用于该层的后续浮点运算。

模型继续使用标准反向传播 (backpropagation)进行训练，但权重和激活会不断受到模拟量化噪声的推动。这促使模型学习对精度降低具有更强内在适应性的参数 (parameter)值。

> 量化感知训练 (QAT) 期间，层内计算前插入量化-反量化 (QDQ) 节点的工作流程图。

### 梯度处理：直通式估计器 (STE)

在反向传播 (backpropagation)过程中，会出现一个重要问题。量化 (quantization)中固有的舍入操作是不可微分的，这意味着其梯度几乎处处为零。这将阻碍学习过程，因为梯度无法通过 QDQ 节点反向传播以更新原始的高精度权重 (weight)。

标准的解决办法是直通式估计器 (STE)。在反向传播过程中，STE 有效地将量化函数视为在梯度计算方面的恒等函数。它只是简单地将传入的梯度未经修改地通过 QDQ 节点，忽略不可微分的舍入步骤。

数学上，如果 $y = \text{量化}(x)$，使用 STE 的梯度计算近似为：


$$
\frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial y} \times \frac{\partial y}{\partial x} \approx \frac{\partial L}{\partial y} \times 1
$$


虽然这在数学上不完全精确，但在实践中效果非常好。它使得基于量化噪声前向传播计算出的梯度能够更新底层的浮点权重，引导它们趋向于相对于量化扰动位于损失函数 (loss function)平坦区域的值。

### LLM 的实际 QAT 应用：实现细节

将 QAT 有效应用于大型语言模型需要仔细考量：

1. **初始化：** QAT 几乎总是作为微调 (fine-tuning)步骤执行。你从一个训练良好的 FP32 LLM 开始，然后启用 QAT 对其进行相对较少轮次或步骤的微调。从头开始使用 QAT 训练大型 LLM 通常不切实际。
2. **微调策略：** QAT 微调阶段通常比原始预训练 (pre-training)阶段短得多。与初始预训练学习率相比，学习率通常会降低（例如，降低 10 倍-100 倍）。短暂的学习率热身，然后是衰减策略是常见的做法。
3. **QDQ 节点的放置：** 确定在哪里插入 QDQ 节点很重要。通常，它们应用于：
   - 线性层 (`nn.Linear`) 和嵌入 (embedding)层 (`nn.Embedding`) 的权重 (weight)。
   - 作为计算密集型操作或非线性操作输入的激活（例如，GELU 的输入、注意力机制 (attention mechanism)的输出、残差连接）。
     对于层归一化 (normalization) (Layer Normalization) 和 Softmax 等操作需要注意。通常为了稳定性，这些操作会保持较高精度（FP16/FP32），尽管对其输入/输出进行量化 (quantization)很常见。对于 LLM 中的权重，由于通道间的参数 (parameter)范围差异，按通道量化通常是必需的。
4. **量化参数（尺度/零点）：** 这些参数定义了从浮点到量化域的映射。
   - **固定：** 它们可以在开始 QAT 微调前使用校准数据进行估算（类似于 PTQ 校准），并在训练期间保持固定。这更简单。
   - **可学习：** 另外，尺度和零点（或控制裁剪范围的参数）可以被视为可学习参数，并在 QAT 期间通过反向传播 (backpropagation)进行更新。这通常会带来更好的结果，但会增加复杂性。训练期间使用可学习权重裁剪 (LWC) 或通过指数移动平均 (EMA) 跟踪统计数据等技术。
5. **批归一化折叠：** 如果模型使用批归一化 (Batch Normalization)（在标准 Transformer 中不常见，但在变体中可能会出现），则通常在 QAT 开始前或 QAT 过程中，将 BN 参数折叠到前一个线性层的权重和偏置 (bias)中。

### QAT 的权衡：精度与成本

QAT 的主要优势在于其实现更高精度的可能性，尤其是在低于 8 位精度时。通过将量化 (quantization)噪声集成到训练循环中，模型会适应，通常能够恢复 PTQ 方法损失的大部分甚至全部精度。

然而，这也有其代价：

- **计算开销：** QAT 需要对模型进行微调 (fine-tuning)，这涉及到反向传播 (backpropagation)和权重 (weight)更新，使其计算密集程度显著高于 PTQ。
- **数据需求：** 需要访问代表性的训练或微调数据集。
- **复杂性：** 正确实施 QAT，包括 QDQ 节点放置、微调策略和量化参数 (parameter)处理，比应用 PTQ 更复杂。训练稳定性有时也需要注意。



[交互图表：精度与位宽对比 (PTQ vs. QAT)](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-2-advanced-quantization-techniques/quantization-aware-training-qat#plot-1lpsvg4)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "精度与位宽对比 (PTQ vs. QAT)",
    "xaxis": {
      "title": "量化位宽",
      "autorange": "reversed",
      "dtick": 1
    },
    "yaxis": {
      "title": "模型精度 (归一化)",
      "range": [
        0.7,
        1.05
      ]
    },
    "legend": {
      "yanchor": "bottom",
      "y": 0.01,
      "xanchor": "right",
      "x": 0.99
    },
    "font": {
      "family": "sans-serif"
    }
  },
  "data": [
    {
      "x": [
        32,
        16,
        8,
        6,
        4,
        3
      ],
      "y": [
        1.0,
        1.0,
        0.99,
        0.98,
        0.96,
        0.92
      ],
      "mode": "lines+markers",
      "name": "QAT",
      "line": {
        "color": "#1c7ed6"
      }
    },
    {
      "x": [
        32,
        16,
        8,
        6,
        4,
        3
      ],
      "y": [
        1.0,
        1.0,
        0.98,
        0.94,
        0.85,
        0.75
      ],
      "mode": "lines+markers",
      "name": "PTQ",
      "line": {
        "color": "#f03e3e"
      }
    }
  ]
}
```

</details>



> 随着量化精度的降低，尤其是在低于 8 位时，QAT 通常比 PTQ 保持更高的精度。

QAT 是一种强大的技术，用于实现高水平的压缩和加速，同时保持模型保真度。当微调成本可接受且训练基础设施可用时，QAT 通常是 LLM 量化到较低位宽的首选方法，为高效部署做好准备。PyTorch（使用 `torch.ao.quantization`）和 TensorFlow/Keras（通常通过 TFLite 转换器的 QAT 功能）等框架提供了有助于其实现的工具。

## 参考资料

- [Quantization API (torch.ao.quantization)](https://pytorch.org/docs/stable/quantization.html) — PyTorch Documentation (2024)
  PyTorch量化API的官方文档，包含QAT的实现细节。

---

[上一节](02-%E8%AE%AD%E7%BB%83%E5%90%8E%E9%87%8F%E5%8C%96%20%28PTQ%29.md) · [下一节](04-%E6%9E%81%E8%87%B4%E9%87%8F%E5%8C%96.md)

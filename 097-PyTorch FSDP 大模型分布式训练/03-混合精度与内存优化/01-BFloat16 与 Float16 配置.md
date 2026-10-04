# BFloat16 与 Float16 配置

来源：[原文](https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization/bfloat16-vs-float16-configurations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在标准GPU集群上，以完整的32位精度（FP32）训练高参数 (parameter)模型效率低下，并且由于内存限制通常无法实现。混合精度训练通过在大多数算术操作中使用16位格式来处理这种情况，同时在FP32中保留权重 (weight)的原始副本以进行更新。对于PyTorch FSDP，在脑浮点（BFloat16）和标准浮点（Float16）之间选择，不仅仅是个人偏好，而是由硬件能力和收敛稳定性需求所决定的。

### 16位格式的构成

要做出明智的决定，必须了解这些格式的位级别表示。主要区别在于16位在指数（范围）和尾数（精度）之间是如何分配的。

标准 **Float16 (IEEE 754)** 为符号分配1位，为指数分配5位，为尾数（有效数）分配10位。这种格式在狭窄的动态范围内提供更高的精度。有限的指数宽度意味着可表示的最大数字是65,504，最小的正规数大约是$6.1 \times 10^{-5}$。在深度学习 (deep learning)中，梯度经常低于此阈值（下溢）或激活值超出最大值（上溢），这要求进行积极的损失缩放。

由Google Brain开发的 **BFloat16** 改变了这种权衡。它为符号分配1位，为指数分配8位，为尾数分配7位。指数宽度与FP32相同。因此，BFloat16保留了标准32位浮点的动态范围，实际上作为截断的FP32。尽管它损失了显著的精度（与FP16相比，尾数少了3位），但扩展的范围使其固有地防止下溢和上溢，无需复杂的缩放逻辑。

> 浮点格式位分配的比较。BFloat16保持FP32的指数宽度以保留动态范围。

### 收敛稳定性与损失缩放

这些格式之间的操作差异主要体现在训练循环的稳定性上。

使用 **Float16** 时，梯度常变得非常小以至于消失（下溢至零）。为应对此情况，PyTorch 使用 `GradScaler`。此工具在反向传播 (backpropagation)前将损失乘以一个缩放因子（例如 $2^{16}$），将梯度移入FP16的可表示范围。反向传播后，梯度在优化器步骤前被取消缩放。这会带来计算开销和额外的环节。如果缩放因子过高，梯度将上溢至无穷大；如果过低，则会下溢。缩放器必须动态调整此因子，这可能在检测到 `Inf` 或 `NaN` 值时导致跳过步骤。

**BFloat16** 完全消除了损失缩放的需要。由于其动态范围与FP32匹配（$\approx 10^{-38} \text{ 至 } 10^{38}$），梯度在标准训练运行中很少下溢或上溢。这种稳定性对于使用Transformer架构训练的大型语言模型（LLM）尤其重要，因为注意力分数和激活峰值可能不稳定。

下表展示了可表示范围的限制。请注意Float16与BFloat16相比，达到上限的速度有多快。



[交互图表：交互图表](https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization/bfloat16-vs-float16-configurations#plot-1085wzm)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "幅值 (对数刻度)",
      "type": "log",
      "range": [
        -45,
        45
      ],
      "showgrid": true,
      "gridcolor": "#dee2e6"
    },
    "yaxis": {
      "showticklabels": false,
      "showgrid": false
    },
    "shapes": [
      {
        "type": "rect",
        "x0": 1e-38,
        "x1": 3.4e+38,
        "y0": 2,
        "y1": 3,
        "fillcolor": "#d0bfff",
        "opacity": 0.7,
        "line": {
          "width": 0
        }
      },
      {
        "type": "rect",
        "x0": 6e-05,
        "x1": 65504,
        "y0": 0,
        "y1": 1,
        "fillcolor": "#99e9f2",
        "opacity": 0.7,
        "line": {
          "width": 0
        }
      }
    ],
    "annotations": [
      {
        "x": 1.0,
        "y": 2.5,
        "text": "BFloat16 / FP32 范围",
        "showarrow": false,
        "font": {
          "color": "#5f3dc4"
        }
      },
      {
        "x": 1.0,
        "y": 0.5,
        "text": "Float16 范围",
        "showarrow": false,
        "font": {
          "color": "#0c8599"
        }
      }
    ],
    "height": 250,
    "margin": {
      "t": 30,
      "b": 40,
      "l": 40,
      "r": 40
    }
  },
  "data": []
}
```

</details>



> BFloat16的有效动态范围明显宽于Float16，与FP32的操作界限相符。

### FSDP混合精度配置

在FSDP中，混合精度通过 `MixedPrecision` 配置对象来控制。此类指定了训练生命周期中三个特定阶段的数据类型（dtype）：

1. **`param_dtype`**: 参数 (parameter)在正向传播前被转换成的格式。
2. **`reduce_dtype`**: 用于跨进程梯度同步（AllReduce）的格式。
3. **`buffer_dtype`**: 用于缓冲区（例如 BatchNorm 统计数据）的格式。

正确设置这些参数决定了您的运行的内存节省和数值安全性。

#### BFloat16 配置 (推荐用于 Ampere+)

如果您的集群配备了NVIDIA Ampere (A100)、Hopper (H100) 或更新的架构，BFloat16是常规选择。它通过Tensor Cores为矩阵乘法提供硬件加速。

```python
import torch
from torch.distributed.fsdp import MixedPrecision

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,  # 关于归约的注意事项见下文
    buffer_dtype=torch.bfloat16,
)
```

请注意 `reduce_dtype`。尽管将其设置为 `bfloat16` 以减少通信带宽很诱人，但这样做是有风险的。BFloat16精度较低（只有7个尾数位）。当在数百个GPU上累加梯度（AllReduce）时，会遇到“淹没”现象，即向大的累积缓冲区添加小的梯度更新会导致小值完全丢失。保持 `reduce_dtype=torch.float32` 可确保梯度平均保持精确，而 `param_dtype=torch.bfloat16` 则确保繁重的计算（正向/反向）使用更快、更轻的格式。

#### Float16 配置 (旧版硬件)

对于像V100 (Volta) 或T4 (Turing) 这样的旧版硬件，BFloat16不被原生支持。您必须使用Float16并管理 `GradScaler`。

```python
fp16_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

# 需要外部缩放器管理
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
scaler = ShardedGradScaler()
```

当使用 `reduce_dtype=torch.float16` 时，您会获得通信速度的提升，但会增加归约期间上溢的风险。不过，由于FP16在尾数方面比BF16具有更高的精度，它较不容易受到淹没效应的影响，这使得16位归约比BF16情况略微安全，前提是数值保持在范围内。

### 精度-吞吐量 (throughput)权衡

精度的选择会影响内存吞吐量（DRAM带宽）和算术吞吐量（TFLOPS）。

1. **内存带宽**: 相较于FP32，BF16和FP16都将模型权重 (weight)和激活的内存流量减半。这通常是LLM训练的主要加速因素，因为这些工作负载通常受限于内存而非计算。
2. **计算吞吐量**: 在A100 GPU上，BF16和FP16 Tensor Cores提供理论上相同的峰值吞吐量（312 TFLOPS）。然而，BF16通常会带来略好的性能，因为它避免了与动态损失缩放检查相关的内核启动开销和内存读写操作。

在FSDP中，`param_dtype` 带来的内存节省使您能够增加本地批次大小。如果模型层在FP32中占用 $W$ 字节，那么使用混合精度将该层正向传播所需的活动内存减少到 $W/2$，加上在优化器状态中保留FP32主权重的开销（这些在FSDP中是分片的）。

### 建议总结

| 特性 | BFloat16 | Float16 |
| --- | --- | --- |
| **硬件要求** | NVIDIA Ampere (A100) 或更新版本 | Volta (V100) 或更新版本 |
| **尾数精度** | 低 (7 位) | 高 (10 位) |
| **动态范围** | 高 (8 位指数) | 低 (5 位指数) |
| **损失缩放** | 不需要 | 需要 (`GradScaler`) |
| **归约稳定性** | 差 (归约保持在FP32) | 中等 (易于上溢) |
| **应用场景** | LLM、Transformer、大型集群 | 旧版硬件、卷积网络 |

在现代集群上训练大型语言模型时，**使用FP32归约的BFloat16** 是主流策略。它在没有损失缩放器管理开销的情况下，最大程度地减少内存使用并最大化稳定性。仅当硬件代次受到很大限制时才使用Float16。

---

[上一节](../02-%E6%A8%A1%E5%9E%8B%E5%8C%85%E8%A3%85%E4%B8%8E%E5%88%9D%E5%A7%8B%E5%8C%96%E7%AD%96%E7%95%A5/05-%E4%BB%A3%E7%A0%81%E5%AE%9E%E8%B7%B5-%20%E9%AB%98%E7%BA%A7%E5%8C%85%E8%A3%85%E9%85%8D%E7%BD%AE.md) · [下一节](02-%E6%BF%80%E6%B4%BB%E6%A3%80%E6%9F%A5%E7%82%B9%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86.md)

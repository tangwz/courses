# Transformer 封装策略

来源：[原文](https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-2-model-wrapping-initialization-policies/transformer-wrapping-policies)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在完全分片数据并行 (FSDP) 中，高效的内存管理依赖于分片单元的定义。默认情况下，如果没有提供封装策略，FSDP 会将整个根模块视为一个单一单元。这种配置迫使系统在正向传播开始时将所有参数 (parameter)从所有秩收集到 GPU 内存中。结果，正向传播期间的内存峰值消耗实际上退化为分布式数据并行 (DDP) 的水平，从而抵消了参数分片的优势。为了实现 ZeRO-3 的内存节省，模型必须被划分为更小、独立的 FSDP 单元，这些单元可以动态地收集和腾出。

对于 Transformer 架构，这些单元的自然边界是 Transformer 块（或层）。本节说明了如何实现 `ModuleWrapPolicy`，使 FSDP 单元与模型的架构深度对齐 (alignment)。

### 分片的粒度

当模型被正确封装时，FSDP 会为每个单元执行一个“取消分片-正向传播-重新分片”循环。随着执行流程通过计算图，FSDP 会收集当前单元的参数 (parameter)，计算激活，并立即腾出（重新分片）参数，然后移动到下一个单元。

在正向传播过程中，对于在块级别封装的模型，任何时间点 $t$ 的内存消耗可以近似表示为：

$M_{\text{峰值}} \approx \frac{M_{\text{总计}}}{N_{\text{GPU数量}}} + M_{\text{块}}$

这里，$\frac{M_{\text{总计}}}{N_{\text{GPU数量}}}$ 代表驻留在设备上的分片参数的基准内存使用量，而 $M_{\text{块}}$ 代表实例化单个 Transformer 块的完整权重 (weight)所需的临时内存峰值。

相比之下，整体封装会导致：

$M_{\text{峰值}} \approx M_{\text{总计}}$

当训练数十亿参数的模型时，这两种运行模式之间的差异很大。目标是强制实现锯齿形内存曲线，其中内存分配仅在单个块的大小处达到峰值，并立即回到基准水平。

### FSDP 单元层次结构

为了实现这一点，我们必须识别架构中负责重复层的特定类。在标准 Transformer 中，这是包含自注意力 (self-attention)机制 (attention mechanism)、前馈网络 (MLP) 和归一化 (normalization)层的类。

下面的图表说明了 FSDP 如何封装这些特定块，创建嵌套在主模型封装器内的递归 FSDP 单元。

> 分层封装结构，其中单个 Transformer 块充当原子 FSDP 单元。

### 实现 ModuleWrapPolicy

PyTorch 提供了 `ModuleWrapPolicy`（之前可通过 `transformer_auto_wrap_policy` 访问），以自动化此过程。此策略接受一组目标层类。在初始化期间，FSDP 会遍历模块树；每当它遇到目标类的实例时，它就会将该子模块封装到自己的 FSDP 实例中。

下面的实现展示了如何为标准 `Llama` 模型结构配置此功能，尽管通过更改目标类，该原理适用于任何 Transformer 变体（BERT、GPT、T5）。

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ModuleWrapPolicy,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def get_llama_wrapper(model, mesh=None):
    """
    为 Llama 架构配置 FSDP 封装。
    """
    # 识别重复的层类
    # 对于 GPT-2 使用 GPT2Block，对于 BERT 使用 BertLayer 等。
    llama_auto_wrap_policy = ModuleWrapPolicy({LlamaDecoderLayer})

    wrapped_model = FSDP(
        model,
        auto_wrap_policy=llama_auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # torch.compile 所需
    )
    
    return wrapped_model
```

识别正确的类是主要要求。如果使用自定义架构，检查 `model.named_modules()` 有助于确认精确的类类型。如果类未正确指定目标，FSDP 将默认采用整体封装，并且 OOM (内存不足) 错误很可能在训练循环的早期发生。

### 内存配置文件分析

封装策略的效果最好通过内存分析来观察。下面的图表模拟了 70亿参数 (parameter)模型在正向传播过程中 GPU 内存的分配情况。



[交互图表：GPU 内存分配：整体封装 vs. Transformer 封装](https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-2-model-wrapping-initialization-policies/transformer-wrapping-policies#plot-52eze4)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "GPU 内存分配：整体封装 vs. Transformer 封装",
    "xaxis": {
      "title": "执行步骤（层）",
      "showgrid": false
    },
    "yaxis": {
      "title": "已分配内存 (GB)",
      "range": [
        0,
        30
      ]
    },
    "legend": {
      "x": 0.1,
      "y": 1.1,
      "orientation": "h"
    },
    "plot_bgcolor": "#f8f9fa",
    "paper_bgcolor": "#f8f9fa"
  },
  "data": [
    {
      "x": [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100
      ],
      "y": [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        28,
        28,
        28,
        4
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "整体封装（无策略）",
      "line": {
        "color": "#fa5252",
        "width": 3,
        "dash": "dash"
      }
    },
    {
      "x": [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100
      ],
      "y": [
        4,
        6,
        4,
        6,
        4,
        6,
        4,
        6,
        4,
        6,
        4,
        6,
        4,
        6,
        4,
        6,
        4,
        6,
        4,
        6,
        4
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "Transformer 块封装",
      "line": {
        "color": "#228be6",
        "width": 3
      }
    }
  ]
}
```

</details>



> 内存占用对比。整体方法累积权重 (weight)，直到整个模型被实例化。Transformer 块封装为每个块分配和腾出权重，从而形成稳定的锯齿形模式。

在整体场景（红色）中，系统逐步收集所有参数。对于使用 FP16 的 70亿参数模型，仅权重就大约占用 14GB。当考虑梯度和优化器状态时，这很容易超出不分片情况下标准 GPU 的容量。Transformer 块封装（蓝色）保持基准线较低，仅在适应当前工作层时达到峰值。

### 通信开销和延迟

细粒度封装虽然优化了内存，但它引入了通信延迟。每个 FSDP 单元在计算前会触发一次 `AllGather` 集合操作，并在之后执行腾出内存的逻辑。

如果封装粒度过细，例如，封装每个单独的 `nn.Linear` 层而不是 `TransformerBlock`，系统会因启动数千个小型 CUDA 内核和 NCCL 操作而产生很大的开销。网络延迟（握手时间）开始主导传输时间。

Transformer 块代表了内存效率和通信效率之间的几何平均值。它提供了一个足够大的参数 (parameter)块（通常为 100MB 到 500MB），足以在 `AllGather` 期间饱和网络带宽，最大限度地减少延迟的影响，同时又足够小，可以轻松适应 GPU 的可用内存。

在定义非标准架构的策略时，如果干净的重复类结构不可用，可以采用 `min_num_params` 参数并结合 `size_based_auto_wrap_policy`。然而，对于大型语言模型 (LLM)，显式基于类的封装仍然是实现确定性内存行为的标准做法。

---

[上一节](../01-%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7%E4%B8%8E%20ZeRO%20%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/04-%E5%AE%9E%E7%8E%B0%E5%9F%BA%E7%A1%80FSDP%E5%8C%85%E8%A3%85%E5%99%A8.md) · [下一节](02-%E8%87%AA%E5%AE%9A%E4%B9%89%E5%8C%85%E8%A3%85%E7%AD%96%E7%95%A5.md)

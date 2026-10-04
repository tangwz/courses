# Transformer的适配器模块

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-14-advanced-architectural-modifications/adapter-modules-transformers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为特定下游任务微调 (fine-tuning)庞大的预训练 (pre-training)语言模型（如Transformer）带来了显著的计算和存储难题。为每个新任务重新训练整个模型（可能包含数十亿参数 (parameter)）通常不切实际。参数高效微调（PEFT）方法旨在通过仅使用少量新增或修改的参数来适应模型，从而应对这一问题。适配器模块是PEFT技术中最早且最有影响力的策略之一。

适配器的核心理念是在保持预训练Transformer原有权重 (weight)不变的前提下，向其现有架构中注入小型、可训练的模块。在微调过程中，仅更新这些新增适配器模块的参数。与完全微调相比，这极大地减少了可训练参数的数量，通常能降低几个数量级。

### 适配器架构

适配器模块通常由一个瓶颈结构构成，旨在将输入维度映射到一个小得多的中间维度，然后再将其映射回原始维度。该结构包含：

1. 下投影线性层。
2. 非线性激活函数 (activation function)（例如，GeLU、ReLU）。
3. 上投影线性层。
4. 一个残差连接，将适配器的输出加到其接收到的原始输入上。

瓶颈维度（即中间层的大小）是一个重要的超参数 (parameter) (hyperparameter)。它控制着适配器中的参数数量，并影响参数效率与任务性能之间的平衡。更小的瓶颈维度会带来更少的参数，但可能会限制适配器学习特定任务特征的能力。

> 适配器模块的基本结构，展示了下投影、非线性处理、上投影和残差连接。

### 在Transformer块中的位置

适配器通常被插入到每个Transformer块中，通常位于多头注意力 (multi-head attention)（MHA）子层和前馈网络（FFN）子层之后，但在该子层的最终残差连接和层归一化 (normalization)之前。

> 适配器模块在标准Transformer块中的位置，通常位于MHA和FFN子层之后。请注意适配器自身周围的残差连接。

### 实现范例

以下是适配器模块的简化PyTorch实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim, dropout=0.1):
        super().__init__()
        self.down_project = nn.Linear(d_model, bottleneck_dim)
        self.non_linear = nn.GELU() # 常见选择，也可以是ReLU等。
        self.up_project = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # 将up_project的权重初始化为零或接近零
        # 这使得适配器在初始时表现为恒等函数
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        # x 是来自上一层的输入（例如，MHA或FFN的输出）
        adapter_input = x
        x = self.down_project(x)
        x = self.non_linear(x)
        x = self.up_project(x)
        x = self.dropout(x)
        # 添加残差连接
        output = adapter_input + x
        return output

# Transformer层前向传播中的使用示例
# 假设 `self.mha_adapter` 和 `self.ffn_adapter` 是
# Adapter的实例
# hidden_states = ... MHA的输出 ...
# adapted_mha_output = self.mha_adapter(hidden_states)
# hidden_states = layer_norm(adapted_mha_output + residual_mha_input)
# # 添加主残差和层归一化
#
# feed_forward_output = ... FFN的输出 ...
# adapted_ffn_output = self.ffn_adapter(feed_forward_output)
# hidden_states = layer_norm(adapted_ffn_output + residual_ffn_input)
# # 添加主残差和层归一化
```

请注意 `up_project` 层的初始化策略。将其权重 (weight)和偏置 (bias)初始化为零，可确保在微调 (fine-tuning)开始时，适配器模块基本上表现为恒等函数（输出 = 输入 + 0），从而保持原始模型的行为。这有助于稳定微调过程的初期。

### 使用适配器训练

使用适配器进行微调 (fine-tuning)的过程包括以下步骤：

1. **加载预训练 (pre-training)模型：** 从预训练的Transformer模型开始。
2. **冻结基础模型：** 将原始Transformer模型所有参数 (parameter)的 `requires_grad` 设置为 `False`。这会阻止它们在训练期间被更新。
3. **注入适配器：** 在模型架构中所需位置添加适配器模块。
4. **训练适配器：** 在目标任务的数据集上训练模型。只有新增适配器模块中的参数（以及可能的层归一化 (normalization)参数或最终的分类头部）将 `requires_grad` 设置为 `True` 并接收梯度更新。

由于只有总参数的一小部分（通常为0.5%到5%）被训练，因此优化器状态和梯度的内存需求大大减少，并且与更新整个模型相比，训练过程也快得多。

### 优点与考量

使用适配器有以下几点优势：

- **效率：** 显著减少微调 (fine-tuning)所需的计算资源（GPU时间、内存）。
- **存储：** 无需为每个任务存储一个完整的数十亿参数 (parameter)模型，只需保存少量适配器权重 (weight)。基础模型在所有任务之间共享。
- **模块化：** 不同任务的适配器可以方便地进行插拔。
- **性能：** 研究表明，适配器在许多任务上能够取得与完全微调非常接近的性能，尤其是在数据充足的情况下。

然而，也有一些需要考量的因素：

- **超参数 (hyperparameter)调整：** 瓶颈维度和放置策略会影响性能。
- **潜在性能差距：** 尽管通常接近，但在某些复杂任务或数据量较少的情况下，适配器的性能可能略微落后于完全微调。
- **推理 (inference)延迟：** 添加适配器会引入额外的计算（每个适配器包含两个线性层和一个非线性处理），这可能会使推理延迟相较于原始模型略有增加。然而，这种增加通常很小。

总之，适配器模块为将大型预训练 (pre-training)语言模型应用于各种下游任务提供了一种实用且有效的方法，避免了与完全微调相关的过高成本。它们是参数高效微调这一发展中方向的一项基本技术。

## 参考资料

- [PEFT library documentation: Adapters](https://huggingface.co/docs/peft/en/conceptual_guides/adapter) — Hugging Face (2024)
  Publisher: Hugging Face
  提供了Hugging Face PEFT库的官方文档，涵盖了用于参数高效微调的适配器实现和使用。

---

[上一节](01-%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E7%9A%84%E9%9C%80%E6%B1%82.md) · [下一节](03-%E4%B8%93%E5%AE%B6%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B%20%28MoE%29%20%E7%AE%80%E4%BB%8B.md)

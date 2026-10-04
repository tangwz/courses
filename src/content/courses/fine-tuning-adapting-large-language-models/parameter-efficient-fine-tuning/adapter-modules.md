---
course: "fine-tuning-adapting-large-language-models"
chapter: "parameter-efficient-fine-tuning"
lesson: "adapter-modules"
sourceId: 3031
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning/adapter-modules"
title: "适配器模块"
description: "介绍将小型、可训练的适配器层插入到LLM结构中的方法。"
order: 4
plots: []
sourceHash: "8d186bbde9339475808d0251c961a0098140f2a22efe12f3c45aef07e4891930"
sourceCorrections: []
---

适配器是参数 (parameter)高效微调 (fine-tuning) (PEFT) 系列中一种先行且有影响力的方案。与完全微调或 LoRA 那样修改全部甚至部分原始模型权重 (weight)不同，适配器背后的核心思想巧妙而简洁：将小型、新初始化、可训练的神经网络 (neural network)模块插入到冻结预训练 (pre-training)大型语言模型 (LLM) 的结构中。

可以把庞大的预训练LLM想象成一个固定结构。适配器就像是小巧、专门的扩展包，你可以将其插入到这个结构中指定的插槽里。在微调过程中，只有这些紧凑的适配器模块内的参数会被更新，而基础LLM中数十亿参数保持不变。这种方法大幅减少了可训练参数的数量，通常不到原始模型尺寸的1%，使得即使计算资源有限也能进行微调。

### 适配器结构：瓶颈设计

适配器模块最常见的设计遵循瓶颈结构。这通常包括：

1. 一个**下投影**层：通常是一个线性层，将输入隐藏状态的维度从模型的维度 ($d_{model}$) 降低到小得多的适配器维度 ($d_{adapter}$)。
2. 一个**非线性层**：一个激活函数 (activation function)，如 GELU 或 ReLU，应用于下投影的输出。
3. 一个**上投影**层：另一个线性层，将激活的、低维表示重新投影回原始模型维度 ($d_{model}$)。
4. 一个**残差连接**：上投影的输出加回到进入适配器模块的原始输入隐藏状态。

从数学上讲，如果 $h \in \mathbb{R}^{d_{model}}$ 是适配器的输入隐藏状态：

1. 下投影: $h_{down} = W_{down}h$，其中 $W_{down} \in \mathbb{R}^{d_{adapter} \times d_{model}}$
2. 非线性: $h_{act} = f(h_{down})$，其中 $f$ 是激活函数。
3. 上投影: $h_{up} = W_{up}h_{act}$，其中 $W_{up} \in \mathbb{R}^{d_{model} \times d_{adapter}}$
4. 残差连接: $h_{out} = h + h_{up}$

这里的超参数 (parameter) (hyperparameter)是适配器维度，$d_{adapter}$（也称为瓶颈维度）。较小的 $d_{adapter}$ 意味着更少的可训练参数，但适配器学习任务特定修改的能力可能较弱。$d_{adapter}$ 的典型值可能范围在 8 到 128 之间，远小于模型的隐藏维度（例如，4096 或更多）。

这些适配器模块通常被插入到 Transformer 结构中每个模块的特定位置。常见的插入点是依次在多头自注意力 (self-attention)子层之后和在前馈网络 (FFN) 子层之后。适配器之前的层归一化 (normalization)层也可能进行微调 (fine-tuning)。

> 适配器模块在 Transformer 模块中的插入点和结构。适配器通常放置在主要子层之后，并采用带有残差连接的瓶颈结构。

### 训练适配器

训练过程利用了基础模型的冻结特性：

1. **冻结基础模型：** 原始预训练 (pre-training) LLM 的所有参数 (parameter)被冻结；其梯度不被计算或更新。
2. **初始化适配器：** 新添加适配器模块的参数 ($W_{down}$、$W_{up}$ 和任何关联的偏置 (bias)) 被初始化，通常使用标准初始化方案。
3. **训练适配器：** 标准优化算法（如 AdamW）用于根据为特定下游任务（例如，分类、序列生成）计算的损失，*仅*更新适配器参数。有时，Transformer 模块内的层归一化 (normalization)参数也会被解冻并与适配器一起训练，这可以提高稳定性和性能。

这种有针对性的训练大幅减少了存储梯度和优化器状态（如 Adam 中的动量和方差估计）所需的内存，使得在大模型上进行微调 (fine-tuning)在消费级或中等规模的企业级 GPU 上成为可能。

重要的可调超参数 (hyperparameter)包括适配器维度 ($d_{adapter}$)、适配器参数的学习率，以及如果偏离标准做法，可能还包括适配器的具体插入位置。

### 适配器的优点

- **高参数 (parameter)效率：** 适配器只引入少量新参数，与完全微调 (fine-tuning)相比，大幅降低了计算成本和存储需求。单个基础模型可以用于多个任务，只需求存储小型的、特定于任务的适配器权重 (weight)。
- **模块化：** 由于适配器权重与基础模型分离，它们可以易于共享、加载或组合。这种模块化在管理不同任务适应性方面提供了灵活性。
- **降低训练内存需求：** 冻结基础模型显著降低了微调过程中所需的 GPU 内存。
- **更快的训练迭代：** 更新更少的参数通常能加快训练步骤，相比于更新整个模型。

### 缺点与考量

- **潜在的推理 (inference)延迟：** 由于适配器在推理过程中增加了额外的计算步骤（通过适配器层的正向传播），与完全微调 (fine-tuning)的模型或 PEFT 修改（如 LoRA）已合并回基础权重 (weight)的模型相比，它们可能会引入延迟。
- **性能权衡：** 尽管通常能达到接近完全微调的性能，但适配器在某些复杂任务或基准测试上可能略微落后于完全微调或 LoRA 等方法，这取决于适配器配置和任务具体情况。
- **结构敏感性：** 性能可能对瓶颈维度 ($d_{adapter}$) 的选择以及 Transformer 结构中确切的插入点敏感。找到最优配置可能需要实验。
- **组合复杂性：** 尽管适配器是模块化的，但简单地加载独立训练的多个适配器，在组合或多任务场景中可能不总能带来最佳性能。可能需要专门的技术来实现有效的适配器组合。

### 实现说明

像 Hugging Face 的 `adapter-transformers` 库（`transformers` 主库的一个扩展）提供了方便的 API 来添加、训练和管理各种适配器配置（包括不同的架构变体，如 Pfeiffer 或 Houlsby 适配器），适用于许多预训练 (pre-training)模型。在 PyTorch 中定义适配器模块可能如下所示：

```python
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, model_dim, bottleneck_dim, activation=nn.GELU()):
        super().__init__()
        self.down_project = nn.Linear(model_dim, bottleneck_dim)
        self.activation = activation
        self.up_project = nn.Linear(bottleneck_dim, model_dim)
        # 将上投影层的权重初始化为零或接近零
        # 有助于在训练初期稳定模型
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        # x 是输入隐藏状态（例如，MHA 或 FFN 的输出）
        down = self.down_project(x)
        activated = self.activation(down)
        up = self.up_project(activated)
        # 添加残差连接
        output = x + up
        return output

# 示例用法（- 在 Transformer 模块内部）
# ... 前置层（例如，MHA + LayerNorm） ...
# hidden_states = layer_norm(hidden_states + attention_output)
# adapter1 = Adapter(model_dim=config.hidden_size, bottleneck_dim=64)
# hidden_states_after_adapter1 = adapter1(hidden_states)
# ... 前馈网络 ...
```

总之，适配器模块提供了一种引人注目的 PEFT 策略，其特点是向冻结的基础模型添加小型、可训练的瓶颈层。它们的参数 (parameter)效率和模块化使其成为适应 LLM 的一个有价值的工具，特别是在管理多个任务或面临严格计算限制时，尽管潜在的推理 (inference)延迟是一个需要考虑的因素，相比于允许将修改合并回基础模型的方法。

## 参考资料

- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly (2019)
  Journal: Proceedings of the 36th International Conference on Machine Learning (ICML); Pages: 2790-2799; DOI: [10.48550/arXiv.1902.00751](https://doi.org/10.48550/arXiv.1902.00751)
  这篇基础论文介绍了适配器模块的概念，展示了其在自然语言处理中进行参数高效迁移学习的有效性。

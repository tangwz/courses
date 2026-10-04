---
course: "how-to-build-a-large-language-model"
chapter: "initialization-techniques-deep-networks"
lesson: "small-initialization-final-layers"
sourceId: 6010
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks/small-initialization-final-layers"
title: "末尾层的小初始化"
description: "讨论模型输出附近层的特定初始化考量。"
order: 5
plots: []
sourceHash: "0b7ad0c393c13bad24b24a5d0e4668ff2b89d3306c6a382244d739bd9449f472"
sourceCorrections: []
---

Kaiming和Xavier等有章法的初始化方法为深度网络中的大多数层提供了很好的起始点，但通常会特别考虑最终层，特别是将最后隐藏状态映射到语言模型中词汇表 (vocabulary)逻辑值的输出投影层。这一层的初始化能对初始损失值和训练过程的稳定性产生很大影响，尤其是在最开始的几步。

### 末尾层不同初始化的理由

输出层直接计算词汇表 (vocabulary)中每个词元 (token)的Softmax前逻辑值。如果这些初始逻辑值有较大的方差，可能会出现一些情况：

1. **初始损失过大：** 较大的正或负逻辑值可能导致Softmax函数后的概率极高或极低，进而可能引发非常大的初始交叉熵损失。这会使得训练刚开始时就发生梯度爆炸。
2. **Softmax饱和：** 如果初始逻辑值非常大，Softmax函数可能会饱和，这意味着许多词汇项的反向传播 (backpropagation)梯度会接近零。这会减缓学习速度。
3. **训练动态：** 一些研究表明，将最终层初始化为使得初始输出分布更接近均匀（或至少具有较低方差）的状态，可以促成更稳定且有时更快的收敛。直观来说，模型在开始时对任何特定词元的“信心”较少，这使得梯度在早期能更有效地流动。

考虑标准的Kaiming或Xavier初始化。这些方法旨在保持网络层中的方差。然而，对于将隐藏维度 $d_{model}$ 映射到大型词汇表 $V$ 的最终层，产生的尺度可能仍然过大，不利于稳定的初始训练，特别是在考虑到 $V$ 的规模时。

### 常见做法

一种常见做法是，将最终线性层的权重 (weight)矩阵初始化为一个较小的标准差，与Kaiming或Xavier初始化通常建议的值相比。偏置 (bias)项通常初始化为零。

例如，不必使用Kaiming均匀/正态初始化中默认的标准差，而是可以手动指定一个较小的标准差，通常与网络深度成反比缩放，或基于经验结果。一些架构或代码库采用特定的较小标准差，例如 $0.02$，或将标准Kaiming/Xavier标准差乘以一个系数（例如 $0.5$）。

另一种观点来自一些架构，其中最终层是残差连接通路的一部分。在像GPT-2/3这样的模型中，对残差流有贡献的层有时会根据层数来缩放其初始化值，以避免残差信号增长过大。虽然这通常适用于残差块内的中间前馈或注意力输出投影，但如果观察到稳定性问题，类似的原则也可能应用于最终输出投影。核心思路是控制初始输出的大小。

### 实现示例

让我们看看如何在PyTorch模型中对最终线性层应用自定义的较小初始化。假设你的模型有一个名为 `lm_head` 的最终层，它是 `torch.nn.Linear` 的一个实例。

```python
import torch
import torch.nn as nn
import math

class SimpleTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward):
        super().__init__()
        self.d_model = d_model
        # ... (embedding, positional encoding, transformer encoder layers) ...
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                       num_layers)
        # 偏置通常为False或归零
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights) # 递归应用初始化

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 大多数线性层的标准Kaiming初始化
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            # 或合适的非线性
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 嵌入层的标准初始化
            # 示例标准差
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # 对最终输出层 (lm_head) 的特殊处理
        # 检查模块是否为lm_head实例
        if hasattr(self, 'lm_head') and module is self.lm_head:
            # 对最终层权重使用较小的标准差
            # 示例：缩放Kaiming标准差或使用固定的较小值
            std_dev = 0.01 # 或其他经验确定的较小值
            # 备选：基于该层的Kaiming标准差进行缩放
            # kaiming_std = math.sqrt(2.0 / (module.in_features *
            #                            (1 + math.pow(0, 2)))) # ReLU示例
            # std_dev = kaiming_std * 0.5 # 缩小它

            print(
                f"对lm_head应用特殊初始化，std={std_dev}"
            )
            nn.init.normal_(module.weight, mean=0.0, std=std_dev)
            if module.bias is not None:
                # 确保如果存在偏置，则将其归零
                nn.init.zeros_(module.bias)

    def forward(self, src):
        # ... (前向传播逻辑) ...
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        # 在此处添加位置编码
        output = self.transformer_encoder(embedded)
        logits = self.lm_head(output)
        return logits

# 示例用法：
vocab_size = 10000
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048

model = SimpleTransformerLM(
    vocab_size, d_model, nhead, num_layers, dim_feedforward
)

# 在模型实例化时，你会看到打印输出：
# "Applying special initialization to lm_head with std=0.01"
```

在这个示例中，`_init_weights` 函数被递归应用于所有模块。它首先对普通线性层应用标准Kaiming正态初始化，并对嵌入 (embedding)层应用正态初始化。然后，它特别检查当前正在初始化的模块是否为 `lm_head` 实例。如果是，它会使用一个小得多的标准差（本例中 `std=0.01`）的正态分布来覆盖标准初始化。

为最终层选择精确的标准差或缩放因子通常涉及一些经验性调整，依据是特定的模型架构、词汇表 (vocabulary)大小和初始训练观察结果。监测初始损失值和梯度范数有助于判断所选初始化是否合适。目标是让训练在一个稳定状态下开始，使得梯度包含有用信息但不过大。

## 参考资料

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Pages: 1026-1034; DOI: [10.48550/arXiv.1502.01852](https://doi.org/10.48550/arXiv.1502.01852)
  介绍了Kaiming初始化（MSRA初始化），这是一种初始化深度神经网络权重的方法，对ReLU激活层后的层特别有效。
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (2019)
  Journal: Technical Report; Publisher: OpenAI
  描述了GPT-2模型及其训练，包括缩放残差层权重的做法，这有助于控制信号幅度并与最终层初始化策略相关。
- [Llama: Open and Efficient Foundation Models](https://arxiv.org/abs/2302.13971) — Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2302.13971](https://doi.org/10.48550/arXiv.2302.13971)
  详细介绍了Llama模型的架构和训练，明确提到使用小标准差（例如0.02）进行权重初始化，这是现代大型语言模型的常见做法。
- [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) — PyTorch Developers (2022)
  Publisher: PyTorch
  PyTorch初始化模块的官方文档，提供了初始化神经网络权重的函数和指南，与实际实现相关。

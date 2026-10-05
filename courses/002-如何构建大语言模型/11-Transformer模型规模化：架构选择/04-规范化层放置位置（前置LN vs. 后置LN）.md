# 规范化层放置位置（前置LN vs. 后置LN）

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-11-scaling-transformers-architectural-choices/normalization-layer-placement)

[返回章节目录](README.md) · [返回课程目录](../README.md)

随着Transformer模型规模的扩大，增加深度会带来潜在的训练难题。尽管残差连接旨在缓解梯度消失，但规范化层的具体放置位置对训练时的表现有重要影响，尤其是在非常深的神经网络 (neural network)中。层规范化（LN）本身通过在特征维度上规范化层的输入来稳定训练，确保其具有零均值和单位方差。问题在于，这种规范化在Transformer模块中应该相对于子层（例如自注意力 (self-attention)或前馈网络）和残差连接发生*在何处*。

两种主要方法是后置层规范化（Post-Layer Normalization，简称Post-LN）和前置层规范化（Pre-Layer Normalization，简称Pre-LN）。

### 后置层规范化（原始Transformer）

原始论文《Attention Is All You Need》提出了Post-LN配置。在这种设置下，子层的输出会与输入（残差连接）相加，*然后*应用层规范化。

数据流如下所示：

`output = LayerNorm(x + Sublayer(x))`

> Post-LN Transformer模块中的数据流。规范化发生在残差相加之后。

尽管对于中等深度的模型有效，但随着层数显著增加（例如，在12-24层中），Post-LN会遇到稳定性问题。主要问题在于，残差通路的直接输出（图中的`x`）在规范化*之前*被添加到转换（`Sublayer(x)`）的输出上。如果子层输出的幅度变化很大或逐层增大，这些相加可能导致进入下一个LayerNorm的激活值产生很大的方差。这可能导致网络深处出现梯度爆炸或梯度消失，通常需要仔细的学习率预热策略（在训练开始时逐渐增加学习率）以及精确的超参数 (parameter) (hyperparameter)调整来防止发散。训练深层Post-LN模型可能会比较困难。

### 前置层规范化

为了解决Post-LN在非常深的模型中的稳定性问题，提出了前置层规范化（最早在GPT-2等模型中受到关注，尽管早期也有类似变体）。在这里，层规范化应用于输入，*在*其通过子层*之前*。残差连接随后将未规范化的输入`x`添加到子层的输出上。

数据流变为：

`output = x + Sublayer(LayerNorm(x))`

> Pre-LN Transformer模块中的数据流。规范化发生在子层之前。

这种看似微小的改变对训练稳定性有重要影响。通过对每个子层的输入进行规范化，Pre-LN确保了注意力网络和前馈网络处理的激活值具有一致的尺度（零均值，单位方差），无论网络深度如何。反向传播 (backpropagation)通过网络的梯度也通常表现更好，因为规范化步骤有效地“重置”了每个残差块输入的尺度。这使得训练深层Transformer模型更加稳定，通常允许更高的学习率，并减少对非常长的预热期（warmup periods）的严格要求（尽管预热通常仍然有益）。

这里是一个简化的PyTorch示例，说明了Transformer模块中的结构差异：

```python
import torch
import torch.nn as nn

# 假设'sublayer'是一个预定义的nn.Module（例如，SelfAttention或FeedForward）
# 假设'd_model'是嵌入维度

class PostLNBlock(nn.Module):
    def __init__(self,
                 d_model,
                 sublayer,
                 dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 应用子层，添加残差，然后规范化
        residual = x
        x = self.dropout(self.sublayer(x))
        x = residual + x
        x = self.norm(x)
        return x

class PreLNBlock(nn.Module):
    def __init__(self,
                 d_model,
                 sublayer,
                 dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 规范化，应用子层，然后添加残差
        residual = x
        x = self.norm(x) # 区别在这里
        x = self.dropout(self.sublayer(x))
        x = residual + x
        return x

# 使用示例
d_model = 512
# 替换为实际的注意力/FFN层
dummy_sublayer = nn.Linear(d_model, d_model)

post_ln_block = PostLNBlock(d_model, dummy_sublayer)
pre_ln_block = PreLNBlock(d_model, dummy_sublayer)

input_tensor = torch.randn(
    32, 10, d_model
) # 批次，序列，维度

output_post = post_ln_block(input_tensor)
output_pre = pre_ln_block(input_tensor)

print("Post-LN 输出形状：", output_post.shape)
print("Pre-LN 输出形状：", output_pre.shape)

# 预期输出：
# Post-LN Output Shape: torch.Size([32, 10, 512])
# Pre-LN Output Shape: torch.Size([32, 10, 512])
```

### 权衡与建议

主要的权衡是**稳定性与潜在的最高性能**。

- **Pre-LN：** 提供了显著更好的训练稳定性，尤其对于具有数十或数百层的模型。它对学习率和预热时长等超参数 (parameter) (hyperparameter)选择不太敏感。这种稳固性使其成为当前大多数大规模LLM训练的首选。梯度在整个训练过程中表现更好。
- **Post-LN：** 在训练成功时，有时可以获得略好一些的最终性能指标（例如，更低的困惑度）。然而，实现成功的训练通常需要更细致地调整学习率计划、预热步数和初始化。在大规模训练中，它通常被认为更难稳定训练。

考虑到训练大型语言模型的巨大计算成本和时长，**Pre-LN**提供的改善的稳定性和稳固性通常胜过Post-LN可能带来的微小性能提升。在数周的训练过程中，由于不稳定性而出现意外发散是一种代价高昂的挫折。因此，在构建大型、深层Transformer模型时，Pre-LN架构是推荐且被广泛采用的标准。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762v7) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.5555/3295222.3295349](https://doi.org/10.5555/3295222.3295349)
  介绍了原始的Transformer架构，其中使用了Post-Layer Normalization。
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (2016)
  Journal: arXiv preprint arXiv:1607.06450; DOI: [10.48550/arXiv.1607.06450](https://doi.org/10.48550/arXiv.1607.06450)
  介绍了Layer Normalization的概念，它是Transformer架构中的基本组成部分。
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (2019)
  Publisher: OpenAI
  一篇重要论文，推广了Pre-Layer Normalization在训练超深Transformer模型时保持稳定性的应用。
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu (2020)
  Journal: ICML 2020; DOI: [10.48550/arXiv.2002.04745](https://doi.org/10.48550/arXiv.2002.04745)
  深入分析了Pre-LN和Post-LN Transformer配置在训练动态和稳定性方面的差异。

---

[上一节](03-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E9%80%89%E6%8B%A9%20%28ReLU%2C%20GeLU%2C%20SwiGLU%29.md) · [下一节](05-%E7%A8%80%E7%96%8F%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E7%AE%80%E4%BB%8B.md)

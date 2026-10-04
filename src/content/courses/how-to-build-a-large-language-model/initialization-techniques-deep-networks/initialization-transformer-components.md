---
course: "how-to-build-a-large-language-model"
chapter: "initialization-techniques-deep-networks"
lesson: "initialization-transformer-components"
sourceId: 6009
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks/initialization-transformer-components"
title: "Transformer组件中的初始化"
description: "将初始化方法具体应用于嵌入层、注意力层和FFN层。"
order: 4
plots: []
sourceHash: "d1fbc6092b2f29f749b0c654d2ca4409c5717c657db52b8eb896bb5c9a08945c"
sourceCorrections: []
---

Xavier和Kaiming初始化的基本原理是深度学习 (deep learning)中必不可少的。本文讨论这些技术如何具体应用于标准Transformer模型中不同类型的层。确保每个组件得到恰当的初始化，对于保持信号传播的稳定并使基于梯度的学习在这些深层架构中得以有效进行具有主要作用。

### 初始化嵌入 (embedding)层

Transformer模型以嵌入层开始，这些层将输入令牌ID及其位置转换为密集的向量 (vector)表示。

- **令牌嵌入（`nn.Embedding`）：** 此层将离散令牌ID映射到连续向量。一个普遍的做法是将这些嵌入权重 (weight)从一个均值为0且标准差相对较小的正态分布中初始化，例如$\mathcal{N}(0, 0.02)$。这种方法在嵌入中提供了初始多样性，同时避免引入过大的值，这些值可能在训练早期造成不稳定。虽然它不严格遵循Xavier/Kaiming的扇入/扇出逻辑（因为输入是稀疏的），但这种经验方法在实际中运行良好。有些实现可能会调整Xavier初始化，将嵌入维度视为输出大小。
- **位置嵌入：** 如果使用学习到的位置嵌入（也通常是一个`nn.Embedding`层），类似的初始化策略适用，通常从$\mathcal{N}(0, \sigma^2)$中抽取权重，其中$\sigma$较小。对于正弦位置编码 (positional encoding)，不需要初始化，因为它们是固定的、确定的值。

以下是在PyTorch中初始化嵌入层的方法：

```python
import torch
import torch.nn as nn

vocab_size = 30000
hidden_dim = 768
max_position_embeddings = 512

# 令牌嵌入
token_embedding_layer = nn.Embedding(vocab_size, hidden_dim)
nn.init.normal_(token_embedding_layer.weight, mean=0.0, std=0.02)

# 学习到的位置嵌入（示例）
positional_embedding_layer = nn.Embedding(
    max_position_embeddings, hidden_dim
)
nn.init.normal_(positional_embedding_layer.weight, mean=0.0, std=0.02)

print("令牌嵌入层权重形状:", token_embedding_layer.weight.shape)
print(
    "位置嵌入层权重形状:",
    positional_embedding_layer.weight.shape
)
```

### 初始化注意力机制 (attention mechanism)层

Transformer的核心在于其自注意力 (self-attention)和交叉注意力机制。这些机制涉及多个线性投影层：

- **查询（Q）、值（V）投影（`nn.Linear`）：** 这些层将输入嵌入 (embedding)投影到Q、K和V空间。由于它们是标准线性变换，且通常后接受益于受控方差的操作（如缩放点积注意力），Kaiming初始化通常是合适的选择，特别是Kaiming均匀或正态初始化。这有助于在初始投影步骤中保持方差。
- **输出投影（`nn.Linear`）：** 在计算注意力输出后（通常来自多个注意力头），另一个线性层将连接后的输出投影回模型的隐藏维度。与Q/K/V投影类似，Kaiming初始化是此层的常见且合理默认方法。

考虑多头注意力 (multi-head attention)块中的一个线性投影层：

```python
# 示例：初始化一个QKV投影层
hidden_dim = 768
projection_layer = nn.Linear(hidden_dim, hidden_dim * 3) # Q、K、V的组合

# 应用Kaiming均匀初始化
# 假设在更广泛的计算图或前馈层中可能隐含地跟随ReLU类非线性
nn.init.kaiming_uniform_(
    projection_layer.weight,
    a=0,
    mode='fan_in',
    nonlinearity='leaky_relu'
)
# 将偏置初始化为零（常见做法）
if projection_layer.bias is not None:
    nn.init.zeros_(projection_layer.bias)

print("投影层权重形状:", projection_layer.weight.shape)

# 示例：初始化输出投影层
output_projection = nn.Linear(hidden_dim, hidden_dim)
# 另一种选择是使用Kaiming正态初始化
nn.init.kaiming_normal_(
    output_projection.weight,
    mode='fan_out',
    nonlinearity='relu'
)
if output_projection.bias is not None:
    nn.init.zeros_(output_projection.bias)

print("输出投影权重形状:", output_projection.weight.shape)
```

请注意，`mode`（`fan_in`或`fan_out`）和`nonlinearity`的选择可以根据理论思考或实验结果进行调整。`fan_in`在前向传播中保持方差，而`fan_out`在反向传播 (backpropagation)中保持方差。对于ReLU类激活（包括GeLU、SwiGLU），指定`nonlinearity='relu'`或`nonlinearity='leaky_relu'`可以适当调整缩放因子。

### 初始化逐位置前馈网络（FFN）

每个Transformer块都含有一个FFN，它通常由两个线性层组成，中间带有一个非线性激活函数 (activation function)（例如，ReLU、GeLU、SwiGLU）。

- **第一个线性层（`nn.Linear`）：** 此层通常扩展维度。在此处强烈推荐使用Kaiming初始化，并与所用的非线性函数匹配（例如，对于ReLU/GeLU使用`nonlinearity='relu'`）。这可以确保激活函数后输出的方差保持受控。
- **第二个线性层（`nn.Linear`）：** 此层将维度投影回模型的隐藏大小。虽然可以使用Kaiming初始化，但一些研究和实现（例如遵循GPT-2/3做法的）建议为此层使用较小的方差进行初始化。其原因通常与FFN块后的残差连接有关。通过缩小贡献给残差分支的层的初始化比例，它有助于训练的稳定，特别是在非常深的神经网络 (neural network)中。这种缩放可能与$1/\sqrt{N}$成比例，其中$N$是残差块或层的数量。

```python
hidden_dim = 768
ffn_intermediate_dim = hidden_dim * 4 # 常见的扩展因子
num_layers = 12 # Transformer层数的示例

# 第一个FFN层（扩展）
ffn_layer1 = nn.Linear(hidden_dim, ffn_intermediate_dim)
# 使用Kaiming匹配激活函数（例如，类似ReLU的GeLU）
nn.init.kaiming_uniform_(
    ffn_layer1.weight,
    a=0,
    mode='fan_in',
    nonlinearity='leaky_relu'
)
if ffn_layer1.bias is not None:
    nn.init.zeros_(ffn_layer1.bias)

# 第二个FFN层（投影回）
ffn_layer2 = nn.Linear(ffn_intermediate_dim, hidden_dim)

# 选项1：标准Kaiming
# nn.init.kaiming_uniform_(ffn_layer2.weight, a=0, mode='fan_out',
#                          nonlinearity='linear') # 此后无激活函数

# 选项2：缩放初始化（GPT风格，用于残差连接）
# 使用较小的标准差进行初始化，并按层数缩放
residual_scaling_factor = 2 * num_layers # 缩放因子的启发式示例
std_dev = 0.02 / (residual_scaling_factor**0.5)
nn.init.normal_(ffn_layer2.weight, mean=0.0, std=std_dev)

if ffn_layer2.bias is not None:
    nn.init.zeros_(ffn_layer2.bias)

print("FFN层1权重形状:", ffn_layer1.weight.shape)
print("FFN层2权重形状:", ffn_layer2.weight.shape)
```

### 初始化层归一化 (normalization)参数 (parameter)

层归一化（`nn.LayerNorm`）具有可学习的仿射参数：增益（$\gamma$）和偏置 (bias)（$\beta$）。标准做法是将$\gamma$初始化为1，将$\beta$初始化为0。这使得层归一化在初始时对归一化输出表现得接近恒等变换，从而允许网络在训练过程中根据需要学习偏差。

```python
layer_norm = nn.LayerNorm(hidden_dim)

# PyTorch nn.LayerNorm中的默认初始化对权重（gamma）已是1
# （gamma）和对偏置（beta）已是0，但显式初始化看起来像：
# nn.init.ones_(layer_norm.weight) # Gamma
# nn.init.zeros_(layer_norm.bias)  # Beta

print("层归一化Gamma（权重）形状:", layer_norm.weight.shape)
print("层归一化Beta（偏置）形状:", layer_norm.bias.shape)
```

### 初始化最终输出层

仅解码器或编码器-解码器Transformer的最终层通常将隐藏状态投影到词汇大小，通常后接一个softmax以进行概率分布。

- **输出到词汇的投影（`nn.Linear`）：** 此层将最终隐藏维度映射到词汇的大小。与第二个FFN层类似，直接应用标准的Xavier或Kaiming初始化可能会导致初始输出过大，可能在训练早期产生过度自信的预测和不稳定。一个普遍的做法是为此层使用较小的标准差进行初始化，类似于令牌嵌入 (embedding)（例如，$\mathcal{N}(0, 0.02)$），或者可能将其权重 (weight)与令牌嵌入矩阵绑定（权重绑定），尽管如果它们没有绑定，初始化仍然适用。

```python
output_projection_vocab = nn.Linear(hidden_dim, vocab_size)

# 与内部层相比，可能以较小的尺度初始化
nn.init.normal_(output_projection_vocab.weight, mean=0.0, std=0.02)
if output_projection_vocab.bias is not None:
    nn.init.zeros_(output_projection_vocab.bias)

print("最终输出投影形状:", output_projection_vocab.weight.shape)
```

总而言之，虽然Xavier和Kaiming等普遍原理提供了很好的起点，但有效初始化一个深度Transformer通常需要对每个组件审慎地应用这些方法，有时还会根据架构选择以及从大型模型训练动态中观察到的情况进行经验性调整（例如，对嵌入或特定残差层使用较小的标准差）。细致的初始化为更稳定和高效的训练奠定了基础。

## 参考资料

- [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 30; DOI: [10.5555/3295222.3295252](https://doi.org/10.5555/3295222.3295252)
  介绍了Transformer架构，是关于组件初始化讨论的基础。
- [Understanding the difficulty of training deep feedforward neural networks](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF6eLbKwpJBPf_u3npHtpnKx1wMzMFt0mPzsgHxAr3GwtF3fkwr4L5tpWbQ0ySz93GIWhbqGyejYStpWKfXx-PUwhb8fvRoOpN1k_g93hHslGpmEkwFcsIbGRzFRDuo9OOrv5abBG_4Nqxy) — Xavier Glorot, Yoshua Bengio (2010)
  Journal: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics; Publisher: PMLR; Volume: 9; Pages: 249-256
  提出了Xavier初始化，一种通过保持信号方差来稳定深度网络训练的方法。
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://doi.org/10.1109/ICCV.2015.122) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV 2015); Publisher: IEEE; Pages: 1026-1034; DOI: [10.1109/ICCV.2015.122](https://doi.org/10.1109/ICCV.2015.122)
  介绍了Kaiming初始化，一种专门为ReLU及类似非线性激活函数设计的方差保持方法。
- [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) — PyTorch Contributors (2022)
  Publisher: PyTorch Foundation
  PyTorch初始化函数的官方文档，包括`normal_`、`kaiming_uniform_`、`zeros_`和`ones_`。

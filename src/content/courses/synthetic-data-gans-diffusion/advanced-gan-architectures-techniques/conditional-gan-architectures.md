---
course: "synthetic-data-gans-diffusion"
chapter: "advanced-gan-architectures-techniques"
lesson: "conditional-gan-architectures"
sourceId: 5654
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-2-advanced-gan-architectures-techniques/conditional-gan-architectures"
title: "条件生成对抗网络：架构与控制"
description: "实现多种方法来控制GAN的生成结果。"
order: 4
plots: []
sourceHash: "4eb445e90421384d7095465185b0e387a329725e9699eacf392ed5af13c6c3c4"
sourceCorrections: []
---

虽然标准生成对抗网络（GAN）擅长学习数据分布，但它们对生成的特定输出的控制能力有限。给定一个噪声向量 (vector) $z$，生成器会产生一个样本，但我们常常希望能够指导这个过程，生成具有特定特征或属性的样本。条件生成对抗网络（cGAN）通过将辅助信息（通常表示为 $y$）整合到生成器和判别器中，提供了一个实现这种控制的结构。

这种条件信息 $y$ 可以采用多种形式，例如类别标签（例如，“生成数字‘7’”）、描述性属性（例如，“生成戴眼镜的人脸”）、文本描述，甚至是其他图像（如在图像到图像转换任务中所示）。主要思想是使生成过程不仅依赖于随机噪声 $z$，还依赖于条件 $y$。

### 基本条件设置：拼接

Mirza和Osindero（2014）提出的最早和最简单的条件设置方法，是将条件信息 $y$ 直接输入到生成器和判别器中。通常，$y$ 被表示为一个向量 (vector)（例如，独热编码的类别标签或嵌入 (embedding)向量），并与生成器的噪声向量 $z$ 拼接，以及与判别器的真实/虚假数据 $x$ 拼接。

- **生成器输入：** 拼接向量 $[z, y]$
- **判别器输入：** 拼接向量 $[x, y]$

生成器的任务是产生与条件 $y$ 相对应的真实样本 $G(z, y)$。判别器则必须学习区分真实对 $(x, y)$ 和虚假对 $(G(z, y), y)$。它需要评估的不仅是生成样本的真实性，还有其是否符合所提供的条件。

目标函数与标准GAN的目标函数相似，但引入了条件 $y$：


$$
\min_G \max_D V(D, G) = \mathbb{E}_{(x,y) \sim p_{数据}(x,y)} [\log D(x, y)] + \mathbb{E}_{z \sim p_z(z), y \sim p_y(y)} [\log(1 - D(G(z, y), y))]
$$


这种简单的拼接方法对许多任务都有效，特别是当条件信息维度相对较低时，如类别标签。

> 比较无条件GAN（顶部）和条件GAN（底部）基本结构的图表，显示了条件 $y$ 如何作为输入提供给生成器和判别器。

### 高级条件设置架构

虽然简单的拼接有效，但已开发出更精妙的方法来提高条件设置的有效性，特别是针对高分辨率合成和复杂条件。

#### 投影判别器

在判别器中将 $y$ 直接拼接图像特征 $x$ 的一个局限性是，网络必须从这种组合表示中学习分离 $x$ 和 $y$ 的影响。投影判别器（Miyato & Koyama, 2018）提供了一种替代方案。

不同于拼接，主要的判别器架构处理图像 $x$ 以产生特征激活 $\phi(x)$。条件信息 $y$ 被嵌入 (embedding)到一个向量 (vector) $V_y$ 中。判别器的最终输出是根据图像特征和条件嵌入之间的相似性计算的，通常使用点积，并加上仅来自图像特征的贡献：


$$
D(x, y) = \sigma(\psi(\phi(x))^T V_y + b_y + \alpha(\phi(x)))
$$


其中 $\psi$ 是另一个学习到的映射，$V_y$ 是条件 $y$ 的学习嵌入，$b_y$ 是类别特异性偏差，而 $\alpha(\phi(x))$ 表示判别器预测的无条件部分。这种结构使得判别器能够直接加强图像特征 $\phi(x)$ 和条件 $y$ 之间的关联，这通常会带来更好的条件一致性和更高的样本质量。

#### 条件归一化 (normalization)层

另一种有效的技术是根据条件信息 $y$ 调节生成器内的归一化层。标准归一化层，如批量归一化（Batch Normalization）或实例归一化（Instance Normalization），通常会学习仿射参数 (parameter)（$\gamma$, $\beta$）来缩放和移动归一化后的激活。在条件变体中，这些参数由一个以 $y$ 为输入的小型神经网络 (neural network)预测。

- **条件批量归一化（CBN）：** $\gamma$ 和 $\beta$ 是 $y$ 的函数。
- **条件实例归一化（CIN）：** 类似于CBN，但按实例应用。
- **自适应实例归一化（AdaIN）：** 广泛用于StyleGAN中，其中风格向量（源自 $z$ 和可能的条件 $y$）决定了归一化层的 $\gamma$ 和 $\beta$ 参数，从而有效地控制了在不同特征图尺度上生成图像的风格。

通过使归一化具有条件性，辅助信息 $y$ 可以影响整个生成器网络中的特征统计，从而对合成过程提供细致的控制，影响与条件相关的纹理、颜色和风格等方面。

```python
# 条件批量归一化层的简化伪代码

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False) # 标准批量归一化，不带学习参数

        # 小型网络（或嵌入层）用于从条件 y 预测 gamma 和 beta
        self.embed_gamma = nn.Embedding(num_classes, num_features)
        self.embed_beta = nn.Embedding(num_classes, num_features)

    def forward(self, x, y):
        # 1. 归一化输入特征
        out = self.bn(x)

        # 2. 根据条件 y 预测 gamma 和 beta
        # 假设 y 是整数类别标签的张量
        gamma = self.embed_gamma(y) # 形状: (批量大小, 特征数量)
        beta = self.embed_beta(y)   # 形状: (批量大小, 特征数量)

        # 3. 重塑 gamma 和 beta 以匹配特征图维度
        # 示例: (批量大小, 特征数量) -> (批量大小, 特征数量, 1, 1)
        gamma = gamma.view(x.size(0), self.num_features, 1, 1)
        beta = beta.view(x.size(0), self.num_features, 1, 1)

        # 4. 应用条件缩放和偏移
        out = gamma * out + beta
        return out
```

> 伪代码示例，说明了如何将在标准批量归一化后应用源自条件 $y$ 的条件参数（$\gamma$, $\beta$）。

### 条件设置的类型和实现

条件设置机制的选择通常取决于 $y$ 的性质：

- **类别标签：** 通常使用独热编码或学习到的嵌入 (embedding)来处理，这些编码或嵌入被输入到拼接层、投影判别器或条件归一化 (normalization)层中。
- **属性：** 可以表示为二进制向量 (vector)（存在/不存在）或连续值，通常在使用前进行嵌入。多个属性可以组合成一个单一的条件向量。
- **文本：** 需要更复杂的文本编码模型（例如，RNN、Transformer或像CLIP这样的预训练 (pre-training)嵌入）来生成一个条件向量。注意力机制 (attention mechanism)在这里常常很重要，以将文本特征与图像区域对齐 (alignment)。
- **图像：** 用于图像到图像转换（例如，Pix2Pix）。整个输入图像作为条件，通常通过编码器网络进行处理，其特征会传递到生成器的解码器（例如，通过U-Net中的跳跃连接）。

实现cGAN需要仔细考虑 $y$ 的表示方式和集成方式。嵌入层常用于将离散条件转换为适合网络输入或调节的密集向量 (dense vector)。

训练cGAN的一个重要挑战是确保生成器真正学会遵循条件 $y$，并且判别器能够有效使用它。有时，判别器可能会发现只关注真实性而忽略生成的样本是否符合条件更为容易。像投影判别器这样的技术就是专门设计来减轻这个问题的。仔细的超参数 (parameter) (hyperparameter)调整和架构选择对于实现强大的条件控制是必要的。

## 参考资料

- [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — Mehdi Mirza, Simon Osindero (2014)
  Journal: NIPS 2014 Workshop on Deep Learning; DOI: [10.48550/arXiv.1411.1784](https://doi.org/10.48550/arXiv.1411.1784)
  这篇原始论文介绍了条件GAN，展示了如何通过拼接将辅助信息整合到生成器和判别器中。

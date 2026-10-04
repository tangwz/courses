---
course: "generative-adversarial-networks-gans"
chapter: "gan-implementation-optimization"
lesson: "weight-initialization-gans"
sourceId: 2773
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-7-gan-implementation-optimization/weight-initialization-gans"
title: "权重初始化技术"
description: "正确权重初始化对稳定GAN训练的意义。"
order: 4
plots: []
sourceHash: "8a1d92c284b4b788799e28e5b36094e15cfc61a38345c067bfe03bc47f480dda"
sourceCorrections: []
---

生成对抗网络 (GAN)（尤其是复杂架构）的稳定性和表现对网络权重 (weight)的初始赋值非常敏感。虽然现代深度学习 (deep learning)框架提供了默认的初始化方法（通常是He或Glorot/Xavier），但GAN训练的对抗特性引入了独特的挑战，这可能需要更仔细地考量在训练开始时如何设定权重。不当的初始化可能立即导致梯度消失或爆炸、模式崩溃，或一个网络在学习有效开始之前就压倒另一个网络等问题。

### 控制初始化方法的意义

标准初始化方法，如Xavier/Glorot初始化（常与`tanh`或sigmoid激活函数 (activation function)一起使用）和He初始化（专为ReLU激活函数设计），旨在保持激活和梯度在网络层中传播时的方差。这有助于避免梯度过小或过大，从而促进典型监督学习 (supervised learning)场景中更平稳的训练。

然而，GAN训练涉及生成器（$G$）和判别器（$D$）之间的精妙平衡。如果初始权重 (weight)导致判别器过于强大，其梯度可能早期消失，无法向生成器提供有用信号。反之，如果生成器初始产生的输出容易区分，判别器可能学习过快并饱和，再次阻碍生成器的学习。因此，初始化方法的选择会明显影响早期训练动态和整体收敛。

### GAN中常见的初始化策略

虽然标准初始化通常是一个合理的起点，但在GAN的背景下，有几种策略被发现特别有帮助：

#### 1. 小标准差的高斯初始化

早期的GAN论文，包括DCGAN，通常建议从零均值正态（高斯）分布初始化权重 (weight)，并使用小标准差，例如$\mathcal{N}(0, 0.02^2)$。


$$
W \sim \mathcal{N}(0, \sigma^2) \quad \text{这里 } \sigma = 0.02
$$


这样做的理由是使初始激活和梯度保持较小，可能立即避免一个网络压倒另一个网络。这对于卷积层和全连接层尤其相关。虽然简单，但这种方法可能不适合非常深的网络的扩展，因为在这些网络中方差控制变得更重要。

#### 2. 正交初始化

正交初始化旨在设置权重矩阵$W$，使得$W^T W = I$（或$W W^T = I$）。这个特性有助于保持向量 (vector)（和梯度）在前向和反向传播 (backpropagation)过程中的范数。通过防止梯度因重复矩阵乘法而爆炸或消失，正交初始化有助于更稳定的训练，特别是在更深的神经网络 (neural network)或RNN中。

它常与谱归一化 (normalization)等技术结合使用，谱归一化也约束了判别器层的Lipschitz常数。实现正交初始化通常涉及将奇异值分解（SVD）或QR分解等技术应用于初始随机矩阵。许多深度学习 (deep learning)库为此提供了函数（例如PyTorch中的`torch.nn.init.orthogonal_`）。

> 示意图：在反向传播过程中，良好初始化下梯度范数的保持与不良初始化下可能的爆炸或消失。

#### 3. 重新审视Xavier/Glorot和He初始化

尽管有专门技术，但He初始化（用于后接ReLU或LeakyReLU的层）和Xavier/Glorot初始化（用于后接`tanh`或类似有界激活函数 (activation function)的层）仍然是GAN的常见且通常有效的选择。它们提供了一种按层维度调整初始权重的原则方法。当使用标准卷积层或线性层时，这些通常是默认设置和明智的初始选择。

重要的是根据后续层中使用的激活函数正确应用它们。如果您的生成器广泛使用ReLU，通常更青睐He初始化。如果输出层使用`tanh`，最后一层的权重可以考虑使用Xavier。

### 特定架构中的初始化

高级GAN架构有时采用专门的初始化方案，这些方案根据其独特组件量身定制：

- **StyleGAN/StyleGAN2:** 这些模型通常对标准卷积层使用He初始化，但对于涉及AdaIN（自适应实例归一化 (normalization)）调制或映射网络特定部分的层，可能会使用定制初始化（例如，将权重 (weight)缩放到1.0）。目标通常是确保学到的风格向量 (vector)具有可预测的初始效果。
- **BigGAN:** 以大规模训练而闻名，BigGAN经常使用正交初始化，这补充了其谱归一化的使用，并有助于管理大批量大小下的训练动态。

如果您旨在忠实复现，建议参考这些特定架构的原始论文或可靠的实现。

### 实际建议

- **从标准方法开始：** 除非训练不稳定，否则对于ReLU/LeakyReLU层从He初始化开始，对于`tanh`/sigmoid层使用Xavier/Glorot。这些通常是PyTorch和TensorFlow等框架中的默认设置。
- **考虑$\mathcal{N}(0, 0.02^2)$：** 如果您在训练早期遇到不稳定，特别是对于类似于DCGAN的架构，尝试小方差高斯初始化作为替代。
- **试验正交初始化：** 对于更深的神经网络 (neural network)或如果使用谱归一化 (normalization)等技术，正交初始化是一个值得尝试的有力选择。
- **初始化偏置 (bias)：** 偏置通常初始化为零。这通常是安全且标准的操作。
- **一致性：** 在生成器和判别器内部的相似层类型中，始终如一地应用所选的初始化方案。
- **代码示例（PyTorch）：**

  ```python
  import torch
  import torch.nn as nn
  import torch.nn.init as init

  def weights_init_normal(m):
      classname = m.__class__.__name__
      if classname.find('Conv') != -1:
          init.normal_(m.weight.data, 0.0, 0.02) # 高斯初始化
          if hasattr(m, 'bias') and m.bias is not None:
              init.constant_(m.bias.data, 0.0)
      elif classname.find('BatchNorm2d') != -1:
          init.normal_(m.weight.data, 1.0, 0.02) # 批量归一化权重通常初始化接近1
          init.constant_(m.bias.data, 0.0)
      elif classname.find('Linear') != -1:
           init.normal_(m.weight.data, 0.0, 0.02) # 线性层的高斯初始化
           if hasattr(m, 'bias') and m.bias is not None:
               init.constant_(m.bias.data, 0.0)

  # 使用示例：
  # model = Generator(...) or Discriminator(...)
  # model.apply(weights_init_normal)
  ```

  *注意：您可以使用`init.xavier_uniform_`、`init.kaiming_normal_` (He) 或 `init.orthogonal_` 以类似方式。*
- **与归一化的相互影响：** 请记住，归一化技术（批量归一化、实例归一化、层归一化、谱归一化）与权重 (weight)初始化有明显相互影响。例如，谱归一化直接约束权重矩阵的谱范数，可能覆盖初始尺度的一些效果，但不覆盖初始方向/结构（如正交性）。

选择合适的权重初始化方法并非总是直接明了的，有时可能需要基于特定的架构、数据集和训练配置进行试验。然而，理解这些常见策略为应对实现和稳定高级GAN的实际挑战提供了一个坚实的基础。

## 参考资料

- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.pdf) — Xavier Glorot, Yoshua Bengio (2010)
  Journal: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics; Publisher: JMLR.org; Volume: 9; Pages: 249-256
  介绍了Xavier/Glorot初始化，一种稳定深度神经网络训练的基础技术，尤其适用于sigmoid或tanh激活函数。
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Pages: 1026-1034; DOI: [10.1109/ICCV.2015.123](https://doi.org/10.1109/ICCV.2015.123)
  提出了He初始化（又称Kaiming初始化），专门用于使用ReLU类激活函数的网络。
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf) — Alec Radford, Luke Metz, and Soumith Chintala (2016)
  Journal: International Conference on Learning Representations (ICLR)
  一篇重要的GAN论文，常推荐使用零均值、小标准差（如0.02）的正态分布初始化权重。
- [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/pdf/1312.6120.pdf) — Andrew M. Saxe, James L. McClelland, and Surya Ganguli (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6120](https://doi.org/10.48550/arXiv.1312.6120)
  从理论角度阐明了正交初始化如何保持梯度范数并促进深度线性网络的学习，为非线性深度模型中的使用提供了依据。
- [torch.nn.init: Initialization](https://pytorch.org/docs/stable/nn.init.html) — PyTorch Developers (2022)
  Publisher: PyTorch
  PyTorch中权重初始化函数的官方文档，包括Kaiming (He)、Xavier/Glorot、正态和正交初始化方法。

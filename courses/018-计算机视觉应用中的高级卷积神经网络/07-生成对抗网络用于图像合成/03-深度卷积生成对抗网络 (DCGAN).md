# 深度卷积生成对抗网络 (DCGAN)

来源：[原文](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-7-gans-image-synthesis/deep-convolutional-gans-dcgan)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管原始的生成对抗网络 (GAN)框架提供了一个强大的理念，但早期的尝试常常面临训练不稳定，并生成低分辨率、不真实的图像。Radford、Metz和Chintala在2015年引入的深度卷积生成对抗网络 (DCGANs) 标志着一项显著的进步，它提供了一套架构指导原则，使得基于卷积的深度生成模型训练更加稳定和有效。DCGANs表明CNN可以成功地用于无监督学习 (supervised learning) (unsupervised learning)，尤其是图像生成。

DCGANs的成功主要归因于一些特定的架构选择，这些选择解决了常见的训练问题：

1. **用步长卷积替代池化**：DCGAN鉴别器在空间下采样时，不使用确定性池化层（如最大池化），而是使用步长卷积。同样，生成器使用分数步长卷积（常称为转置卷积或“反卷积”）进行空间上采样。这使得网络能够*学习*其自身的空间下采样和上采样，从而获得比固定池化方法可能更好的特征表示。
2. **引入批量归一化 (normalization)**：批量归一化（BatchNorm）应用于生成器和鉴别器。它通过归一化每层输入来帮助稳定学习，减轻了不良初始化相关问题，并改进了梯度流动。这在深度模型中尤为重要。也有例外：BatchNorm通常*不*应用于生成器的输出层或鉴别器的输入层。
3. **在较深架构中移除全连接层**：传统的CNN在最终输出前通常会有一个或多个全连接层。DCGANs在较深的卷积架构中大部分取消了这些层。在生成器中，输入噪声向量 (vector)$z$可能通过一个全连接层进行投影，但随后的层都是卷积层。在鉴别器中，最终卷积层的特征通常会被展平并直接送入单个Sigmoid输出节点。这减少了参数 (parameter)数量，并可能促进学习到更多与空间相关的特征。
4. **使用合适的激活函数 (activation function)**：生成器主要使用修正线性单元 (ReLU) 激活函数，$ReLU(x) = max(0, x)$，用于除输出层外的所有层。输出层使用 Tanh 激活函数，$tanh(x)$，它将输出缩放到 $[-1, 1]$ 的范围。这很方便，因为图像像素值在训练期间通常被归一化到此范围。
5. **在鉴别器中使用LeakyReLU**：鉴别器所有层都使用Leaky修正线性单元 (LeakyReLU) 激活函数。LeakyReLU定义为$LeakyReLU(x) = max(\alpha x, x)$，其中$\alpha$是一个小的正常量（例如0.2），它允许在单元不活跃（$x < 0$）时产生一个小的非零梯度。这能防止梯度消失并帮助学习，尤其是在对抗环境中，鉴别器需要向生成器提供有用的梯度。

### 生成器架构

DCGAN生成器将随机噪声向量 (vector)$z$（通常从标准正态分布或均匀分布中采样）作为输入，并将其转换为图像。该过程通常遵循以下步骤：

1. **输入**：一个通常100维的潜在向量$z$。
2. **投影**：向量$z$通常首先通过全连接层或初始转置卷积被投影并重塑为一个具有大量通道的小空间体积。这构建了空间结构。
3. **上采样**：应用一系列转置卷积层。每个层都会增加空间尺寸（高度和宽度），同时通常减少特征通道的数量。
4. **归一化 (normalization)与激活**：在每次转置卷积之后（最后一次除外），应用批量归一化，然后是ReLU激活函数 (activation function)。
5. **输出**：最后一层使用转置卷积达到所需的输出图像大小（例如64x64像素）和通道数（例如RGB的3个通道）。应用Tanh激活函数将输出像素值限制在 $[-1, 1]$ 范围内。

> 描述了DCGAN生成器的一个典型流程图，通过学习到的上采样将噪声向量转换为图像。

### 鉴别器架构

DCGAN鉴别器接收图像（来自数据集的真实图像或来自生成器的虚假图像）作为输入，并输出一个表示图像是真实还是虚假的概率。其结构本质上是一个标准CNN，适用于二元分类，与生成器架构相反：

1. **输入**：一个图像（例如64x64x3）。
2. **下采样**：应用一系列步长卷积层。每个层通常会减少空间尺寸，同时增加特征通道的数量。
3. **归一化 (normalization)与激活**：在每个卷积层之后（除了第一个输入层和最终输出层），应用LeakyReLU激活。此处也可使用批量归一化（尽管有时根据经验结果省略）。
4. **展平**：在最终卷积层之后，所得的特征图被展平为向量 (vector)。
5. **输出**：此向量被送入一个带有Sigmoid激活函数 (activation function)$\sigma(x) = 1 / (1 + e^{-x})$的单个输出节点。输出表示输入图像是真实的概率。

> 描述了DCGAN鉴别器的一个典型流程图，通过学习到的下采样将输入图像分类为真实或虚假。

### DCGAN的意义

DCGANs具有重要影响，因为它提供了一种可靠且相对稳定的架构，用于在图像数据上训练GAN。它表明GAN可以无监督地从图像中学习有意义的特征表示，并生成视觉上合理的图像。许多后续的GAN架构都建立在DCGAN所确定的原则之上，融入了修改和改进，但通常保留了使用卷积、批量归一化 (normalization)和仔细选择激活函数 (activation function)的核心思路。了解DCGAN为后续研究更复杂的生成模型（如条件GAN或StyleGAN）奠定了坚实的根基。

## 参考资料

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) — Alec Radford, Luke Metz, Soumith Chintala (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1511.06434](https://doi.org/10.48550/arXiv.1511.06434)
  介绍了深度卷积GAN (DCGAN) 及其用于通过卷积稳定训练深度生成模型的架构指南。
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems (NIPS) 27; Volume: 27; Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  提出了原始的生成对抗网络 (GAN) 框架，建立了对抗训练的基本思想。

---

[上一节](02-%E8%AE%AD%E7%BB%83%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%E7%9A%84%E6%8C%91%E6%88%98.md) · [下一节](04-%E6%9D%A1%E4%BB%B6GANs%E7%94%A8%E4%BA%8E%E5%8F%AF%E6%8E%A7%E7%94%9F%E6%88%90.md)

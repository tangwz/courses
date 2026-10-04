---
course: "generative-adversarial-networks-gans"
chapter: "gans-beyond-image-generation"
lesson: "gans-video-generation"
sourceId: 2760
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-6-gans-beyond-image-generation/gans-video-generation"
title: "视频生成与预测"
description: "使用GAN生成连贯视频序列的技术与难点。"
order: 5
plots: []
sourceHash: "20b7faa288a41bdfc029d7f9471f19cf9e507742332bc7cda38de1c6a2a8c6d3"
sourceCorrections: []
---

将生成模型从静态图像扩展到动态视频序列带来了很多复杂性，主要集中在建模运动和保持时间上的一致性。视频数据由一系列帧组成，其中空间外观和时间动态都很重要。一个成功的视频GAN不仅要生成逼真的单个帧，还要确保这些帧随时间形成合理且一致的序列。

### 视频生成难点

对视频数据建模带来了与图像生成不同的障碍:

1. **时间一致性：** 连续帧必须显示逼真的运动和外观变化。如果未能充分捕捉时间依赖关系，生成的视频常出现闪烁伪影或对象运动不一致。
2. **长距离依赖：** 逼真的视频常涉及跨越许多帧的事件和运动。捕捉这些长期关联需要模型能在较长时间内保持状态或上下文 (context)，这在计算上要求很高。
3. **高维度：** 视频数据本质上是高维的（帧 $\times$ 高度 $\times$ 宽度 $\times$ 通道）。这增加了训练的计算成本，并需要大型数据集和模型容量。
4. **运动建模：** 显式或隐式地对底层运动动态进行建模是必需的。简单的逐帧生成通常无法产生令人信服的运动。

### 视频生成架构

为了解决这些难点，已开发出几种架构方法:

#### 1. 3D卷积网络

类似于对图像使用2D卷积，3D卷积 (Conv3D) 在时空体上运行（例如，`时间 x 高度 x 宽度`）。在生成器和判别器中应用Conv3D层，使模型能够直接学习时空特征。

- **生成器：** 接收噪声向量 (vector) $z$ 并生成一系列帧。Conv3D转置层（有时称为3D反卷积）用于在空间和时间维度上进行上采样。
- **判别器：** 接收一系列真实或生成的帧，并输出一个概率得分，使用Conv3D层处理时空输入。

VGAN (VideoGAN) 等模型开创了这种方法，证明了在GAN框架中使用3D CNN进行视频处理的可行性。然而，3D卷积显著增加了参数 (parameter)数量和计算负载。

#### 2. 循环神经网络 (neural network) (RNN) (RNNs) 与时间条件化

另一种方法是将2D卷积网络（用于每帧的空间特征）与循环网络（如LSTM或GRU）结合，以建模时间动态。

- 生成器可以使用RNN来维持一个随时间演变的隐藏状态，根据前一帧和初始噪声向量 $z$ 影响后续每一帧的生成。
- 判别器也可以加入循环层来处理帧序列及其时间关系。

这使得建模可能比固定核3D卷积更长的依赖关系，但更难稳定训练。通常，输入噪声 $z$ 只在第一个时间步输入，或者在每个时间步重复输入，影响RNN的状态转换。

#### 3. 分解内容与运动 (MoCoGAN)

一些架构试图将内容（外观）与运动分离。例如，运动与内容分解GAN (MoCoGAN) 为内容（时间不变）和运动（时间变化）使用单独的潜在向量。

- 内容潜在向量 $z_c$ 每序列采样一次。
- 运动潜在向量 $z_m(t)$ 为每个时间步 $t$ 采样（通常来自循环过程）。
- 生成器结合 $z_c$ 和 $z_m(t)$ 来生成帧 $x_t$。

这种分解可以实现更可控的生成，并可能改善时间建模。判别器需要评估帧质量和时间一致性，可能使用单独的路径或损失项。

> MoCoGAN式生成器的简化图，它分离了内容和运动输入。RNN随时间处理运动噪声以引导帧生成。

#### 4. 分层与渐进方法

类似于图像的ProGAN，一些视频GAN采用渐进式或分层结构。这可能包括先生成低分辨率视频然后进行精修，或者生成关键帧然后插值中间帧。DVD-GAN (Diverse Video Distribution GAN) 使用分层方法，在不同空间分辨率下使用独立的生成器/判别器。

### 使用GAN进行视频预测

GAN不仅能从随机噪声生成视频，也被用于*视频预测*。此处的任务是给定一系列过去的上下文 (context)帧，预测未来的帧。

在此设置中:

- 生成器接收过去的帧 $x_{1:t}$ 作为输入（并可能包含用于随机性的噪声向量 (vector) $z$），然后输出预测的未来帧 $\hat{x}_{t+1:T}$。
- 判别器接收真实帧序列 $(x_{1:t}, x_{t+1:T})$ 或包含预测帧的序列 $(x_{1:t}, \hat{x}_{t+1:T})$，并尝试区分它们。

对抗损失促使生成器生成与真实未来帧无区别的未来帧，并以过去帧为条件。与纯粹基于重建的损失（如均方误差）相比，这通常会带来更清晰的预测，后者倾向于产生可能未来的模糊平均值。通常，重建损失（例如 $\hat{x}_{t+k}$ 和 $x_{t+k}$ 之间的L1或L2距离）会与对抗损失结合使用:


$$
\mathcal{L}_{Total} = \mathcal{L}_{GAN} + \lambda \mathcal{L}_{Recon}
$$


其中 $\lambda$ 用于平衡对抗和重建目标的影响。

### 视频GAN的评估

评估视频生成质量比评估图像更具挑战。标准指标如Inception Score (IS) 和 Fréchet Inception Distance (FID) 可以逐帧应用，但它们无法捕捉时间一致性。

已有针对视频的指标提出，例如:

- **Fréchet视频距离 (FVD)：** FID的扩展，它使用从预训练 (pre-training)视频识别网络（如I3D）中提取的特征，在同时考虑外观和运动的特征空间中比较生成视频与真实视频的分布。FVD值越低，表示真实视频和生成视频的分布相似性越好。

由人类评估员进行的定性评估对于判断生成运动的真实性和连贯性仍然重要。

生成逼真且时间上连贯的视频仍是活跃的研究方面。当前模型可以生成短小、合理的片段，特别是在受限范围内，但生成长篇、多样化、高分辨率并保持复杂叙事或交互的视频仍然是一个前沿难题。这里讨论的技术代表了实现该目标的重要步骤。

## 参考资料

- [Video Generative Adversarial Networks](https://proceedings.neurips.cc/paper/2016/file/c399862d3b4b884f73a6e386c99af31d-Paper.pdf) — Carl Vondrick, Hamed Pirsiavash, Antonio Torralba (2016)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 29; Pages: 613-621; DOI: [10.48550/arXiv.1609.02612](https://doi.org/10.48550/arXiv.1609.02612)
  该论文将3D卷积网络引入GAN框架，用于生成视频序列。
- [DVD-GAN: A Differentiable Video Discriminator for Training Conditional GANs](https://papers.nips.cc/paper_files/paper/2019/hash/a7e37e174b025f1ceb79930f9d989c77-Abstract.html) — Alexia Clark, Anna Lucic, Kosta Derpanis, Marcus Brubaker (2019)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 32; Pages: 5451-5461
  提出了一种分层GAN方法，利用多个判别器来提高视频生成的时空真实性和多样性。
- [Assessing Generative Models via Fréchet Video Distance](http://proceedings.mlr.press/v80/sajjadi18a/sajjadi18a.pdf) — Mehdi Sajjadi, Oleksiy Lukashenko, Anil Sharma, Bernhard Schölkopf (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); Volume: 80; Pages: 4402-4411
  该论文定义了弗雷歇视频距离（FVD），这是一种量化生成视频内容质量和时间一致性的指标。
- [Deep multi-scale video prediction beyond mean square error](https://openreview.net/forum?id=Bk0rZ4KXl) — Michael Mathieu, Camille Couprie, Yann LeCun (2016)
  Journal: International Conference on Learning Representations (ICLR); Publisher: OpenReview.net; DOI: [10.48550/arXiv.1511.05440](https://doi.org/10.48550/arXiv.1511.05440)
  对视频预测的对抗性学习的早期应用，与传统的L2损失相比，产生了更清晰的未来帧预测。

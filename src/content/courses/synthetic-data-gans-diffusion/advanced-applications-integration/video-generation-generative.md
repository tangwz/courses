---
course: "synthetic-data-gans-diffusion"
chapter: "advanced-applications-integration"
lesson: "video-generation-generative"
sourceId: 5732
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-6-advanced-applications-integration/video-generation-generative"
title: "使用生成模型进行视频生成"
description: "将图像生成技术应用于序列视频数据。"
order: 4
plots: []
sourceHash: "79a9201c3dffce618f968ae2d2f71fc4e32feae40f16ef072e29a94f7a5017b3"
sourceCorrections: []
---

“与合成静态图像相比，生成图像序列（即视频）会引入显著的复杂性。虽然GANs和扩散模型的核心原理仍然适用，但视频生成要求对时间动态进行明确建模，以确保帧间的一致性和运动流畅性。简单地独立生成帧通常会导致闪烁伪影或物体外观不一致，无法捕捉视频中固有的平滑过渡。”

### 视频合成的特有挑战

将生成模型从图像扩展到视频带来了一些明显的挑战：

1. **时间一致性：** 在帧间保持物体身份、外观、背景和运动的一致性可以说是主要挑战。生成的视频必须看起来像一段连续的录像，而不是一系列不相关的图像。
2. **运动建模：** 准确捕捉运动的物理特性和模式很困难。这包括全局场景变化以及场景中物体或角色的动作表现。
3. **高维度：** 视频数据在图像已有的高维空间 (high-dimensional space)数据基础上增加了时间维度，显著提高了待学习数据分布的复杂性。一个短视频片段所包含的信息量远超单张图像。
4. **长距离依赖：** 早期帧中的事件或外观可能会影响后续很远的帧。模型需要有能力捕捉这些长期时间关系。
5. **计算成本：** 由于序列长度以及通常需要3D卷积或循环层等专用架构，训练视频模型所需的内存和计算量远高于图像模型。

### 生成模型在视频中的扩展

为了使GANs和扩散模型适应视频生成，并超越简单的逐帧合成，已经开发出多种策略：

- **3D卷积网络：** 正如2D卷积对图像中的空间模式有效一样，3D卷积可以同时处理帧内的空间信息和帧间的时间信息。视频GAN (VGAN) 中的生成器和判别器都可以采用3D卷积层，直接从视频数据中学习时空特征。生成器将潜在向量 (vector) $z$ 映射到帧序列，而判别器接收帧序列并输出其真实性的概率。

  
  $$
  G: z \rightarrow \{x_1, x_2, ..., x_T\}
  $$
  
  
  $$
  D: \{x_1, x_2, ..., x_T\} \rightarrow [0, 1]
  $$
  

  这里，$x_t$ 代表时间 $t$ 的帧， $T$ 是序列中的总帧数。
- **循环架构：** 循环神经网络 (neural network) (RNN)（RNNs），特别是LSTM或GRU，可以集成到生成器中。例如，RNN可以处理一系列潜在向量，其中每个输出状态都为相应帧的生成提供信息，通常与卷积层结合使用。这明确地对视频的序列特性进行建模。
- **分解潜在空间：** 一些架构试图分离变异因素，例如将静态内容（如物体外观、背景）的潜在表示与动态运动的表示分开。MoCoGAN（运动和内容分解GAN）就是一个例子，其中内容编码在每个序列中采样一次，而一系列运动编码驱动帧间变化。

> 视频GAN的一种常见结构，可能会使用单独的潜在编码来表示内容和运动，并由时空生成器和判别器网络进行处理。

- **时间注意力：** 与其在图像生成中的应用类似，注意力机制 (attention mechanism)可以适用于视频，以建模跨时间和空间的长期依赖。Transformer模型，利用自注意力 (self-attention)，正越来越多地用于视频生成任务。

### 视频扩散模型

扩散模型也可以扩展到视频。其核心思想保持不变：逐渐对数据添加噪声（前向过程），并学习逆转这一过程（去噪）。

- **前向过程：** 通常在每个扩散时间步独立地向每帧添加高斯噪声，可能会在帧间共享噪声调度。

  
  $$
  q(x_{1:T}^k | x_{1:T}^{k-1}) = \prod_{t=1}^T \mathcal{N}(x_t^k; \sqrt{1-\beta_k} x_t^{k-1}, \beta_k I)
  $$
  

  其中 $x_{1:T}^k = \{x_1^k, ..., x_T^k\}$ 是扩散步骤 $k$ 时的帧序列，$\beta_k$ 是噪声方差调度。
- **逆向过程：** 去噪网络必须预测整个序列中添加的噪声（或噪声较少的视频）。这通常涉及时空网络架构，通常是用3D卷积修改或加入时间注意力层的U-Net，以处理跨空间和时间的信息。

  
  $$
  p_\theta(x_{1:T}^{k-1} | x_{1:T}^k) = \mathcal{N}(x_{1:T}^{k-1}; \mu_\theta(x_{1:T}^k, k), \Sigma_\theta(x_{1:T}^k, k))
  $$
  

  网络 $\mu_\theta$（由 $\theta$ 参数 (parameter)化）在给定当前噪声序列 $x_{1:T}^k$ 和步骤 $k$ 的情况下，预测前一步视频序列分布的均值。

对视频扩散模型进行条件控制（例如，用于文本到视频或动作条件生成）遵循与图像扩散相似的原理，通常使用分类器引导或无分类器引导，以处理序列输入和输出。由于计算负荷较高，潜在扩散等技术（其中扩散在压缩的潜在空间中运行）也正在适用于视频。

### 评估生成视频

评估视频质量需要评估逐帧真实感和时间特性。诸如Fréchet Inception距离（FID）之类的度量已扩展到视频，成为Fréchet视频距离（FVD）。FVD使用从预训练 (pre-training)视频分类网络（如I3D）中提取的特征，比较生成视频与真实视频的分布，同时考虑外观和运动。定性评估仍然重要，需要仔细检查视频的闪烁、运动伪影和整体时间平滑度。

生成高质量、时间一致的视频是一个活跃的研究方向。尽管扩展图像生成技术提供了基础，但应对时间建模的独特挑战通常需要专用架构和训练策略，以及大量的计算资源。

## 参考资料

- [Generating Videos with Spatio-Temporal Generative Adversarial Networks](https://papers.nips.cc/paper_files/paper/2016/file/490212e690022d25ab2541a547214534-Paper.pdf) — Carl Vondrick, Hamed Pirsiavash, Antonio Torralba (2016)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: NeurIPS Foundation; Volume: 29
  引入使用3D卷积生成对抗网络进行视频合成的奠基性工作。
- [MoCoGAN: Decomposing Motion and Content for Video Generation](https://doi.org/10.1109/CVPR.2018.00165) — Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE Computer Society; Pages: 1526-1535; DOI: [10.1109/CVPR.2018.00165](https://doi.org/10.1109/CVPR.2018.00165)
  提出了一种GAN架构，用于解耦视频生成中的运动和内容，以提高时间连贯性。

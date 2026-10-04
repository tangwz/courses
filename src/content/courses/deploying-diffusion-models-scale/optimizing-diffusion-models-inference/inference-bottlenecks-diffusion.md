---
course: "deploying-diffusion-models-scale"
chapter: "optimizing-diffusion-models-inference"
lesson: "inference-bottlenecks-diffusion"
sourceId: 5377
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-2-optimizing-diffusion-models-inference/inference-bottlenecks-diffusion"
title: "扩散模型推理中的瓶颈"
description: "识别扩散模型采样过程中的计算热点。"
order: 1
plots: []
sourceHash: "428e19cff1d78c4399c230258655cef126298f2958d169f88668eb5eb1403fe1"
sourceCorrections: []
---

如前所述，扩散模型通过迭代细化过程生成高质量数据。此过程从噪声开始，并分多步逐步去噪，通常由提示或条件引导。尽管功能强大，但这种迭代特性是推理 (inference)过程中性能瓶颈的主要原因。与仅需一次正向传播的模型不同，扩散模型会多次执行核心计算循环。清晰了解时间与资源在此循环中的使用情况，对实现有效优化非常重要。

推理过程，常被称为采样，通常包含以下阶段，重复 $N$ 次（$N$ 值可从10到超过1000，具体取决于采样器）：

1. **噪声预测**：一个神经网络 (neural network)，通常采用U-Net架构，将当前噪声数据（例如，步骤 $t$ 的噪声图像）和当前步骤索引 $t$（以及可能的条件信息，如文本嵌入 (embedding)）作为输入。它的目的是预测为达到此状态而添加的噪声。
2. **去噪步骤**：利用预测的噪声和预定义的噪声时间表，采样器算法计算步骤 $t-1$ 时噪声较少的数据的估计值。
3. **迭代**：步骤2的输出成为下一次迭代（步骤 $t-1$）的输入。

此循环持续进行，直到 $t=0$，生成最终输出。我们来分析此过程中的主要瓶颈。

### 重复的神经网络 (neural network)评估

到目前为止，最突出的瓶颈是噪声预测网络（U-Net）的重复执行。这个网络通常非常大，包含数十亿参数 (parameter)和计算开销大的操作，如自注意力 (self-attention)机制 (attention mechanism)，尤其是在生成高分辨率图像时。

考虑一个常见情况：使用DDIM采样器和50步的Stable Diffusion模型生成一张512x512图像。这意味着核心U-Net，连同文本编码器（如果提供了提示）以及可能的VAE解码器，必须为*单次图像生成*执行50次正向传播。每次传播都涉及数十亿次浮点运算（FLOPs）。

> 扩散模型采样的核心循环。U-Net正向传播占据了主要的计算开销，并为每个生成请求重复执行。

这种重复的繁重计算直接影响：

- **延迟**：生成单张图像所需时间大致与采样步数乘以每次U-Net评估所需时间成正比。
- **吞吐量 (throughput)**：在给定硬件配置下，单位时间内可生成的图像数量与每张图像的延迟成反比。
- **成本**：长期运行GPU等昂贵硬件会产生高昂的运营成本。

### 模型大小、内存使用和带宽

扩散模型，尤其是最先进的版本，通常很大。参数 (parameter)以16位浮点格式（FP16）存储的模型，可以轻松占用数GB存储空间，并且仅加载权重 (weight)就需要大量GPU显存（VRAM）。

在U-Net的正向传播过程中，称为激活的中间结果也必须存储在VRAM中。对于高分辨率图像和复杂架构（例如具有许多注意力层的架构），激活所需的内存可能超过权重本身所需的内存。

这导致了几个与内存相关的瓶颈：

- **显存容量**：所需的总显存（权重 + 激活 + 可能的其他缓冲区）决定了所需GPU的最低类别。显存不足会完全阻止推理 (inference)或迫使采用缓慢的变通方法。
- **内存带宽**：在GPU计算单元与其显存之间持续读取模型权重和读写激活会消耗内存带宽。现代GPU上的高带宽内存（HBM）非常重要，但它仍然可能成为限制因素，特别是在计算本身非常快的情况下。缓慢的数据移动会导致计算单元空闲，从而降低整体效率。
- **模型加载时间**：将大型模型权重从存储（例如，磁盘或网络存储）加载到GPU显存的初始时间会增加冷启动延迟，尤其是在无服务器或缩减到零的场景中。

### 采样步数

采样器算法的选择直接影响U-Net必须评估的次数（$N$）。早期采样器，如DDPM，需要数百甚至数千步。较新的采样器（DDIM、PNDM、DPM-Solver++等）在少得多的步数（例如10-50步）内就能取得良好效果，大幅减少了总计算量。然而，即使是20步也代表20次完整的U-Net评估。在不牺牲输出质量的前提下进一步减少 $N$ 是采样器优化的主要目标。

### 步骤间的串行依赖

在采样循环中，计算步骤 $t-1$ 的状态通常需要步骤 $t$ 的结果。这种固有的序列依赖性使得为单张图像生成在不同时间步之间并行化计算变得困难。尽管单个U-Net正向传播内的操作可以在GPU上高度并行化，但整个过程从一步到另一步仍主要呈串行。这限制了加快每个单独步骤的延迟降低潜力。

了解这些核心瓶颈，U-Net计算的主导地位，对内存大小和带宽的需求，采样步数的影响，以及过程的串行特性，是实现优化的第一步。后续章节将介绍量化 (quantization)、蒸馏、采样器改进以及硬件/编译器优化等技术，这些技术专门设计用于缓解这些痛点，使扩散模型推理 (inference)更快、更具成本效益。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍了去噪扩散概率模型的开创性论文，阐述了构成扩散模型推理基础的迭代优化过程。
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Jiaming Song, Chenlin Meng, and Stefano Ermon (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.02502](https://doi.org/10.48550/arXiv.2010.02502)
  介绍了去噪扩散隐式模型（DDIMs），提供了一种比DDPMs更快、更高效的采样策略，直接解决了采样步数过多的瓶颈问题。
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)
  Journal: Medical Image Computing and Computer-Assisted Intervention (MICCAI); Publisher: Springer, Cham; Volume: 9351; Pages: 234-241; DOI: [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28)
  提出了U-Net架构，它是扩散模型中反复评估的核心神经网络，也是推理过程中计算成本的主要来源。
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (2022)
  Journal: Conference on Computer Vision and Pattern Recognition (CVPR); DOI: [10.48550/arXiv.2112.10752](https://doi.org/10.48550/arXiv.2112.10752)
  介绍了像Stable Diffusion这样的潜在扩散模型（LDMs），它们在压缩的潜在空间中运行以实现高分辨率图像生成，展示了现代扩散模型的计算规模。
- [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) — Tim Salimans and Jonathan Ho (2022)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2202.00512](https://doi.org/10.48550/arXiv.2202.00512)
  提出了一种蒸馏方法，使扩散模型能够以显著减少的采样步数生成高质量样本，直接解决了重复U-Net评估的瓶颈问题。

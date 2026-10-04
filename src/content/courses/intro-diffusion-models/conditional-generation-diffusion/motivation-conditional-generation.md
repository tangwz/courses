---
course: "intro-diffusion-models"
chapter: "conditional-generation-diffusion"
lesson: "motivation-conditional-generation"
sourceId: 5436
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-6-conditional-generation-diffusion/motivation-conditional-generation"
title: "条件生成的动因"
description: "为什么控制生成模型的输出通常很有必要。"
order: 1
plots: []
sourceHash: "9ee1d96e5f90979aeeba7d6a0a362f475eb9820ac70069988a574238bfdd6e82"
sourceCorrections: []
---

扩散模型擅长学习数据集 $p(x)$ 的潜在分布，并通过逆转渐进的加噪过程生成高质量样本。从纯高斯噪声 $x_T$ 开始，逆向过程会迭代地对其去噪，以生成一个看起来来自原始数据的样本 $x_0$。

然而，这种标准生成过程是*无条件的*。尽管它能生成逼真的输出，但它不提供对生成*何种*特定输出的明确控制。如果你在一个多样化的动物数据集上训练扩散模型，运行标准采样程序可能会生成狗、猫、鸟或训练数据中存在的任何其他动物的图像。你实际上是从学到的分布 $p(x)$ 中获得一个随机样本，但你无法直接要求模型生成，例如，*只*生成猫的图像。

这种缺乏直接控制的情况限制了无条件模型在许多场景下的实用性。通常，我们需要根据特定要求或输入来引导生成过程。考虑以下常见应用情况：

- **类别条件生成：** 我们可能希望生成属于特定类别的图像，例如从数据集中只生成“桥梁”的图片，或者只生成“轿车”类型的汽车。
- **文本到图像合成：** 一个非常流行的应用是根据描述性文本提示生成图像，例如为“宇航员在月球上骑马”创建图像。
- **图像编辑和转换：** 诸如图像修复（填充缺失区域）、风格迁移（将一幅图像的风格应用到另一幅）或超分辨率等任务通常需要以现有输入图像为条件进行生成。
- **数据增强：** 生成符合某些特定标准，用于训练下游模型的合成数据。

在所有这些例子中，目标不只是从整体数据分布 $p(x)$ 中采样，而是从*条件*分布 $p(x|y)$ 中采样，而 $y$ 代表条件信息。这个条件变量 $y$ 可能是：

- 一个类别标签（例如，代表“猫”或“狗”的整数）。
- 一个源自提示的文本嵌入 (embedding)。
- 另一幅图像（用于图像转换或编辑等任务）。
- 与生成任务相关的其他形式的结构化数据。

因此，我们需要机制来将这些条件信息 $y$ 整合到扩散模型的生成过程中。我们需要方法来引导迭代去噪步骤，以便最终输出 $x_0$ 不仅看起来逼真（属于数据流形），而且与提供的条件 $y$ 对齐 (alignment)。

本章将重点介绍正是这些机制。我们将研究允许我们控制扩散模型输出的技术，将其从随机样本生成器转变为可控的合成引擎。我们首先会了解外部模型如何引导过程，然后转向更整合的方法，例如分类器无关引导，它已成为条件扩散模型的一种标准技术。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  这篇基础性论文介绍了去噪扩散概率模型（DDPM），为后续条件生成方法奠定了无条件图像生成的核心框架。
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — Prafulla Dhariwal and Alex Nichol (2021)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2105.05233](https://doi.org/10.48550/arXiv.2105.05233)
  这项工作显著推进了扩散模型，展示了其卓越的图像生成质量，并引入了分类器引导，这是一种早期且有效的条件生成方法。
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — Jonathan Ho and Tim Salimans (2022)
  Journal: arXiv preprint arXiv:2207.12598; DOI: [10.48550/arXiv.2207.12598](https://doi.org/10.48550/arXiv.2207.12598)
  引入了无分类器扩散引导，这是一种广泛采用的技术，无需外部分类器即可实现有效的条件生成，简化了许多应用中的过程。
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer (2022)
  Journal: CVPR 2022; DOI: [10.48550/arXiv.2112.10752](https://doi.org/10.48550/arXiv.2112.10752)
  介绍了潜在扩散模型（LDMs），它实现了高效的高分辨率图像生成，是像Stable Diffusion等流行文本到图像模型的基础架构，展示了实用的条件生成。

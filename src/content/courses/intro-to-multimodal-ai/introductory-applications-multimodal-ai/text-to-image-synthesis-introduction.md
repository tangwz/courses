---
course: "intro-to-multimodal-ai"
chapter: "introductory-applications-multimodal-ai"
lesson: "text-to-image-synthesis-introduction"
sourceId: 6534
sourceUrl: "https://apxml.com/zh/courses/intro-to-multimodal-ai/chapter-5-introductory-applications-multimodal-ai/text-to-image-synthesis-introduction"
title: "文本到图像生成：根据描述创建视觉内容（简介）"
description: "对文本到图像生成技术的介绍，AI在此技术中根据文本描述创建图像。"
order: 3
plots: []
sourceHash: "d2b522c4893c7baa83eafc19f4d4d3ff817bf86f97a99d9f4064eebb11476578"
sourceCorrections: []
---

多模态 (multimodal)AI最引人入胜的应用之一是**文本到图像生成**。想象一下，您输入一个描述场景的句子，例如“一个快乐的机器人在火星表面挥手”，然后一个AI系统试图为您创建该场景的图片。这种能力表明AI正在从主要理解现有数据，发展到也能根据不同类型的输入生成新的、富有创意的内容。

文本到图像生成是多模态AI的直接体现，因为它处理来自一种模态（文本）的信息，以在另一种模态（图像）中生成输出。为了实现这一点，AI模型必须学习将词语和短语与视觉元素、风格和排列联系起来。例如，它需要将文本“快乐的机器人”与可能在机器人身上传达喜悦的视觉特征关联起来，理解“火星表面”的样子，以及“挥手”应该如何描绘。这不仅涉及识别单个物体，还要理解它们之间的关系、颜色和纹理等属性，甚至描述所暗示的整体情绪。

可以将AI学习此任务的方式看作与人类艺术家学习的方式有些相似。艺术家学习大量图像，观察周围环境，并练习绘制他们所见或想象的内容。文本到图像AI模型在配对图像及其对应文本描述的数据集合上进行训练。例如，一个数据集可能包含一张猫玩毛线球的图像，并附带文本，例如“一只毛茸茸的姜黄色猫在木地板上拍打着一个红色毛线球。”通过处理数百万这样的对，AI模型学习文本模式（例如“猫”这个词或“木地板”这个短语）和视觉模式（例如毛茸茸的纹理、尖耳朵或木纹）之间的统计关联。这并非人类意义上的理解，而是文本与视觉之间复杂关联图的构建。

尽管基础技术涉及先进的机器学习 (machine learning)模型，但总体过程可以分为两个主要阶段来理解：

1. **理解文本（文本编码）：** 当您提供文本提示时，例如“日出时宁静的湖面，背景是雾蒙蒙的山脉”，AI首先处理此文本。专用组件（通常使用神经网络 (neural network)构建）将词语及其结构转换为数字格式。这种数字表示旨在获取提示中的重要信息，包括物体、它们的属性以及它们之间的关联。
2. **生成图像（图像生成/解码）：** 文本的这种数字“摘要”随后引导模型的图像生成部分。该组件，通常也是一个精密的神经网络，工作以构建图像。这可能涉及从噪声模式开始并逐步完善它，或者以其他方式构建，以创建与编码文本高度一致的视觉内容。目标是生成一张人们可能使用与最初提供的提示相似的语句来描述的图像。

这是一个简化图表，说明了这一流程：

> 此图显示文本提示输入到文本到图像AI模型中，然后输出对应的图像。

文本到图像生成不仅仅是技术上的新奇事物。它有多个实际和创意用途：

- **创意辅助：** 艺术家和设计师可以使用这些工具来快速地将想法可视化，生成草图以获取灵感，或创建独特的视觉素材。
- **内容创作：** 它可以用于为文章、故事或演示文稿生成插图，特别是当需要定制或独特的图像时。
- **个性化图像：** 用户可以为特定需求或仅为娱乐创建定制图像，例如“一个以太空为主题的烘焙店的标志，其羊角面包形状像火箭。”

值得注意的是，这个方面正在快速发展。早期的文本到图像模型生成的结果可能有些抽象或模糊，但新系统可以生成非常详细且通常高度逼真的图像。

然而，它们并非没有局限性：

- **提示细节：** 生成图像的质量和准确性通常很大程度上取决于文本提示的措辞是否清晰和详细。措辞上的微小变化有时可能导致截然不同的图像。
- **意料之外的输出：** AI有时会错误理解复杂或非常规的提示，导致图像令人惊讶、幽默，或与预期不完全一致。例如，非常抽象的请求可能使AI难以准确描绘。
- **习得偏差：** 由于这些AI系统从互联网和其他来源的大量数据中学习，它们会无意中学习并再现数据中存在的偏差。这是一个正在进行的研究和伦理考量方向。

即使存在这些方面，文本到图像生成也是一个强大的例证，说明AI如何整合不同类型的数据来执行结合解释与创造的任务。它强调了AI不仅正在发展分析现有信息的能力，而且还在发展生成新的、多样化输出的能力。

## 参考资料

- [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092) — Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever (2021)
  DOI: [10.48550/arXiv.2102.12092](https://doi.org/10.48550/arXiv.2102.12092)
  介绍了DALL-E，一个从文本描述生成多样化图像的开创性模型，展示了大规模文本-图像预训练的有效性。
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_with_Latent_Diffusion_Models_CVPR_2022_paper.html) — Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (2022)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 10684-10695; DOI: [10.1109/CVPR52688.2022.01042](https://doi.org/10.1109/CVPR52688.2022.01042)
  提出了潜在扩散模型，即Stable Diffusion的架构，用于高效高质量的文本到图像合成，使先进的生成技术变得易于使用。

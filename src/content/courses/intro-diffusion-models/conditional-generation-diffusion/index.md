---
course: "intro-diffusion-models"
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-6-conditional-generation-diffusion"
sourceId: 1000
chapter: "conditional-generation-diffusion"
title: "扩散模型的条件生成"
order: 6
description: "引导扩散模型生成过程的方法，这些方法基于类别标签或文本等条件信息。"
hasQuiz: true
---

迄今为止，我们已经了解了扩散模型如何通过逆转加噪过程来生成数据，其起点是纯随机噪声 $x_T \sim \mathcal{N}(0, \mathbf{I})$。这会产生反映训练数据分布的样本，但无法对输出进行精确控制。

然而，我们通常希望引导生成过程。例如，我们可能想生成特定物体类别的图像，或者根据文本描述创建图像。本章将介绍用于 *条件* 生成的方法，使我们能够根据额外信息（通常表示为条件 $y$）影响扩散模型的输出。

我们将考察诸如分类器引导等技术，其中一个独立的分类模型有助于将采样引向所需属性。随后我们将研究无分类器引导（CFG），这是一种广泛使用的方法，通过修改训练和采样流程，无需外部分类器即可实现类似控制。最后，我们将提及基于文本提示的条件设置的基础知识，这是现代文本到图像模型背后的重要技术。

本章结束时，您将理解如何实现并应用这些引导技术，以便更好地控制扩散模型的生成能力。

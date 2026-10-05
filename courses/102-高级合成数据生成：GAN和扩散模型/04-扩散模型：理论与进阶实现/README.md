# 第 4 章：扩散模型：理论与进阶实现

来源：[原章节](https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-4-diffusion-models-theory-implementation)

[返回课程目录](../README.md)

在我们学习了生成对抗网络之后，本章介绍扩散模型，这是一种独特且有效的生成建模方法。这些模型通过逐步向数据点添加噪声，然后学习逆向去噪过程，在生成高质量数据（特别是图像）方面取得了卓越成果。

本章涵盖以下几个方面：
*   将扩散过程与随机微分方程（$SDE$）联系起来的数学基础。
*   去噪扩散概率模型（DDPM）的公式和实现细节，包括其具体目标函数，例如实践中常用的简化目标函数：$$L_{simple}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, t) \|^2 \right]$$。
*   扩散模型与基于分数的生成建模之间的关联，通过分数匹配和朗之万动力学等思想。
*   加速采样过程的方法，例如去噪扩散隐式模型（DDIM），以及使用分类器引导或无分类器引导来控制生成的技术。
*   常用的神经网络结构，尤其是用于噪声预测的U-Net结构。
*   通过动手编码环节构建基础的DDPM，以巩固理解。

您将理解扩散模型背后的理论，以及有效实现和改进它们的实际考量。

## 小节

- 1. [数学基本原理：随机微分方程](01-%E6%95%B0%E5%AD%A6%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%EF%BC%9A%E9%9A%8F%E6%9C%BA%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B.md)
- 2. [去噪扩散概率模型 (DDPM)](02-%E5%8E%BB%E5%99%AA%E6%89%A9%E6%95%A3%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B%20%28DDPM%29.md)
- 3. [基于得分的生成模型](03-%E5%9F%BA%E4%BA%8E%E5%BE%97%E5%88%86%E7%9A%84%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B.md)
- 4. [改进技术：DDIM和方差调度](04-%E6%94%B9%E8%BF%9B%E6%8A%80%E6%9C%AF%EF%BC%9ADDIM%E5%92%8C%E6%96%B9%E5%B7%AE%E8%B0%83%E5%BA%A6.md)
- 5. [分类器引导与无分类器引导](05-%E5%88%86%E7%B1%BB%E5%99%A8%E5%BC%95%E5%AF%BC%E4%B8%8E%E6%97%A0%E5%88%86%E7%B1%BB%E5%99%A8%E5%BC%95%E5%AF%BC.md)
- 6. [扩散模型 (U-Net) 的结构考量](06-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%20%28U-Net%29%20%E7%9A%84%E7%BB%93%E6%9E%84%E8%80%83%E9%87%8F.md)
- 7. [动手实践：构建基础DDPM](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%9F%BA%E7%A1%80DDPM.md)

---
course: "generative-adversarial-networks-gans"
chapter: "evaluation-of-gans"
lesson: "qualitative-assessment-visual-turing"
sourceId: 2731
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans/qualitative-assessment-visual-turing"
title: "定性评估：视觉图灵测试"
description: "Using human evaluation and visual inspection to assess sample quality."
order: 2
plots: []
sourceHash: "6ee0313290590d2a5ceda585ca8bedb8f51da0d458e2783cb1b674a4a45a91c0"
sourceCorrections: []
---

尽管像FID和IS这样的定量指标提供了有价值的数值分数，但它们并不总能完整反映GAN的性能，尤其是在生成样本的*感知质量*和*真实性*方面。自动化指标有时会被那些符合统计特性但包含不真实伪影，或未能捕捉到使图像对人类观察者有说服力的微小之处的生成内容所蒙蔽。在这种情况下，定性评估变得不可或缺。

定性评估最直接的形式之一来源于艾伦·图灵著名的机器智能测试：**视觉图灵测试**。其核心思想很直接：人类评估者能否可靠地区分来自训练数据分布的真实样本和GAN生成器产生的合成样本？

### 进行视觉图灵测试

该设置通常涉及以随机顺序向人类参与者展示一组图像，其中一些是真实的，一些是生成的。参与者（常被称为评判员或评估员）不知道每张特定图像的来源。他们的任务是将每张图像分类为“真实”或“虚假”（生成）。

存在几种变体：

1. **单幅图像分类：** 评估者每次看到一张图像并做出判断。这是与经典图灵测试最直接的类比。
2. **并排比较：** 评估者可能会同时看到一张真实图像和一张生成图像，并被要求识别虚假的那张，或者简单地评价哪一张看起来更真实。
3. **评分量表：** 替代二元分类，评估者可能会在李克特量表（例如，1-5）上对样本进行真实性、质量或伪影存在情况的评分。

结果随后被汇总。如果评估者的表现接近随机（即，区分真实和虚假的准确率在50%左右），这表明生成器正在产生高度真实的样本，人类难以与真实品区分。相反，高准确率表明生成的样本存在明显的缺陷。

> 用于GAN评估的视觉图灵测试的基本流程。真实样本和生成样本被盲目地展示给人类评估者进行分类。

### 定性评估的优点

- **直接感知评估：** 它直接衡量我们通常最关心的内容：生成样本是否在人类看来是真实的。自动化指标近似此点，但可能遗漏微小线索。
- **对伪影的敏感性：** 人类善于发现视觉不一致、奇异纹理或不自然物体形状，这些可能不会显著扰乱像FID这样的统计指标。
- **真实性的黄金标准：** 最终，如果目标是照片真实感或有说服力的风格化生成，人类判断仍是衡量标准。

### 挑战与局限性

尽管它具有直观吸引力，通过视觉图灵测试进行定性评估存在明显缺点：

- **主观性：** 判断会因评估者的专业知识、对细节的关注以及个人偏见而有很大差异。一个人认为真实的，另一个人可能认为有缺陷。
- **可扩展性和成本：** 进行这些测试需要大量人力，使其耗时且昂贵，特别是对于大型数据集或大量模型比较。在日常开发迭代中，进行具有统计意义的参与者数量的测试通常不切实际。
- **可复现性：** 评估过程（说明、界面、参与者池）的标准化很困难，使得结果可能难以在不同的研究或实验室中复现。
- **评估者疲劳：** 要求评估者判断许多图像可能导致疲劳，并随时间降低准确性或一致性。
- **诊断能力不足：** 尽管它能告诉你人们*是否*能识别虚假，但它不一定能告诉你*原因*。与像Precision/Recall或FID这样的指标相比，它在特定失败模式（如多样性不足与样本质量差）上提供的反馈颗粒度较低。

### 互补作用

视觉图灵测试和其他定性方法很少单独使用，尤其是在模型开发的迭代过程中。它们是定量指标的宝贵补充。像FID这样的定量分数可以在训练和超参数 (parameter) (hyperparameter)调整期间提供快速、自动化的反馈。定性评估通常保留用于最终模型比较、里程碑评估，或当调查自动化指标指示的特定感知失败时。它们提供了必要的“现实检验”，确保定量衡量的进展转化为真正更好、更真实的生成输出。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems 27 (NIPS 2014); DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  介绍了生成对抗网络框架，其中判别器学习区分真实样本和生成样本，这在概念上类似于视觉图灵测试中的人类评判。
- [Do GANs Always Fool Humans? A Study on Perceptual Distinguishability](https://ieeexplore.ieee.org/document/8446221) — Ali Borji (2019)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 41; Pages: 2516-2529; DOI: [10.1109/TPAMI.2018.2867909](https://doi.org/10.1109/TPAMI.2018.2867909)
  研究人类观察者区分真实图像和GAN生成图像的能力，为视觉图灵测试的有效性提供了见解。
- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) — Tero Karras, Samuli Laine, Timo Aila (2019)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 4401-4410; DOI: [10.1109/CVPR.2019.00453](https://doi.org/10.1109/CVPR.2019.00453)
  介绍了StyleGAN架构，这是高质量图像合成的突出工作，并广泛使用人类感知研究（视觉图灵测试）来评估真实感。

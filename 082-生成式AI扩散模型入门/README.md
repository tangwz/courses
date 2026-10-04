# 生成式AI扩散模型入门

来源：[生成式AI扩散模型入门](https://apxml.com/zh/courses/intro-diffusion-models)

理解用于生成任务的扩散模型的原理与实现。本课程讲解正向和反向扩散过程、U-Net等网络结构、训练步骤以及采样与条件生成输出的方法。学习相关知识，以创建AI生成图像及其他数据。

预计学时：18 小时

先修要求：Python与机器学习知识

## 课程目录

### 1. [生成模型简介](01-%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E7%AE%80%E4%BB%8B/README.md)

- 1. [生成模型概述](01-%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E7%AE%80%E4%BB%8B/01-%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0.md)
- 2. [扩散模型的缘由](01-%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E7%AE%80%E4%BB%8B/02-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E7%BC%98%E7%94%B1.md)
- 3. [核心思路：噪声与去噪](01-%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E7%AE%80%E4%BB%8B/03-%E6%A0%B8%E5%BF%83%E6%80%9D%E8%B7%AF%EF%BC%9A%E5%99%AA%E5%A3%B0%E4%B8%8E%E5%8E%BB%E5%99%AA.md)
- 4. [概率框架概述](01-%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E7%AE%80%E4%BB%8B/04-%E6%A6%82%E7%8E%87%E6%A1%86%E6%9E%B6%E6%A6%82%E8%BF%B0.md)
- [章节测验](https://apxml.com/zh/courses/intro-diffusion-models/chapter-1-generative-modeling-fundamentals/quiz)

### 2. [前向扩散过程](02-%E5%89%8D%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/README.md)

- 1. [定义马尔可夫链](02-%E5%89%8D%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/01-%E5%AE%9A%E4%B9%89%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE.md)
- 2. [高斯噪声调度](02-%E5%89%8D%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/02-%E9%AB%98%E6%96%AF%E5%99%AA%E5%A3%B0%E8%B0%83%E5%BA%A6.md)
- 3. [每一步的数学表示](02-%E5%89%8D%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/03-%E6%AF%8F%E4%B8%80%E6%AD%A5%E7%9A%84%E6%95%B0%E5%AD%A6%E8%A1%A8%E7%A4%BA.md)
- 4. [从中间步骤采样](02-%E5%89%8D%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/04-%E4%BB%8E%E4%B8%AD%E9%97%B4%E6%AD%A5%E9%AA%A4%E9%87%87%E6%A0%B7.md)
- 5. [前向过程的性质](02-%E5%89%8D%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/05-%E5%89%8D%E5%90%91%E8%BF%87%E7%A8%8B%E7%9A%84%E6%80%A7%E8%B4%A8.md)
- 6. [实践：模拟正向扩散](02-%E5%89%8D%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/06-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%A8%A1%E6%8B%9F%E6%AD%A3%E5%90%91%E6%89%A9%E6%95%A3.md)
- [章节测验](https://apxml.com/zh/courses/intro-diffusion-models/chapter-2-forward-diffusion-process/quiz)

### 3. [逆向扩散过程](03-%E9%80%86%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/README.md)

- 1. [目标：逆转马尔可夫链](03-%E9%80%86%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/01-%E7%9B%AE%E6%A0%87%EF%BC%9A%E9%80%86%E8%BD%AC%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE.md)
- 2. [逼近逆向转移](03-%E9%80%86%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/02-%E9%80%BC%E8%BF%91%E9%80%86%E5%90%91%E8%BD%AC%E7%A7%BB.md)
- 3. [使用神经网络参数化逆向过程](03-%E9%80%86%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/03-%E4%BD%BF%E7%94%A8%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%8F%82%E6%95%B0%E5%8C%96%E9%80%86%E5%90%91%E8%BF%87%E7%A8%8B.md)
- 4. [预测噪声分量](03-%E9%80%86%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/04-%E9%A2%84%E6%B5%8B%E5%99%AA%E5%A3%B0%E5%88%86%E9%87%8F.md)
- 5. [去噪步骤的数学表达](03-%E9%80%86%E5%90%91%E6%89%A9%E6%95%A3%E8%BF%87%E7%A8%8B/05-%E5%8E%BB%E5%99%AA%E6%AD%A5%E9%AA%A4%E7%9A%84%E6%95%B0%E5%AD%A6%E8%A1%A8%E8%BE%BE.md)
- [章节测验](https://apxml.com/zh/courses/intro-diffusion-models/chapter-3-reverse-diffusion-process/quiz)

### 4. [模型架构与训练](04-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%8E%E8%AE%AD%E7%BB%83/README.md)

- 1. [用于噪声预测的U-Net架构](04-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%8E%E8%AE%AD%E7%BB%83/01-%E7%94%A8%E4%BA%8E%E5%99%AA%E5%A3%B0%E9%A2%84%E6%B5%8B%E7%9A%84U-Net%E6%9E%B6%E6%9E%84.md)
- 2. [整合时间步信息](04-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%8E%E8%AE%AD%E7%BB%83/02-%E6%95%B4%E5%90%88%E6%97%B6%E9%97%B4%E6%AD%A5%E4%BF%A1%E6%81%AF.md)
- 3. [定义训练目标](04-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%8E%E8%AE%AD%E7%BB%83/03-%E5%AE%9A%E4%B9%89%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87.md)
- 4. [简化训练损失的推导](04-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%8E%E8%AE%AD%E7%BB%83/04-%E7%AE%80%E5%8C%96%E8%AE%AD%E7%BB%83%E6%8D%9F%E5%A4%B1%E7%9A%84%E6%8E%A8%E5%AF%BC.md)
- 5. [训练算法](04-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%8E%E8%AE%AD%E7%BB%83/05-%E8%AE%AD%E7%BB%83%E7%AE%97%E6%B3%95.md)
- 6. [动手实践：搭建U-Net](04-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E4%B8%8E%E8%AE%AD%E7%BB%83/06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%90%AD%E5%BB%BAU-Net.md)
- [章节测验](https://apxml.com/zh/courses/intro-diffusion-models/chapter-4-model-architecture-training/quiz)

### 5. [采样与生成过程](05-%E9%87%87%E6%A0%B7%E4%B8%8E%E7%94%9F%E6%88%90%E8%BF%87%E7%A8%8B/README.md)

- 1. [从噪声生成数据](05-%E9%87%87%E6%A0%B7%E4%B8%8E%E7%94%9F%E6%88%90%E8%BF%87%E7%A8%8B/01-%E4%BB%8E%E5%99%AA%E5%A3%B0%E7%94%9F%E6%88%90%E6%95%B0%E6%8D%AE.md)
- 2. [DDPM采样算法](05-%E9%87%87%E6%A0%B7%E4%B8%8E%E7%94%9F%E6%88%90%E8%BF%87%E7%A8%8B/02-DDPM%E9%87%87%E6%A0%B7%E7%AE%97%E6%B3%95.md)
- 3. [理解采样方差](05-%E9%87%87%E6%A0%B7%E4%B8%8E%E7%94%9F%E6%88%90%E8%BF%87%E7%A8%8B/03-%E7%90%86%E8%A7%A3%E9%87%87%E6%A0%B7%E6%96%B9%E5%B7%AE.md)
- 4. [加速采样方法介绍：DDIM](05-%E9%87%87%E6%A0%B7%E4%B8%8E%E7%94%9F%E6%88%90%E8%BF%87%E7%A8%8B/04-%E5%8A%A0%E9%80%9F%E9%87%87%E6%A0%B7%E6%96%B9%E6%B3%95%E4%BB%8B%E7%BB%8D%EF%BC%9ADDIM.md)
- 5. [DDIM采样算法](05-%E9%87%87%E6%A0%B7%E4%B8%8E%E7%94%9F%E6%88%90%E8%BF%87%E7%A8%8B/05-DDIM%E9%87%87%E6%A0%B7%E7%AE%97%E6%B3%95.md)
- 6. [DDPM与DDIM的权衡](05-%E9%87%87%E6%A0%B7%E4%B8%8E%E7%94%9F%E6%88%90%E8%BF%87%E7%A8%8B/06-DDPM%E4%B8%8EDDIM%E7%9A%84%E6%9D%83%E8%A1%A1.md)
- 7. [动手实践：实现采样循环](05-%E9%87%87%E6%A0%B7%E4%B8%8E%E7%94%9F%E6%88%90%E8%BF%87%E7%A8%8B/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E9%87%87%E6%A0%B7%E5%BE%AA%E7%8E%AF.md)
- [章节测验](https://apxml.com/zh/courses/intro-diffusion-models/chapter-5-sampling-generation-process/quiz)

### 6. [扩散模型的条件生成](06-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90/README.md)

- 1. [条件生成的动因](06-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90/01-%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90%E7%9A%84%E5%8A%A8%E5%9B%A0.md)
- 2. [分类器引导](06-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90/02-%E5%88%86%E7%B1%BB%E5%99%A8%E5%BC%95%E5%AF%BC.md)
- 3. [无分类器引导 (CFG)](06-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90/03-%E6%97%A0%E5%88%86%E7%B1%BB%E5%99%A8%E5%BC%95%E5%AF%BC%20%28CFG%29.md)
- 4. [实现分类器无关引导](06-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90/04-%E5%AE%9E%E7%8E%B0%E5%88%86%E7%B1%BB%E5%99%A8%E6%97%A0%E5%85%B3%E5%BC%95%E5%AF%BC.md)
- 5. [文本条件要点](06-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90/05-%E6%96%87%E6%9C%AC%E6%9D%A1%E4%BB%B6%E8%A6%81%E7%82%B9.md)
- 6. [用于条件生成的架构修改](06-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90/06-%E7%94%A8%E4%BA%8E%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90%E7%9A%84%E6%9E%B6%E6%9E%84%E4%BF%AE%E6%94%B9.md)
- 7. [动手实践：应用引导](06-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BA%94%E7%94%A8%E5%BC%95%E5%AF%BC.md)
- [章节测验](https://apxml.com/zh/courses/intro-diffusion-models/chapter-6-conditional-generation-diffusion/quiz)

## 学习目标

- **扩散过程机制**：解释正向（加噪）和反向（去噪）过程的数学表述。
- **模型结构**：理解U-Net结构在扩散模型中的作用与构成。
- **训练与损失**：描述扩散模型中使用的训练目标和损失函数。
- **采样技术**：实现DDPM等采样步骤，并了解DDIM等更快的方法。
- **条件生成**：应用基本方法，利用条件信息引导生成过程。
- **实现要点**：使用深度学习框架构建并训练一个简单的扩散模型。

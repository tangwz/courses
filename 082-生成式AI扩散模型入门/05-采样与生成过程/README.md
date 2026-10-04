# 第 5 章：采样与生成过程

来源：[原章节](https://apxml.com/zh/courses/intro-diffusion-models/chapter-5-sampling-generation-process)

[返回课程目录](../README.md)

在我们已经弄清扩散模型如何训练以预测损坏数据中的噪声之后，本章将侧重于其逆向操作：生成新的数据样本。我们将了解如何从纯噪声（通常从高斯分布 $x_T \sim \mathcal{N}(0, \mathbf{I})$ 中采样）开始，并迭代应用所学到的逆向扩散过程，以生成一个干净的数据点 $x_0$。

您将学习标准去噪扩散概率模型（DDPM）采样算法的标准步骤。接下来，我们将介绍去噪扩散隐式模型（DDIM），这是一种相关但通常更快的采样技术，可在速度和样本质量之间提供不同的平衡。最后，我们将讨论在代码中构建这些采样循环所需的实现细节。

## 小节

- 1. [从噪声生成数据](01-%E4%BB%8E%E5%99%AA%E5%A3%B0%E7%94%9F%E6%88%90%E6%95%B0%E6%8D%AE.md)
- 2. [DDPM采样算法](02-DDPM%E9%87%87%E6%A0%B7%E7%AE%97%E6%B3%95.md)
- 3. [理解采样方差](03-%E7%90%86%E8%A7%A3%E9%87%87%E6%A0%B7%E6%96%B9%E5%B7%AE.md)
- 4. [加速采样方法介绍：DDIM](04-%E5%8A%A0%E9%80%9F%E9%87%87%E6%A0%B7%E6%96%B9%E6%B3%95%E4%BB%8B%E7%BB%8D%EF%BC%9ADDIM.md)
- 5. [DDIM采样算法](05-DDIM%E9%87%87%E6%A0%B7%E7%AE%97%E6%B3%95.md)
- 6. [DDPM与DDIM的权衡](06-DDPM%E4%B8%8EDDIM%E7%9A%84%E6%9D%83%E8%A1%A1.md)
- 7. [动手实践：实现采样循环](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E9%87%87%E6%A0%B7%E5%BE%AA%E7%8E%AF.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-diffusion-models/chapter-5-sampling-generation-process/quiz)

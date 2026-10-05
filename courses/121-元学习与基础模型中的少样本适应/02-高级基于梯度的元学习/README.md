# 第 2 章：高级基于梯度的元学习

来源：[原章节](https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-2-advanced-gradient-based-meta-learning)

[返回课程目录](../README.md)

本章着重介绍基于梯度的元学习方法，这是一类旨在学习模型初始化或更新规则、以利于快速适应的主要算法。其主要思想是优化参数 $\theta$，使得在新任务支持集上经过少量梯度步长后，能在其查询集上获得良好性能。

我们将首先细致考察模型无关元学习 (MAML)，介绍其优化目标及所涉及的计算方面。接着，我们分析计算高效的近似方法，例如一阶MAML (FOMAML) 和 Reptile，比较它们的机制和性能表现。隐式MAML (iMAML) 将作为一种替代方法进行介绍，它可能带来稳定性与内存方面的益处。我们还会讨论优化稳定性、梯度方差等常见问题，并介绍缓解这些问题的方法。最后，我们将考量如何将这些基于梯度的方法扩展到大型基础模型的具体挑战和应对策略，并以一个聚焦于FOMAML的实操练习作结。

## 小节

- 1. [模型无关元学习 (MAML)](01-%E6%A8%A1%E5%9E%8B%E6%97%A0%E5%85%B3%E5%85%83%E5%AD%A6%E4%B9%A0%20%28MAML%29.md)
- 2. [一阶MAML (FOMAML) 与 Reptile](02-%E4%B8%80%E9%98%B6MAML%20%28FOMAML%29%20%E4%B8%8E%20Reptile.md)
- 3. [隐式MAML (iMAML)](03-%E9%9A%90%E5%BC%8FMAML%20%28iMAML%29.md)
- 4. [处理稳定性和梯度方差](04-%E5%A4%84%E7%90%86%E7%A8%B3%E5%AE%9A%E6%80%A7%E5%92%8C%E6%A2%AF%E5%BA%A6%E6%96%B9%E5%B7%AE.md)
- 5. [基础模型的可扩展性考量](05-%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%8F%AF%E6%89%A9%E5%B1%95%E6%80%A7%E8%80%83%E9%87%8F.md)
- 6. [动手实践：实现 FOMAML 用于模型适应](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%20FOMAML%20%E7%94%A8%E4%BA%8E%E6%A8%A1%E5%9E%8B%E9%80%82%E5%BA%94.md)

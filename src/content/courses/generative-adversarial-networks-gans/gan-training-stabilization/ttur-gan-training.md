---
course: "generative-adversarial-networks-gans"
chapter: "gan-training-stabilization"
lesson: "ttur-gan-training"
sourceId: 2701
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-3-gan-training-stabilization/ttur-gan-training"
title: "双时间尺度更新规则 (TTUR)"
description: "为生成器和判别器使用不同的学习率。"
order: 7
plots: []
sourceHash: "085084bbef13aa4c75f2d7913123ca9d7ed88c9fa9f43b4d28b4869569452d8d"
sourceCorrections: []
---

在GAN训练过程中，生成器 ($G$) 和判别器 ($D$) 之间的互动需要精细调整。标准方法对两个网络使用相同的学习率。然而，实践观察和理论分析表明，这可能并非总是最合适的。如果判别器相对于生成器学习得太快，它很容易在早期就区分真实样本和生成样本，导致生成器出现梯度消失。反之，如果判别器学习得太慢，生成器可能无法获得有效反馈以改进。

双时间尺度更新规则 (TTUR) 提供了一种简单而有效的训练动态调整，它建议为生成器和判别器的优化器使用不同的学习率。这个想法源于对随机近似算法的分析，表明不同的学习速度可以在特定优化情况中带来更好的收敛特性，这包括GAN特有的鞍点优化。

### TTUR的原则

TTUR 的核心原则直接明了：**判别器的学习率要高于生成器的学习率。**

设 $\alpha_D$ 为判别器优化器的学习率，$\alpha_G$ 为生成器优化器的学习率。TTUR 要求：


$$
\alpha_D > \alpha_G
$$


常用比例可能设定 $\alpha_D$ 是 $\alpha_G$ 的2到5倍，例如。

### 理由与好处

为什么这个看起来很小的改变会有帮助呢？

1. **判别器反馈改进：** 通过让判别器学习得更快 ($\alpha_D > \alpha_G$)，它能更快地接近当前生成器的最优判别功能。一个更接近其最优状态的判别器能为生成器提供更准确、更有用的梯度。这有助于生成器根据判别器的当前状态，明白如何调整其参数 (parameter)以生成更真实的样本。
2. **动态稳定：** 使用不同的学习率解耦了更新速度，这可以抑制振荡，并避免一个网络持续压倒另一个。与需要进行复杂手动调度（例如每次生成器更新时判别器更新 $k$ 次）不同，TTUR 通过学习率在标准优化步骤中直接达到类似的效果。
3. **更快收敛：** 通常，与使用单一学习率相比，采用TTUR可以带来更快的整体收敛，因为生成器在整个训练过程中会收到更持续有用的梯度信号。

### 实现方法

在现代深度学习 (deep learning)框架中，实现TTUR通常非常简单。它涉及定义两个独立的优化器实例，一个用于判别器的参数 (parameter)，一个用于生成器的参数，每个实例都配置其相应的学习率 ($\alpha_D$ 和 $\alpha_G$ )。

```python
# 使用类似PyTorch的伪代码示例

# 根据TTUR定义学习率
learning_rate_D = 0.0004
learning_rate_G = 0.0001 # 通常小于learning_rate_D

# 创建独立的优化器
optimizer_D = Adam(discriminator.parameters(), lr=learning_rate_D, betas=(0.0, 0.9))
optimizer_G = Adam(generator.parameters(), lr=learning_rate_G, betas=(0.0, 0.9)) # 注意：beta1=0常与TTUR一起使用

# --- 训练循环内部 ---

# 1. 更新判别器
optimizer_D.zero_grad()
# 计算判别器损失（例如，WGAN-GP损失）
d_loss.backward()
optimizer_D.step()

# 2. 更新生成器
optimizer_G.zero_grad()
# 计算生成器损失
g_loss.backward()
optimizer_G.step()
```

> 一个简化的训练循环结构，展示了为判别器和生成器分别使用不同学习率的独立优化器，遵循TTUR原则。

### 与其他技术的关联

TTUR 与本章讨论的其他稳定方法并非互相排斥，例如 Wasserstein 损失、梯度惩罚 (WGAN-GP) 或谱归一化 (normalization)。事实上，它经常与这些技术*结合使用*。WGAN-GP 和谱归一化等方法通过修改损失函数 (loss function)或网络结构来提高稳定性，而 TTUR 则直接通过学习率调整优化动态。将这些方法结合使用通常比单独使用任何一种技术能产生更优的结果。

虽然 TTUR 为设置 $\alpha_D > \alpha_G$ 提供了强大的启发式和理论基础，但具体值仍是超参数 (parameter) (hyperparameter)。找到最佳学习率及其比例可能仍然需要根据特定数据集、架构和正在使用的其他稳定技术进行一些实验。然而，TTUR 提供了一个有原则的起点，与平衡单一学习率或临时更新调度相比，它通常简化了调整过程。它代表着一个有价值的工具，用于增强 GAN 训练的稳定性和收敛速度。

## 参考资料

- [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron C. Courville (2017)
  Journal: Advances in Neural Information Processing Systems 30; Volume: 30; Pages: 5767-5777; DOI: [10.48550/arXiv.1704.00028](https://doi.org/10.48550/arXiv.1704.00028)
  这篇论文介绍了带有梯度惩罚的Wasserstein GAN (WGAN-GP)，并广泛推广了用于稳定GAN训练的双时间尺度更新规则 (TTUR)，包括对判别器和生成器学习率以及Adam优化器beta参数的具体建议。
- [Which Training Methods for GANs does your Application Need? A Benchmark Study](https://ieeexplore.ieee.org/document/9189689) — Simon Lucas, Koki Nonaka, Akira Matsui (2020)
  Journal: IEEE Access; Publisher: Institute of Electrical and Electronics Engineers; Volume: 8; Pages: 164292-164303; DOI: [10.1109/ACCESS.2020.3023402](https://doi.org/10.1109/ACCESS.2020.3023402)
  这项基准研究实证评估了各种GAN训练方法和稳定技术，为TTUR在不同应用场景中的有效性及其与其他方法的相互作用提供了实践见解。

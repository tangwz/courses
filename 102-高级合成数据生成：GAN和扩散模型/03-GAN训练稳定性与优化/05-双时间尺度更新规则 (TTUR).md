# 双时间尺度更新规则 (TTUR)

来源：[原文](https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-3-gan-training-stability-optimization/gan-ttur-optimization)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练生成对抗网络 (GAN)需要精细的平衡。生成器 $G$ 和判别器 $D$ 处于一场最小-最大博弈中，其目标函数定义如下：


$$
\min_G \max_D V(D, G)
$$


如果一方明显强于另一方，训练过程可能会变得不稳定，导致诸如模式崩溃（生成器 $G$ 仅生成有限种类的输出）或发散（梯度爆炸或消失，学习停止）等问题。标准的优化方法通常使用相同的学习率同时更新 $G$ 和 $D$。然而，这种双玩家博弈的动态性通常意味着 $G$ 和 $D$ 以不同的有效速度学习。判别器执行更标准的监督分类任务（真实与虚假），其收敛速度可能快于生成器，后者处理学习整个数据分布的更难问题。

当判别器过快地变得过于准确时，它向生成器提供的梯度信息用处不大，实际上导致生成器的学习停滞。相反，如果判别器明显落后，生成器可能无法获得足够强的信号以有效提升。

### 使用不同学习率平衡更新

双时间尺度更新规则 (TTUR)，由 Heusel 等人在其 2017 年论文《通过双时间尺度更新规则训练的 GAN 收敛到局部纳什均衡》中提出，提供了一个直接而有效的解决方法：为生成器和判别器优化器使用不同的学习率。

这个主要思想源于双玩家博弈的随机逼近理论。理论分析表明，为生成器设置独立的学习率 $\eta_G$ 和为判别器设置 $\eta_D$，可以在学习率选择得当的情况下，促使收敛到一个均衡点。具体来说，TTUR 通常涉及为判别器设置更高的学习率，而为生成器设置较低的学习率 ($\eta_D > \eta_G$)。这使得判别器能够更好地估计真实分布和生成分布之间的差异，从而向生成器提供更多有用的梯度信息，而较慢的生成器更新则能防止它因采取过大的步长而破坏学习过程的稳定性。

> 图示：独立的优化器使用不同的学习率 ($\eta_G$ 和 $\eta_D$)，根据各自的损失更新生成器和判别器参数 (parameter)。通常，$\eta_D$ 设置得高于 $\eta_G$。

### 实际操作中的实现

在 PyTorch 或 TensorFlow 等现代深度学习 (deep learning)框架中实现 TTUR 通常很简单。您不是为两个网络使用单个优化器，而是实例化两个独立的优化器，一个用于生成器的参数 (parameter)，一个用于判别器的参数，每个优化器都配置有自己的学习率。

这里是一个简化的 PyTorch 风格伪代码片段：

```python
# 假设生成器和判别器模型已定义
generator = Generator(...)
discriminator = Discriminator(...)

# 定义独立的学习率
lr_g = 0.0001
lr_d = 0.0004 # TTUR: 判别器通常更高

# 定义独立的优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

# 在训练循环内：
# 1. 训练判别器
optimizer_D.zero_grad()
# ... 计算判别器损失 (loss_d) ...
loss_d.backward()
optimizer_D.step() # 使用 lr_d 更新 D

# 2. 训练生成器
optimizer_G.zero_grad()
# ... 计算生成器损失 (loss_g) ...
loss_g.backward()
optimizer_G.step() # 使用 lr_g 更新 G
```

具体的学习率 $\eta_G$ 和 $\eta_D$ 是超参数 (hyperparameter)，需要根据您的特定数据集和模型架构进行调整。尽管原始论文提出了具体的值（例如 $\eta_G = 0.0001, \eta_D = 0.0004$），但这仅作为起始点。网格搜索或其他超参数优化技术可能需要用来寻找最佳值。

### 优点与考量

使用 TTUR 带来多项优势：

1. **提高稳定性：** 通过让判别器提供更一致的反馈，同时避免变得过于强大，TTUR 有助于防止生成器的振荡和梯度消失。
2. **加快收敛：** 尽管单个步骤可能看起来相似，但与使用单一学习率相比，达到稳定的均衡点通常在实际运行时间或生成高质量样本所需的迭代次数方面更快。
3. **更好的样本质量：** 稳定的训练通常与更高的保真度和更多样化的生成样本相关联。

然而，请记住：

- **超参数 (parameter) (hyperparameter)调整：** TTUR 引入了两个学习率（以及可能的其他优化器参数，例如 Adam 的 beta 值），需要仔细调整。最佳比率 $\eta_D / \eta_G$ 可能有所不同。
- **并非万灵药：** TTUR 是一种有价值的技术，但不能保证收敛或解决所有 GAN 训练问题。它与本章讨论的其他增强稳定性的方法结合使用时最为有效，例如改进的损失函数 (loss function)（WGAN-GP、LSGAN）和正则化 (regularization)技术（谱范数归一化 (normalization)）。

TTUR 在最先进的 GAN 训练方案中被广泛采用，正是因为它直接处理了生成器-判别器动态中固有的异步学习速度，显著地促成了更稳定和有效的训练。

## 参考资料

- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Günter Klambauer, Andreas Maier, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Advances in Neural Information Processing Systems; Volume: 31; Pages: 6626-6637
  提出了双时间尺度更新规则（TTUR），通过为生成器和判别器设置不同的学习率来稳定GAN训练，并提供了支持其收敛特性的理论分析。
- [Generative Adversarial Networks](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS Foundation; Volume: 27; Pages: 2672-2680
  介绍了生成对抗网络（GANs）及其最小-最大博弈目标的基础性论文，为后续的GAN研究和优化挑战奠定了框架。
- [Improved Training of Wasserstein GANs](https://proceedings.neurips.cc/paper_files/paper/2017/file/892c91e0a653ba196a570df52c0e816d-Paper.pdf) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron C. Courville (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; Pages: 2226–2234
  通过引入带梯度惩罚的 Wasserstein GAN (WGAN-GP) 在GAN训练稳定性方面取得了显著进展，通过修改损失函数和正则化解决了常见的GAN训练失败问题。

---

[上一节](04-GAN%20%E7%9A%84%E6%AD%A3%E5%88%99%E5%8C%96%E6%96%B9%E6%B3%95.md) · [下一节](06-GAN%E7%9A%84%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E6%95%B4%E7%AD%96%E7%95%A5.md)

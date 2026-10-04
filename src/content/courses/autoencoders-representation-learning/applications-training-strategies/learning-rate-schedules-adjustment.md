---
course: "autoencoders-representation-learning"
chapter: "applications-training-strategies"
lesson: "learning-rate-schedules-adjustment"
sourceId: 2932
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-7-applications-training-strategies/learning-rate-schedules-adjustment"
title: "学习率计划与调整"
description: "实施有效的学习率计划策略，以实现稳定高效的训练。"
order: 7
plots: ["plots/2932-0.json"]
sourceHash: "fd6c85af0f2143faaf1f3144da3220ab94db72c23dab2e00d495f4664a8704d0"
sourceCorrections: []
---

优化自编码器的训练过程，通常不只涉及选择一个好的初始学习率。随着训练的进行，梯度下降 (gradient descent)的理想步长通常会发生变化。最初，较大的步长可能加快收敛到损失函数 (loss function)中有前景的区域，但随后，需要较小的步长来细致地处理更精微之处，并稳定在损失函数 $L$ 的良好最小值处。这时，学习率计划便成为非常实用的方法。

在整个训练过程中使用恒定的学习率会带来挑战。如果学习率过高，优化过程可能会不稳定地摆动或越过最小值，从而阻碍收敛。如果学习率过低，训练可能会变得非常缓慢，并且优化器可能陷入次优的局部最小值或鞍点。学习率计划会根据训练周期或迭代次数动态调整学习率，旨在平衡在损失曲面上的全局搜索和局部优化。

### 常见的学习率计划策略

训练期间存在多种调整学习率的策略。选择哪种策略通常取决于具体的自编码器架构、数据集、所使用的优化器（如Adam或SGD）以及经验结果。

1. **分步衰减（Step Decay）：** 这是最简单的计划之一。学习率在固定数量的训练周期内保持不变，然后按特定因子减小。这个过程可能会重复多次。例如，你可以从 $0.001$ 的学习率开始，运行30个训练周期，然后将其减小到 $0.0001$ 再运行30个周期，接着再减小到 $0.00001$。虽然易于实现，但这种突然的下降有时会暂时性地导致训练不稳定。
2. **指数衰减（Exponential Decay）：** 这种计划使学习率随时间平稳下降。在训练周期 $t$ 时的学习率 $lr_t$ 通常按以下方式计算：

   
   $$
   lr_t = lr_0 \times \gamma^{t / d}
   $$
   

   这里，$lr_0$ 是初始学习率，$\gamma$ 是衰减率（一个小于1的值，例如0.95），$t$ 是当前训练周期数，$d$ 是衰减步长（决定衰减应用的频率）。这种逐渐的减少有助于稳步地调整模型参数 (parameter)。
3. **余弦退火（Cosine Annealing）：** 这种常用且通常有效的计划使学习率遵循余弦曲线下降。它从初始最大学习率 $lr_{max}$ 开始，在指定的 $T_{max}$ 个训练周期内平稳地退火到最小学习率 $lr_{min}$。在训练周期 $t$ 时（其中 $T_{cur}$ 是自上次重启以来的训练周期数，通常就是 $t \pmod{T_{max}}$）的学习率由以下公式给出：

   
   $$
   lr_t = lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi))
   $$
   

   这种计划很平滑，并且在处理复杂的损失曲面时表现良好。诸如“带重启的余弦退火”等变体，会周期性地将学习率重置为 $lr_{max}$ 并重复退火过程，这可能有助于优化器摆脱不佳的最小值。
4. **学习率预热（Learning Rate Warmup）：** 尤其在使用较大的初始学习率或复杂的架构（如基于Transformer的自编码器）时，立即使用目标学习率可能会导致训练早期不稳定或发散。预热阶段通过从非常小的学习率开始，并在一定数量的初始训练周期或步数内逐渐将其提高到所需的初始学习率 $lr_0$ 来解决这个问题。这种增加可以是线性的，也可以遵循类似余弦段的曲线。预热为模型提供了稳定时间，然后再应用较大的梯度更新。

### 学习率计划的可视化

不同的计划会产生随时间变化的不同学习率轨迹。通过可视化了解这些轨迹有助于选择合适的策略。



![学习率计划对比](plots/2932-0.json)



> 不同学习率计划在100个训练周期内的对比，初始学习率为0.001（Y轴为对数刻度）。分步衰减显示出突然的下降，而指数衰减和余弦退火则提供更平滑的降低。

### 自编码器的实用考量

- **与优化器的互动：** 尽管像Adam这样的自适应优化器会根据每个参数 (parameter)调整学习率，但应用全局学习率计划仍然有益。该计划指导更新的整体幅度，与优化器的自适应特性互补。
- **调整：** 选择最佳计划及其参数（初始学习率、衰减因子、周期长度）通常是一个经验过程。可以从余弦退火或分步/指数衰减等常用计划开始。密切关注训练和验证损失曲线。如果损失过快地趋于平稳，可以考虑更慢的衰减或更小的衰减步长。如果损失不稳定，可能需要更低的初始学习率、预热阶段或更快的衰减。
- **任务依赖性：** 最佳计划可能因自编码器的应用而异。对于使用VAE的生成任务，仔细平衡重建损失和KL散度项可能需要精细调整的计划，或许需要将KL项的权重 (weight)与学习率一同退火。对于异常检测，确保模型能学到正常数据的良好表征可能更倾向于允许在小学习率下进行长时间微调 (fine-tuning)的计划。
- **框架支持：** 现代深度学习 (deep learning)框架（TensorFlow/Keras、PyTorch）为大多数常用学习率调度器提供了内置实现，使其易于集成到你的训练循环中。

通过计划有效管理学习率是成功训练自编码器的一种重要方法。它有助于实现更快的收敛，避免陷入损失空间中不佳的区域，并最终使模型学到更好的表征，在诸如降维、生成或异常检测等后续任务上表现良好。需要进行实验和仔细监控，才能为你的特定模型和数据找到最合适的计划策略。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教科书，全面涵盖了深度学习原理，包括优化算法、梯度下降以及动态学习率调整的基本原理。
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — Ilya Loshchilov, Frank Hutter (2017)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983)
  介绍了带有热重启的余弦退火概念，这是一种学习率调度技术，常能在深度学习训练中取得更优异的表现。
- [A Disciplined Approach to Neural Network Hyper-parameters: Part 1 -- Learning Rate, Batch Size, Momentum, and Weight Decay](https://doi.org/10.48550/arXiv.1803.09820) — Leslie N. Smith (2018)
  Journal: arXiv preprint arXiv:1803.09820; Publisher: US Naval Research Laboratory; DOI: [10.48550/arXiv.1803.09820](https://doi.org/10.48550/arXiv.1803.09820)
  为神经网络超参数设置提供了实用指南，其中包括学习率调度的系统策略，例如结合预热和退火的One-Cycle策略。
- [How to adjust learning rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — PyTorch Documentation (2023)
  Publisher: PyTorch
  PyTorch官方文档，提供了在PyTorch深度学习框架中实现各种学习率调度器的实用指南和示例。

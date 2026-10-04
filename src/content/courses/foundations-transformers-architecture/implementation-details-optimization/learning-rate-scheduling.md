---
course: "foundations-transformers-architecture"
chapter: "implementation-details-optimization"
lesson: "learning-rate-scheduling"
sourceId: 2062
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/learning-rate-scheduling"
title: "学习率调度 (热身, 衰减)"
description: "实现学习率热身和衰减策略对Transformer收敛很重要。"
order: 4
plots: ["plots/2062-0.json"]
sourceHash: "5122f2a832c004fd68a1d2327c2c2e4d5c97cb7e4142f72152410ed365b1a2a6"
sourceCorrections: []
---

选择合适的优化器，如Adam或AdamW，只是成功训练大型Transformer模型的一部分。学习率作为控制优化过程中步长的基本超参数 (parameter) (hyperparameter)，通常需要在整个训练过程中仔细调整。简单使用固定学习率通常是次优的，并可能导致不稳定或收敛缓慢，尤其考虑到涉及的复杂损失曲面和深度结构。因此，学习率调度变得必要。

学习率调度定义了一种在训练期间动态改变学习率的策略。对于Transformer模型，一种常见且高效的方法是结合“热身”阶段和随后的“衰减”阶段。

### 热身阶段: 稳定早期训练

在训练的初始阶段，模型的参数 (parameter)是随机初始化的，梯度可能很大且不稳定。从一开始就使用高学习率可能会导致显著的更新，从而破坏训练过程的稳定性，可能使优化器偏离参数空间的良好区域。这对于Transformer模型来说尤其如此，因为层归一化 (normalization)、残差连接和注意力机制 (attention mechanism)之间的互相影响对早期大参数变化很敏感。

热身阶段通过从非常小的学习率（通常为零）开始，并在预设的初始训练步数（即 `warmup_steps`）内逐渐增加学习率来解决这个问题。线性增加是常见的：


$$
lr_{step} = lr_{peak} \times \frac{step}{warmup\_steps} \quad \text{当 } step < warmup\_steps
$$


在此，$lr_{peak}$ 是调度在热身阶段后将达到的最大学习率。这种渐进式增加允许模型在应用更大参数更新之前稳定下来，防止早期发散并促进更平滑的收敛。

### 衰减阶段: 优化收敛

一旦热身阶段完成且学习率达到峰值 ($lr_{peak}$)，逐步降低它是有益的。这个衰减阶段有以下几个作用：

1. **精细调整:** 随着训练的进行，当模型接近一个良好的最小值时，较小的学习率允许更精细的调整，防止优化器越过最小值。
2. **避免震荡:** 训练后期的高学习率可能导致在最小值附近震荡。降低学习率有助于抑制这些震荡。
3. **提高泛化能力:** 一些研究表明，降低学习率有助于找到泛化能力更好的最小值，使其在未见过的数据上表现更佳。

热身阶段后，几种衰减策略常被使用：

- **逆平方根衰减:** 这种策略在原始论文“Attention Is All You Need”中被采用。热身之后，学习率与步数的逆平方根成比例地减小：

  
  $$
  lr_{step} = lr_{peak} \times \sqrt{\frac{warmup\_steps}{step}} \quad \text{当 } step \ge warmup\_steps
  $$
  

  或者，更常见的是直接参照论文中的公式实现（使用模型维度 $d_{model}$ 和一个缩放因子）：

  
  $$
  lr_{step} = d_{model}^{-0.5} \times \min(step^{-0.5}, step \times warmup\_steps^{-1.5})
  $$
  

  此公式将线性热身和逆平方根衰减结合到一个表达式中。请注意，$lr_{peak}$ 由 $d_{model}^{-0.5}$ 和 $warmup\_steps$ 隐式定义。
- **余弦衰减 (余弦退火):** 学习率遵循余弦曲线，在剩余的训练步数中从 $lr_{peak}$ 降至最小值（通常为零）。这提供平滑、渐进的下降。

  
  $$
  lr_{step} = lr_{min} + 0.5 \times (lr_{peak} - lr_{min}) \times \left(1 + \cos\left(\frac{(step - warmup\_steps) \times \pi}{total\_steps - warmup\_steps}\right)\right)
  $$
  

  其中 $lr_{min}$ 是目标最小学习率（例如0），$total\_steps$ 是计划的总训练步数。
- **线性衰减:** 学习率在剩余步数中从 $lr_{peak}$ 线性下降到最小值（通常为零）。

  
  $$
  lr_{step} = lr_{min} + (lr_{peak} - lr_{min}) \times \frac{total\_steps - step}{total\_steps - warmup\_steps}
  $$
  
- **指数衰减:** 学习率以固定间隔或每一步乘以一个小于1的衰减因子。

衰减策略的选择会影响最终模型性能，经验评估通常是必要的。余弦衰减和逆平方根衰减是训练Transformer模型时尤其受欢迎的选择。

### 组合调度可视化

热身和衰减的组合随时间形成一个典型的学习率曲线。以下图表示例一个典型的调度，包含线性热身和随后的逆平方根衰减。



![学习率调度示例](plots/2062-0.json)



> 学习率调度，包含10步热身阶段，达到0.0005的峰值，随后是逆平方根衰减。

### 实现方面的考量

大多数深度学习 (deep learning)框架都提供内置支持，使学习率调度器可以轻松与优化器集成。

- **PyTorch:** `torch.optim.lr_scheduler` 模块提供多种调度器，例如 `LambdaLR`（用于逆平方根等自定义函数）、`CosineAnnealingLR`、`LinearLR` 和 `SequentialLR`（用于连接热身和衰减）。通常在每次优化器 `step()` 调用后调用调度器的 `step()` 方法。
- **TensorFlow/Keras:** `tf.keras.optimizers.schedules` 模块提供诸如 `PolynomialDecay`（可以实现线性衰减）、`CosineDecay` 的类，并允许通过继承 `LearningRateSchedule` 创建自定义调度。这些调度在初始化时直接传递给优化器。

调度的具体参数 (parameter)（$lr_{peak}$、$warmup\_steps$、衰减类型、$total\_steps$）是重要的超参数 (hyperparameter)，通常需要根据所使用的特定模型大小、数据集和批次大小进行调整。选择合适的调度并调整其参数是获得最佳性能和Transformer模型稳定训练的必要步骤。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008
  介绍Transformer架构及其涉及预热和逆平方根衰减的特定学习率调度。
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/pdf/1608.03983.pdf) — Ilya Loshchilov, Frank Hutter (2017)
  Journal: International Conference on Learning Representations
  提出余弦退火作为一种学习率调度策略，因其平滑的衰减特性而被广泛使用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: 281-326
  涵盖基础优化算法和学习率策略，为调度技术提供理论背景。
- [tf.keras.optimizers.schedules](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  在TensorFlow和Keras框架中实现学习率调度的官方文档。

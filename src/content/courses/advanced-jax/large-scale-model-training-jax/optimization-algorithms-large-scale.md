---
course: "advanced-jax"
chapter: "large-scale-model-training-jax"
lesson: "optimization-algorithms-large-scale"
sourceId: 2946
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-6-large-scale-model-training-jax/optimization-algorithms-large-scale"
title: "大规模优化算法"
description: "简要概述适用于大规模分布式训练的优化器（例如，AdamW，LAMB）。"
order: 9
plots: []
sourceHash: "0814e3c834e2f01a2ae989b2f21a8273fcf6699e8b25da8a01c221ba36af1f6e"
sourceCorrections: []
---

在训练大规模模型时，尤其是在分布式环境中，选择合适的优化算法是一个重要因素。尽管带有动量或Adam的标准优化器（如随机梯度下降 (gradient descent)SGD）构成了基础，但扩展训练通常需要设计用于有效处理超大批量大小或特定正则化 (regularization)需求的优化器。数据并行（例如使用`pmap`）和梯度累积等技术直接导致更大的有效批量大小，从而影响优化器行为。

### 大型模型常用优化器

#### AdamW

Adam因其自适应学习率而仍是广泛使用的优化器。然而，Adam中L2正则化 (regularization)的标准实现通常是次优的。AdamW通过将权重 (weight)衰减计算与自适应学习率相关的梯度更新解耦来修改Adam。

在带有L2正则化的标准Adam中，衰减项与自适应矩（$m_t$ 和 $v_t$）发生作用，可能导致历史梯度大的权重衰减程度小于梯度小的权重。AdamW在Adam步骤*之后*直接将权重衰减应用于权重，其行为更像SGD中使用的权重衰减。

参数 (parameter) $\theta$ 的更新步骤如下：

1. 计算梯度 $g_t = \nabla L(\theta_{t-1})$。
2. 更新偏置 (bias)的一阶和二阶矩估计 $m_t, v_t$。
3. 计算偏置校正后的估计 $\hat{m}_t, \hat{v}_t$。
4. 使用Adam逻辑更新参数：$\theta'_{t} = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$。
5. 应用权重衰减：$\theta_t = \theta'_{t} - \eta \lambda \theta_{t-1}$。（注意：衰减与学习率 $\eta$ 和衰减因子 $\lambda$ 成比例。）

与带有L2正则化的原始Adam实现相比，这种解耦通常能为像Transformer这样的大型模型带来更好的泛化性能和更稳定的训练。它已成为许多大型语言模型训练方法的标准选择。

#### LAMB (层级自适应矩优化器，用于批次训练)

当使用超大批量大小时（通常通过使用`pmap`在多个加速器上进行分布式训练实现），Adam等标准优化器有时会变得不稳定或需要仔细调整学习率（特别是进行充分的预热）。LAMB的开发专门旨在实现使用极大批量大小（数万或更多）的稳定训练。

LAMB背后的主要思想是对参数更新应用层级归一化 (normalization)。它计算Adam更新步骤的方式与AdamW相似，但随后根据权重范数与该层Adam更新范数之比对每个层的更新进行归一化。

对于每一层 $l$：

1. 计算Adam(W)更新步长 $\Delta \theta_l$。
2. 计算比率 $r_l = \frac{\|\theta_l\|}{\|\Delta \theta_l\|}$。
3. 缩放该层的更新：$\theta_{l, t} = \theta_{l, t-1} - \eta \cdot r_l \cdot \Delta \theta_l$。（这是一个简化观点；实际实现涉及信任比率计算。）

这种层级信任比率缩放有助于防止更新对于某些层变得过大或过小，这可能发生在批量很大、梯度方差减小的情况下，可能导致Adam采取过于激进的步骤。LAMB已被证明在训练BERT等模型时有效，使用非常大的批量大小，比以前显著更快。

### 其他重要考量

其他因素对于成功的大规模优化也很重要：

- **学习率调度：** 几乎所有大型模型训练都高度依赖于精心设计的学习率调度。常见策略包括：

  - **预热：** 从较小的学习率开始，并在前几千步中线性增加，在早期稳定训练，尤其是在权重 (weight)随机初始化时。
  - **衰减：** 在预热阶段之后，学习率通常在训练的其余部分进行衰减。常见的衰减调度包括线性衰减、余弦退火或逆平方根衰减。选择通常取决于具体的模型架构和数据集。
- **优化器状态管理：** Adam(W)和LAMB等自适应优化器为每个参数 (parameter)维护状态（例如，动量和方差估计）。对于大型模型，这种状态会消耗大量内存，有时甚至与模型参数本身相当。在使用`pmap`的分布式设置中，这种优化器状态也必须与参数和梯度一同在设备间分布（分片）。像`optax`（一个流行的JAX梯度处理和优化库）这样的库通常在`pmap`装饰的函数中正确使用时会自动处理这一点。请确保您的训练设置正确地在设备间划分优化器状态。

### JAX中的优化实践

像`optax`这样的库提供了许多常用和高级优化器的实现，与JAX的函数式编程模型和变换平滑集成。使用`optax`通常包括：

1. **初始化：** 创建优化器状态，通常基于模型参数 (parameter)。
2. **更新步骤：** 一个函数，接收梯度和当前优化器状态，以生成参数更新和新的优化器状态。

这是一个使用`optax`的更新步骤示例：

```python
import jax
import jax.numpy as jnp
import optax

# 假设 'params' 是模型参数（例如，PyTree）
# 假设 'grads' 是由 jax.grad 计算的梯度
# 假设 'opt_state' 是当前的优化器状态

# 定义优化器（例如，带有余弦衰减调度的AdamW）
learning_rate_schedule = optax.warmup_cosine_decay_schedule(...)
optimizer = optax.adamw(learning_rate=learning_rate_schedule, weight_decay=0.01)

# 初始化优化器状态（通常在训练循环外部完成一次）
# opt_state = optimizer.init(params)

# 在训练步骤内部（可能经过 pmap 处理）
@jax.jit
def update_step(params, grads, opt_state):
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state

# 应用更新
# params, opt_state = update_step(params, grads, opt_state)
```

当 `update_step` 函数在 `pmap` 中使用时，`optax` 有助于确保梯度计算、更新和状态管理在设备间得到正确处理，包括在计算优化器更新之前必要的梯度聚合（例如，使用 `lax.pmean`）。选择和调整优化器，以及其相关的学习率调度和权重 (weight)衰减，是一个迭代过程。虽然像AdamW这样的优化器提供了一个很好的起点，LAMB为超大批量提供了优势，但通常需要通过实验来找到特定大型训练任务的最佳组合，考虑模型架构、数据集特点和可用的计算资源。

## 参考资料

- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  介绍了 AdamW，证明了将权重衰减与自适应梯度更新解耦可以提高自适应优化器的泛化能力。
- [Large Batch Optimization for Deep Learning](https://arxiv.org/abs/1904.00962) — Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1904.00962](https://doi.org/10.48550/arXiv.1904.00962)
  提出了 LAMB，一种旨在以极大批次大小稳定高效训练深度学习模型的优化器。
- [Optax: A JAX library for gradient processing and optimization](https://optax.readthedocs.io/en/latest/) — DeepMind (2024)
  Optax 的官方文档，提供了在 JAX 中使用各种优化算法和调度方案的实用指南和 API 参考。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了 Transformer 架构及其相关的学习率调度策略（热身期后接逆平方根衰减），该策略被大型语言模型广泛采用。

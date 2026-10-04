---
course: "advanced-pytorch"
chapter: "advanced-network-architectures"
lesson: "meta-learning"
sourceId: 2700
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-2-advanced-network-architectures/meta-learning"
title: "元学习算法"
description: "实现元学习技术，如用于少样本学习的模型无关元学习（MAML）。"
order: 6
plots: []
sourceHash: "37bfa0f6c26c021ef764719682a62448c615ee03164449e755f95b5a2d5e6648"
sourceCorrections: []
---

“标准深度学习 (deep learning)模型在大型标记 (token)数据集上训练时通常表现优异。然而，许多情况下需要用少量示例快速适应新任务，这种场景称为少样本学习 (few-shot learning)。元学习，即“学习如何学习”，提供了一个训练模型的体系，使其能够有效泛化到数据有限的新任务。元学习算法不是学习如何很好地执行一个特定任务，而是学习一个过程或一个初始化方法，从而能够快速适应新的、相关任务。”

主要说明如何在PyTorch中实现元学习算法，特别是介绍一种流行且多功能的方案：模型无关元学习（MAML）。

### 元学习问题设置

在典型的监督学习 (supervised learning)设置中，我们有一个数据集 $D = \{(x_i, y_i)\}$，目标是学习一个由 $\theta$ 参数 (parameter)化的函数 $f_\theta$，使其在数据集上最小化损失 $\mathcal{L}(f_\theta(x_i), y_i)$。

元学习重新定义了这个问题。我们假设存在任务分布 $p(\mathcal{T})$。在元训练期间，我们从 $p(\mathcal{T})$ 中采样批次任务 $\mathcal{T}_i$。对于每个任务 $\mathcal{T}_i$，我们通常有一个小的*支持集* $D_i^{supp}$ 用于任务内部学习，以及一个*查询集* $D_i^{query}$ 用于评估该任务的学习效果。目标是学习模型参数 $\theta$（通常称为元参数），使得模型能够利用*新的*、以前未见的任务 $\mathcal{T}_{new}$ 的支持集快速适应，从而在其查询集 $D_{new}^{query}$ 上获得良好性能。

### 模型无关元学习（MAML）

MAML 由 Finn 等人于 2017 年提出，其目标是找到对任务变化敏感的元参数 (parameter) $\theta$，仅用少量梯度步长就能在小支持集上进行有效微调 (fine-tuning)。它之所以“模型无关”，是因为它不对模型架构 $f_\theta$ 做强假设；它可以应用于 CNN 或 RNN 等多种模型。

其核心思想涉及一个两层优化过程：

1. **内循环（任务特定适应）：** 对于每个采样的任务 $\mathcal{T}_i$，从当前元参数 $\theta$ 开始。仅使用任务的支持集 $D_i^{supp}$ 执行一次或几次梯度下降 (gradient descent)步骤，以获得任务特定参数 $\theta'_i$。对于学习率为 $\alpha$ 的单个梯度步长：

   
   $$
   \theta'_i = \theta - \alpha \nabla_{\theta} \mathcal{L}_{\mathcal{T}_i}(f_{\theta}(D_i^{supp}))
   $$
   

   这里，$\mathcal{L}_{\mathcal{T}_i}$ 是任务 $\mathcal{T}_i$ 的损失函数 (loss function)，$f_{\theta}(D_i^{supp})$ 表示模型使用参数 $\theta$ 对支持集进行预测的结果。请注意，此梯度是相对于*初始*参数 $\theta$ 计算的。
2. **外循环（元优化）：** 评估*已适应*参数 $\theta'_i$ 在任务查询集 $D_i^{query}$ 上的表现。元目标是在适应*之后*最小化跨任务的损失。元参数 $\theta$ 根据这些适应后查询集损失的总和（或平均值）进行更新，使用元学习率 $\beta$：

   
   $$
   \theta \leftarrow \theta - \beta \nabla_{\theta} \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i}(D_i^{query}))
   $$
   

关键在于，外循环中的梯度 $\nabla_{\theta} \sum \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i}(...))$ 涉及到对内循环更新步骤的求导。这意味着我们需要计算相对于 $\theta$ 的梯度，并考虑 $\theta'_i$ 是如何从 $\theta$ 推导出来的。这导致梯度计算涉及二阶导数（梯度的梯度）。

> 流程图说明了 MAML 优化过程。内循环使用支持集损失，将参数 $\theta$ 适应为任务特定的 $\theta'$。外循环根据使用已适应参数 $\theta'$ 的查询集损失计算元梯度，该元梯度随后用于更新原始元参数 $\theta$。

### 在 PyTorch 中实现 MAML

实现外循环的梯度计算需要谨慎。标准的 PyTorch `backward()` 调用会丢弃梯度中梯度计算所需的中间图信息。

有两种主要的方法来处理这个问题：

1. **使用 `torch.autograd.grad`：** 使用 `torch.autograd.grad` 并设置 `create_graph=True` 参数 (parameter)来手动计算内部梯度。这会告诉 PyTorch 为梯度计算本身构建一个计算图，从而允许稍后进行反向传播 (backpropagation)。

# 示意图：内循环梯度计算

````
inner_loss = calculate_loss(model(support_x), support_y)
grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True)

# 计算已适应参数（函数式方法在这里通常更简单）
adapted_params = [p - alpha * g for p, g in zip(model.parameters(), grads)]

# 使用 adapted_params 计算外部损失（需要函数式模型调用）
# ... outer_loss = calculate_loss(functional_model(adapted_params, query_x), query_y) ...

# 外循环梯度计算稍后会汇总跨任务的 outer_loss
# 并对总和调用 backward()。
```
````

2. **使用高阶梯度库：** 像 `higher` 这样的库能显著简化此过程。`higher` 提供了上下文 (context)管理器，让你可以创建模型的临时可微分副本。你在此临时副本上执行内循环更新，库会自动处理外循环梯度所需的计算跟踪。

````
```python
````

# 使用 'higher' 的示意图

````
import higher

meta_optimizer.zero_grad()
total_outer_loss = 0.0

for task_i in batch_of_tasks:
    support_x, support_y, query_x, query_y = get_task_data(task_i)

    with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=True) as (fmodel, diffopt):
        # 内循环更新
        for _ in range(num_inner_steps):
            inner_loss = calculate_loss(fmodel(support_x), support_y)
            diffopt.step(inner_loss) # 更新 fmodel 的参数

        # 外循环评估
        outer_loss = calculate_loss(fmodel(query_x), query_y)
        total_outer_loss += outer_loss

# 反向传播元目标
total_outer_loss.backward()
meta_optimizer.step()
```
````

`higher` 方法因其更简洁的实现而常受青睐，它抽象了 `create_graph=True` 的手动处理和函数式参数更新。

### MAML 变体

- **一阶 MAML (FOMAML)：** 计算二阶导数在计算上可能开销很大。FOMAML 通过忽略二阶项来近似 MAML 更新。本质上，它计算内部梯度 $\nabla_{\theta} \mathcal{L}_{\mathcal{T}_i}(f_{\theta}(D_i^{supp}))$，然后使用已适应参数 (parameter)计算外部梯度 $\nabla_{\theta'} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i}(D_i^{query}))$，但在外部反向传播 (backpropagation)期间，它将内部梯度步骤视为与初始 $\theta$ 无关。这更快，但性能可能略逊于完整的 MAML。在 PyTorch 的手动实现中，这对应于调用 `torch.autograd.grad` *不带* `create_graph=True`。
- **Reptile：** 另一种一阶元学习算法（Nichol 等人，2018），它简化了更新过程。它在内循环中执行多个梯度步骤，然后通过简单地将元参数 $\theta$ 稍微朝已适应参数 $\theta'_i$ 的方向移动来更新它们：$\theta \leftarrow \theta + \beta (\theta'_i - \theta)$。这完全避免了显式的二阶导数计算。

### 应用与考虑

元学习，特别是 MAML 及其变体，已在以下方面获得应用：

- **少样本图像分类：** 学习能够从极少量示例中识别新物体类别的分类器。
- **强化学习 (reinforcement learning)：** 训练能够快速适应新环境或动态变化的智能体。
- **域适应：** 将在一个数据分布（源域）上训练的模型适应到相关但不同分布（目标域）上，并使用有限的目标数据实现良好性能。

**挑战：**

- **计算成本：** MAML 的二阶梯度计算和存储开销可能很大，特别是对于大型模型。FOMAML 和 Reptile 提供了替代方案。
- **训练稳定性：** 元学习优化场景可能很复杂，有时需要仔细调整超参数 (parameter) (hyperparameter)（例如，内外部学习率、内循环步数）。
- **任务定义：** 元学习的有效性很大程度上取决于任务 $p(\mathcal{T})$ 的定义和分布。任务需要共享某种元学习器可以加以运用的潜在结构。

元学习代表了一种转变，从训练单个任务的模型转向训练具备高效学习能力的模型。像 MAML 这样的算法为实现这一目标提供了一个具体机制，通过优化可适应的初始化，使模型能够在数据稀缺的情况下快速适应。实现这些需要仔细处理梯度计算，这通常通过专门的库或手动应用 PyTorch 的自动求导功能来简化。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1703.03400](https://doi.org/10.48550/arXiv.1703.03400)
  介绍了模型无关元学习（MAML）算法。
- [higher: A library for high-level gradient-based meta-learning in PyTorch](https://arxiv.org/abs/1910.01727) — Grefenstette, Edward and Amos, Brandon and Yarats, Denis and Htut, Phu Mon and Molchanov, Artem and Meier, Franziska and Kiela, Douwe and Cho, Kyunghyun and Chintala, Soumith (2019)
  Journal: arXiv preprint arXiv:1910.01727; Publisher: arXiv; DOI: [10.48550/arXiv.1910.01727](https://doi.org/10.48550/arXiv.1910.01727)
  PyTorch库，用于实现如MAML等元学习算法。
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Core Team (2017)
  Publisher: PyTorch
  解释了PyTorch中的自动微分系统，包括高阶梯度。

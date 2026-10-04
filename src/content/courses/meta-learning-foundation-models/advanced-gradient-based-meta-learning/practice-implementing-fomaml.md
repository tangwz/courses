---
course: "meta-learning-foundation-models"
chapter: "advanced-gradient-based-meta-learning"
lesson: "practice-implementing-fomaml"
sourceId: 4321
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-2-advanced-gradient-based-meta-learning/practice-implementing-fomaml"
title: "动手实践：实现 FOMAML 用于模型适应"
description: "FOMAML 代码实现，用于适应较小的预训练模型。"
order: 6
plots: []
sourceHash: "0484488e835d23afcb2fb278b721301d2c75fb18a257cb21c75d6e2413411eb5"
sourceCorrections: []
---

一阶模型无关元学习 (FOMAML) 的实用实现涉及调整一个预训练 (pre-training)的卷积神经网络 (neural network) (CNN) 以处理少样本图像分类任务。目标是学习一组初始模型参数 (parameter) $\theta$，使其能仅用少量样本便快速适应新的、未曾见过的分类任务。FOMAML 通过简化 MAML 更新来达成此目的，它忽略二阶导数以提高计算效率，这使得它在处理更大模型时尤为适用，尽管为便于说明使用了较小的模型。

我们假定一个标准的少样本学习 (few-shot learning)配置，常被称为 N 类 K 样本分类。在每次元训练迭代中，我们抽取一批不同的任务。对于每个任务 $T_i$，我们获得一个小的*支持集* $D_{S}^{(i)} = \{(x_j, y_j)\}_{j=1}^{N \times K}$（N 个类别中每个类别有 K 个样本）和一个用于评估的*查询集* $D_{Q}^{(i)}$。

### FOMAML 元训练过程

其主要思路是在元训练期间模拟适应过程。

1. **任务抽样：** 抽取一批任务 $\{T_i\}$。
2. **内循环（适应模拟）：** 对于每个任务 $T_i$：
   a. 使用当前的元参数 (parameter) $\theta$ 初始化一个临时模型。
   b. 在支持集 $D_{S}^{(i)}$ 上执行一步或多步梯度下降 (gradient descent)，使用内部学习率 $\alpha$。设任务 $T_i$ 的损失函数 (loss function)为 $L_{T_i}$。单步更新规则为：
   
   $$
   \phi_i = \theta - \alpha \nabla_{\theta} L_{T_i}(\theta, D_{S}^{(i)})
   $$
   
   重要的是，对于 FOMAML，我们处理 $\phi_i$ 的方式，就好像在*外*部更新的梯度计算中，它与 $\theta$ 无关。在实现时，这通过使用基于已适应参数计算的梯度来达成。
3. **外循环（元优化）：** 使用已适应的参数 $\phi_i$ 计算每个任务在其查询集 $D_{Q}^{(i)}$ 上的损失。总体元损失是批次中所有任务的平均损失：
   
   $$
   L_{meta} = \sum_{T_i} L_{T_i}(\phi_i, D_{Q}^{(i)})
   $$
   
4. **元更新：** 使用元损失的梯度更新元参数 $\theta$。由于我们在内循环中执行了一阶近似，因此元梯度近似为：
   
   $$
   \nabla_{\theta} L_{meta} \approx \sum_{T_i} \nabla_{\phi_i} L_{T_i}(\phi_i, D_{Q}^{(i)})
   $$
   
   使用元学习率 $\beta$ 的更新规则为：
   
   $$
   \theta \leftarrow \theta - \beta \sum_{T_i} \nabla_{\phi_i} L_{T_i}(\phi_i, D_{Q}^{(i)})
   $$
   

### 实现草图 (PyTorch 风格)

让我们使用 PyTorch 概述主要组成部分。我们假设有一个 `model`（继承自 `torch.nn.Module`），一个 `loss_fn`（例如 `CrossEntropyLoss`），以及提供任务批次的数据加载器，每个任务都会生成支持集和查询集。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

# 假设 'model' 是我们的基础网络（例如，一个 CNN）
# 假设 'meta_optimizer' 是用于元参数 theta 的优化器（例如，Adam）
# 假设 'task_batch' 已加载，包含多个任务的 support_data, support_labels, query_data, query_labels

inner_lr = 0.01 # Alpha
num_inner_steps = 5 # 适应步骤的数量

# --- 元训练迭代 ---
meta_optimizer.zero_grad()
total_meta_loss = 0.0

for task_idx in range(len(task_batch['support_data'])): # 遍历批次中的任务
    support_x = task_batch['support_data'][task_idx]
    support_y = task_batch['support_labels'][task_idx]
    query_x = task_batch['query_data'][task_idx]
    query_y = task_batch['query_labels'][task_idx]

    # 为内循环适应创建一个临时模型
    # 使用 deepcopy 以避免过早修改原始元参数
    # 但要跟踪相对于原始权重的梯度，以便稍后的外部步骤使用。
    # 注意：对于纯 FOMAML，不需要跟踪高阶梯度，
    # 但库的处理方式可能不同。更简单的显式方法：

# 步骤 2a: 初始化临时模型
    # 实际上，我们计算相对于当前 theta 的梯度

    # 步骤 2b: 内循环适应
    adapted_params = list(model.parameters()) # 从当前 theta 开始

    for step in range(num_inner_steps):
        # 使用当前 adapted_params 在支持集上计算损失
        # 如果不使用像 higher 这样的库，需要手动进行函数式前向传播或类似技术。
        # 简化版本，假设模型可以接受参数覆盖：

        # 使用当前 adapted_params 计算损失（需要仔细实现）
        # 计算带特定参数的损失的示例占位符：
        # support_preds = functional_forward(model_definition, adapted_params, support_x)
        # inner_loss = loss_fn(support_preds, support_y)

        # 计算相对于 adapted_params 的梯度
        # grads = torch.autograd.grad(inner_loss, adapted_params) # 对于 FOMAML，Create_graph=False

        # 更新 adapted_params（手动 SGD 更新）
        # adapted_params = [p - inner_lr * g for p, g in zip(adapted_params, grads)]

    # --- 一种更实用的 PyTorch 方法（使用克隆模型）---
    fast_model = deepcopy(model) # 克隆模型用于任务特定的适应
    fast_model.train()

    # 对克隆模型使用标准的优化器进行内循环
    inner_optimizer = optim.SGD(fast_model.parameters(), lr=inner_lr)

    for step in range(num_inner_steps):
        inner_optimizer.zero_grad()
        support_preds = fast_model(support_x)
        inner_loss = loss_fn(support_preds, support_y)
        inner_loss.backward() # 在 fast_model 上计算梯度
        inner_optimizer.step() # 更新 fast_model 的参数

    # 步骤 3: 使用适应后的模型 (fast_model) 在查询集上评估
    fast_model.eval() # 确保 dropout/batchnorm 处于评估模式
    query_preds = fast_model(query_x)
    outer_loss = loss_fn(query_preds, query_y)

    # 累加元损失以进行外部更新
    total_meta_loss += outer_loss

# 步骤 4: 元更新
# 对批次中的任务损失取平均
average_meta_loss = total_meta_loss / len(task_batch['support_data'])

# 计算元损失相对于原始元参数 theta 的梯度
# 这是外循环更新的核心。因为 outer_loss 是使用
# 派生自*原始*模型参数（通过内循环）的参数计算的，
# 通过 average_meta_loss 的反向传播会更新原始模型。
# PyTorch 的 autograd 会跟踪这一点，即使是通过 deepcopy 和内部步骤，
# 但 FOMAML 的主要思想是我们*不需要*复杂的二阶项。
# 这里计算的梯度就是 FOMAML 梯度。
average_meta_loss.backward()

# 应用元更新
meta_optimizer.step()
```

> **注意：** 上述 PyTorch 代码片段阐明了原理。实现时通常需要细致处理模型状态、梯度流，并可能使用函数式编程方法或像 `higher` 这样的库来进行更清晰的梯度管理，特别是对于复杂架构或计算图复杂度增加的多个内步。`deepcopy` 方法简单，但对于大型模型而言可能占用大量内存。重点在于 `average_meta_loss.backward()` 计算了原始 `model` 参数 (parameter)进行 FOMAML 更新所需的梯度。

### FOMAML 更新流程的可视化

> FOMAML 中元批次内单个任务的流程。元参数 (parameter) $\theta$ 用于计算支持集上的初始损失。该损失的梯度用于更新 $\theta$，以获得任务特定的 $\phi_i$。外部损失是使用 $\phi_i$ 在查询集上计算的。该外部损失的梯度（相对于 $\phi_i$ 取得）用于更新原始元参数 $\theta$。指向元更新的虚线表示近似步骤。

### 实现时的考量

- **内部与外部学习率（$\alpha$ 对比 $\beta$）：** 这些是重要的超参数 (parameter) (hyperparameter)。$\alpha$ 控制任务内的适应速度，而 $\beta$ 控制元参数的学习率。通常，$\alpha$ 可能大于 $\beta$。调整这些参数需要实验。
- **内部步数：** 更多的内部步数可以实现更精细的适应，但会增加计算量，并且如果 $\alpha$ 过大或 K 过小，可能导致不稳定或对支持集过拟合 (overfitting)。一步或几步（例如 1-10 步）是常见的做法。
- **模型架构：** 尽管 FOMAML 与模型无关，但所选架构的容量和归纳偏置 (bias)会明显影响性能。适合任务一般范围的架构是优选的。
- **批归一化 (normalization)：** 在内循环中处理批归一化统计数据需要小心。常见做法包括为每个内步重置统计数据，使用转导式批归一化（计算支持集和查询集组合的统计数据，这略微偏离了纯粹的少样本设置），或者改用层归一化/组归一化。
- **计算成本：** 即使没有二阶导数，在多个任务上通过网络的内循环和外循环进行前向和反向传播 (backpropagation)仍然可能要求很高。对于更大的模型或数据集，高效批处理和潜在的分布式训练（稍后会介绍）变得重要。

本次实践练习阐明了实现 FOMAML 的主要机制。通过元学习得到一个适当的初始化点 $\theta$，模型能够迅速适应新任务，只需少量数据，这在处理基础模型时是一项宝贵的能力，因为对众多任务进行全面微调 (fine-tuning)通常不可行。请记住，将此想法应用到真实的大规模基础模型时，需要应对后续章节中讨论的可扩展性难题。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research (PMLR); Volume: 70; Pages: 1126-1135; DOI: [10.5598/v70/finn17a](https://doi.org/10.5598/v70/finn17a)
  介绍了MAML算法，它是FOMAML的基础，并讨论了其一阶近似实现。
- [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999) — Alex Nichol, Joshua Achiam, John Schulman (2018)
  Journal: arXiv preprint arXiv:1803.02999; Publisher: arXiv; DOI: [10.48550/arXiv.1803.02999](https://doi.org/10.48550/arXiv.1803.02999)
  介绍了Reptile，另一种基于一阶梯度的元学习算法，为FOMAML提供了另一种选择。

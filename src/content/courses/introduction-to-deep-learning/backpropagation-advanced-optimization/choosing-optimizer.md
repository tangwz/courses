---
course: "introduction-to-deep-learning"
chapter: "backpropagation-advanced-optimization"
lesson: "choosing-optimizer"
sourceId: 5050
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-4-backpropagation-advanced-optimization/choosing-optimizer"
title: "选择优化算法"
description: "提供为您的模型选择合适优化器的实用考量。"
order: 8
plots: []
sourceHash: "0e45034fdea30fadfdfe40c047ccb1de813c4a8ea425ef4c07725e3c0f895360"
sourceCorrections: []
---

深度学习 (deep learning)任务需要高效的梯度计算，通常通过反向传播 (backpropagation)完成，并利用动量法（Momentum）、RMSprop和Adam等高级优化算法。鉴于这些要素，一个主要问题是如何为特定的深度学习任务选择最合适的优化器。

这些高级优化器旨在解决普通随机梯度下降 (gradient descent)（SGD）的不足，以期实现更快的收敛，并更可靠地应对复杂高维的损失平面。虽然SGD仅根据当前梯度调整权重 (weight)，但动量法整合了过去的梯度，RMSprop根据近期梯度的幅度调整学习率，而Adam则结合了这两种思路。

### 没有单一的最佳优化器

需要理解的是，并没有一个普遍更优的优化算法。优化器的表现通常很大程度上取决于问题结构、具体数据集、模型架构以及所选的超参数 (parameter) (hyperparameter)（尤其是学习率）。可以将优化器视为工具箱中不同的工具，每种都适用于稍有差异的任务。

### 考量因素

选择优化器时，请考量以下方面：

1. **收敛速度：** 优化器接近一个好的解决方案（低损失）的速度有多快？自适应方法如Adam通常比SGD初期收敛更快，尤其是在复杂问题上或没有仔细调整学习率的情况下。
2. **泛化性能：** 训练好的模型在新数据上的表现如何？一些研究表明，精心调整的SGD加动量法，在某些情况下，可能会找到更平坦的最小值，从而泛化能力比自适应方法找到的稍好。然而，Adam通常以更少的调整工作提供出色的泛化能力。
3. **对超参数 (parameter) (hyperparameter)的敏感性：** 需要多少工作量来找到好的超参数，特别是学习率？与SGD相比，Adam和RMSprop对初始学习率的选择通常不那么敏感。Adam的默认学习率常能提供合理的性能，使其更容易上手。
4. **内存占用：** 自适应方法如Adam和RMSprop需要维护额外的每个参数信息（动量向量 (vector)、平方梯度的移动平均值）。与SGD或SGD加动量法相比，这会增加内存使用，尽管对于大多数常见模型大小来说，这很少成为瓶颈。
5. **每步计算成本：** 尽管每次更新步骤的计算存在细微差异，但与前向传播和反向传播 (backpropagation)的成本相比，这些差异通常较小。总训练时间受收敛所需步骤数量的影响更大。

### 对比不同选项

让我们简要总结一下我们讨论过的优化器的特点：

- **SGD加动量法：**

  - *优点：* 经过仔细调整，可取得当前最佳性能。可能带来泛化能力非常好的解决方案。比自适应方法内存使用量更低。
  - *缺点：* 通常需要仔细调整学习率和动量参数 (parameter)。为获得最佳结果，可能需要学习率调度（训练期间降低学习率）。收敛速度可能比自适应方法慢，尤其是在初期。
- **RMSprop：**

  - *优点：* 根据每个参数调整学习率。通常在非平稳问题（统计量随时间变化的问题）上表现良好。对学习率选择的敏感度低于SGD。
  - *缺点：* 仍然能从学习率调整中受益。在许多应用中普遍被Adam替代，Adam在自适应学习率中增加了动量。
- **Adam（自适应矩估计）：**

  - *优点：* 结合了动量和自适应学习率（类似RMSprop）的优点。通常收敛速度快。对超参数 (hyperparameter)相对不敏感（默认设置通常表现良好）。普遍被认为是很好的默认选择。
  - *缺点：* 比SGD需要更多内存。一些研究表明，它有时可能收敛到比SGD更尖锐的最小值，在特定情境下可能影响泛化能力，但实际上它表现非常出色。
- **AdamW：** 一种常见且推荐的Adam变体。它修改了权重 (weight)衰减（L2正则化 (regularization)）的应用方式，将其与自适应学习率机制解耦。与将L2正则化直接应用于损失函数 (loss function)的标准Adam相比，这通常会带来更好的正则化性能和更高的最终模型精度。使用Adam时，考虑AdamW通常有益。

### 实用建议

那么，您应该从何开始呢？

1. **从Adam或AdamW开始：** 对于大多数深度学习 (deep learning)应用，Adam（或更优选AdamW）是一个很好的首选。它通常能以默认设置（例如学习率为$0.001$）提供快速收敛和强劲的性能。这使您能够快速获得一个不错的基准结果。PyTorch和TensorFlow等框架使得使用AdamW简单明了。

   ```python
   # PyTorch使用示例
   import torch.optim as optim

   # model = YourNeuralNetwork()
   # learning_rate = 0.001

   # 使用AdamW
   optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

   # --- 训练循环 ---
   # loss.backward()
   # optimizer.step()
   # optimizer.zero_grad()
   ```
2. **如有必要，考虑SGD加动量法：** 如果您有足够的计算预算进行大量超参数 (parameter) (hyperparameter)调整（学习率、动量、学习率调度），或者在特定基准上最大化泛化能力是您的首要任务，那么尝试SGD加动量法是值得的。从$0.9$等常见动量值开始。找到合适的学习率调度（例如，步进衰减、余弦退火）在这里很重要。
3. **实践并评估：** 最好的方法通常是经验性方法。如果您的初始选择（很可能是Adam/AdamW）未能达到您的性能目标，请尝试SGD加动量法，或者为您选择的优化器尝试不同的学习率。务必在单独的验证集上评估性能，而不仅仅是训练损失。
4. **调整学习率：** 即使使用自适应优化器，学习率仍然是一个重要的超参数。虽然默认值（Adam为$0.001$）通常是好的起点，但您可以通过尝试$0.0001$、$0.0005$、$0.005$或$0.01$等值来获得更好的结果。
5. **使用学习率调度：** 无论使用哪种优化器，在训练期间降低学习率（学习率调度）是一种常见技术，常能改善最终性能。这允许在训练初期采取更大步长，并在模型接近收敛时进行更精细的调整。

> 这是一个选择优化器的简化决策流程。从Adam/AdamW开始，进行评估，并在需要时考虑已调整的SGD加动量法。

总之，尽管Adam/AdamW对于大多数深度学习任务而言是高效且推荐的起点，但了解不同优化器的特点并愿意根据验证性能进行尝试，对于您的神经网络 (neural network)取得最佳结果是必不可少的。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2014)
  Journal: International Conference on Learning Representations (ICLR 2015); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化算法，结合了自适应学习率和动量。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov and Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR 2019); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  提出了AdamW，是Adam的一个变体，将权重衰减与自适应学习率机制解耦，从而改进了正则化。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面的教材，涵盖了各种优化算法，包括SGD、动量、RMSprop和Adam，以及它们的理论基础和实际考量。
- [The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/abs/1705.08292) — Ashia C. Wilson, Rebecca Roelofs, Mitchell Stern, Nathan Srebro, Benjamin Recht (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS 2017); DOI: [10.48550/arXiv.1705.08292](https://doi.org/10.48550/arXiv.1705.08292)
  批判性分析了自适应梯度方法（如Adam）与SGD的泛化性能，指出SGD可以找到更平坦的极小值。

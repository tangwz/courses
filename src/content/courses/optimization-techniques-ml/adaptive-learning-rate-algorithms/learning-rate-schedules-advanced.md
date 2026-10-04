---
course: "optimization-techniques-ml"
chapter: "adaptive-learning-rate-algorithms"
lesson: "learning-rate-schedules-advanced"
sourceId: 1272
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-3-adaptive-learning-rate-algorithms/learning-rate-schedules-advanced"
title: "了解学习率调整策略"
description: "结合自适应方法与学习率调整策略（例如，衰减、周期性学习率）。"
order: 7
plots: ["plots/1272-0.json"]
sourceHash: "ada2edaefe01c725b83de4c3aecc2bc118cb20fcbc4a60819cd3bfa776d1eed6"
sourceCorrections: []
---

调整学习率是训练机器学习 (machine learning)模型的重要步骤。虽然Adam、RMSprop和AdaGrad等自适应方法根据梯度历史动态调整学习率，常比固定学习率带来显著性能提升，但采用全局学习率调整策略能够为模型的稳定性和性能带来独特的优势。学习率调整策略会系统地调整训练过程中的学习率 $\eta$，通常随训练的进行而降低。

主要思想很直接：在训练初期，当参数 (parameter)可能远离最优值时，使用相对较大的学习率以便快速取得进展。随着训练收敛，参数接近良好解决方案时，降低学习率以允许更精细的调整，从而减少越过最小值的风险，并帮助更有效地处理复杂损失面。即使使用管理每个参数学习率的自适应优化器，将调整策略应用于全局基础学习率 $\eta$ 也是一种常见且通常有效的方法。

接下来，我们考察几种流行的调整策略。

### 常见学习率调整策略

几种函数常用于定义学习率的变化方式，通常是回合数或迭代次数 $t$ 的函数。

1. **步长衰减：**
   这是最简单的策略之一。学习率在固定数量的回合（一个“步长”）内保持不变，然后按一定因子降低。例如，你可能每10个回合将学习率减半。

   - **机制：** $\eta_t = \eta_0 \times \text{衰减因子}^{\lfloor t / \text{步长} \rfloor}$
   - **参数 (parameter)：** 初始学习率 $\eta_0$，衰减因子（例如，0.5，0.1），步长（以回合为单位）。
   - **考量：** 需要调整步长和衰减因子。离散的下降有时可能导致训练动态的突然变化。
2. **指数衰减：**
   提供比步长衰减更平滑的下降。学习率在每个回合（有时是每次迭代）后乘以一个小于1的衰减因子。

   - **机制：** $\eta_t = \eta_0 \times e^{-kt}$（连续形式）或 $\eta_t = \eta_0 \times \text{衰减因子}^t$（离散回合形式）。这里 $k$ 或 `decay_factor` 控制下降速度。
   - **参数：** 初始学习率 $\eta_0$，衰减率 $k$ 或 `decay_factor`。
   - **考量：** 与步长衰减相比，提供更平滑的降低。衰减率需要调整。
3. **逆时间衰减 (1/t 衰减)：**
   学习率与迭代或回合数的倒数成比例下降。

   - **机制：** $\eta_t = \frac{\eta_0}{1 + kt}$
   - **参数：** 初始学习率 $\eta_0$，衰减率 $k$。
   - **考量：** 这种策略在特定条件下与随机梯度下降 (gradient descent)的收敛保证有理论关联。在训练后期，学习率下降可能变得相当缓慢。
4. **余弦退火：**
   这种流行的策略在定义的周期 $T$ 内，学习率遵循余弦曲线的形状下降。它从初始学习率 $\eta_{max}$ 开始，平滑地退火到最小学习率 $\eta_{min}$（通常为0）。

   - **机制：** $\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min}) \left(1 + \cos\left(\frac{t_{mod} \pi}{T}\right)\right)$，这里 $t_{mod} = t \pmod T$。
   - **参数：** 最大学习率 $\eta_{max}$（通常是初始学习率 $\eta_0$），最小学习率 $\eta_{min}$，周期 $T$（一个完整周期的回合数/迭代次数）。
   - **考量：** 常与“重启”结合使用（SGDR：带热启动的随机梯度下降），即策略每 $T$ 个回合重置。学习率的这种周期性增加可以帮助优化器摆脱次优局部最小值或鞍点。在许多深度学习 (deep learning)任务中，它在实践中非常有效。

   
   
   ![学习率调整策略比较](plots/1272-0.json)
   
   

> 不同学习率调整策略在100个回合中的表现，初始学习率为0.1。步长衰减使学习率骤降，指数衰减平滑下降，余弦退火遵循周期性模式（这里展示的是每50个回合重启一次）。

5. **周期性学习率 (CLR)：**
   CLR 不是单调递减学习率，而是在最小 ($\eta_{min}$) 和最大 ($\eta_{max}$) 边界之间周期性变化。常见形式包括三角形或基于余弦的周期。

   - **机制：** 学习率在固定的周期长度内波动。例如，三角形策略在周期前半段从 $\eta_{min}$ 线性增加到 $\eta_{max}$，然后在后半段线性下降回 $\eta_{min}$。
   - **参数：** 最小学习率 $\eta_{min}$，最大学习率 $\eta_{max}$，周期长度（以回合/迭代为单位）。
   - **考量：** 由Leslie N. Smith提出，其理由是周期性增加学习率可以帮助优化器更快地通过鞍点，并更广泛地查看损失面，可能使得模型最终位于更宽泛、泛化能力更好的最小值。调整边界 ($\eta_{min}$, $\eta_{max}$) 和周期长度很重要。通常建议进行学习率范围测试以找到合适的边界。
6. **预热：**
   尤其适用于训练非常深的网络（如Transformer）或使用大批量大小时，预热阶段常在训练的*开始*阶段使用。在预热期间，学习率从很低（例如，0或接近0）开始，并在指定数量的初始回合或迭代中逐渐增加到目标初始学习率 ($\eta_0$ 或 $\eta_{max}$)（例如，线性或二次方增加）。

   - **机制：** 在 $N_{warmup}$ 步/回合内将 $\eta$ 从接近零增加到 $\eta_0$。
   - **考量：** 有助于稳定早期训练，此时参数随机初始化，梯度可能很大或不稳定。可以防止由于大的初始更新导致的发散，尤其是在与Adam等自适应优化器结合使用时，它们的初始估计可能存在噪声。预热阶段结束后，通常会采用其中一种衰减策略（如余弦或步长衰减）。

### 将调整策略与自适应优化器结合

学习率调整策略经常与Adam或RMSprop等自适应优化器结合使用。该策略通常控制每一步 $t$ 的*全局基础学习率* $\eta_t$。然后，自适应优化器会使用这个 $\eta_t$ 以及其内部状态（例如，梯度的一阶和二阶矩估计）来计算最终的参数 (parameter)更新。

对于Adam，更新规则是：
$\theta_{t+1} = \theta_t - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
这里，$\eta_t$ 是在时间步 $t$ 由当前生效策略确定的学习率。该策略提供全局的、依赖时间的调整，而Adam则基于梯度统计信息提供局部、参数特定的适应。这种组合在实践中通常非常有效。

### 实际考量与调整

- **选择策略：** 最优策略因问题而异。步长衰减简单易用，而余弦退火（常带重启）和预热阶段后的线性衰减是深度学习 (deep learning)中非常常见且有效的默认选择。CLR需要更细致的调整，但在特定情况下能提供优势。
- **参数 (parameter)调整：** 所有策略都有需要调整的参数（初始学习率、衰减率/因子、步长、周期长度、最小/最大学习率、预热持续时间）。这通常通过监控训练和验证损失曲线来经验性地完成。可以使用网格搜索、随机搜索或贝叶斯优化等更复杂的方法。
- **实现：** 现代深度学习框架（TensorFlow/Keras、PyTorch）为大多数常见学习率调整器提供了便捷的实现。你通常会定义好调整器，然后将其与你的优化器关联。

```python
# 示例（伪代码）

initial_lr = 0.001
optimizer = Adam(parameters, lr=initial_lr) # Adam使用初始学习率

# 示例：带预热的余弦退火
num_epochs = 100
warmup_epochs = 10
scheduler = CosineAnnealingWithWarmup(optimizer,
                                     warmup_epochs=warmup_epochs,
                                     total_epochs=num_epochs,
                                     min_lr=1e-6)

for epoch in range(num_epochs):
    # 训练循环...
    # train_one_epoch(model, dataloader, optimizer)

    # 在每个回合结束时更新学习率
    scheduler.step()

    # 可选：记录当前学习率
    # current_lr = scheduler.get_last_lr()[0]
    # print(f"Epoch {epoch+1}, LR: {current_lr}")

    # 验证循环...
    # validate(model, val_dataloader)
```

> 该结构展示了调整器如何集成到训练循环中，根据回合数更新优化器的学习率。

了解并恰当应用学习率调整策略，能为你的优化方法增添一项有效技能。虽然自适应方法处理大部分参数的学习率调整，但调整策略对学习过程提供了必要的全局控制，有助于加快收敛，提高稳定性，并可能改善最终模型性能。实验和仔细监控在为你的特定机器学习 (machine learning)任务寻找最适合的调整策略时，仍是重要部分。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  为深度学习优化技术提供了全面基础，包括学习率的作用和基本调度方法。
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — Ilya Loshchilov and Frank Hutter (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983)
  介绍了带有周期性热重启的余弦退火学习率调度，并通过实验证明了其有效性。
- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) — Leslie N. Smith (2017)
  Journal: arXiv; DOI: [10.48550/arXiv.1506.01186](https://doi.org/10.48550/arXiv.1506.01186)
  提出了循环学习率技术，旨在稳定训练并提高性能。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化器，这是一种广泛使用的自适应方法，常与全局学习率调度结合使用。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，并展示了包含预热阶段的学习率调度策略的有效性。

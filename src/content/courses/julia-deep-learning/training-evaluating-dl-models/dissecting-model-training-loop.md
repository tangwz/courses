---
course: "julia-deep-learning"
chapter: "training-evaluating-dl-models"
lesson: "dissecting-model-training-loop"
sourceId: 6913
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-4-training-evaluating-dl-models/dissecting-model-training-loop"
title: "剖析模型训练循环"
description: "详细解释了Flux.jl中典型深度学习模型训练循环所包含的组件和步骤。"
order: 1
plots: []
sourceHash: "f84368214dff960f05f53ee4d7dfd7d8b83488e17a9caf0c95adeeb6e9455975"
sourceCorrections: []
---

为了使神经网络 (neural network)架构能够从数据中学习，一个称为训练的过程非常重要。这个训练的中心组成部分是**模型训练循环**。这个循环包括模型反复处理数据，衡量其误差，并调整其内部参数 (parameter)以求改进。尽管 Flux.jl 提供了像 `Flux.train!` 这样封装此循环的高级抽象，但了解其底层机制对于调试、定制和理解更高级的训练过程非常有帮助。

### 迭代特性：周期与批次

训练深度学习 (deep learning)模型是一个迭代优化过程。我们不会仅仅让模型查看整个数据集一次，并期望它学会一切。相反，我们通常会对数据进行多次处理。

- **周期（Epochs）**：一个周期表示对整个训练数据集的一次完整遍历。例如，如果你有 10,000 个训练样本并完成一个周期，你的模型已经看过这 10,000 个样本各一次。训练通常涉及运行多个周期，使模型能够反复查看数据并学习更复杂的模式。
- **批次（Batches）**：一次性处理整个数据集可能会带来高计算开销和高内存占用，特别是对于大型数据集。这还可能导致学习过程不够稳定。为了管理这种情况，数据集通常被分成更小、更易于处理的块，称为**小批次（mini-batches）**。模型一次处理一个迷你批次，计算误差，并更新其权重 (weight)。

  - **批次大小（Batch Size）**：迷你批次中的样本数量是一个超参数 (parameter) (hyperparameter)，通常称为 `batch_size`。常见的批次大小范围在 32 到 256 之间，但可能根据数据集大小和可用内存而异。
  - **为什么要使用小批次？**
    1. **内存效率**：更小的批次处理所需内存更少。
    2. **计算效率**：由于并行化，现代硬件（CPU，特别是 GPU）通常能更高效地处理批次。
    3. **更好的泛化能力**：来自小批次更新的随机性（因为每个批次都对整体梯度提供略微不同的估计）可以帮助模型避开局部最小值，并对未见过的数据有更好的泛化表现。

在 Julia 中，你通常会使用数据加载器（例如第三章中提及的 `MLUtils.jl` 提供的加载器）来以小批次方式遍历数据集。

### 单次训练步骤（每批次）的构成

在每个周期内，模型会遍历所有小批次。对于每个小批次，会发生几个必要的操作：

> 训练循环中每个小批次执行的操作序列。

1. **获取一个小批次**：数据加载器提供下一批次的输入特征（例如 `x_batch`）和对应的目标标签（例如 `y_batch`）。
2. **前向传播**：输入特征（`x_batch`）被输入到神经网络 (neural network)。网络通过其层处理这些输入，应用权重 (weight)和激活函数 (activation function)，以生成预测。

   ```julia
   predictions = model(x_batch)
   ```

   这里，`model`是你的Flux.jl神经网络。
3. **计算损失**：使用预定义的损失函数 (loss function)（例如，分类任务使用 `Flux.crossentropy`，回归任务使用 `Flux.mse`）将预测与真实目标标签（`y_batch`）进行比较。损失量化 (quantization)了模型对当前批次的预测“错误”程度。

   ```julia
   current_loss = loss_function(predictions, y_batch)
   ```

   较低的损失值通常表示在该批次上的表现更好。
4. **计算梯度（反向传播 (backpropagation)）**：这是自动微分发挥作用的地方，通常由 Flux 生态系统中的 Zygote.jl 处理。我们计算损失函数对模型中所有可训练参数 (parameter)（权重和偏置 (bias)）的梯度。这些梯度指示了每个参数为减少损失所需改变的方向和大小。

   ```julia
   # ps 包含模型的可训练参数（例如，Flux.params(model)）
   grads = Flux.gradient(() -> loss_function(model(x_batch), y_batch), ps)
   ```

   `Flux.gradient` 函数接受一个匿名函数（第一个参数）来计算损失，以及需要计算梯度的参数 `ps`（例如通过 `Flux.params(model)` 获取）。它返回一个包含梯度的 `Zygote.Grads` 对象。
5. **更新参数（优化步骤）**：优化器（例如 Adam、SGD）使用计算出的梯度来更新模型的参数。学习率（优化器的一个超参数 (hyperparameter)）控制这些更新的大小。目标是将参数朝着最小化损失的方向微调 (fine-tuning)。

   ```julia
   # 优化器是你的优化器（例如，ADAM()）
   Flux.Optimise.update!(opt, ps, grads)
   ```

   `Flux.Optimise.update!` 函数根据梯度 `grads` 和优化器的逻辑修改 `ps` 中的参数。
6. **隐式梯度重置**：对于大多数标准训练循环，梯度是基于该批次的特定损失为每个批次重新计算的。Zygote 的 `Flux.gradient` 为特定的函数调用计算梯度，因此除非你明确地设计你的循环来实现梯度累积，否则不会跨批次进行持久的梯度累积。参数更新后，当前批次的梯度就完成了其作用。

这些步骤会对训练数据集中的每个小批次重复执行。一旦所有小批次处理完成，一个周期就完成了。整个过程（所有周期）会持续进行，直到达到停止条件，例如预定义的周期数，或者模型在验证集上的表现不再提升。

### Julia中Flux.jl的基本训练循环

让我们看看这些组件如何在简化的 Julia 代码结构中结合起来。假设 `model`、`loss_function`、`opt`（优化器）和 `train_loader`（你的批次数据）都已定义。

```julia
# 假设：
# 模型 = 你的Flux模型（例如，Chain(...)）
# 损失函数 = 你选择的损失函数（例如，Flux.crossentropy）
# 优化器 = 你的优化器（例如，ADAM()）
# 训练加载器 = 你的数据加载器，提供(x_batch, y_batch)元组
# 周期数 = 训练的周期数

# 获取模型参数
ps = Flux.params(model)

for epoch in 1:num_epochs
    epoch_loss = 0.0
    num_batches = 0

    for (x_batch, y_batch) in train_loader
        # 步骤4：计算梯度
        # 损失计算（步骤3）包含在梯度计算中
        # 前向传播（步骤2）也隐含在其中
        current_loss, grads = Flux.withgradient(ps) do
            # 前向传播和损失计算
            predictions = model(x_batch)
            loss_function(predictions, y_batch)
        end

        # 步骤5：更新参数
        Flux.Optimise.update!(opt, ps, grads)

        epoch_loss += current_loss
        num_batches += 1
    end

    average_epoch_loss = epoch_loss / num_batches
    println("周期: $epoch, 平均损失: $average_epoch_loss")
end

println("训练完成。")
```

在这个例子中，`Flux.withgradient` 是一个有用的函数，它同时计算所提供的匿名函数（即我们的损失）的值及其对 `ps` 的梯度。如果你也需要损失值，这通常比在计算损失后单独调用 `Flux.gradient` 更高效。

这个循环遍历周期，然后在每个周期内遍历批次。对于每个批次，它计算损失和梯度，然后更新模型参数 (parameter)。它还会累积损失以打印该周期的平均值，为你提供一个监控训练进度的基本方法。

### 为什么要理解循环细节？

尽管 Flux.jl 提供了 `Flux.train!(loss, params, data, opt; cb = ...)` 这样一个高级实用函数来处理整个循环，但像我们这样拆解它，可以提供多种好处：

- **调试**：当出现问题时（例如，损失变为 `NaN`，模型不学习），理解每个步骤可以让你检查中间值（预测、损失、梯度），并更有效地查明问题。
- **定制**：你可能需要在训练循环中实现自定义逻辑。例如，应用梯度裁剪、对每个批次执行更复杂的指标计算，或实现非标准优化方案。手动循环能让你完全掌控。
- **高级技术**：许多高级深度学习 (deep learning)技术都涉及对标准训练循环的修改。在处理更复杂的情况（例如涉及多个优化器或自定义梯度操作的情况）之前，了解基本情况很重要。
- **清晰度**：它阐明了像 `Flux.train!` 这样的抽象在底层做了什么，使其使用不再是“黑箱”。

随着学习的推进，你将看到这个基本循环结构如何作为训练各种类型模型和实现更精密的训练策略的依据，我们将在本章的后续部分介绍这些策略，例如使用回调函数以及更正式地评估模型性能。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8: Optimization for Training Deep Models (and related chapters)
  为神经网络训练提供了全面的理论基础，包括梯度下降、反向传播、损失函数、优化器、迭代次数和mini-batch处理。
- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl Contributors (2025)
  Flux.jl的官方文档，为在Julia中构建模型、定义损失函数、使用优化器和实现训练循环提供了实用指导。
- [Zygote.jl Documentation](https://fluxml.ai/Zygote.jl/dev/) — The Zygote.jl Contributors (2024)
  解释了Julia中自动微分的官方文档，它为Flux.jl训练生态系统中的梯度计算（反向传播）提供了支持。
- [CS231n: Deep Learning and Computer Vision - Lecture Notes](http://cs231n.stanford.edu/) — Fei-Fei Li, Ehsan Adeli, Justin Johnson, Zane Durante (2023)
  Publisher: Stanford University
  提供了关于神经网络训练核心组件的清晰易懂的解释，包括前向传播、反向传播、损失函数和优化算法。

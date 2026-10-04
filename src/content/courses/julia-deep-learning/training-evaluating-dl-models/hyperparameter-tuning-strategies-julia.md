---
course: "julia-deep-learning"
chapter: "training-evaluating-dl-models"
lesson: "hyperparameter-tuning-strategies-julia"
sourceId: 6918
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-4-training-evaluating-dl-models/hyperparameter-tuning-strategies-julia"
title: "超参数调优策略"
description: "了解为深度学习模型寻找最佳超参数的策略，例如网格搜寻或随机搜寻。"
order: 6
plots: []
sourceHash: "213562bcc89666e17f42c8d89fbb601ad938e94ff3c79b9a1c8e42153dbf9094"
sourceCorrections: []
---

为深度学习 (deep learning)模型寻找最佳设置，这些设置并非在训练期间学习，是模型优化中的一项主要挑战。这些设置称为超参数 (parameter) (hyperparameter)，有效选择它们可以显著影响模型的表现。超参数包括优化器的学习率、隐藏层中的神经元数量、用于训练的批次大小，或正则化 (regularization)项的强度（例如$L_2$正则化中的$\lambda$）。有多种策略可用于搜索超参数空间，并确定能带来更好表现模型的配置。

### 什么是超参数 (parameter) (hyperparameter)？

在讨论调优策略之前，让我们澄清参数和超参数之间的区别：

- **模型参数**：这些是模型在训练过程中学习到的值。对于神经网络 (neural network)，它们主要是层中的权重 (weight)和偏置 (bias)。优化器根据损失函数 (loss function)迭代更新这些参数。
- **超参数**：这些是模型和训练过程的外部配置。您作为实践者，在训练开始*前*设置它们。例子包括：
  - 学习率 ($\alpha$)
  - 迭代周期数
  - 批次大小
  - 隐藏层数量
  - 每层单元数量
  - 激活函数 (activation function)的选择（例如，ReLU，sigmoid）
  - 优化器的选择（例如，Adam，SGD）
  - 正则化 (regularization)参数（例如，丢弃率，$L_2$惩罚强度）

找到一组好的超参数通常更像一门艺术而非一门精确的科学，它涉及实验和迭代改进。

### 手动调优

手动调优，或称“有根据的猜测”，是最直接的方法，也是实践者通常首先尝试的方法。它依赖于：

- **经验和直觉**：如果您曾处理过类似问题，可能对某些超参数 (parameter) (hyperparameter)有一个好的起点。
- **已发表研究**：论文通常会报告其实验中使用的超参数，这可以作为参考。
- **经验法则**：例如，常试的学习率可能是0.1、0.01、0.001。批次大小通常是2的幂（例如，32、64、128）。

您通常会使用一组初始超参数来训练模型，在验证集上评估其表现，然后根据结果调整超参数。例如，如果训练损失下降非常缓慢，您可能会尝试增加学习率。如果模型出现过拟合 (overfitting)，您可能会增加正则化 (regularization)或降低模型复杂度。

**优点：**

- 如果您有良好的领域知识或正在处理一个已充分了解的问题，这可能很有效。
- 无需在标准训练脚本中添加特殊工具。

**缺点：**

- 可能非常耗时且效率不高。
- 高度依赖实践者的专业知识。
- 可能无法系统地寻找超参数空间，从而可能错过最佳配置。

手动调优通常是任何超参数搜寻的一部分，即使在使用更自动化方法时也是如此，因为初始范围和选择仍然需要设定。

### 网格搜寻

网格搜寻是一种更系统的方法。您定义一个要测试的超参数 (parameter) (hyperparameter)值的“网格”。然后算法会穷尽式地训练和评估模型，针对这些值的每种可能组合进行操作。

例如，如果您想调整学习率和批次大小：

- 学习率：`[0.1, 0.01, 0.001]`
- 批次大小：`[32, 64]`

网格搜寻将评估以下 $3 \times 2 = 6$ 种组合：

1. 学习率 = 0.1, 批次大小 = 32
2. 学习率 = 0.1, 批次大小 = 64
3. 学习率 = 0.01, 批次大小 = 32
4. 学习率 = 0.01, 批次大小 = 64
5. 学习率 = 0.001, 批次大小 = 32
6. 学习率 = 0.001, 批次大小 = 64

评估所有组合后，选择在验证集上表现最佳的组合。

> 网格搜寻中评估的两个超参数点。每个点代表一次模型训练和评估运行。

**优点：**

- 简单易懂，易于实现。
- 在指定网格内穷尽。

**缺点：**

- **维度灾难**：组合数量随超参数的数量及其每个值的数量呈指数增长。如果您有5个超参数，每个有5个值，那么就是 $5^5 = 3125$ 次评估，这可能在计算上难以承受。
- 最佳值可能位于网格中的点之间。对于像学习率这样的连续超参数，选择离散点是一种简化。
- 假定超参数在影响上是独立的，但这并非总是如此。

在使用Flux.jl在Julia中实现网格搜寻时，您通常会编写嵌套循环，其中每个循环遍历一个超参数的可能值。在最内层循环中，您配置、训练并评估您的Flux模型。

### 随机搜寻

随机搜寻由Bergstra和Bengio（2012）提出，提供了一种出乎意料有效的网格搜寻替代方案。您不是尝试离散网格中的所有组合，而是为每个超参数 (parameter) (hyperparameter)定义一个范围或分布，然后从这些分布中随机抽取组合，进行固定次数的迭代。

例如：

- 学习率：从$10^{-4}$到$10^{-1}$均匀抽样（对数均匀通常更好）。
- 批次大小：从离散值`[16, 32, 64, 128]`中均匀抽样。
- 层中神经元数量：从50到500之间的整数中均匀抽样。

然后，您将运行，比如50次试验，每次使用一组随机抽样的超参数。

> 随机搜寻中评估的点。随机抽样比固定网格能更有效地寻觅空间，尤其当某些超参数比其他超参数影响更大时。

**优点：**

- 通常比网格搜寻更高效，特别是在只有少数超参数显著影响表现时。随机搜寻更有可能命中那些重要超参数的好值。
- 可以为连续超参数寻觅更宽广的值范围。
- 更容易管理预算：您决定运行多少次随机试验。

**缺点：**

- 系统性较差，因此纯粹凭运气可能错过最佳值，尽管随着试验次数的增加，这种情况会变得不那么可能。
- 不保证找到某个区域内的绝对最佳组合，但通常能更快找到“足够好”的配置。

在Julia中实现随机搜寻涉及为每个超参数抽样值（例如，使用适当缩放的`rand()`或`Distributions.jl`中特定分布），然后运行您的训练循环。

### 贝叶斯优化

贝叶斯优化是一种寻找最佳超参数 (parameter) (hyperparameter)的更精巧策略。它为目标函数（例如，验证损失作为超参数的函数）构建一个概率模型（通常是高斯过程）。这个模型在每次评估后更新。“采集函数”（例如，期望改进）被用来决定接下来尝试哪组超参数，平衡新颖区域的尝试（尝试新的、不确定区域）和已知优秀区域的利用（尝试已知良好的区域）。

**核心思想：**

1. **概率模型**：假定性能函数（例如，验证准确度与超参数的关系）未知，但可以通过代理函数（如高斯过程）建模。
2. **采集函数**：此函数使用代理模型的预测和不确定性来建议要评估的下一个超参数组合。它量化 (quantization)一个点“有多大前景”。
3. **迭代**：评估所选超参数，用新的数据点更新代理模型，然后重复。

**优点：**

- 通常比网格或随机搜寻更具样本效率，这意味着它可以用更少的评估找到好的超参数，这在每次模型训练成本较高时非常有价值。
- 有效地应对复杂搜寻空间。

**缺点：**

- 从零开始理解和实现更为复杂。
- 贝叶斯优化的表现可能取决于代理模型和采集函数的选择，它们本身也可能有参数。
- 由于模型拟合，每一步的计算强度可能更高，但这通常可以通过减少对主要模型的评估次数来弥补。

在Julia中，您可以使用`Hyperopt.jl`等包进行贝叶斯优化。虽然将此类工具整合到基本的Flux.jl工作流程中超出了本文范围，但理解其原理很有价值。

### 进阶策略和自动化机器学习 (machine learning) (AutoML)

这些技术包括进化算法（例如，粒子群优化、遗传算法）以及基于强化学习 (reinforcement learning)的方法等进阶手段。其中许多归属于自动化机器学习（AutoML）的范畴，其目标是尽可能自动化机器学习流程，包括超参数 (parameter) (hyperparameter)调优。Google Vizier、Optuna或Hyperopt（Python库）等工具为这些进阶方法提供了框架。

### 超参数 (parameter) (hyperparameter)调优的实际考量

无论您选择哪种策略，请记住以下实用提示：

- **使用专用验证集**：始终根据模型在验证集上的表现来调整超参数，该验证集应与训练集和最终测试集分离。这可以防止超参数对测试数据过拟合 (overfitting)，并提供更真实的泛化能力估计。
- **定义合理的搜寻空间**：
  - 对于学习率，通常在对数尺度上搜寻（例如，$10^{-5}, 10^{-4}, ..., 10^{-1}$）。
  - 对于像神经元数量或批次大小这样的整数参数，根据您的问题和资源选择合理的范围。
- **从简开始，迭代改进**：首先在较大范围内调整最具影响力的超参数（通常是学习率和模型复杂度）。一旦找到有前景的区域，您可以进行更集中的搜寻，采用更精细的粒度或包含更多超参数。
- **提前停止**：在超参数试验中考虑使用提前停止。如果特定超参数组合在训练早期导致表现不佳，您可以提前终止该试验，从而节省计算资源。Flux.jl的回调函数在此处可能有用。
- **并行化**：网格搜寻和随机搜寻高度可并行化，因为每个试验都是独立的。如果您可以访问多个核心或机器，则可以同时评估许多超参数组合。
- **记录所有信息**：仔细记录每次实验：使用的超参数、产生的验证表现，甚至训练曲线。这有助于您理解不同超参数如何影响您的模型，并为未来的试验做出明智决定。`TensorBoardLogger.jl`或自定义日志脚本等工具可能会有帮助。
- **注意计算预算**：超参数调优可能非常耗费计算资源。设定一个您可以承受的时间或试验次数预算。随机搜寻对于固定预算特别适用。

### 使用Flux.jl实现超参数 (parameter) (hyperparameter)调优

在典型的Julia和Flux.jl设置中，您可以通过编写脚本来实现网格搜寻或随机搜寻，该脚本应包含：

1. 定义每个超参数的范围或值集。
2. 循环所需的试验次数（或网格搜寻的所有组合）。
3. 在循环内部：
   - 设置当前超参数。
   - 使用这些超参数构建您的Flux模型（`Chain`、`Dense`、`Conv`等）。
   - 使用当前学习率定义优化器（例如，`ADAM(learning_rate)`）。
   - 运行训练循环（`Flux.train!`）指定的迭代周期数。
   - 在验证集上评估训练好的模型。
   - 记录超参数和验证分数。
4. 所有试验结束后，选择产生最佳验证分数的超参数。

这里是Julia中随机搜寻循环的大致结构：

```julia
# （假设您已定义了 data_loader, build_model, loss_function, train_model!, eval_model）

best_val_loss = Inf
best_hyperparams = Dict()

num_trials = 50

for trial in 1:num_trials
    # 1. 抽样超参数
    lr = 10^(rand() * -4 -1) # 在 1e-5 和 1e-1 之间对数均匀抽样学习率
    batch_size = rand([32, 64, 128])
    num_neurons = rand(50:500)
    # ... 其他超参数

    current_hyperparams = Dict(:lr => lr, :batch_size => batch_size, :num_neurons => num_neurons)
    println("试验 $trial: 使用 $current_hyperparams 进行训练")

    # 2. 构建模型和优化器
    # model = build_model(num_neurons, ...) # 您用于构建 Flux 模型的函数
    # opt = ADAM(lr)

    # 3. 使用当前批次大小创建数据迭代器
    # train_data_iter = # ... 使用 MLUtils.jl DataLoader 与批次大小
    # val_data_iter = # ...

    # 4. 训练模型
    # try
    #     for epoch in 1:num_epochs
    #         # Flux.train!(loss_function, Flux.params(model), train_data_iter, opt; cb=...)
    #     end
    #
    #     # 5. 在验证集上评估
    #     val_loss = # eval_model(model, val_data_iter, loss_function)
    #     println("试验 $trial: 验证损失 = $val_loss")
    #
    #     # 6. 记录并更新最佳值
    #     if val_loss < best_val_loss
    #         best_val_loss = val_loss
    #         best_hyperparams = current_hyperparams
    #         println("找到新的最佳超参数：$best_hyperparams，损失为 $best_val_loss")
    #     end
    # catch e
    #     println("试验 $trial 失败，错误：$e")
    #     # （可选）记录错误并继续
    # end
end

println("找到的最佳超参数：$best_hyperparams，验证损失为：$best_val_loss")
```

这段伪代码说明了大致结构。您需要使用Flux.jl函数填充模型创建、训练和评估的细节。请记住为`eval_model`使用单独的验证集。

### 总结

超参数 (parameter) (hyperparameter)调优是开发有效深度学习 (deep learning)模型的重要一步。虽然手动调优提供了一个起点，但网格搜寻和随机搜寻等系统方法提供了更结构化的方式来寻觅超参数空间。对于计算成本高的模型，贝叶斯优化可能是一种更高效的替代方案。通过仔细选择策略、定义合理的搜寻空间并严谨跟踪实验，您可以显著提升模型在未见数据上的表现。这个过程是迭代的，但模型质量的提升通常非常值得付出努力。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础性教材，提供了深度学习全面的数学和概念背景，其中有专门章节讨论优化、正则化和超参数选择。
- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Publisher: JMLR Editorial Board; Volume: 13; Pages: 281-305
  这篇论文首次提出随机搜索，作为比网格搜索更高效的超参数优化方法，尤其在仅部分超参数对性能有显著影响时表现出色。
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://papers.nips.cc/paper_files/paper/2012/hash/05311655a15b75ffa9cfe05b0d0616b3-Abstract.html) — Jasper Snoek, Hugo Larochelle, Ryan P. Adams (2012)
  Journal: Advances in Neural Information Processing Systems 25; Publisher: NeurIPS Proceedings; Volume: 25; Pages: 2951-2959
  一篇有影响力的论文，展示了贝叶斯优化在调整机器学习算法方面的有效性，并提供了其应用的实用建议和经验结果。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125964/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广泛使用的实用指南，涵盖了包括网格搜索和随机搜索在内的各种超参数调整方法，附有具体示例和实践建议。第三版。

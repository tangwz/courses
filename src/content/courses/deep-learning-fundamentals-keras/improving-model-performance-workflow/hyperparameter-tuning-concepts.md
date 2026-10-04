---
course: "deep-learning-fundamentals-keras"
chapter: "improving-model-performance-workflow"
lesson: "hyperparameter-tuning-concepts"
sourceId: 4984
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-6-improving-model-performance-workflow/hyperparameter-tuning-concepts"
title: "超参数调优概念"
description: "介绍超参数调优的观念和寻找最优配置的方法。"
order: 7
plots: ["plots/4984-0.json"]
sourceHash: "cd72b993dd1a707bf44af586aff3c520416e539c2177884aa9a9f8f79bb1800b"
sourceCorrections: []
---

本章中，我们侧重于优化已训练模型，处理过拟合 (overfitting)等问题，并简化训练流程。然而，优化模型性能的另一个重要方面是在训练开始前，就选择恰当的*超参数 (parameter) (hyperparameter)*。与从数据中习得的模型参数（权重 (weight)和偏置 (bias)）不同，超参数是由实践者选定的配置项。正确设定这些参数能显著影响训练速度、收敛性和模型最终的预测能力。

### 什么是超参数 (parameter) (hyperparameter)？

可以把超参数看作是你在启动神经网络 (neural network)构建机器并开始处理数据*之前*设定的旋钮和刻度盘。它们定义了模型的结构，并控制学习过程本身。相比之下，模型参数是机器在运行（训练）*期间*自动调整的内部变量，以最小化损失函数 (loss function)。

你已经遇到或将使用的常见超参数包括：

- **学习率：** 控制优化器在权重 (weight)更新时所采取的步长。
- **训练轮数：** 整个训练数据集通过网络的次数。
- **批次大小：** 在更新模型权重前处理的训练样本数量。
- **层数：** 神经网络的深度。
- **每层神经元/单元数量：** 每层的宽度或容量。
- **激活函数 (activation function)：** 不同层的ReLU、Sigmoid、Tanh或Softmax等选择。
- **优化器选择：** 选择Adam、RMSprop或SGD等算法。
- **正则化 (regularization)参数：** L1、L2正则化（通常表示为$\lambda$）或Dropout率的强度。
- **卷积核大小和数量（针对CNN）：** 卷积层中卷积核的尺寸和数量。

选择这些超参数的恰当值非常重要，因为很少有单一的“最佳”组合适用于所有问题或数据集。

### 超参数 (parameter) (hyperparameter)调优的挑战

寻找超参数的最优组合通常更偏向于艺术而非科学，并且它面临着几项挑战：

1. **搜索空间很大：** 即使只有少量超参数，可能的组合数量也能呈指数增长。例如，测试5种学习率、3种批次大小和4种层架构，就已产生$5 \times 3 \times 4 = 60$种组合。
2. **相互依赖：** 超参数通常相互影响。最优学习率可能取决于所选的优化器或批次大小。
3. **计算成本：** 评估单一组合需要训练模型，这通常需要很长时间。搜索许多组合可能会非常耗时且资源密集。
4. **反馈延迟：** 只有在训练（和评估）过程完成后，你才能知道一个组合有多好。

寻找最佳超参数组合的这个过程被称为**超参数调优**或**超参数优化**。

### 常见调优策略

虽然基于经验的手动调整起到一定作用，但更系统的方法通常更受青睐。

#### 手动调优

这通常是起始点。你根据常见做法、先前的经验或已发表的研究做出有根据的猜测，训练模型，评估其性能，然后根据结果调整超参数 (parameter) (hyperparameter)。例如，如果模型过拟合 (overfitting)，你可能会尝试增加正则化 (regularization)强度或降低模型复杂性（减少层/神经元数量）。如果训练过慢或不稳定，你可能会调整学习率。尽管直观，但这可能效率低下，并可能错过最优组合。

#### 网格搜索

网格搜索是一种穷举方法。你为每个想要调优的超参数定义一个特定的范围或离散值列表。然后，算法系统地训练并评估模型，针对这些值的*每种可能组合*。

例如，如果你指定：

- 学习率：`[0.01, 0.001, 0.0001]`
- 批次大小：`[32, 64]`

网格搜索将训练并评估$3 \times 2 = 6$个不同的模型。

> 网格搜索在由学习率和批次大小定义的二维超参数空间中查看所有组合（蓝点）的视图。

尽管对于较小的搜索空间来说很全面，但随着超参数数量或每个超参数的值的数量增加，网格搜索在计算上变得难以处理（受“维度诅咒”影响）。

#### 随机搜索

随机搜索提供了一种更高效的替代方案。它不是尝试所有组合，而是从每个超参数指定的范围或分布中随机抽取固定数量的组合。

研究（例如，Bergstra和Bengio的研究）表明，在相同的计算预算下，随机搜索通常能找到比网格搜索更好的模型。这是因为并非所有超参数都同等主要；随机搜索有更高的机会为*主要*超参数找到好的值，而网格搜索则花费大量时间在只改变不重要参数的组合上。



![网格搜索与随机搜索采样比较](plots/4984-0.json)



> 比较网格搜索如何在固定网格上采样点，以及随机搜索如何在超参数空间中更自由地采样点。随机搜索可能在相同试验次数下，为每个超参数尝试更广泛的值范围。

你为超参数定义范围（例如，学习率在0.0001到0.01之间）或分布，并指定要尝试的组合总数（`n_iter`）。

### 高级调优策略

存在更复杂的算法，例如：

- **贝叶斯优化：** 建立超参数 (parameter) (hyperparameter)与模型性能之间关系的概率模型，然后使用该模型智能地选择下一个要尝试的组合，侧重于搜索空间中有潜力的区域。
- **进化算法：** 利用受生物进化启发的设计思想（变异、交叉、选择）来迭代优化超参数集合的群体。

这些方法可能更高效，但实现和理解起来也更复杂。

### 实际考量

- **使用交叉验证：** 使用训练数据上的交叉验证来进行超参数 (parameter) (hyperparameter)调优很重要。直接针对单一验证集进行调优可能导致超参数对该特定验证集过拟合 (overfitting)，从而导致对未见过测试数据的泛化能力差。
- **使用调优库：** 像KerasTuner这样的库专门设计用于与Keras集成，并自动化网格搜索、随机搜索甚至贝叶斯优化过程，为你处理训练和评估循环。Scikit-learn也提供`GridSearchCV`和`RandomizedSearchCV`，它们可以适用于Keras模型，尽管KerasTuner通常提供更紧密的集成。
- **从小处开始：** 在考虑更细致的细节或影响力较小的参数之前，先从调优最有影响力的超参数（如学习率、优化器选择和基本架构）开始。

超参数调优是一个迭代过程，随着你对模型和数据获得更多见解，通常会再次进行。这是将模型性能从可接受提升到优异的重要一步。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  深度学习基础的全面参考，涵盖超参数选择和优化策略的各个方面。
- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — James Bergstra, Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Volume: 13; Pages: 281-305
  介绍并实证展示了随机搜索在超参数优化方面优于网格搜索的有效性。
- [KerasTuner Documentation](https://keras.io/keras_tuner/) — O'Malley, Tom, Bursztein, Elie, Long, James, Chollet, François, Jin, Haifeng, Invernizzi, Luca, and others (2019)
  KerasTuner的官方文档，提供使用Keras模型实现各种超参数调优策略的实践示例和指南。
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://papers.nips.cc/paper/2012/file/05311655a15b75f8e0293fdc8623f99b-Paper.pdf) — Jasper Snoek, Hugo Larochelle, Ryan P. Adams (2012)
  Journal: Advances in Neural Information Processing Systems 25; Publisher: NeurIPS; Volume: 25
  一篇关于将贝叶斯优化应用于高效调整机器学习算法超参数的开创性工作。

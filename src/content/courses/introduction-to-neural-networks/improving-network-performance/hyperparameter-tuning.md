---
course: "introduction-to-neural-networks"
chapter: "improving-network-performance"
lesson: "hyperparameter-tuning"
sourceId: 2037
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-6-improving-network-performance/hyperparameter-tuning"
title: "超参数调优策略"
description: "查找学习率、层大小等最佳超参数的方法。"
order: 6
plots: []
sourceHash: "9b437e297c728f4fb3452693a021fb9722038a23d83c80beb8652d11124b7af4"
sourceCorrections: []
---

正如之前所讨论的，找到合适的网络结构和正则化 (regularization)参数 (parameter)只是优化神经网络 (neural network)的一部分。许多在训练过程开始*之前*设定的其他配置，对模型学习和泛化的效果有显著影响。这些配置被称为**超参数 (hyperparameter)**。与模型在训练中学习的参数（权重 (weight)和偏置 (bias)）不同，超参数由您，也就是实践者来设定。调整它们是获得良好模型性能的必要步骤。

您可能需要调优的常见超参数包括：

- **学习率：** 控制梯度下降 (gradient descent)时的步长。
- **批量大小：** 模型参数更新前处理的样本数量。
- **隐藏层数量：** 决定网络的深度。
- **每隐藏层神经元数量：** 控制每层的宽度或容量。
- **激活函数 (activation function)：** 隐藏层中使用的非线性函数（例如，ReLU，Tanh）。
- **优化器：** 使用的特定梯度下降算法（例如，Adam，SGD，RMSprop）。
- **正则化强度：** $L1$ 或 $L2$ 正则化的系数（例如，lambda）。
- **Dropout 比率：** 在 Dropout 层训练期间停用的神经元比例。

正确设定这些配置很重要。学习率过高可能导致训练过程发散，而过低则会使训练变得异常缓慢或陷入次优解。网络的容量（层和神经元）需要足以捕获数据中的隐含模式，但过高的容量会增加过拟合 (overfitting)的风险。优化器的选择会影响收敛的速度和稳定性。实际上，超参数定义了模型学习的条件。

### 超参数 (parameter) (hyperparameter)调优策略

找到超参数的最佳组合可能感觉像大海捞针。幸好，有一些系统性的方法可以使用。

#### 手动调优

最基础的方法是手动调优。基于经验、直觉和对训练过程的观察（例如，监测验证损失），您迭代地调整超参数。您可能会训练一个模型，观察其表现，微调 (fine-tuning)一个超参数（例如，如果损失趋于平稳或爆炸，则将学习率减半），然后重新训练。

手动调优虽然简单易懂，但通常耗时，并且非常依赖实践者的专业知识。分析不同超参数之间的相互作用可能很困难，而且您可能仅仅因为没有想到去尝试而错失了更好的组合。

#### 网格搜索

网格搜索通过系统地检查预设的超参数值集合来自动化此过程。您为每个超参数指定一个要尝试的值列表，算法会为*每种可能的组合*训练并评估一个模型。

例如，如果您想调优学习率和单个隐藏层中的神经元数量：

- 学习率：`[0.1, 0.01, 0.001]`
- 神经元数量：`[32, 64, 128]`

网格搜索是一种通过在预定义的超参数值组合上系统地训练和评估模型来寻找最佳超参数的方法。例如，网格搜索会训练并评估 $3 \times 3 = 9$ 个不同的模型：(0.1, 32), (0.1, 64), (0.1, 128), (0.01, 32), ..., (0.001, 128)。然后，您将选择在验证集上产生最佳性能的组合。

> 网格搜索在由学习率和神经元数量定义的二维网格上尝试组合的示意图。每个点代表一个使用该特定超参数组合训练的模型。

网格搜索的主要优点是其系统性。然而，它受到“维度诅咒”的影响。组合的数量随着超参数的数量和每个超参数考虑的值的数量呈指数级增长。如果您有5个超参数，每个有5个可能的值，则需要训练 $5^5 = 3125$ 个模型！这很快就会变得计算上不可行。

#### 随机搜索

随机搜索通常提供一种更高效的替代方案。您不是在固定的网格上尝试所有组合，而是为每个超参数定义一个*范围*或*分布*（例如，学习率在0.0001到0.1之间均匀采样，神经元数量在16到256之间的整数中采样）。然后，您从这个搜索空间中随机采样固定数量的组合。

> 随机搜索在相同的二维空间内随机采样点的示意图。与网格搜索的固定步长相比，它更广泛地查看不同的值。

研究（特别是Bergstra和Bengio的研究）表明，随机搜索通常比网格搜索更有效，特别是当只有一部分超参数明显影响最终性能时。通过随机采样，您更有可能比穷尽检查网格上的每个点更快地找到重要超参数的良好值。您通常会设定一个预算（例如，训练50个随机组合），然后选择在该预算内找到的最佳组合。

#### 更高级的方法（简述）

除了网格搜索和随机搜索之外，还存在更复杂的技术，通常归属于贝叶斯优化。这些方法建立一个概率模型（一个“替代模型”），用于表示超参数如何影响验证性能。它们使用此模型智能地选择下一个要尝试的超参数组合，重点关注搜索空间中根据过去结果看起来最有希望的区域。Optuna、Hyperopt或Keras Tuner等工具实现了此类策略。虽然这些工具功能强大，但它们增加了复杂性，通常在掌握了更简单的方法之后才考虑使用。

### 调优的实际考虑

- **定义合理的搜索空间：** 不要选择任意大的范围。对于学习率，通常在对数尺度上进行搜索（例如，$10^{-4}, 10^{-3}, 10^{-2}$）。对于像 Dropout 比率这样的参数 (parameter)，线性尺度（例如，0.1, 0.25, 0.5）可能更合适。使用先验知识或研究论文的结果来指导您的初始范围。
- **使用验证集：** 始终使用*验证集*上的性能来评估超参数 (hyperparameter)组合。测试集应保持不动，直到您选择了最终模型（及其调优后的超参数），以获得对其泛化性能的无偏估计。
- **考虑交叉验证：** 对于较小的数据集，在您的调优循环中使用 k 折交叉验证可以为每个超参数组合提供更好的性能估计，从而降低选择恰好在一个特定验证分割上表现良好的设置的风险。
- **保持耐心和迭代：** 超参数调优通常是一个迭代过程。您可以从广泛的随机搜索开始，以识别搜索空间中有希望的区域，然后对这些区域进行更细致的搜索（无论是网格搜索还是随机搜索）。早期停止等技术也可以应用于超参数搜索循环*内部*，以快速放弃表现不佳的组合。

超参数调优是机器学习 (machine learning)工作流程中一个必要但有时繁琐的部分。通过使用网格搜索或随机搜索等方法系统地尝试不同的设置，并对它们在验证数据上进行严格评估，您可以显著提高模型对新的、未见过数据的泛化能力。这个过程与之前讨论的正则化 (regularization)技术协同工作，以构建既强大又可靠的模型。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  提供神经网络的全面理论和实践基础，详细讨论超参数、优化方法和训练策略。
- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html) — James Bergstra, Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 13; Pages: 281-305; DOI: [10.5555/2188385.2188395](https://doi.org/10.5555/2188385.2188395)
  提出随机搜索方法，并证明其在超参数优化方面比网格搜索更有效，尤其在并非所有超参数都同等重要的情况下。
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://proceedings.neurips.cc/paper/2012/file/05311655a15b75ffa88a8d189ae47fcd-Paper.pdf) — Jasper Snoek, Hugo Larochelle, Ryan P. Adams (2012)
  Journal: Advances in Neural Information Processing Systems (NIPS) 25; Publisher: Elsevier B.V.; Pages: 2952-2960; DOI: [10.5555/2999134.2999201](https://doi.org/10.5555/2999134.2999201)
  一篇在机器学习超参数调整中应用贝叶斯优化的开创性研究，提供比穷举或随机搜索更智能的方法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=z12JEAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  涵盖网格搜索和随机搜索等超参数调整策略的指南，并提供使用流行深度学习框架的实现细节。（第四版）

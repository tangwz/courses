---
course: "fine-tuning-adapting-large-language-models"
chapter: "full-parameter-fine-tuning"
lesson: "hyperparameter-tuning-full"
sourceId: 3002
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-3-full-parameter-fine-tuning/hyperparameter-tuning-full"
title: "超参数调优策略"
description: "针对学习率、批次大小和其他超参数进行调优的先进方法，以获得最佳性能。"
order: 3
plots: ["plots/3002-0.json"]
sourceHash: "6a65cc20c4895fdd9030f75d1b427e18c8af7fa147d07763e9803e99434fc858"
sourceCorrections: []
---

全参数 (parameter)微调 (fine-tuning)虽然强大，但对其配置敏感。选择合适的超参数 (hyperparameter)是获得良好性能的根本，同时避免浪费大量计算资源和时间。与预训练 (pre-training)不同，预训练的超参数设定常通过大量试错确定，而微调需要根据具体任务、数据集大小和模型结构进行仔细调整。我们来看看最有影响力的超参数，以及有效调整它们的方法。

### 学习率 ($\eta$)

学习率可以说是最重要的超参数 (parameter) (hyperparameter)。它决定了梯度下降 (gradient descent)过程中更新模型权重 (weight)$\theta$的步长：


$$
\theta_{new} = \theta_{old} - \eta \nabla L(\theta_{old})
$$


这里 $\nabla L(\theta_{old})$ 是损失函数 (loss function) $L$ 相对于旧权重的梯度。

- **过高：** 学习率过大可能导致训练过程不稳定，损失剧烈波动甚至发散（无限增大）。优化器可能会越过最优权重配置。
- **过低：** 学习率过小会导致收敛非常慢。模型可能会陷入次优的局部最小值，或者需要过多的训练步数才能达到一个好的解。

对于大型预训练 (pre-training)模型的全参数微调 (fine-tuning)，学习率通常设置远低于预训练时使用的学习率。这是因为我们希望温和地调整现有知识，保留预训练阶段学到的强大表示，同时让模型适应新任务。微调大型语言模型（LLMs）的常见起始值常常在 $1 \times 10^{-5}$ 到 $5 \times 10^{-5}$ 的范围内。全参数微调很少会超过 $1 \times 10^{-4}$。

**学习率调度器：** 在整个训练过程中使用恒定的学习率通常是次优的。学习率调度器会在训练期间动态调整学习率，经常能提高收敛速度和最终性能。常见的方法包括：

1. **线性预热与衰减：** 这是微调Transformer模型非常常用的调度方法。训练开始时学习率非常小，在设定的“预热”步数内逐渐线性增加。预热阶段结束后，学习率通常在剩余训练步数内线性（或有时多项式）衰减到零。预热阶段有助于早期训练的稳定，特别是当梯度可能很大或有噪声时，能防止模型发散。随后的衰减使得模型接近收敛时可以进行更精细的调整。

   
   
   ![线性预热与衰减学习率调度](plots/3002-0.json)
   
   

   > 示例：线性预热（0到100步）后接线性衰减的学习率调度。
2. **余弦退火：** 在这种方式中，学习率从最大值（预热后）开始，遵循余弦曲线衰减到最小值（通常为零）。与线性衰减相比，这提供了更平滑的衰减，有时能更好地优化损失面。
3. **恒定预热：** 类似于线性预热，但学习率在预热阶段结束后立即跳到最大值，然后根据某种调度（例如线性、余弦）衰减。

调度器的选择及其参数（预热步数、衰减函数）本身就是超参数，通常需要进行调整。

### 批次大小

批次大小指定了在单次前向和反向传播 (backpropagation)中用于计算梯度并更新模型权重 (weight)的训练样本数量。

- **更大的批次大小：**
  - *优点：* 能更准确地估计整个数据集的真实梯度，带来更稳定的训练更新。能更有效地发挥硬件并行性优势，通常每轮训练时间更快。
  - *缺点：* 需要更多的GPU内存。有时会导致收敛到“尖锐”的局部最小值，其泛化效果可能不如小批次找到的“平坦”局部最小值。
- **更小的批次大小：**
  - *优点：* 需要更少的内存。小批次梯度估计中的固有噪声可以作为一种正则化 (regularization)形式，有可能帮助模型避免糟糕的局部最小值，找到泛化效果更好的平坦局部最小值。
  - *缺点：* 由于硬件利用率较低，每轮训练可能更慢。梯度更新噪声更大，可能需要更小的学习率来保持稳定。

可行的最大批次大小通常受限于可用的GPU内存。**梯度累积**（将在第7章讨论）等方法允许通过在几次小批次上计算梯度后再进行权重更新来模拟更大的批次大小，缓解内存限制，代价是计算时间略有增加。全参数 (parameter)微调 (fine-tuning)的典型批次大小范围从4到64，这很大程度上取决于模型大小和GPU内存。

### 轮数与早期停止

一轮（epoch）表示完整遍历整个训练数据集一次。

- **轮数过少：** 模型可能对数据没有足够的学习，无法有效完成任务（欠拟合 (underfitting)）。
- **轮数过多：** 模型可能开始记忆训练数据，包括其噪声，导致在未见数据上表现不佳（过拟合 (overfitting)）。

确定最佳轮数通常是使用**验证集**凭经验完成的。在每轮（或部分轮）训练后，监控验证集上的性能（例如损失、任务特定指标）。当验证集上的性能停止提升或开始下降时，停止训练，这种技术称为**早期停止**。由于预训练 (pre-training)模型提供了良好初始化，微调 (fine-tuning)通常只需要几轮（例如1-5轮），特别是在数据集较大时。较小的数据集可能看起来需要更多轮次，但这会增加过拟合的风险，使得正则化 (regularization)更加重要。

### 优化器选择

尽管存在各种优化器，**AdamW**（带有解耦权重 (weight)衰减的Adam）是微调 (fine-tuning)Transformer模型的标准且普遍推荐的优化器。

- **Adam：** 结合了动量（使用过去的梯度）和自适应学习率（根据梯度大小对每个参数 (parameter)进行调整）。
- **AdamW：** 修改了Adam中应用L2正则化 (regularization)（权重衰减）的方式。标准Adam将权重衰减与自适应学习率计算混合，可能使其效果降低。AdamW将权重衰减计算与梯度更新解耦，在Transformer模型中通常带来更好的正则化和改进的泛化能力。

在内存极度受限的场景下，可能会考虑Adafactor等其他优化器，但AdamW是常见的起点。AdamW中优化器特有的超参数 (hyperparameter)，如beta值（$\beta_1, \beta_2$）和epsilon（$\epsilon$），通常保留其默认值（例如 $\beta_1=0.9, \beta_2=0.999, \epsilon=1e-8$），尽管调整它们偶尔能带来少量提升。

### 权重 (weight)衰减

权重衰减是一种正则化 (regularization)技术（等同于L2正则化），它在损失函数 (loss function)中添加一个与模型权重平方大小成比例的惩罚项。这通过保持权重较小来抑制模型学习过于复杂的模式，从而减少过拟合 (overfitting)。

权重衰减系数是另一个需要调整的超参数 (parameter) (hyperparameter)。微调 (fine-tuning)的典型值通常在0.01到0.1之间。将其设置为0会禁用权重衰减。如前所述，其有效性通常与使用AdamW等能正确处理它的优化器有关。

### 系统化调优方法

找到超参数 (parameter) (hyperparameter)的最佳组合可能很复杂。与手动试错不同，当计算预算允许时，经常采用更系统化的方法：

1. **网格搜索：** 为每个要调优的超参数定义一组离散值。对这些值的每种可能组合训练模型。尽管这种方法很详尽，但随着超参数数量和每个超参数值的增加，它在计算上很快变得不可行（维度灾难）。
2. **随机搜索：** 为每个超参数定义一个范围或分布。从这些范围或分布中随机抽取超参数组合，并为每个样本训练模型。令人意外的是，随机搜索通常比网格搜索更高效，因为它更广泛地在超参数空间中搜索，有可能更快地找到好的组合，特别是当只有少数超参数对性能有显著影响时。

   > 网格搜索与随机搜索在两个超参数（学习率和批次大小）上的搜索模式比较。网格搜索系统地覆盖点，而随机搜索在空间中采样。
3. **贝叶斯优化：** 一种更高级的方法，通过之前训练的结果指导下一次要尝试的超参数组合选择。它构建一个概率模型（通常使用高斯过程），将超参数映射到性能指标（例如验证损失）。它使用这个模型来平衡*广域搜索*（在不确定性高的区域尝试超参数）和*局部优化*（在目前表现最好的点附近尝试超参数）。Optuna、Hyperopt或Ray Tune等工具提供了贝叶斯优化和其他高级调优算法的实现。

### 调优的实用建议

- **从合理的默认值开始：** 从文献中为类似模型和任务报告的超参数 (parameter) (hyperparameter)开始。除非有特殊原因，否则对优化器参数使用标准值。
- **优先考虑超参数：** 最初的调优工作应集中在学习率、批次大小以及可能的权重 (weight)衰减上，因为这些通常影响最大。
- **使用验证集：** 始终根据单独验证集（而非测试集）上的性能评估超参数设置，并使用早期停止来确定每次尝试的训练时长。
- **记录实验：** 使用实验跟踪工具（如MLflow、Weights & Biases）系统地记录每次运行的超参数、代码版本、数据集和结果。这对于可重现性和分析是不可或缺的。
- **迭代与优化：** 在随机搜索或贝叶斯优化中，从更宽的范围开始，然后在有希望的值周围缩小范围，进行更精细的调优。
- **考虑资源限制：** 全参数微调 (fine-tuning)成本较高。如果资源有限，优先调优学习率和轮数。考虑使用更小的模型变体或数据子集进行初步的、更快的超参数调整，但要注意，当扩展回原规模时，最佳设置可能会略有变化。

掌握全参数微调的超参数调优，是融合了对基本原理的理解、系统化搜索方法的应用，以及结合基于具体任务和可用资源的实际考虑。它在一定程度上仍然是经验性的过程，但有序的方法显著增加了找到能带来最佳性能的配置的可能性。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: 3rd International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化器，该优化器被广泛用于训练和微调包括大型语言模型在内的深度学习模型。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: ICLR 2019; DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  介绍了AdamW，这是Adam的改进版本，它正确地应用了权重衰减，使其成为基于Transformer模型的优化器。
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — Ilya Loshchilov, Frank Hutter (2017)
  Journal: ICLR 2017; DOI: [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983)
  提出了余弦退火学习率调度策略，这是一种常用于提升模型训练和收敛的技术。
- [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Volume: 13; Pages: 281-305
  证明了随机搜索在超参数优化方面通常比网格搜索更有效。
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/abs/1206.2944) — Jasper Snoek, Hugo Larochelle, and Ryan P. Adams (2012)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.1206.2944](https://doi.org/10.48550/arXiv.1206.2944)
  讨论了贝叶斯优化在超参数调整中的应用，提供了一种比网格搜索和随机搜索更系统的方法。

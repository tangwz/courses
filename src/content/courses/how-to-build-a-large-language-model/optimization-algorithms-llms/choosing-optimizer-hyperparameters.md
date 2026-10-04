---
course: "how-to-build-a-large-language-model"
chapter: "optimization-algorithms-llms"
lesson: "choosing-optimizer-hyperparameters"
sourceId: 6037
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-17-optimization-algorithms-llms/choosing-optimizer-hyperparameters"
title: "选择优化器超参数 (lr, betas, eps, weight_decay)"
description: "讨论设置优化器超参数的常见做法和注意事项。"
order: 5
plots: []
sourceHash: "19e41f997bb62939898f0de0f59e630c33d60dbe5ba82d633fa4757584c175db"
sourceCorrections: []
---

为优化器选择合适的超参数 (parameter) (hyperparameter)对于成功训练大型语言模型来说非常重要。尽管像AdamW这样的自适应优化器与传统SGD相比简化了一些方面，但找到学习率、动量项、epsilon以及权重 (weight)衰减的最佳设置仍然是训练过程中的主要组成部分，特别是在考虑到LLM训练的规模和成本时。这些超参数直接影响模型的收敛速度、稳定性和最终性能。本文探讨如何选择这些主要参数的值，包括AdamW、学习率调度和梯度裁剪等概念的考量。

### 学习率 ($\eta$)

学习率也许是唯一最重要的超参数 (parameter) (hyperparameter)。它决定了梯度下降 (gradient descent)过程中采取的步长。设置过高，训练可能会变得不稳定，导致发散或损失波动。设置过低，训练会慢得不切实际，可能陷入次优的局部最小值。

- **LLM的典型范围：** 对于使用AdamW训练的大型Transformer模型，峰值学习率通常在$1e-5$到$6e-4$的范围内。具体值很大程度上取决于模型大小、批次大小和架构。较大模型有时能从略小的峰值学习率中获益。例如，对于数十亿参数规模的模型，常见的起始点可能是$1e-4$到$3e-4$左右。
- **与批次大小的关联：** LLM训练通常涉及非常大的批次大小（通常每批数百万个token，通过数据和梯度累积实现）。一个常见的经验法则是*线性缩放规则*：如果将批次大小乘以$k$，那么学习率也应乘以$k$。然而，这条规则并非总是完全适用，尤其对于特别大的批次大小。一些研究表明，*平方根缩放规则*（$\eta \propto \sqrt{k}$）可能更适用于某些批次大小。实际操作中，虽然批次大小会影响选择，但最优学习率通常在所述典型范围内通过经验发现，通常从训练类似大小模型的论文中报告的值开始。
- **与调度的关联：** 请记住，我们通常使用包含预热和衰减的学习率调度（第17章，“学习率调度策略”）。当我们谈论“学习率”时，通常是指在预热阶段后达到的*峰值*学习率。预热时长和衰减策略（例如，线性或余弦）也与峰值学习率的有效性有关。更长的预热可能允许设置略高的峰值学习率。
- **找到最优值：** 考虑到计算成本，穷举网格搜索通常不可行。常见的策略是：
  1. 从文献中报告的、针对相似大小和架构模型（例如，Llama、GPT-3论文）的值开始。
  2. 如果可能，在较短的训练运行或模型的较小规模版本/数据集上，围绕此值进行小范围尝试（例如，$1e-4, 2e-4, 3e-4$）。
  3. 在初始阶段密切关注训练损失曲线。快速下降后出现不稳定表明学习率过高。进展极其缓慢则表明学习率过低。

### Adam/AdamW Betas ($\beta_1, \beta_2$)

Adam和AdamW使用两个类似动量的项，由$\beta_1$和$\beta_2$控制。

- $\beta_1$：控制梯度的指数移动平均（第一阶矩）。典型默认值为$0.9$。
- $\beta_2$：控制梯度平方的指数移动平均（第二阶矩）。典型默认值为$0.999$。此项针对每个参数 (parameter)调整学习率。

对于大多数LLM训练场景，$\beta_1 = 0.9$和$\beta_2 = 0.999$的默认值效果非常好，通常无需修改即可使用。

- **调整考量：** 尽管调整这些参数不如调整学习率常见，但一些研究已尝试替代方案。
  - 降低$\beta_2$（例如，到$0.95$或$0.98$）会使优化器更快地适应梯度方差的变化，这有时有助于跳出尖锐的局部极小值或提高训练初期的稳定性，但也可能导致后期收敛稳定性下降。一些大型模型训练方案使用$\beta_2 = 0.95$或$\beta_2 = 0.98$。
  - 调整$\beta_1$的情况较少。
  - 除非您观察到通过调整学习率或使用梯度裁剪无法解决的特定不稳定情况，否则通常建议保留默认值（$\beta_1=0.9, \beta_2=0.999$，或者可以尝试$\beta_2=0.98$）。更改betas会增加调整过程的复杂性。

### Epsilon ($\epsilon$)

Adam/AdamW中的epsilon项（$\epsilon$）是一个小值，在自适应学习率计算时加到分母中（具体来说，是在计算第二阶矩估计的平方根之前）。它的主要目的是防止除以零并提高数值稳定性，尤其当第二阶矩估计非常接近零时。

- **常见值：** 标准默认值为$1e-8$。
- **调整考量：** Epsilon很少调整。与学习率或权重 (weight)衰减相比，它对性能的影响通常很小。在涉及混合精度训练（第20章）的场景中，特别是FP16，数值精度较低时，一些实践者可能会略微增加epsilon（例如，到$1e-7$或$1e-6$）以进一步提高数值稳定性。然而，如果怀疑存在与优化器分母相关的稳定性问题，这通常是最后的办法。坚持使用默认值$1e-8$是标准做法。

### 权重 (weight)衰减 ($\lambda$)

权重衰减是一种正则化 (regularization)技术，通过向损失函数 (loss function)添加与模型权重平方大小成比例的惩罚项来防止过拟合 (overfitting)。如前所述，AdamW实现了*解耦*的权重衰减，与自适应梯度一起使用时，通常优于原始Adam优化器固有的L2正则化方法。

- **LLM的典型范围：** LLM训练中权重衰减（$\lambda$）的常见值通常在$0.01$到$0.1$的范围内。$0.1$的值经常用作起始点。
- **调整策略：** 最优的权重衰减值取决于数据和模型。
  - 通常在找到合理的学习率和调度后进行调整。
  - 监测验证损失（或困惑度）。如果验证损失开始增加而训练损失持续下降，模型可能正在过拟合，增加权重衰减可能有所帮助。反之，如果训练和验证损失都过早进入平台期，减少权重衰减可能有利。
  - 小范围尝试（例如，$0.01,$0.05, $0.1$等值）通常就足够了。
  - 请注意，对于在海量数据集上训练的超大型模型，数据规模带来的隐式正则化可能会减少对强显式权重衰减的需求，但$0.1$左右的值仍然常见。

### 实际实现和关联

请记住这些超参数 (parameter) (hyperparameter)彼此关联。更改学习率可能需要调整权重 (weight)衰减。学习率调度的有效性与所选的峰值学习率有关。

以下是如何在PyTorch中定义AdamW优化器并指定这些超参数：

```python
import torch

# 假设 'model' 是您的大型语言模型实例
# 定义超参数
peak_lr = 2e-4
beta1 = 0.9
beta2 = 0.98 # 示例，使用了略微调整的beta2
epsilon = 1e-8
weight_decay_lambda = 0.1

# 过滤不应应用权重衰减的参数
# （例如，偏置、层归一化权重）
decay_params = []
no_decay_params = []
for pn, p in model.named_parameters():
    if p.requires_grad:
        # 检查偏置或1D参数（如层归一化权重）
        if (pn.endswith("bias") or
                len(p.shape) == 1):
            no_decay_params.append(p)
            # print(f"不衰减的参数: {pn}") # 解开注释用于调试
        else:
            decay_params.append(p)
            # print(f"衰减的参数: {pn}") # 解开注释用于调试

# 创建优化器参数组
optimizer_grouped_parameters = [
    {'params': decay_params, 'weight_decay': weight_decay_lambda},
    {
        'params': no_decay_params,
        'weight_decay': 0.0
        # 这些参数不进行权重衰减
    }
]

# 实例化优化器
optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=peak_lr,
    # 注意：学习率调度器将在训练期间调整此值
    betas=(beta1, beta2),
    eps=epsilon
)

print(
    f"优化器已创建，包含 {len(optimizer_grouped_parameters)} "
    f"个参数组。"
)
print(
    f"组 0（衰减）："
    f"{len(optimizer_grouped_parameters[0]['params'])} 个张量，"
    f"weight_decay="
    f"{optimizer_grouped_parameters[0]['weight_decay']}"
)
print(
    f"组 1（无衰减）："
    f"{len(optimizer_grouped_parameters[1]['params'])} 个张量，"
    f"weight_decay="
    f"{optimizer_grouped_parameters[1]['weight_decay']}"
)

# 学习率调度器通常的使用示例
# （此处未显示调度器设置）
# 假设调度器已在其他地方根据峰值学习率、预热和衰减定义
# for step in range(num_training_steps):
#     current_lr = scheduler.get_last_lr()[0] # 从调度器获取当前学习率
#     # 在优化器步进*之前*设置优化器的学习率
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = current_lr
#     # ... 前向传播，损失计算，反向传播 ...
#     optimizer.step()
#     scheduler.step() # 更新调度器
#     optimizer.zero_grad()
```

> 此PyTorch代码片段示例展示了如何使用独立的参数组初始化AdamW优化器，以选择性地应用权重衰减。偏置 (bias)和归一化 (normalization)层参数通常会关闭权重衰减。

**要点：**

- 从研究论文中针对类似LLM训练设置被证明有效的超参数开始。
- 优先调整峰值学习率及其相关调度（预热步数、衰减类型）。
- 根据验证性能调整权重衰减以控制过拟合 (overfitting)。
- 保持AdamW的$\beta_1$和$\beta_2$为默认值（$0.9$, $0.999$），如果观察到稳定性问题，可以尝试$\beta_2=0.98$。除非必要，否则避免大量调整。
- 保持$\epsilon$为默认值（$1e-8$），除非出现特定的数值精度问题，尤其是在混合精度训练中。
- 采用日志记录和监测（第24章）来跟踪损失曲线、梯度范数和其他指标，以诊断与超参数选择相关的问题。

找到最优的超参数集合是一个迭代过程。仔细选择，依据既定实践并通过监测进行经验观察，对于应对大规模模型优化的复杂性非常重要。

## 参考资料

- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov and Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  介绍了AdamW，它将权重衰减与自适应梯度更新解耦，这是LLM优化中的标准做法。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化器的原始论文，解释了其一阶和二阶矩估计。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  讨论了模型性能如何随模型大小、数据集大小和计算量而扩展，为LLM的超参数选择提供信息。
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) — Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He (2017)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.1706.02677](https://doi.org/10.48550/arXiv.1706.02677)
  展示了学习率随大批量大小的线性缩放规则，与LLM训练相关。

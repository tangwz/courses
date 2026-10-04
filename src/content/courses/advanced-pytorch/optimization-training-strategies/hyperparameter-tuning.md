---
course: "advanced-pytorch"
chapter: "optimization-training-strategies"
lesson: "hyperparameter-tuning"
sourceId: 2725
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-3-optimization-training-strategies/hyperparameter-tuning"
title: "自动化超参数调整"
description: "将PyTorch模型与Optuna或Ray Tune等超参数优化库结合。"
order: 7
plots: []
sourceHash: "86ed69c4bf389ccd6d23047ea7150f405cadaa558319a61a36f8a1abc8e38131"
sourceCorrections: []
---

为深度学习 (deep learning)模型找到最佳超参数 (parameter) (hyperparameter)组合可以大幅影响其表现，但手动调整通常是一个繁琐且依赖经验的过程。随着模型和数据集变得复杂，手动遍历庞大的可能配置空间变得不切实际。这里对自动化超参数优化（HPO）技术进行分析，并提供使用常用库将它们整合到PyTorch工作流程中的方法。

自动化HPO提供了一个系统方法来搜索超参数空间，旨在找到使预定义目标指标（通常与验证表现有关）最小化或最大化的配置。

### 自动化超参数 (parameter) (hyperparameter)优化的核心要点

在应用HPO工具之前，理解其基本组成部分是必不可少的：

1. **超参数：** 这些是在训练过程开始*前*指定的配置设置。它们不像模型权重 (weight)那样在训练期间学习得到。例子包括学习率、优化器类型（及其参数，如Adam的beta值）、权重衰减强度、Dropout概率、批大小、层数、每层单元数、激活函数 (activation function)，以及学习率调度器或数据增强策略的参数。
2. **目标函数：** 这是HPO算法旨在优化（最小化或最大化）的函数。它以一组特定的超参数作为输入，使用这些超参数训练模型，在验证集上评估模型，并返回一个表示模型表现的单一标量值（例如，验证损失、准确率、F1分数）。
3. **搜索空间：** 这定义了每个待调整超参数的可能值范围或集合。例如，学习率可以定义为对数范围内的浮点数（例如，$10^{-5}$到$10^{-1}$），层数可以定义为特定范围内的整数（例如，2到6），优化器类型可以定义为分类选择（例如，'Adam'、'AdamW'、'SGD'）。
4. **搜索算法/策略：** 这是用于在搜索空间中寻找并选择下一组要评估的超参数的方法。不同的算法在计算成本和找到的解的质量之间提供不同的权衡。

### 常见的HPO策略

有几种算法可用于自动化HPO：

- **网格搜索：** 穷举评估在离散网格上定义的所有可能的超参数 (parameter) (hyperparameter)组合。虽然简单，但它受到“维度灾难”的困扰，其计算成本随超参数数量呈指数增长。如果某些超参数对目标影响不大，则效率低下。
- **随机搜索：** 从定义的搜索空间中随机采样超参数配置。出乎意料地有效，随机搜索在相同的计算预算下通常优于网格搜索，特别是当只有少数超参数对性能有明显影响时（如Bergstra和Bengio在2012年所展示）。
- **贝叶斯优化：** 构建目标函数$f(x)$的概率代理模型（通常使用高斯过程），其中$x$代表一个超参数配置。它使用采集函数（例如，预期改进、上置信界）来平衡勘探（尝试不确定、可能高回报的配置）和专注于当前最佳配置附近（的尝试），以选择下一组要评估的超参数。这种方法通常比网格或随机搜索的样本效率更高，特别适用于计算成本高的目标函数。
- **早期停止算法：** 诸如HyperBand和异步逐次减半（ASHA）等技术专注于有效分配固定预算（例如，计算时间、训练周期）。它们启动许多配置，并根据它们的中间表现迭代地修剪掉前景较差的配置，将更多资源分配给表现更好的试验。这些在训练单个模型耗时较长时特别有用。

> 自动化超参数优化过程的简化视图。HPO算法建议一个配置，使用该配置训练并评估模型，得到的性能指标为算法的下一次建议提供依据。

### 将HPO库与PyTorch结合

像Optuna和Ray Tune这样的库简化了将HPO结合到PyTorch项目中的过程。典型的工作流程包括：

1. **定义目标函数：** 创建一个接受特殊`trial`对象（不同库的术语可能略有差异）的Python函数。
2. **建议超参数 (parameter) (hyperparameter)：** 在目标函数内部，使用`trial`对象提供的方法（例如，`trial.suggest_float`、`trial.suggest_int`、`trial.suggest_categorical`）根据定义的搜索空间为当前试验采样超参数值。
3. **构建和训练模型：** 实例化你的PyTorch模型、优化器、数据加载器等，使用建议的超参数。实现你的标准训练和验证循环。
4. **评估并返回指标：** 训练后（或在中间步骤），在验证集上评估模型，并返回HPO算法应优化的目标指标（例如，验证损失或准确率）。
5. **实施剪枝（可选但推荐）：** 对于早期停止算法，定期使用`trial.report(metric, step)`向HPO库报告中间验证指标（例如，每个训练周期后）。然后，调用`trial.should_prune()`并在它返回真时抛出一个特殊异常（例如，`optuna.TrialPruned`）。这允许库提前停止无前景的试验，从而节省资源。
6. **创建并运行研究：** 使用库的API创建一个“研究”或实验实例。通过指定目标函数、优化方向（“最小化”或“最大化”）、搜索算法（采样器/调度器）、要运行的试验次数以及可能的并行执行设置来配置该研究。
7. **分析结果：** 研究完成后，库提供结果访问，包括找到的最佳超参数配置及其对应的目标值。

#### Optuna示例代码片段

这里是一个使用Optuna的示例，以说明其结构：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

# 假设get_model, get_dataloaders, train_one_epoch, evaluate_model已在其他地方定义

def objective(trial):
    # 1. 建议超参数
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    num_layers = trial.suggest_int("num_layers", 2, 5)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, log=True)

    # 2. 构建模型、优化器等
    model = get_model(num_layers=num_layers, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer_class = getattr(optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), lr=lr)

    train_loader, valid_loader = get_dataloaders()
    num_epochs = 20 # 或者也可以是一个超参数

    # 3. 带有剪枝的训练循环
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation_accuracy = evaluate_model(model, valid_loader, device)

        # 5. 报告中间结果以进行剪枝
        trial.report(validation_accuracy, epoch)

        # 根据中间值处理剪枝。
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 4. 返回最终目标值
    final_validation_accuracy = evaluate_model(model, valid_loader, device)
    return final_validation_accuracy # 如果未指定，Optuna默认最大化

# 6. 创建并运行研究
study = optuna.create_study(
    direction="maximize", # 最大化验证准确率
    pruner=optuna.pruners.MedianPruner() # 示例剪枝器
)
study.optimize(objective, n_trials=100) # 运行100个试验

# 7. 分析结果
print("完成的试验次数：", len(study.trials))
print("最佳试验：")
trial = study.best_trial

print("  值: ", trial.value)
print("  参数: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
```

> Optuna目标函数的结构，与PyTorch训练工作流程相结合，包括超参数建议和剪枝。

### 注意事项和最佳实践

- **搜索空间设计：** 仔细定义搜索空间。过窄可能错过最佳区域；过宽则增加计算成本。对学习率等参数 (parameter)使用对数尺度。结合先验知识设定合理的边界。
- **目标指标：** 选择一个真正反映期望的模型行为的指标（例如，验证准确率，不平衡数据集的F1分数，验证损失）。
- **计算预算：** 根据可用资源确定试验次数或时间预算。早期停止算法（Hyperband，ASHA）以及Ray Tune等库中的并行执行支持对于有效地管理预算很有帮助。
- **剪枝：** 积极实施剪枝，通过提前停止无前景的试验来节省大量计算。根据学习动态选择合适的剪枝器。
- **可复现性：** 为PyTorch、NumPy和HPO库本身设置随机种子，以确保结果可复现。
- **复杂度：** 首先调整影响最大的超参数 (hyperparameter)（通常是学习率、优化器选择、正则化 (regularization)），然后再扩展搜索空间。

自动化超参数优化是高级深度学习 (deep learning)从业者的工具箱中一个有价值的工具。通过系统地审视超参数配置并使用智能搜索策略和早期停止，与手动调整相比，你可以大幅提高模型性能和开发效率，从而节省时间以专注于模型架构和训练过程的其他方面。将Optuna或Ray Tune等库整合到你的PyTorch管道中，使你能够有效应用这些技术。

## 参考资料

- [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Volume: 13; Pages: 281-305; DOI: [10.5555/2330613.2330674](https://doi.org/10.5555/2330613.2330674)
  这份论文介绍了随机搜索作为超参数优化的有效且计算高效的替代方法，尤其是在只有少数超参数具有显著影响时。
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://proceedings.neurips.cc/paper/2012/file/05311655a15b75ffa9f2e007270e653d-Paper.pdf) — Jasper Snoek, Hugo Larochelle, and Ryan P. Adams (2012)
  Journal: Advances in Neural Information Processing Systems 25 (NIPS 2012); Publisher: Curran Associates, Inc.; Volume: 25; Pages: 2951-2959; DOI: [10.5555/2999486.2999602](https://doi.org/10.5555/2999486.2999602)
  这篇开创性论文提出了使用高斯过程进行贝叶斯优化的实用框架，展示了其在调整机器学习算法方面的出色样本效率。
- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](http://www.jmlr.org/papers/volume18/16-460/16-460.pdf) — Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar (2017)
  Journal: Journal of Machine Learning Research; Volume: 18; Pages: 1-52; DOI: [10.5555/3305381.3305404](https://doi.org/10.5555/3305381.3305404)
  介绍了Hyperband，一种高效的超参数优化算法，它通过基于bandit的策略和逐次减半方法，提前修剪无前景的配置来智能分配资源。
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/pdf/1907.10902.pdf) — Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama (2019)
  Journal: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19); Pages: 2623–2631; DOI: [10.1145/3292500.3330705](https://doi.org/10.1145/3292500.3330705)
  这篇原始研究论文介绍了Optuna，一个开源超参数优化框架，具有define-by-run API以及高效的采样和修剪算法，并在本节中具体提及。
- [Hyperparameter Optimization: A Review of Methods and Applications](https://ieeexplore.ieee.org/document/7339841) — Bob Shahriari, Kevin Swersky, Ziyu Wang, Ryan P. Adams, and Nando de Freitas (2016)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 104; Pages: 148-175; DOI: [10.1109/JPROC.2015.2494218](https://doi.org/10.1109/JPROC.2015.2494218)
  一篇综合性综述论文，涵盖了广泛的超参数优化技术，包括网格搜索、随机搜索和贝叶斯优化，提供了坚实的理论和实践概述。

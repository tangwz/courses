---
course: "deep-learning-regularization-optimization"
chapter: "optimization-refinements-tuning"
lesson: "practice-hyperparameter-tuning"
sourceId: 5008
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-7-optimization-refinements-tuning/practice-hyperparameter-tuning"
title: "实践：模型超参数调整"
description: "亲自动手调整深度学习模型的超参数。"
order: 11
plots: ["plots/5008-0.json"]
sourceHash: "75357f4c38b674ffdf2a9bd262916711318cd773971cc726215035e984d08c4c"
sourceCorrections: []
---

让我们将本章提到的方法付诸实践。我们讨论了权重 (weight)初始化的作用，学习率调度如何帮助模型收敛，以及网格搜索和随机搜索等寻找合适超参数 (parameter) (hyperparameter)值的方法。现在，您将亲身体验如何使用这些方法来调整深度学习 (deep learning)模型。

调整超参数更多的是一门艺术而非科学，需要多加尝试。然而，系统化的方法会大大提高您找到一个能带来更好模型性能和泛化能力的配置的可能性。

### 设置实验

一个常见的场景是使用简单的卷积神经网络 (neural network)（CNN）在CIFAR-10数据集上进行图像分类。CIFAR-10包含60,000张32x32彩色图像，分为10个类别。为完成此任务，需要具备基本的PyTorch环境，并熟悉模型定义、数据加载和训练循环的编写。

我们的目的不是构建一个绝对最优的CIFAR-10分类器，而是呈现超参数 (parameter) (hyperparameter)调整的*过程*。

首先，我们使用PyTorch定义一个简单的CNN架构。这将是我们的基础模型：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # 输入: 3x32x32 -> 输出: 16x32x32
        self.pool = nn.MaxPool2d(2, 2) # 输出: 16x16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 输出: 32x16x16
        # 池化后: 32x8x8

        # 全连接层
        self.fc1 = nn.Linear(32 * 8 * 8, 128) # 展平尺寸: 32*8*8 = 2048
        self.dropout = nn.Dropout(dropout_rate) # 应用Dropout
        self.fc2 = nn.Linear(128, 10) # 10个输出类别

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 展平除批量维度外的所有维度
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 注意：权重初始化（如He初始化）通常由PyTorch层默认处理
# 但如果需要，也可以在此处明确设置。
```

我们还需要用于CIFAR-10的标准数据加载和转换流程。我们假设您有函数`load_cifar10_data(batch_size)`，它返回用于训练集和验证集的PyTorch `DataLoader`实例。请记住包含归一化 (normalization)处理。

### 确定超参数 (parameter) (hyperparameter)和搜索范围

根据本章内容，有几个超参数是值得调整的候选：

1. **学习率 ($\alpha$):** 可能是最重要的超参数。影响收敛速度和最终性能。我们将查看通常在$1e-4$到$1e-2$范围内的值。对学习率进行搜索时，对数尺度通常有效。
2. **优化器:** 我们可以比较`Adam`和带有动量的`SGD`。Adam通常是一个不错的默认选择，但SGD+动量在细致调整后有时能取得稍好的泛化能力。为简单起见，我们在此使用`Adam`，但调整其学习率。
3. **权重 (weight)衰减 (L2 正则化 (regularization)强度, $\lambda$):** 控制对大权重的惩罚以防止过拟合 (overfitting)。常用值从$0$（无衰减）到$1e-3$不等。
4. **Dropout率:** 训练期间丢弃神经元的概率。已包含在我们的`SimpleCNN`中。值通常介于$0.1$到$0.5$之间。
5. **批量大小:** 影响梯度估计的噪声和训练速度。也与学习率相互影响。常用值是2的幂，如32、64、128、256。

我们将使用**随机搜索**以提高效率。让我们定义搜索范围：

- **学习率:** 从$10^{-4}$到$10^{-2}$的对数尺度上均匀采样。
- **权重衰减:** 从$10^{-5}$到$10^{-3}$的对数尺度上均匀采样。
- **Dropout率:** 在$0.1$到$0.5$之间均匀采样。
- **批量大小:** 从集合{64, 128, 256}中随机选择。

### 调整循环的实现

主要思想是运行多个训练实验（试验），每个实验都使用从我们定义的范围内随机采样的超参数 (parameter) (hyperparameter)集。我们进行固定且相对较少轮次的训练（例如10-15轮），以快速获得信号，记录验证性能，然后比较不同试验的结果。

以下是调整循环的概要：

```python
import random
import numpy as np
import torch.optim as optim

# 假设SimpleCNN, load_cifar10_data已定义
# 假设train_one_epoch()和evaluate()函数存在

num_trials = 20 # 要尝试的随机配置数量
num_epochs_per_trial = 10 # 短时间训练

results = []

for trial in range(num_trials):
    print(f"--- 试验 {trial+1}/{num_trials} ---")

    # 1. 采样超参数
    lr = 10**np.random.uniform(-4, -2) # 学习率的对数均匀采样
    weight_decay = 10**np.random.uniform(-5, -3) # 权重衰减的对数均匀采样
    dropout_rate = random.uniform(0.1, 0.5)
    batch_size = random.choice([64, 128, 256])

    print(f"采样结果: lr={lr:.6f}, wd={weight_decay:.6f}, dropout={dropout_rate:.4f}, batch_size={batch_size}")

    # 2. 设置数据加载器、模型、优化器
    train_loader, val_loader = load_cifar10_data(batch_size=batch_size)
    model = SimpleCNN(dropout_rate=dropout_rate)
    # 如果可用，考虑使用CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0

    # 3. 固定轮次训练
    for epoch in range(num_epochs_per_trial):
        # train_one_epoch(model, train_loader, criterion, optimizer, device)
        # val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        # 用于说明结构的模拟训练/评估
        print(f"  轮次 {epoch+1}/{num_epochs_per_trial} - 模拟训练...")
        # 在实际运行时，根据evaluate()结果更新best_val_accuracy
        # 对于本示例，我们模拟一个结果
        simulated_val_accuracy = 0.3 + trial*0.01 + epoch*0.02 + random.uniform(-0.05, 0.05) # 占位符
        best_val_accuracy = max(best_val_accuracy, simulated_val_accuracy)

    print(f"试验 {trial+1} 完成。最佳验证准确率: {best_val_accuracy:.4f}")

    # 4. 记录结果
    results.append({
        'trial': trial + 1,
        'lr': lr,
        'weight_decay': weight_decay,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'best_val_accuracy': best_val_accuracy
    })

# 5. 分析结果（见下一节）
print("\n--- 调整完成 ---")
# 按验证准确率排序结果
results.sort(key=lambda x: x['best_val_accuracy'], reverse=True)

print("前5个配置：")
for i in range(min(5, len(results))):
    print(f"排名 {i+1}: 准确率={results[i]['best_val_accuracy']:.4f}, "
          f"学习率={results[i]['lr']:.6f}, 权重衰减={results[i]['weight_decay']:.6f}, "
          f"Dropout={results[i]['dropout_rate']:.4f}, 批量大小={results[i]['batch_size']}")
```

*注意：`train_one_epoch`和`evaluate`函数是标准的PyTorch训练组件，为简洁起见此处省略。您可以像往常一样实现它们。*

### 分析结果

运行调整循环后，`results`列表包含每个超参数 (parameter) (hyperparameter)配置的性能。简单地按验证准确率排序，就能得到搜索过程中表现最优的配置集。

将超参数与性能之间的关系可视化可以提供一些发现。例如，让我们绘制验证准确率与学习率（对数尺度）的关系图：



![验证准确率与学习率（对数尺度）](plots/5008-0.json)



> 经过10个训练轮次后，不同随机采样学习率所达到的验证准确率。在此模拟运行中，大约在$10^{-3}$到$3 \times 10^{-3}$之间的值似乎表现最佳。

可以为权重 (weight)衰减和Dropout率制作类似的图表。例如，您可能会发现，非常低或非常高的Dropout率会损害性能，或者适度的权重衰减是有益的。分析表现最佳的试验有助于您把握哪些超参数值（或范围）最有前景。

### 后续步骤和考量

- **精细搜索:** 根据初始的随机搜索，您可能会找到有前景的范围（例如，学习率介于$5e-4$和$3e-3$之间）。然后您可以在这些更窄的范围内进行第二次、更集中的搜索（无论是随机搜索还是网格搜索）。
- **更长时间的训练:** 在短期运行中找到的最佳超参数 (parameter) (hyperparameter)可能不适合完整的训练。一旦您有了一些有前景的候选，就让它们进行更多轮次的训练，并根据验证性能进行早期停止。
- **相互影响:** 请记住超参数之间存在相互影响。最优学习率可能会根据批量大小或所使用的优化器而变化。随机搜索会涉及其中一些影响。更高级的方法，如贝叶斯优化，会明确地对它们进行建模。
- **学习率调度:** 我们在此保持学习率不变。加入学习率调度（例如PyTorch中的StepLR、CosineAnnealingLR）会增加更多超参数（衰减因子、步长），但通常可以提高最终结果。您可以将调度选择或其参数包含在您的搜索范围中。

这项实践表明了改进深度学习 (deep learning)模型的基础工作流程。虽然存在自动化超参数优化工具（例如Optuna、Ray Tune），但理解手动过程能提供宝贵的直觉，以有效设置搜索范围和解读结果。尝试是必不可少的，因此请尝试将此过程应用于您自己的模型和数据集。

## 参考资料

- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Publisher: JMLR; Volume: 13; Pages: 281-305
  介绍并实证了随机搜索在超参数优化方面相对于网格搜索的有效性，这是本节使用的核心方法。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习基本概念的综合性教科书，包括优化算法、正则化技术和模型训练的实践方法。
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/) — Andrej Karpathy and Fei-Fei Li (2024)
  Publisher: Stanford University
  提供设置和训练卷积神经网络的实践指南，包括数据预处理、权重初始化、优化、正则化和超参数调优策略的讨论。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: 3rd International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了 Adam 优化算法，该算法在深度学习中被广泛使用，并在本节示例代码中明确采用。

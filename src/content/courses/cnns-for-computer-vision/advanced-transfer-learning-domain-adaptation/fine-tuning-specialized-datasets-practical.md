---
course: "cnns-for-computer-vision"
chapter: "advanced-transfer-learning-domain-adaptation"
lesson: "fine-tuning-specialized-datasets-practical"
sourceId: 2611
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-6-advanced-transfer-learning-domain-adaptation/fine-tuning-specialized-datasets-practical"
title: "动手实践：在特定数据集上微调模型"
description: "一个实操练习，涉及在特定或小型数据集上使用高级微调技术。"
order: 7
plots: ["plots/2611-0.json"]
sourceHash: "2ddb6bd2352260cfa1c1320d1457ece68c54a43f87b7d0dcddc8cf94d2fc85fa"
sourceCorrections: []
---

微调 (fine-tuning)一个在特定数据集上预训练 (pre-training)的CNN模型涉及应用高级方法，例如差异化学习率和逐步解冻。这种做法在将最初在ImageNet等大型数据集上训练的强大模型调整到数据可能有限的更专业任务时非常普遍。

假设我们有一个特定数据集，称之为“精细部件”，其中包含螺丝、螺栓和垫圈等各种工业部件的图像，这些图像被分为50个具体子类别。此数据集远小于ImageNet，并显示出不同的视觉特点（例如，金属纹理、统一背景、类别间的细微差异）。我们的目标是为这些部件构建一个准确的分类器。

### 设置预训练 (pre-training)模型

我们将从一个标准架构（如ResNet50）开始，它已在ImageNet上预训练过。大多数深度学习 (deep learning)框架都易于使用这类模型。我们假设您有一个已配置好的PyTorch或TensorFlow环境。在此，我们将使用类似PyTorch的语法进行说明。

首先，加载预训练模型。我们需要将最初为1000个ImageNet类别训练的最终分类层，替换为一个适用于我们50个“精细部件”类别的新层。

```python
import torch
import torchvision.models as models
import torch.nn as nn

# 加载预训练的ResNet50模型
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 获取分类器的输入特征数量
num_ftrs = model.fc.in_features

# 替换最后的全连接层
# 我们的数据集有50个类别
num_classes = 50
model.fc = nn.Linear(num_ftrs, num_classes)

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("模型已加载，并且最终层已替换。")
# 输出应确认模型结构变化
```

### 朴素微调 (fine-tuning)与高级方法

一种简单的方法是同时使用单个小型学习率对所有层进行微调。然而，正如之前所讨论的，这可能不是最佳选择。预训练 (pre-training)模型中的早期层通常学习普遍的特征（如边缘、纹理），这些特征普遍有用；而后期层则学习更多针对特定任务的特征。从一开始就积极更新所有层，特别是在小型或不相似的数据集上，可能会破坏早期层中有用的已学习表示。这就是高级技术发挥作用的地方。

### 策略1：差异化学习率

此技术涉及对网络的不同部分应用不同的学习率。我们通常对早期层使用较小的学习率（以保留通用特征），而对后期层使用较大的学习率（以适应特定特征和新的分类器头部）。

让我们为ResNet50模型定义参数 (parameter)组。我们可以将初始卷积块、顺序残差块（layer1, layer2, layer3, layer4）以及最终分类层进行分组。

```python
import torch.optim as optim

# 定义基础学习率和乘数
base_lr = 1e-4
lr_multiplier = 10

# 将参数按不同学习率分组
# 早期层使用较小的学习率，后期层/分类器使用较大的学习率
optimizer = optim.AdamW([
    {'params': model.conv1.parameters(), 'lr': base_lr / (lr_multiplier**4)},
    {'params': model.bn1.parameters(), 'lr': base_lr / (lr_multiplier**4)},
    {'params': model.relu.parameters(), 'lr': base_lr / (lr_multiplier**4)},
    {'params': model.maxpool.parameters(), 'lr': base_lr / (lr_multiplier**4)},

    {'params': model.layer1.parameters(), 'lr': base_lr / (lr_multiplier**3)},
    {'params': model.layer2.parameters(), 'lr': base_lr / (lr_multiplier**2)},
    {'params': model.layer3.parameters(), 'lr': base_lr / lr_multiplier},
    {'params': model.layer4.parameters(), 'lr': base_lr},

    {'params': model.avgpool.parameters(), 'lr': base_lr * lr_multiplier},
    {'params': model.fc.parameters(), 'lr': base_lr * lr_multiplier}
], lr=base_lr) # 默认学习率（如果所有参数都已分组，则不会使用）

print("优化器已配置差异化学习率。")
# 验证优化器参数组（可选）
# for group in optimizer.param_groups:
#     print(f"LR: {group['lr']}, Num Params: {sum(p.numel() for p in group['params'])}")
```

此设置将指数级递减的学习率分配给早期层，使得新添加的分类器和后期层能更快适应，同时保护预训练 (pre-training)期间学习到的基础特征。

### 策略2：逐步解冻

另一种有效策略，特别适用于较小的目标数据集，是逐步解冻。最初，我们冻结所有预训练 (pre-training)层，只训练新添加的分类器头部。一旦分类器开始学习，我们就逐步解冻网络中位于深部的层并继续训练，通常随着更多层变为可训练而降低整体学习率。

**阶段1：仅训练分类器头部**

```python
# 冻结除最终分类器外的所有层
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True

# 仅用于分类器参数的优化器
optimizer_phase1 = optim.AdamW(model.fc.parameters(), lr=base_lr * lr_multiplier)

print("阶段1：仅训练分类器头部。")
# 假设存在一个 train_model(model, optimizer, num_epochs) 函数
# train_model(model, optimizer_phase1, num_epochs=5)
```

**阶段2：解冻上层并训练**

在初始阶段之后，解冻一些后期层（例如`layer4`和`layer3`），并以较低的学习率继续训练，也可能使用差异化学习率。

```python
# 解冻 layer4 和 layer3
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True

# 使用来自 fc、layer4、layer3 的参数重新配置优化器
# 此处为简化起见，使用单一的较低学习率示例
# 之前展示的差异化方法也可在此应用
trainable_params = list(model.fc.parameters()) + \
                   list(model.layer4.parameters()) + \
                   list(model.layer3.parameters())
optimizer_phase2 = optim.AdamW(trainable_params, lr=base_lr / 10)

print("阶段2：训练分类器头部、layer4 和 layer3。")
# train_model(model, optimizer_phase2, num_epochs=10) # 继续训练
```

**后续阶段：**

您可以继续此过程，解冻更多层（例如`layer2`，`layer1`）并进一步降低学习率，直到整个网络可训练，或直到性能趋于稳定。

### 策略组合与监测

差异化学习率和逐步解冻可以结合使用。例如，在解冻一个层块后，您可以根据分类器头部和其他层块，为其分配特定的学习率。

在此过程中，仔细监测非常重要。跟踪训练和验证损失，以及准确率（或您特定任务的其他相关指标）。密切关注验证性能，以检测过拟合 (overfitting)，这在小型数据集上进行微调 (fine-tuning)时是常见的风险。第2章讨论的技术，例如数据增强、Dropout和权重 (weight)衰减，在此处尤为重要。

让我们使用“精细部件”数据集，可视化不同微调策略下验证准确率的进展。



![微调验证准确率比较](plots/2611-0.json)



> 图示为“精细部件”数据集上，朴素微调与结合了逐步解冻和差异化学习率的策略在训练轮次中验证准确率的比较。高级策略通常能带来更快的收敛和更高的最终准确率。

### 特定数据集的考虑要点

- **数据集规模：** 如果您的特定数据集非常小，激进的微调 (fine-tuning)（快速解冻多层）会增加过拟合 (overfitting)的风险。逐步解冻和强正则化 (regularization)很重要。
- **领域相似性：** 如果特定数据集在视觉上与ImageNet差异很大（例如，医疗扫描、卫星图像），可能需要更广泛的微调，甚至可能包括解冻早期层。然而，开始时请假设早期特征是有用的。
- **评估指标：** 确保您的评估指标与您特定任务的具体目标一致。准确率对于平衡分类可能已足够，但对于不平衡数据集或特定应用需求，F1分数、精确率、召回率或AUC等指标可能更合适。

本次实操练习展示了高级微调技术如何让您有效调整强大的预训练 (pre-training)模型，即使面对特定数据集样本有限或数据分布与原始预训练数据不同的挑战。请记住，最佳策略通常需要针对您的具体模型、数据集和任务进行实验和仔细监测。

## 参考资料

- [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) — Jeremy Howard, Sebastian Ruder (2018)
  Journal: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL); Volume: 1; Pages: 397-407; DOI: [10.48550/arXiv.1801.06146](https://doi.org/10.48550/arXiv.1801.06146)
  本文介绍并规范了判别性学习率和渐进式解冻等高级微调技术，这些技术在包括计算机视觉在内的各种深度学习领域中非常有效且应用广泛。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本标准教材，全面涵盖了深度学习的基础知识，包括与神经网络架构、训练和迁移学习相关的概念。
- [Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — Sasank Chilamkurthy (2017)
  PyTorch官方教程，演示了使用预训练模型进行图像分类迁移学习和微调的实际实现，可作为设置此类实验的实用指南。

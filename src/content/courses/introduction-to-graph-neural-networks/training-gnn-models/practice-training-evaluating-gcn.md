---
course: "introduction-to-graph-neural-networks"
chapter: "training-gnn-models"
lesson: "practice-training-evaluating-gcn"
sourceId: 7640
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-4-training-gnn-models/practice-training-evaluating-gcn"
title: "实践：训练与评估你的 GCN"
description: "应用本章学习的知识，编写一个完整的脚本来训练和评估之前构建的 GCN 模型。"
order: 7
plots: ["plots/7640-0.json"]
sourceHash: "0394034554b27c92c1d1500983cfc61fd00a639bd2e1e17551932401c5d9ca39"
sourceCorrections: []
---

训练循环将图神经网络 (neural network) (GNN) 的理论原理整合在一起。在这个过程中，模型通过反复处理数据、计算误差并调整参数 (parameter)来减小误差，从而完成学习。我们将提供一个完整的 Python 脚本来训练和评估 GCN 模型。我们将执行半监督节点分类任务，这是 GNN 的一种常见应用。

我们的目标是在仅给定一小部分节点标签的情况下，训练模型预测社交网络中每个节点的所属社区。这个过程直接应用了损失函数 (loss function)、反向传播 (backpropagation)和转导式 (transductive) 数据划分的方法。

### 准备模型和数据

在开始训练之前，我们需要三样东西：一个 GCN 模型、一个图数据集，以及一种划分数据进行训练和评估的方法。

假设我们已经在一个名为 `model.py` 的文件中定义了 GCN 模型（该模型在上一章中已构建）。在本次练习中，我们将使用一个两层 GCN。它接收输入特征矩阵 `X` 和归一化 (normalization)邻接矩阵 `Â`，并输出每个节点的类别 logits。

```python
# 一个简化的 GCN 模型示例
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(GCN, self).__init__()
        # 在实际实现中，这些将是标准的 GCN 层
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, adj):
        # 简化的传播规则：Â * X * W
        # 我们假设 `adj` 是归一化后的邻接矩阵
        x = torch.spmm(adj, x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

数据方面，我们将使用一个简单且著名的图：Zachary's Karate Club（扎卡里空手道俱乐部）。该数据集代表了一个拥有 34 名成员的空手道俱乐部社交网络，边代表成员间的友谊。任务是将每个成员分到四个社区中的一个。在没有内在节点特征时，我们通常使用单位矩阵作为节点特征。

训练设置中最具体的一步是创建掩码 (masks) 来分离数据。在转导式设置中，模型在训练期间可以看到整个图的结构（所有节点和边），但只能通过训练节点的标签进行学习。我们创建布尔张量 `train_mask` 和 `test_mask`，用于选择哪些节点参与损失计算，哪些节点参与性能评估。

### 训练脚本

现在开始组装完整的脚本。我们将执行以下步骤：

1. 加载并预处理数据。这包括归一化 (normalization)邻接矩阵。
2. 实例化模型、优化器（Adam 是一个不错的默认选择）和损失函数 (loss function)。由于模型的最后一层是 `log_softmax`，我们使用负对数似然损失 (`NLLLoss`)。
3. 实现主训练循环，按固定的轮数 (epochs) 进行迭代。
4. 在循环内部执行前向传播，计算训练节点上的损失，并更新模型权重 (weight)。
5. 创建一个单独的函数来评估模型在测试节点上的准确率。
6. 定期调用评估函数，以监控模型在未见数据上的表现。

以下是训练和评估 GCN 的完整代码：

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 假设 'load_karate_club_data' 是一个辅助函数，返回：
# adj: 邻接矩阵 (scipy 稀疏矩阵)
# features: 节点特征 (torch.FloatTensor)
# labels: 节点标签 (torch.LongTensor)
# train_mask, test_mask: 布尔张量 (torch.Tensor)
from utils import load_karate_club_data 

# 占位 GCN 模型类
from model import GCN

# 1. 加载并准备数据
adj, features, labels, train_mask, test_mask = load_karate_club_data()

# 模型和优化器
num_features = features.shape[1]
num_classes = labels.max().item() + 1
model = GCN(in_features=num_features, hidden_features=16, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 模型评估函数
def evaluate(model, features, adj, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

# 2. 训练循环
print("开始训练...")
for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    output = model(features, adj)

    # 仅计算训练节点上的损失
    loss_train = F.nll_loss(output[train_mask], labels[train_mask])

    # 反向传播与优化
    loss_train.backward()
    optimizer.step()

    # 每 10 轮在测试集上评估一次
    if (epoch + 1) % 10 == 0:
        acc_test = evaluate(model, features, adj, labels, test_mask)
        print(f'轮次: {epoch+1:03d}, 损失: {loss_train.item():.4f}, 测试准确率: {acc_test:.4f}')

print("训练完成。")

# 在测试集上进行最终评估
final_accuracy = evaluate(model, features, adj, labels, test_mask)
print(f"最终测试准确率: {final_accuracy:.4f}")
```

### 分析训练过程

运行脚本时，你应该会看到类似下面的输出：

```
开始训练...
轮次: 010, 损失: 1.2567, 测试准确率: 0.5882
轮次: 020, 损失: 1.0453, 测试准确率: 0.7059
轮次: 030, 损失: 0.8122, 测试准确率: 0.7647
...
轮次: 190, 损失: 0.1534, 测试准确率: 0.9412
轮次: 200, 损失: 0.1489, 测试准确率: 0.9412
训练完成。
最终测试准确率: 0.9412
```

请注意两个趋势：

- **训练损失下降**：损失值持续降低。这表明模型正在通过调整权重 (weight)来最小化标记 (token)节点上的预测误差，从而成功学习并拟合训练数据。
- **测试准确率上升**：测试集上的准确率通常会先上升然后趋于平稳。这表明模型能够将其从训练节点中学到的知识泛化到未标记的测试节点上。

我们可以将这一过程可视化，以更清晰地观察模型在各轮次中的学习动态。



![模型训练与评估进度](plots/7640-0.json)



> 随着模型拟合数据，训练损失下降，而测试准确率上升并趋于稳定，表明泛化成功。

这种可视化证实了我们的观察。模型在初始轮次学习迅速，随后对参数 (parameter)进行微调 (fine-tuning)，最终收敛到低训练损失和高测试准确率的状态。通过 `model.train()` 和 `model.eval()` 控制的 Dropout 使用，有助于防止模型简单地死记硬背训练数据，使其具有良好的泛化能力。

你现在已成功实现了一个完整的 GNN 训练和评估流程。你采用了一个模型架构，在图上进行了训练，并衡量了它对未见数据进行预测的能力。虽然为了演示原理我们从零开始实现了这些步骤，但专门的库可以使这个过程更高效。在下一章中，我们将学习如何使用 PyTorch Geometric 以更少的代码实现相同的结果。

## 参考资料

- [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf, Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR) 2017; DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍了图卷积网络及其在半监督节点分类中应用的奠基性论文。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Contributors (2024)
  为GCN模型实现和训练脚本中使用的PyTorch库组件提供了全面的文档。
- [Stanford CS224W: Machine Learning with Graphs](http://cs224w.stanford.edu/) — Jure Leskovec (2025)
  一门成熟的大学课程，涵盖图神经网络、其训练方法、半监督学习和各种图任务。
- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) — William L. Hamilton (2020)
  Journal: Synthesis Lectures on Artificial Intelligence and Machine Learning; Publisher: Morgan & Claypool Publishers; Volume: 14; Pages: 1-159
  详细讨论了图神经网络，包括GCN架构、用于节点分类的训练策略和转导学习设置。

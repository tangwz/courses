# 为节点分类任务搭建 GNN

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-4-training-gnn-models/setup-gnn-node-classification)

[返回章节目录](README.md) · [返回课程目录](../README.md)

GNN 架构是强有力的特征提取器，能够将每个节点的结构和属性信息转化为稠密的向量 (vector)嵌入 (embedding)。这些嵌入（通常表示为矩阵 $Z$）包含的高层表示，在下游任务中比原始输入特征更有用。然而，GNN 本身并不直接输出类别标签。为了执行节点分类等任务，必须在 GNN 模型中添加最后一个组件，将这些学习到的嵌入映射到类别预测上。

这最后一个组件通常被称为**分类头 (classification head)**。在大多数常见的 GNN 应用中，这只是一个以节点嵌入作为输入的标准前馈神经网络 (neural network)。在最简单且最常见的形式中，分类头是一个单一的线性层，不包含任何额外的隐藏层或非线性激活函数 (activation function)。

这个线性层的作用是充当一个可训练的分类器。它接收维度为 $d$ 的嵌入（其中 $d$ 是 GNN 编码器的输出维度），并将其投影到大小为 $C$ 的向量中（其中 $C$ 是数据集中的类别总数）。该输出向量中的每个元素代表特定类别的原始、未归一化 (normalization)的分数。这些分数通常被称为 **logits**。

节点分类的完整模型流程可以看作一个两阶段的过程：

1. **编码 (Encoding)：** 由一个或多个消息传递层组成的 GNN 编码器处理输入图 ($X$, $A$)，生成最终的节点嵌入矩阵 $Z$。$Z$ 中的每一行 $z_i$ 是节点 $i$ 的嵌入。
2. **分类 (Classification)：** 分类头（一个线性层）接收嵌入矩阵 $Z$ 并生成一个 logit 矩阵。该矩阵中的每一行包含对应节点的类别分数。

这种结构使得模型在训练期间能够同时学习图表示和分类任务。

> 节点分类的端到端架构。GNN 编码器生成嵌入，然后将其传递给简单的线性分类器以产生最终的类别 logits。

### 在代码中定义模型

让我们使用 PyTorch 将此架构转换为 Python 类。假设我们要构建一个用于分类的两层图卷积网络 (GCN)。该模型类将包含用于编码的 GCN 层和用于分类的标准 `torch.nn.Linear` 层。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNNodeClassifier(nn.Module):
    """用于节点分类的两层 GCN 模型。"""
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GCNNodeClassifier, self).__init__()

        # GNN 编码器层
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # 分类头
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # 1. GNN 编码器：获取节点嵌入
        # 第一层 GCN
        h = self.conv1(x, edge_index)
        h = F.relu(h)

        # 第二层 GCN
        h = self.conv2(h, edge_index)
        # 最终的节点嵌入现在存储在 'h' 中

        # 2. 分类头：生成 logits
        output = self.classifier(h)

        return output
```

在此实现中：

- `in_channels`：输入节点特征的维度（例如，Cora 数据集为 1433）。
- `hidden_channels`：GNN 层产生的节点嵌入 (embedding)维度。这是一个可以调整的超参数 (parameter) (hyperparameter)。
- `num_classes`：数据集中不同节点标签的数量（例如，Cora 数据集为 7）。
- `forward(self, x, edge_index)`：此方法定义了计算流程。输入节点特征 `x` 和图结构 `edge_index` 通过两个 GCN 层，中间应用了 ReLU 激活函数 (activation function)。得到的嵌入 `h` 随后传递给 `self.classifier` 层以获得最终的 logits。

### 半监督设置

许多节点分类任务的一个显著特点是它们在**半监督**（或更准确地说，**转导式 (transductive)**）环境下运行。这意味着虽然我们只有一小部分节点的标签（训练集），但 GNN 编码器会使用*整个图结构*（包括所有节点和边）来生成嵌入 (embedding)。

> 在转导式设置中，GNN 模型在训练期间可以访问图中所有节点的特征和连接关系，甚至是验证集和测试集中的节点。模型的任务是为这个已知图中的无标签节点预测标签。

我们上面定义的 `GCNNodeClassifier` 就是为此设计的。它的 `forward` 方法为图中的每一个节点计算嵌入和 logits。在下一节讨论损失函数 (loss function)时，我们将看到如何使用掩码来确保模型的误差仅根据有标签训练节点的预测结果来计算。

有了这个端到端的模型结构，接下来的任务是定义一个目标函数来衡量其表现并指导其学习过程。这就引出了损失函数的话题。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR 2017); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍了图卷积网络 (GCN) 的基础论文，GCN 是示例代码和所描述的半监督节点分类设置的核心。
- [Graph Representation Learning](https://www.morganclaypool.com/doi/10.2200/S01045ED1V01Y202009AIM046) — William L. Hamilton (2020)
  Publisher: Morgan & Claypool Publishers; DOI: [10.2200/S01045ED1V01Y202009AIM046](https://doi.org/10.2200/S01045ED1V01Y202009AIM046)
  一本权威教材，涵盖图表示学习的基本概念，包括 GNN 及其在节点分类中的应用。
- [A Comprehensive Survey on Graph Neural Networks](https://ieeexplore.ieee.org/document/9046288) — Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Jing Jiang, and Chengqi Zhang (2021)
  Journal: IEEE Transactions on Neural Networks and Learning Systems; Publisher: IEEE; Volume: 32; Pages: 4-24; DOI: [10.1109/TNNLS.2020.2977118](https://doi.org/10.1109/TNNLS.2020.2977118)
  一份全面的综述，概述了各种 GNN 架构及其应用，有助于理解 GNN 在节点分类中的更广泛背景。

---

[上一节](../03-GNN%20%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20GCN.md) · [下一节](02-%E5%9B%BE%E4%BB%BB%E5%8A%A1%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)

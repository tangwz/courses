---
course: "graph-neural-networks-gnns"
chapter: "advanced-gnn-tasks-techniques"
lesson: "practice-heterogeneous-graph-classification"
sourceId: 2740
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-4-advanced-gnn-tasks-techniques/practice-heterogeneous-graph-classification"
title: "动手实践：异构图节点分类"
description: "在异构图数据集上构建和训练GNN模型，用于节点分类任务。"
order: 7
plots: []
sourceHash: "355a339526193e528cea6ea11640bb23612a42c681746a82692a12c21513f4aa"
sourceCorrections: []
---

构建并训练一个专门用于异构图节点分类的图神经网络 (neural network)在此呈现，其中特别侧重于处理多种节点和边类型。一个异构注意力网络 (HAN) 模型使用PyTorch Geometric (PyG) 实现，并使用DBLP计算机科学文献数据集。本实践练习假定您已安装PyTorch和PyG，并且熟悉它们的基本用法。

### 目标

我们的目标是根据DBLP数据集中作者的发表历史、合著网络以及与其论文相关的术语，预测他们的研究领域（例如，数据库、数据挖掘、人工智能）。

### PyG中的DBLP数据集

首先，我们加载PyG提供的DBLP数据集。该数据集自然地表示一个异构图。

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HANConv, Linear
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# 加载数据集
dataset = DBLP(root='./data/DBLP')
data = dataset[0]

# 应用归一化和转换
# 为自循环添加单位矩阵以提高数值稳定性，然后归一化特征
# 注意：此转换可能会就地修改数据对象，或根据PyG版本返回一个新对象。
# 为了清晰起见，我们显式地将其赋值回data。
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToUndirected() # 确保图是无向的，以便简化关系处理
])
data = transform(data)

print("DBLP数据集概览:")
print(data)
print("\n节点类型:", data.node_types)
print("边类型:", data.edge_types)

# 示例：访问作者特征和标签
print("\n作者节点特征形状:", data['author'].x.shape)
print("作者节点标签形状:", data['author'].y.shape)
print("类别数量:", dataset.num_classes)
```

您会注意到 `HeteroData` 对象很好地组织了按各自类型划分的特征、标签和边索引。对于DBLP，我们通常有 'author'（作者）、'paper'（论文）、'term'（术语）和 'venue'（会议/期刊）等节点类型，以及表示关系的边类型，例如 'author' 撰写 'paper'，'paper' 引用 'paper'，'paper' 使用 'term' 等。具体结构可能因PyG版本和数据集预处理而略有不同。我们的任务侧重于对 'author' 节点进行分类。

### 图模式的可视化

了解不同节点类型之间的关系很重要。我们可以将异构图的模式可视化。

> DBLP异构图的模式，显示了不同的节点类型（作者、论文、术语、会议/期刊）以及它们之间的关系（边类型）。PyG中的`ToUndirected()`通常会创建反向边类型。

### 定义异构注意力网络 (HAN) 模型

HAN架构非常适合异构图。它在两个层级使用注意力机制 (attention mechanism)：

1. **节点层注意力**：学习特定元路径中邻居的重要性。
2. **语义层注意力**：学习不同元路径本身对于特定任务的重要性。

元路径是由边类型连接的节点类型序列，例如 作者$\rightarrow$ 论文$\rightarrow$ 作者 (APA) 或 作者$\rightarrow$ 论文$\rightarrow$ 会议/期刊$\rightarrow$ 论文$\rightarrow$ 作者 (APVPA)。

PyG的 `HANConv` 层简化了实现此功能。它需要指定目标节点类型和与其相关的元路径。

```python
class HAN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata, heads=8):
        """
        初始化HAN模型。

        参数:
            in_channels (int 或 dict): 输入特征的大小。如果为整数，则假定所有节点类型
                                        具有相同的特征大小。如果为字典，则将节点类型
                                        映射到其特征大小。
            hidden_channels (int): 隐藏嵌入的大小。
            out_channels (int): 输出类别数量。
            metadata (tuple): 包含节点类型和边类型的元数据元组，
                              从data.metadata()获得。
            heads (int, optional): 注意力头的数量。默认为8。
        """
        super().__init__()
        # HANConv 层根据元数据自动处理多个元路径
        # 它执行节点层和语义层注意力。
        # 我们将 in_channels 指定为 '-1'，让 HANConv 从元数据和输入数据中推断每种节点类型的输入大小。
        self.conv1 = HANConv(in_channels=-1, out_channels=hidden_channels,
                             metadata=metadata, heads=heads, dropout=0.6)
        self.conv2 = HANConv(in_channels=hidden_channels, out_channels=out_channels,
                             metadata=metadata, heads=1, dropout=0.6) # 最后一层通常使用1个注意力头

    def forward(self, x_dict, edge_index_dict):
        """
        HAN模型的前向传播。

        参数:
            x_dict (dict): 将节点类型映射到其特征张量的字典。
            edge_index_dict (dict): 将边类型映射到其边索引张量的字典。

        返回:
            torch.Tensor: 目标节点类型（'author'）的输出 logits。
        """
        # 注意：HANConv 返回一个字典，其中包含通过源节点定义的元路径可达的所有节点类型的嵌入。
        x_dict = self.conv1(x_dict, edge_index_dict)
        # 应用激活函数（可选，取决于层的实现细节）
        # x_dict = {key: F.elu(x) for key, x in x_dict.items()} # 激活函数示例

        x_dict = self.conv2(x_dict, edge_index_dict)

        # 我们的分类任务只需要 'author' 节点类型的输出
        return x_dict['author']

# 准备模型参数
metadata = data.metadata()
hidden_channels = 128
num_classes = dataset.num_classes

# 实例化模型
# PyG 的 HANConv 在设置为 -1 时可以推断输入通道
model = HAN(in_channels=-1, hidden_channels=hidden_channels,
            out_channels=num_classes, metadata=metadata, heads=8)

print("\nHAN模型架构:")
print(model)

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device) # 将数据对象移动到相同设备
```

我们定义了一个两层HAN模型。`HANConv` 方便地接受整个`HeteroData`的特征字典（`x_dict`）和边索引字典（`edge_index_dict`）作为输入。它根据提供的 `metadata` 在内部确定元路径，并计算注意力加权的表示。请注意，我们将 `in_channels` 指定为 `-1`，以便 `HANConv` 自动推断每种节点类型的输入特征维度。我们关注的输出是 'author' 节点的嵌入 (embedding)，我们将用它进行分类。

### 训练与评估循环

现在，我们来设置标准的PyTorch训练组件。我们将使用Adam优化器和交叉熵损失。我们需要掩码来选择训练、验证和测试节点（特别是针对 'author' 类型）。这些掩码通常随数据集提供。

```python
# 检查标准掩码是否可用，否则创建随机划分
if 'train_mask' not in data['author']:
    print("\n正在为作者节点生成随机掩码...")
    num_authors = data['author'].num_nodes
    indices = torch.randperm(num_authors)
    train_split = int(0.6 * num_authors)
    val_split = int(0.8 * num_authors)

    data['author'].train_mask = torch.zeros(num_authors, dtype=torch.bool)
    data['author'].train_mask[indices[:train_split]] = True
    data['author'].val_mask = torch.zeros(num_authors, dtype=torch.bool)
    data['author'].val_mask[indices[train_split:val_split]] = True
    data['author'].test_mask = torch.zeros(num_authors, dtype=torch.bool)
    data['author'].test_mask[indices[val_split:]] = True

# 确保掩码在正确的设备上
data['author'].train_mask = data['author'].train_mask.to(device)
data['author'].val_mask = data['author'].val_mask.to(device)
data['author'].test_mask = data['author'].test_mask.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    # 传递整个字典
    out = model(data.x_dict, data.edge_index_dict)
    # 仅计算 'author' 类型训练节点的损失
    mask = data['author'].train_mask
    loss = criterion(out[mask], data['author'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test():
    model.eval()
    # 传递整个字典
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)
    accs = []
    # 计算 'author' 节点在训练集、验证集和测试集上的准确率
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['author'][split]
        acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs

print("\n开始训练...")
for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, ',
              f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

print("训练完成。")

# 最终测试准确率
final_train_acc, final_val_acc, final_test_acc = test()
print(f"\n最终表现:\n"
      f"  训练准确率: {final_train_acc:.4f}\n"
      f"  验证准确率: {final_val_acc:.4f}\n"
      f"  测试准确率: {final_test_acc:.4f}")
```

在训练循环中，我们将完整的 `x_dict` 和 `edge_index_dict` 传递给模型。损失和准确率计算仅在相关的 'author' 节点上使用提供的掩码执行。

### 运行与分析

执行代码。您会看到训练损失减少，准确率普遍随 epoch 增加。最终测试准确率表明了HAN模型对未见过的作者节点的泛化能力如何。

性能取决于诸如选择的超参数 (parameter) (hyperparameter)（隐藏维度、学习率、注意力头数量）、节点特征的质量，以及由 `HANConv` 根据图的 `metadata` 隐式考虑的元路径的表达能力等因素。

本实践示例演示了如何：

1. 使用PyG的 `HeteroData` 加载并理解异构图数据集。
2. 定义并实现合适的GNN架构 (HAN)，使用 `HANConv` 显式处理异构信息。
3. 训练并评估模型，以执行特定节点类型（'author'）上的节点分类任务。

您可以通过以下方式进行进一步尝试：

- 调整超参数（`hidden_channels`、`lr`、`heads`、`dropout`）。
- 尝试其他异构GNN层，例如 `HeteroConv`，并为每种关系类型使用特定的聚合函数（例如，`GATConv`、`GCNConv`）。
- 如果使用的层需要，显式定义元路径。
- 应用课程中讨论的其他技术，例如不同的归一化 (normalization)或正则化 (regularization)策略。

“处理异构图是应用中的常见要求，HAN之类的架构提供了一种有效的方法来建模这些复杂的关联结构。”

## 参考资料

- [Heterogeneous Graph Attention Network](https://doi.org/10.1145/3308558.3313562) — Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Peng Cui, Philip S. Yu, Yanfang Ye (2019)
  Journal: The World Wide Web Conference; Publisher: ACM; Pages: 2022–2032; DOI: [10.1145/3308558.3313562](https://doi.org/10.1145/3308558.3313562)
  介绍了异构图注意力网络（HAN）模型，这是本实践练习中用于异构节点分类的主要架构。
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/) — Matthias Fey, Jan E. Lenssen (2023)
  PyTorch Geometric 的官方指南，该库在本实践中用于实现图神经网络和管理异构图数据集。

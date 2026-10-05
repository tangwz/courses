# PyG 中的完整训练脚本

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric/complete-training-script-pyg)

[返回章节目录](README.md) · [返回课程目录](../README.md)

利用 PyTorch Geometric 组件（如 `Data` 对象和模块化 GNN 层），可以构建一个完整的图神经网络 (neural network)（GNN）训练流水线。我们编写了一个可重复使用的脚本，用于在节点分类任务上训练 GNN。该结构提供了一个标准模式，适用于许多基于图的项目。

我们将逐步讲解脚本的每个逻辑块：加载数据、定义模型、创建训练和评估循环，最后协调整个流程。

### PyG 训练脚本剖析

PyTorch Geometric 中的典型训练脚本遵循一个明确的结构，集成了数据处理、模型定义和训练过程。这种关注点分离使代码整洁、易读且易于修改。

> 标准 PyTorch Geometric 训练脚本的工作流，从数据加载到模型评估。

### 1. 导入库与加载数据

首先，我们导入必要的库。这包括用于核心张量操作的 `torch`，用于激活函数 (activation function)和损失函数 (loss function)的 `torch.nn.functional`，以及 `torch_geometric` 中的多个组件。我们将使用 `Planetoid` 数据集集合，其中包括 Cora、CiteSeer 和 PubMed 基准数据集。

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 加载 Cora 数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f'数据集: {dataset.name}')
print(f'图的数量: {len(dataset)}')
print(f'节点数量: {data.num_nodes}')
print(f'边数量: {data.num_edges}')
print(f'特征维度: {dataset.num_node_features}')
print(f'类别数量: {dataset.num_classes}')
```

`Planetoid` 对象会下载数据并将其处理为 PyG 可以使用的格式。由于 Cora 是单张图，数据集仅包含一个 `Data` 对象，我们可以通过 `dataset[0]` 访问它。这个 `data` 对象不仅包含节点特征 (`data.x`) 和图连接关系 (`data.edge_index`)，还包含真实标签 (`data.y`) 以及用于半监督学习 (supervised learning) (semi-supervised learning)的预定义掩码 (`data.train_mask`、`data.val_mask`、`data.test_mask`)。

### 2. 定义 GNN 模型

接下来，我们定义网络架构。我们创建一个继承自 `torch.nn.Module` 的类，这是在 PyTorch 中构建模型的标准方式。这使我们能够像使用其他 PyTorch 层一样使用 PyG 层。

在这里，我们将定义一个简单的双层 GCN。第一层 `GCNConv` 将输入节点特征映射到隐藏维度，第二层将隐藏表示映射到输出类别数。

```python
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        # 第一层 GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层 GCN
        x = self.conv2(x, edge_index)

        # 对 NLLLoss 函数应用 LogSoftmax
        return F.log_softmax(x, dim=1)
```

`forward` 方法定义了数据在网络中的流动方式。它接收节点特征矩阵 `x` 和 `edge_index` 作为输入。请注意，在第一层之后使用了 ReLU 激活函数 (activation function)和 Dropout 进行正则化 (regularization)。最终输出通过 `log_softmax` 函数处理，该函数通常与负对数似然损失 (`F.nll_loss`) 配对用于分类任务。

### 3. 训练循环

训练循环包含优化模型参数 (parameter)的逻辑。我们定义一个执行单次训练迭代的函数。

```python
def train(model, optimizer, data):
    model.train()  # 将模型设置为训练模式
    optimizer.zero_grad()  # 清除梯度

    # 执行单次前向传播
    out = model(data.x, data.edge_index)

    # 仅使用训练节点计算损失
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    # 反向传播推导梯度
    loss.backward()

    # 更新参数
    optimizer.step()

    return loss.item()
```

该函数封装了标准的 PyTorch 训练步骤。对于 GNN 而言，最重要的是损失的计算方式。我们不是使用所有节点的输出，而是应用 `data.train_mask` 仅选择训练集中节点的 Logits 和标签。这是半监督节点分类的核心。

### 4. 评估函数

类似地，我们创建一个函数来评估模型在验证集或测试集上的表现。此函数不应更新模型权重 (weight)，因此我们在 `torch.no_grad()` 下操作。

```python
@torch.no_grad()
def test(model, data):
    model.eval()  # 将模型设置为评估模式
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # 使用概率最高的类别

    # 检查验证集和测试集
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]  # 对比真实标签
        accs.append(int(correct.sum()) / int(mask.sum())) # 计算准确率
    return accs
```

在 `test()` 中，我们将模型设置为 `eval()` 模式，这会禁用 Dropout 等层。我们通过对输出 Logits 取 `argmax` 来获得所有节点的预测结果。然后，通过应用相应的掩码，计算每个数据拆分（训练、验证和测试）的准确率。

### 5. 整合整体流程

最后，我们编写主脚本逻辑来初始化所有组件并运行训练过程。我们实例化模型，定义 Adam 等优化器，然后进行指定轮数（epochs）的循环。在每一轮中，我们调用 `train` 和 `test` 函数并记录结果。

```python
# 实例化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 执行训练
best_val_acc = 0
for epoch in range(1, 201):
    loss = train(model, optimizer, data)
    train_acc, val_acc, test_acc = test(model, data)

    # 简单的早停法
    if val_acc > best_val_acc:
        best_val_acc = val_acc

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}, 测试准确率: {test_acc:.4f}')
```

这个主代码块驱动了学习过程。通过跟踪验证准确率，您可以实现简单的早停法或保存验证集表现最好的模型。经过 200 轮训练后，在小型 Cora 数据集上训练的模型通常会在测试节点上获得较高的准确率。



[交互图表：GNN 训练进度](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric/complete-training-script-pyg#plot-2e0yzc)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "Epoch (轮次)"
    },
    "yaxis": {
      "title": "数值"
    },
    "title": "GNN 训练进度",
    "legend": {
      "traceorder": "normal"
    }
  },
  "data": [
    {
      "type": "scatter",
      "name": "损失",
      "x": [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200
      ],
      "y": [
        1.75,
        1.43,
        1.15,
        0.92,
        0.76,
        0.65,
        0.58,
        0.53,
        0.49,
        0.46,
        0.44,
        0.42,
        0.41,
        0.4,
        0.39,
        0.38,
        0.38,
        0.37,
        0.37,
        0.36
      ],
      "line": {
        "color": "#f03e3e"
      }
    },
    {
      "type": "scatter",
      "name": "验证准确率",
      "x": [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200
      ],
      "y": [
        0.55,
        0.68,
        0.74,
        0.77,
        0.78,
        0.79,
        0.79,
        0.8,
        0.8,
        0.81,
        0.81,
        0.81,
        0.81,
        0.81,
        0.81,
        0.81,
        0.82,
        0.82,
        0.82,
        0.82
      ],
      "yaxis": "y2",
      "line": {
        "color": "#228be6"
      }
    }
  ],
  "frames": [],
  "layout_yaxis2": {
    "title": "准确率",
    "overlaying": "y",
    "side": "right"
  }
}
```

</details>



> GCN 在 Cora 数据集上的典型训练曲线。训练损失下降，而验证准确率上升并趋于稳定。

有了这个完整的脚本，您就拥有了一个强大且灵活的 GNN 训练模版。最后的动手操作环节将要求您应用此结构，从头开始在 Cora 数据集上训练和评估模型。

---

[上一节](05-%E9%92%88%E5%AF%B9%E5%A4%A7%E8%A7%84%E6%A8%A1%E5%9B%BE%E6%95%B0%E6%8D%AE%E7%9A%84%E5%88%86%E6%89%B9%E5%A4%84%E7%90%86.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20PyG%20%E5%9C%A8%20Cora%20%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E8%BF%9B%E8%A1%8C%E8%8A%82%E7%82%B9%E5%88%86%E7%B1%BB.md)

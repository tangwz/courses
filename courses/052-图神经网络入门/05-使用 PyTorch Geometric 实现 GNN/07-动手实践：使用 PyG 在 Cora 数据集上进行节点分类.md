# 动手实践：使用 PyG 在 Cora 数据集上进行节点分类

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric/hands-on-cora-dataset-pyg)

[返回章节目录](README.md) · [返回课程目录](../README.md)

PyTorch Geometric 的一个实际演示包括构建、训练和评估一个用于 Cora 数据集上半监督节点分类的图神经网络 (neural network)。本次练习使用 Cora 数据集（图机器学习 (machine learning)中的标准基准数据集），涵盖了 PyG 框架内的数据处理、模型定义和训练循环。

Cora 数据集是一个引用网络。图中的每个节点代表一篇科学论文，从节点 A 到节点 B 的有向边表示论文 A 引用了论文 B。每篇论文都由一个二进制词向量 (vector)描述，该向量表示固定词典中单词的存在与否，并作为节点特征。任务是将每篇论文分类到七个预定义的学术主题之一。

### 加载并检查 Cora 数据集

PyTorch Geometric 包含了一系列常用的基准数据集，包括 Cora，只需一行代码即可加载。让我们从导入必要的库和加载数据集开始。

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 加载 Cora 数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# 打印数据集信息
print(f'数据集: {dataset}:')
print('======================')
print(f'图的数量: {len(dataset)}')
print(f'特征数量: {dataset.num_features}')
print(f'类别数量: {dataset.num_classes}')

# 打印图的信息
print(f'\n图:')
print('------')
print(f'节点数量: {data.num_nodes}')
print(f'边数量: {data.num_edges}')
print(f'训练节点数: {data.train_mask.sum()}')
print(f'验证节点数: {data.val_mask.sum()}')
print(f'测试节点数: {data.test_mask.sum()}')
```

运行这段代码将下载数据集并打印摘要。你会注意到 `dataset` 对象包含一个单独的图，我们通过 `dataset[0]` 访问它。这个 `data` 对象保存了所有的图信息：

- `data.x`: 形状为 `[num_nodes, num_features]` 的节点特征矩阵。对于 Cora，它是 `[2708, 1433]`。
- `data.edge_index`: 形状为 `[2, num_edges]` 的坐标格式 (COO) 图连接信息。
- `data.y`: 每个节点的真实标签。
- `data.train_mask`, `data.val_mask`, `data.test_mask`: 布尔掩码，用于标识哪些节点用于训练、验证和测试。这种预定义的划分是半监督任务的典型特征，即我们在小部分有标签节点上进行训练。

### 定义图卷积网络

接下来，我们定义 GNN 架构。我们将使用一个简单的两层图卷积网络 (GCN)。第一个 `GCNConv` 层将输入特征映射到低维隐藏表示（例如 16 维）。第二个 `GCNConv` 层将隐藏表示映射到最终的类别数量（Cora 为 7 个）。

我们将模型定义为一个继承自 `torch.nn.Module` 的类，这是标准的 PyTorch 惯例。

```python
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        # 第一层 GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层 GCN
        x = self.conv2(x, edge_index)

        return x

model = GCN(dataset.num_features, dataset.num_classes)
print(model)
```

`forward` 方法指定了每次调用时的计算过程。它应用第一次卷积，然后是 ReLU 激活函数 (activation function)和用于正则化 (regularization)的 dropout 层。第一阶段的输出随后传递给第二个卷积层。最终输出为图中每个节点的 7 个类别提供原始 logits。

### 训练 GNN 模型

数据和模型准备就绪后，我们可以编写训练循环。我们将使用 Adam 优化器和交叉熵损失函数 (loss function)。在此数据集上进行训练的一个显著方面是，我们只计算训练节点上的损失，如 `data.train_mask` 所示。模型在训练期间会查看整个图结构（所有 `edge_index`），但仅使用训练节点的标签来计算梯度。

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    # 执行单次前向传播
    out = model(data.x, data.edge_index)
    # 仅在训练节点上计算损失
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    # 导出梯度
    loss.backward()
    # 更新参数
    optimizer.step()
    return loss

for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
```

训练函数封装了标准的 PyTorch 训练步骤。我们将模型设置为训练模式，清除梯度，执行前向传播，计算训练集上的损失，进行反向传播 (backpropagation)并更新模型权重 (weight)。我们运行此过程 200 个 epoch，每 20 个 epoch 打印一次损失以监控进度。

### 评估模型性能

训练完成后，我们需要在测试集上评估模型的表现。我们创建一个单独的函数用于评估。该函数将模型设置为评估模式 (`model.eval()`) 以停用 dropout 等层。然后，它计算所有节点的预测结果，并将其与测试集中节点的真实标签进行比较。

```python
@torch.no_grad()
def test():
    model.eval()
    # 在整个图上进行前向传播
    out = model(data.x, data.edge_index)
    # 获取预测的类别索引
    pred = out.argmax(dim=1)

    # 计算测试集上的准确率
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

# 训练并打印最终测试准确率
for epoch in range(1, 201):
    train()

final_accuracy = test()
print(f'最终测试准确率: {final_accuracy:.4f}')
```

经过 200 个 epoch 的训练后，你应该会看到 80-82% 左右的测试准确率。这个结果相当不错，特别是考虑到模型在 2708 个节点中仅使用了 140 个有标签节点进行训练。这说明了 GNN 在利用图结构将标签信息从少数节点传播到多数节点方面的效用。

### 可视化节点嵌入 (embedding)

GNN 能够为每个节点学习产生强大的嵌入，这些嵌入同时捕捉了节点特征及其局部邻域。我们可以通过可视化这些嵌入来直观了解模型学到的内容。我们第一层 `GCNConv` 的输出是每个节点的 16 维嵌入。我们可以使用 t-SNE 等降维技术将这些嵌入投影到 2D 空间并进行绘制。训练良好的模型生成的嵌入中，相同类别的节点应形成明显的簇。

让我们从训练好的模型中提取嵌入并进行可视化。

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 获取 16 维节点嵌入
model.eval()
with torch.no_grad():
    h = model.conv1(data.x, data.edge_index)
    h = F.relu(h)

# 应用 t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(h.cpu().numpy())

# 绘制结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=data.y.cpu().numpy(), cmap='jet', alpha=0.7)
plt.legend(handles=scatter.legend_elements()[0], labels=dataset.classes)
plt.title('Cora 节点嵌入的 t-SNE 可视化')
plt.xlabel('t-SNE 特征 1')
plt.ylabel('t-SNE 特征 2')
plt.show()
```

为了避免运行代码，下方显示了输出结果可能样式的预生成图表。



[交互图表：Cora 节点嵌入的 t-SNE 可视化](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric/hands-on-cora-dataset-pyg#plot-2pt7vo)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "Cora 节点嵌入的 t-SNE 可视化",
    "xaxis": {
      "title": "t-SNE 特征 1",
      "showgrid": false,
      "zeroline": false,
      "visible": false
    },
    "yaxis": {
      "title": "t-SNE 特征 2",
      "showgrid": false,
      "zeroline": false,
      "visible": false
    },
    "legend": {
      "title": {
        "text": "论文主题"
      }
    },
    "showlegend": true,
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF"
  },
  "data": [
    {
      "x": [
        1.2,
        1.5,
        1.8,
        -5.5,
        -5.8,
        -6.1,
        8.1,
        8.5,
        8.2,
        -8.2,
        -8.5,
        -8.9,
        4.2,
        4.5,
        4.9,
        -2.1,
        -2.5,
        -2.9,
        0.1,
        0.5,
        0.9
      ],
      "y": [
        8.1,
        8.5,
        8.9,
        -4.2,
        -4.5,
        -4.9,
        5.2,
        5.5,
        5.9,
        -1.1,
        -1.5,
        -1.9,
        0.1,
        0.5,
        0.9,
        6.1,
        6.5,
        6.9,
        -7.2,
        -7.5,
        -7.9
      ],
      "mode": "markers",
      "type": "scatter",
      "marker": {
        "color": [
          0,
          0,
          0,
          1,
          1,
          1,
          2,
          2,
          2,
          3,
          3,
          3,
          4,
          4,
          4,
          5,
          5,
          5,
          6,
          6,
          6
        ],
        "colorscale": [
          [
            "0.0",
            "#7048e8"
          ],
          [
            "0.16",
            "#be4bdb"
          ],
          [
            "0.33",
            "#f06595"
          ],
          [
            "0.5",
            "#f76707"
          ],
          [
            "0.66",
            "#f59f00"
          ],
          [
            "0.83",
            "#37b24d"
          ],
          [
            "1.0",
            "#1c7ed6"
          ]
        ],
        "showscale": false,
        "symbol": "circle",
        "size": 8,
        "opacity": 0.8
      },
      "name": "类别 0",
      "showlegend": false,
      "hoverinfo": "none"
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        1.5
      ],
      "y": [
        8.5
      ],
      "name": "神经网络",
      "marker": {
        "color": "#7048e8",
        "symbol": "circle",
        "size": 10
      }
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        -5.8
      ],
      "y": [
        -4.5
      ],
      "name": "概率方法",
      "marker": {
        "color": "#be4bdb",
        "symbol": "circle",
        "size": 10
      }
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        8.2
      ],
      "y": [
        5.5
      ],
      "name": "遗传算法",
      "marker": {
        "color": "#f06595",
        "symbol": "circle",
        "size": 10
      }
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        -8.5
      ],
      "y": [
        -1.5
      ],
      "name": "强化学习",
      "marker": {
        "color": "#f76707",
        "symbol": "circle",
        "size": 10
      }
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        4.5
      ],
      "y": [
        0.5
      ],
      "name": "理论",
      "marker": {
        "color": "#f59f00",
        "symbol": "circle",
        "size": 10
      }
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        -2.5
      ],
      "y": [
        6.5
      ],
      "name": "规则学习",
      "marker": {
        "color": "#37b24d",
        "symbol": "circle",
        "size": 10
      }
    },
    {
      "type": "scatter",
      "mode": "markers",
      "x": [
        0.5
      ],
      "y": [
        -7.5
      ],
      "name": "基于案例",
      "marker": {
        "color": "#1c7ed6",
        "symbol": "circle",
        "size": 10
      }
    }
  ]
}
```

</details>



> t-SNE 图显示了投影到 2D 平面上的 16 维节点嵌入。每个点代表一篇科学论文，颜色由其学科类别区分。颜色的清晰聚集表明 GNN 已学会在嵌入空间中将具有相似主题的论文分在一组。

可视化结果印证了我们的定量结果。模型成功学习到了能区分不同文档类别的表示，这正是此场景下表示学习的目标。从加载数据到训练模型再到检查输出，这套完整的流程可以作为使用 PyTorch Geometric 解决其他图机器学习 (machine learning)问题的模板。

## 参考资料

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/) — PyTorch Geometric Developers (2023)
  PyTorch Geometric 的官方文档，提供构建和训练 GNN 的全面指南和 API 参考。
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍了图卷积网络 (GCN) 模型（本实践部分使用的核心架构），并展示了其在 Cora 等数据集上半监督节点分类任务中的有效性。

---

[上一节](06-PyG%20%E4%B8%AD%E7%9A%84%E5%AE%8C%E6%95%B4%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC.md)

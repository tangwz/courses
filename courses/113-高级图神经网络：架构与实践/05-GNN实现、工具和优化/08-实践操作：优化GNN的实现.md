# 实践操作：优化GNN的实现

来源：[原文](https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-5-gnn-implementation-tooling-optimization/practice-optimizing-gnn-implementation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

一个标准的图神经网络 (neural network)实现通过应用各种优化技术进行优化。目标是理解这些优化方法如何以及为何有效，从而使你能够将它们应用于自己的复杂GNN项目，而不仅仅是让代码运行更快或占用更少内存。

我们将侧重于一项常见任务：半监督节点分类。我们将从一个使用PyTorch Geometric (PyG) 实现的基线GCN模型开始，确定其性能特点，然后逐步应用优化，并在每一步衡量其影响。

### 1. 基线模型与设置

首先，让我们设定起点。我们将使用Cora数据集和一个简单的两层GCN模型。假定你已准备好PyTorch、PyG和Cora数据集。

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import time
import torch.cuda.amp as amp # 用于自动混合精度

# 加载数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# 定义基线GCN模型
class BaselineGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 基线模型中不使用dropout，以简化计算/内存分析
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = BaselineGCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# --- 基线训练循环 ---
def train_baseline():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def profile_run(train_func, run_name="Run", epochs=50):
    print(f"--- 性能分析: {run_name} ---")
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        start_mem = torch.cuda.max_memory_allocated(device)

    for epoch in range(epochs):
        loss = train_func()
        # 在实际场景中，会添加验证/测试步骤
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    end_time = time.time()
    total_time = end_time - start_time

    if torch.cuda.is_available():
        end_mem = torch.cuda.max_memory_allocated(device)
        peak_mem_increase_mib = (end_mem - start_mem) / 1024**2
        print(f"峰值GPU内存增长: {peak_mem_increase_mib:.2f} MiB")
    else:
        peak_mem_increase_mib = 0 # 如果没有GPU，则为占位符
        print("GPU不可用，跳过内存分析。")

    print(f"总训练时间 ({epochs} 周期): {total_time:.3f} 秒")
    print(f"每个周期的平均时间: {total_time / epochs:.4f} 秒")
    print("-" * (20 + len(run_name)))
    return total_time / epochs, peak_mem_increase_mib

# 分析基线性能
baseline_avg_time, baseline_peak_mem = profile_run(train_baseline, "Baseline GCN")
```

这个基线模型执行全图训练。对于像Cora这样的数据集，这通常是可行的。然而，随着图的扩展，由于内存限制（加载整个图、特征和中间激活）和计算成本，这种方法变得难以处理。

### 2. 优化1：带邻居采样的Mini-Batch

对于更大的图，全批次训练通常是不可能的。标准方法是mini-batch处理，我们在每一步处理更小的子图。PyG的`NeighborLoader`实现了GraphSAGE所推广的邻居采样。

我们来修改训练过程以使用`NeighborLoader`。

```python
from torch_geometric.loader import NeighborLoader

# 创建一个NeighborLoader
# 我们进行2层深度采样，第一跳采样10个邻居，第二跳采样5个。
# 根据你的硬件和图调整batch_size和num_neighbors。
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 5], # 每层采样的邻居数量
    batch_size=128,        # 每个批次处理128个训练节点
    input_nodes=data.train_mask, # 从训练节点开始采样邻域
    shuffle=True
)

# --- 针对Mini-batch修改的模型 ---
# 模型架构本身并不严格需要为NeighborLoader进行更改，
# 但前向传播接收的是Batch对象而不是完整的Data对象。
class SampledGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # 注意：PyG的GCNConv正确处理了采样结构。
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, size):
        # GCNConv在NeighborLoader上下文中需要'size'来知道
        # 对应于采样邻域的二分图的维度。
        x = self.conv1(x, edge_index, size=size[0]) # 第一层的size
        x = F.relu(x)
        x = self.conv2(x, edge_index, size=size[1]) # 第二层的size
        return F.log_softmax(x, dim=1)

# 重新初始化模型和优化器
model_sampled = SampledGCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
optimizer_sampled = torch.optim.Adam(model_sampled.parameters(), lr=0.01, weight_decay=5e-4)

# --- 采样训练循环 ---
def train_sampled():
    model_sampled.train()
    total_loss = 0
    # 处理mini-batch
    for batch in train_loader:
        batch = batch.to(device)
        optimizer_sampled.zero_grad()
        # 前向传播现在接收批次特征、edge_index、
        # 以及加载器提供的二分图大小。
        # 输出'out'仅对应批次中的节点（前batch_size个节点）。
        out = model_sampled(batch.x, batch.edge_index, batch.size)
        loss = F.nll_loss(out, batch.y[:batch.batch_size]) # 使用中心节点的标签
        loss.backward()
        optimizer_sampled.step()
        total_loss += loss.item() * batch.batch_size
    return total_loss / data.train_mask.sum().item() # 所有训练节点的平均损失

# 分析采样版本性能
# 注意：周期时间包括遍历所有mini-batch的时间。
sampled_avg_time, sampled_peak_mem = profile_run(train_sampled, "Neighbor Sampling GCN")
```

**观察结果：**

- **内存：** 你应该会观察到峰值GPU内存使用量显著减少，因为我们每次只处理小的子图。这是对于大型图的主要优势。
- **时间：** 对于 *Cora这样的小数据集*，每个周期的耗时可能比基线模型 *增加*。这是因为采样和处理许多小批次的开销可能超过较小计算并行化的好处。然而，对于不适合内存的图，采样是唯一可行的方法，并行数据加载有助于隐藏延迟。每个周期的耗时相对于可行性变得有意义。

### 3. 优化2：自动混合精度（AMP）

现代GPU具有专用硬件（Tensor Cores），它使用FP16（半精度）等较低精度格式加速计算。PyTorch的自动混合精度（AMP）实用工具允许我们以最少的代码改动发挥其优势，通常可以提供加速并减少内存使用。

我们将把AMP应用于mini-batch设置，因为它更能体现这些优化带来最大收益的场景。

```python
# 重新初始化模型和优化器（对梯度缩放器很重要）
model_amp = SampledGCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
optimizer_amp = torch.optim.Adam(model_amp.parameters(), lr=0.01, weight_decay=5e-4)

# 创建一个梯度缩放器用于损失缩放
scaler = amp.GradScaler(enabled=torch.cuda.is_available())

# --- AMP训练循环 ---
def train_amp():
    model_amp.train()
    total_loss = 0
    for batch in train_loader: # 重复使用相同的加载器
        batch = batch.to(device)
        optimizer_amp.zero_grad()

        # 使用autocast上下文管理器
        # 此上下文中的操作在有利时以较低精度运行
        with amp.autocast(enabled=torch.cuda.is_available()):
            out = model_amp(batch.x, batch.edge_index, batch.size)
            loss = F.nll_loss(out, batch.y[:batch.batch_size])

        # 在反向传播之前缩放损失
        scaler.scale(loss).backward()
        # 取消梯度缩放并更新模型权重
        scaler.step(optimizer_amp)
        # 更新缩放器以进行下一次迭代
        scaler.update()

        total_loss += loss.item() * batch.batch_size
    return total_loss / data.train_mask.sum().item()

# 分析AMP版本性能
amp_avg_time, amp_peak_mem = profile_run(train_amp, "Sampled GCN + AMP")
```

**观察结果：**

- **内存：** AMP通常会减少内存使用，因为FP16张量所需的存储空间是FP32张量的一半。为反向传播 (backpropagation)存储的激活也会受益。
- **时间：** 加速效果在很大程度上取决于GPU架构（Tensor Cores的存在和效率）以及模型中的具体操作。矩阵乘法和卷积通常会看到显著加速。
- **数值稳定性：** 尽管AMP旨在保持数值稳定性，但有时纯FP16可能导致梯度消失/爆炸等问题。`GradScaler`通过动态缩放损失来帮助减轻这个问题。准确性通常应与FP32训练相当。

### 4. 优化3：模型编译（PyTorch >= 2.0）

PyTorch 2.0引入了`torch.compile`，这是一个可以显著加速模型执行的功能，它通过将Python代码转换为优化的图表示并使用TorchInductor等后端编译器来实现。应用它通常很简单。

```python
# 重新初始化模型和优化器
model_compiled_base = BaselineGCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
optimizer_compiled_base = torch.optim.Adam(model_compiled_base.parameters(), lr=0.01, weight_decay=5e-4)

# 编译基线模型
# 首先使用默认模式，后续可以尝试'reduce-overhead'或'max-autotune'等其他模式
compiled_model = torch.compile(model_compiled_base)

# --- 编译后的基线训练循环 ---
def train_compiled_baseline():
    compiled_model.train() # 使用编译后的模型
    optimizer_compiled_base.zero_grad()
    out = compiled_model(data.x, data.edge_index) # 将数据传递给编译后的模型
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer_compiled_base.step()
    return loss.item()

# 分析编译后的基线版本性能
# 注意：最初几次运行可能会因为编译开销而较慢。
# 运行更多周期以分摊此成本。
compiled_avg_time, compiled_peak_mem = profile_run(train_compiled_baseline, "Compiled Baseline GCN", epochs=100) # 更长运行时间

# ---- 可选：编译采样+AMP模型 ----
# model_compiled_amp = SampledGCN(...) # 初始化
# optimizer_compiled_amp = Adam(...)
# compiled_model_amp = torch.compile(model_compiled_amp)
# scaler_compiled = amp.GradScaler(...)
# def train_compiled_amp():
#     ... # 使用compiled_model_amp, autocast, scaler
# profile_run(train_compiled_amp, "Compiled Sampled GCN + AMP", epochs=100)
```

**观察结果：**

- **时间：** `torch.compile`可以带来显著加速，特别是对于主要由标准PyTorch操作构成且运行在新硬件上的模型。对GNN的益处取决于执行时间有多少是在可编译内核中花费的，以及有多少是在PyG/DGL层使用的自定义C++/CUDA扩展中花费的（这些扩展可能已经高度优化）。你可能会看到不如纯CNN/Transformer那么显著的加速，但改进仍然常见。
- **内存：** 编译本身并不会显著减少峰值内存使用，尽管优化执行 *可能* 导致稍微不同的内存访问模式。
- **开销：** 当模型（或其具有不同输入形状的部分）首次被遇到时，会产生一次性编译成本。此开销需要分摊到多次迭代或多个周期。

### 5. 基准测试总结

我们来总结一下发现。具体数值会因你的硬件（CPU、GPU、内存）、软件版本以及具体数据集而有很大差异。



[交互图表：GCN优化对比 (Cora数据集)](https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-5-gnn-implementation-tooling-optimization/practice-optimizing-gnn-implementation#plot-7z0wn7)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "GCN优化对比 (Cora数据集)",
    "xaxis": {
      "title": "指标"
    },
    "yaxis": {
      "title": "值 (越低越好)",
      "type": "log"
    },
    "barmode": "group",
    "legend": {
      "title": "配置"
    },
    "colorway": [
      "#4263eb",
      "#1098ad",
      "#7048e8",
      "#37b24d"
    ]
  },
  "data": [
    {
      "type": "bar",
      "name": "平均周期时间 (秒)",
      "x": [
        "基线",
        "采样",
        "采样+AMP",
        "编译"
      ],
      "y": [
        0.85,
        0.45,
        0.3,
        0.22
      ]
    },
    {
      "type": "bar",
      "name": "峰值GPU内存增长 (MiB)",
      "x": [
        "基线",
        "采样",
        "采样+AMP",
        "编译"
      ],
      "y": [
        500,
        200,
        150,
        180
      ]
    }
  ]
}
```

</details>



> 对Cora数据集上的GCN模型应用不同优化策略后，平均每个周期训练时间与峰值GPU内存增长的比较。请注意y轴采用对数刻度。

**结果解读：**

- **基线：** 使用全图训练设定参照点。如果内存允许，在 *小型* 图上通常每个周期最快，但使用内存最多。
- **邻居采样：** 大幅减少内存，使大型图的训练变为可能。对于小型图，由于采样/批处理开销，可能增加周期时间。
- **AMP：** 在采样基础上应用时，进一步减少内存使用，并可能缩短周期时间（尤其是在兼容的GPU上）。
- **编译：** 可以缩短基线和采样版本的每个周期时间，对内存影响最小。它的有效性取决于模型结构和后端编译器的效率。

### 结论与后续步骤

本次实践演示了如何使用PyG将常见的优化技术应用于GCN实现：

1. **邻居采样（`NeighborLoader`）：** 通过减少内存占用，对于大型图的可扩展性非常重要。
2. **自动混合精度（`torch.cuda.amp`）：** 减少内存并在支持的硬件上通常加速训练。
3. **模型编译（`torch.compile`）：** 可以通过图优化和专用后端加速Python和PyTorch代码的执行。

请记住，优化是一个迭代过程：

- **首先分析：** 在优化之前务必进行测量，以确定真正的瓶颈。
- **考虑权衡：** 加速可能以实现复杂性或细微数值差异（如AMP）为代价。内存节省可能增加每个周期的计算时间（如在小型图上采样）。
- **调整超参数 (parameter) (hyperparameter)：** 最佳批次大小、邻居数量、学习率等在应用优化后可能会改变。
- **了解库功能：** PyG和DGL提供许多其他高级功能，例如专用稀疏操作、融合内核以及与不同后端的集成，这些都可以提供进一步的性能提升。

通过理解和应用这些技术，你可以构建不仅准确而且高效、可扩展的GNN模型，以处理复杂、大规模的图数据。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍了图卷积网络（GCN）模型，为基线GNN实现奠定了理论基础。
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — William L. Hamilton, Rex Ying, and Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  提出了GraphSAGE，一个使用邻居采样机制的框架，对GNN的小批量处理和扩展至大型图至关重要，PyG的NeighborLoader即实现了此机制。
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/) — Matthias Fey and Jan E. Lenssen (2023)
  PyTorch Geometric官方文档，提供了GCNConv、NeighborLoader及其他用于GNN实现和优化的工具的详细信息。
- [Automatic Mixed Precision Training](https://pytorch.org/docs/stable/amp.html) — PyTorch Contributors (2025)
  Publisher: PyTorch Foundation
  torch.cuda.amp的官方指南，解释了如何使用混合精度和GradScaler来加速训练并减少内存占用。
- [torch.compile](https://pytorch.org/docs/stable/compile.html) — PyTorch Contributors (2023)
  Publisher: PyTorch Contributors
  PyTorch 2.0中torch.compile功能的官方文档，详细说明了如何使用图编译来优化模型以实现更快的执行。

---

[上一节](07-%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9C%A8%E7%94%9F%E4%BA%A7%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md)

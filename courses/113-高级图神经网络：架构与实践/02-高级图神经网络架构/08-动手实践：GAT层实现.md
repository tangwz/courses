# 动手实践：GAT层实现

来源：[原文](https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-2-advanced-gnn-architectures/practice-implementing-gat)

[返回章节目录](README.md) · [返回课程目录](../README.md)

图注意力网络（GAT）层通过自注意力 (self-attention)机制 (attention mechanism)，在聚合时赋予邻域内不同节点不同的重要性权重 (weight)。这使得模型能够针对每个节点关注更相关的邻居。GAT层的主要组成部分包括共享线性变换、注意力机制 $a$ 以及用于归一化 (normalization)的 softmax 应用。

现在，我们将构建一个GAT层。我们将使用PyTorch和PyTorch Geometric (PyG) 库，借助其`MessagePassing`基类。该基类通过处理传播逻辑，简化了消息传递GNN的实现。这种方法在理解基本操作和使用高效库工具之间取得了平衡。

### 先决条件

请确保您已安装PyTorch和PyTorch Geometric。您应该熟悉PyTorch的`nn.Module`以及PyG的基本知识，例如`Data`对象和`MessagePassing`接口。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.typing import Adj, OptTensor, PairTensor
```

### 理解GAT层的逻辑

回顾一下计算单个GAT层中节点 $i$ 的输出特征 $h'_i$ 的步骤：

1. **线性变换：** 对输入节点特征 $h_j \in \mathbb{R}^{F}$ 应用一个可学习的权重 (weight)矩阵 $W \in \mathbb{R}^{F' \times F}$。
   $z_j = W h_j$
2. **注意力系数计算：** 对于边 $(j, i)$，计算一个未归一化 (normalization)的注意力得分 $e_{ij}$，表示节点 $j$ 的特征对节点 $i$ 的重要性。这通常通过共享注意力机制 (attention mechanism)完成，该机制通常是一个由可学习权重向量 (vector) $a \in \mathbb{R}^{2F'}$ 参数 (parameter)化的单层前馈网络：
   $e_{ij} = \text{LeakyReLU}(a^T [W h_i || W h_j])$
   其中 $||$ 表示连接。
3. **归一化：** 使用softmax函数对所有邻居 $j \in \mathcal{N}_i \cup \{i\}$（包括节点本身，通常通过自循环实现）的注意力得分 $e_{ij}$ 进行归一化：
   $\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i \cup \{i\}} \exp(e_{ik})}$
4. **加权聚合：** 使用归一化注意力系数 $\alpha_{ij}$，计算转换后的邻居特征的加权和，得到输出特征 $h'_i$。通常会应用一个可选的非线性激活函数 (activation function) $\sigma$（如ELU或ReLU）。
   $h'_i = \sigma\left( \sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij} z_j \right) = \sigma\left( \sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij} W h_j \right)$
5. **多头注意力 (multi-head attention)：** 为了稳定训练并捕捉多样的关联，GAT采用多头注意力。$K$ 个独立的注意力机制（头）并行计算特征。它们的输出通常被连接（concatenated）或平均（averaged）。如果连接 $K$ 个头，每个头的 $W$ 的输出维度通常设置为 $F'/K$。
   - 连接（Concatenation）： $h'_i = \big\|_{k=1}^K \sigma\left( \sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij}^k W^k h_j \right)$
   - 平均（Averaging）： $h'_i = \sigma\left( \frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij}^k W^k h_j \right)$

### 使用PyG实现GAT层

我们将使用`MessagePassing`基类实现一个支持多头注意力 (multi-head attention)的GAT层。该类根据`message`函数中计算的消息处理聚合过程（步骤4）。

```python
class GATLayer(MessagePassing):
    """实现一个具有多头注意力的单个GAT层。"""
    def __init__(self, in_features: int, out_features: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0.0, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        # 使用 'add' 聚合进行加权求和。
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs) # node_dim=0 表示对节点特征进行操作

        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # 每个头的输出维度。如果进行连接，则用总输出特征数除以头数。
        if concat:
            assert out_features % heads == 0
            self.head_dim = out_features // heads
        else:
            self.head_dim = out_features

        # 步骤1：应用于所有节点的线性变换 (W)。
        # 这被实现为K个独立的线性层（每个头一个）。
        self.lin = nn.Linear(in_features, self.heads * self.head_dim, bias=False)

        # 步骤2：注意力机制参数 'a'。
        # 我们使用两个权重向量 (a_l, a_r) 分别用于源节点和目标节点
        # 转换后的特征，稍后会隐式连接。
        # 每个大小为 [1, heads, head_dim]
        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, self.head_dim))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(self.head_dim))
        else:
            self.register_parameter('bias', None)

        self._alpha = None # 用于存储注意力权重以供后续查看

        self.reset_parameters()

    def reset_parameters(self):
        # 权重初始化类似于原始GAT论文和PyG的GATConv
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: Adj,
                size: tuple[int, int] | None = None, return_attention_weights: bool = False):
        """
        GAT层的前向传播。
        参数：
            x (张量或PairTensor): 节点特征 (N, in_features) 或 ((N, F_in), (M, F_in))。
            edge_index (Adj): 图连接 (2, E)。
            size (元组, 可选): 二分图的大小 (N, M)。
            return_attention_weights (布尔值): 如果为True，则同时返回注意力系数。
        """
        # 确保x是张量；如果需要，稍后处理二分图
        if isinstance(x, torch.Tensor):
            x_l: OptTensor = x
            x_r: OptTensor = x
        else: # 二分图情况下的基本处理 PairTensor
            x_l, x_r = x

        assert x_l is not None
        num_nodes = x_l.size(0) # 如果不是二分图，则假设 N = M

        # 步骤1：应用线性变换。为所有头投影特征。
        # 结果形状：[N, heads * head_dim]
        z = self.lin(x_l)
        z = z.view(-1, self.heads, self.head_dim) # 形状：[N, heads, head_dim]

        # 添加自循环，使节点能够关注自身（可选但常见）。
        if self.add_self_loops:
            if isinstance(edge_index, torch.Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None: # 二分图情况
                   num_nodes = (x_l.size(0), x_r.size(0))
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes, fill_value='mean')
            # 注意：如果使用SparseTensor，add_self_loops需要单独处理。

        # --- 消息传递开始 ---
        # 步骤2和3：计算注意力系数并进行归一化。
        # 步骤4：聚合特征。
        # propagate方法协调对message()、aggregate()和update()的调用。
        # 我们传递转换后的特征'z'，它将在message()中使用。
        out = self.propagate(edge_index, x=(z, z), size=size, # 为源节点 (j) 和目标节点 (i) 都传递 z
                             return_attention_weights=return_attention_weights)
        # --- 消息传递结束 ---

        # 步骤5：应用最终变换（连接/平均，偏置，激活）。
        if self.concat:
            # 从 [N, heads, head_dim] 重塑为 [N, heads * head_dim]
            out = out.view(-1, self.heads * self.head_dim)
        else:
            # 在所有头之间求平均：[N, heads, head_dim] -> [N, head_dim]
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if return_attention_weights:
            return out, self._alpha
        else:
            return out

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, # x_j 是边的源节点特征，x_i 是边的目标节点特征
              index: torch.Tensor, # 边索引（用于softmax归一化）
              ptr: OptTensor, # 用于稀疏softmax的指针（如果使用CSR格式）
              size_i: int | None, # 目标节点数量
              return_attention_weights: bool) -> torch.Tensor:
        """
        计算每条边 (j, i) 中从节点j到节点i的消息。
        此函数实现步骤2和3（注意力计算和归一化）。
        参数：
            x_j (张量): 边的源节点特征。形状：[E, heads, head_dim]
            x_i (张量): 边的目标节点特征。形状：[E, heads, head_dim]
            index (张量): 边索引（用于softmax归一化）。形状：[E]
            ptr (OptTensor): 用于稀疏softmax的指针（如果使用CSR格式）。
            size_i (整数): 目标节点数量。
            return_attention_weights (布尔值): 从前向传播中传递的标志。
        返回：
            张量: 沿边传递的消息。形状：[E, heads, head_dim]
        """
        # 步骤2：计算注意力得分 e_ij。
        # 分别计算源节点 (j) 和目标节点 (i) 的分量
        alpha_src = (x_j * self.att_src).sum(dim=-1) # 形状：[E, heads]
        alpha_dst = (x_i * self.att_dst).sum(dim=-1) # 形状：[E, heads]
        # 将它们组合
        alpha = alpha_src + alpha_dst # 形状：[E, heads]

        # 应用LeakyReLU激活
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # 步骤3：使用softmax归一化注意力得分。
        # PyG的 `softmax` 实用函数可以正确处理稀疏softmax。
        # 它对每个目标节点（索引 `i`）的得分进行归一化。
        alpha = softmax(alpha, index, ptr, size_i) # 形状：[E, heads]

        # 如果需要进行分析，则存储注意力权重。
        if return_attention_weights:
            self._alpha = alpha

        # 对注意力权重应用dropout（常见做法）。
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 步骤4（第1部分）：根据注意力对特征进行加权并创建消息。
        # __init__ 中指定的聚合（'add'）将对每个节点的消息求和。
        # 将 alpha 重塑为 [E, heads, 1] 以进行广播。
        message = x_j * alpha.unsqueeze(-1) # 形状：[E, heads, head_dim]
        return message

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_features}, '
                f'{self.out_features}, heads={self.heads}, concat={self.concat})')
```

> 单头GAT层在前向传播过程中的信息流。多头注意力涉及此流程的并行执行，使用独立的参数 (parameter)，随后进行连接或平均。

### 使用GAT层

现在我们来看看如何使用这个`GATLayer`。我们将创建一个简单图并通过该层。

```python
# 使用示例
# 创建一些模拟数据：5个节点，每个节点3个特征
num_nodes = 5
in_channels = 3
out_channels_final = 16 # 期望的最终输出维度
num_heads = 4 # 注意力头的数量

x = torch.randn(num_nodes, in_channels)
# 定义边（源 -> 目标）：0->1, 0->2, 1->2, 1->3, 2->4, 3->4
edge_index = torch.tensor([[0, 0, 1, 1, 2, 3],
                           [1, 2, 2, 3, 4, 4]], dtype=torch.long)

# 实例化层
# 注意：如果 concat=True，out_features 必须能被 heads 整除。
# 这里，out_features=16, heads=4 -> head_dim = 16/4 = 4。
gat_layer = GATLayer(in_features=in_channels,
                     out_features=out_channels_final,
                     heads=num_heads,
                     concat=True, # 连接头输出
                     dropout=0.1) # 训练时应用 dropout

# 执行前向传播
# 在推断或评估期间，设置 model.eval() 以禁用 dropout
gat_layer.train() # 设置为训练模式以启用 dropout
output_features = gat_layer(x, edge_index)

# 检查输出形状
# 预期：[num_nodes, out_channels_final] = [5, 16]
print("输入特征形状:", x.shape)
print("输出特征形状:", output_features.shape)

# 您还可以获取注意力权重
gat_layer.eval() # 禁用 dropout 以供查看
output_features, attention_weights = gat_layer(x, edge_index, return_attention_weights=True)
# attention_weights 的形状大约是 [E + N_loops, heads]
# 其中 E 是原始边的数量，N_loops 是节点数量（如果 add_self_loops=True）
print("注意力权重形状:", attention_weights.shape)
# 用于注意力计算的 edge_index 如果 add_self_loops=True 则包含自循环
print("用于注意力的边索引（可能包含自循环）:", gat_layer.edge_index_prop)
```

### 后续步骤

此实现提供了一个GAT层。要构建一个完整的GNN模型，您通常会：

1. **堆叠多层：** 串联多个`GATLayer`实例，通常在层之间加入激活函数 (activation function)（如ELU或ReLU）和可能的dropout。后续层的输入特征将是前一层的输出特征。请注意，输出维度会根据`concat`设置而变化。如果最终层使用`concat=True`，其`out_features`将是模型最终的节点嵌入 (embedding)维度。如果使用`concat=False`（平均），其`out_features`将直接定义最终维度。
2. **添加读出/池化：** 对于图级别任务（如图分类），在GAT层之后应用池化或读出函数（例如，全局平均/最大池化）以获得单个图表示。
3. **定义损失和优化器：** 选择一个合适的损失函数 (loss function)（例如，用于节点分类的CrossEntropyLoss）和一个优化器（例如，Adam）。
4. **训练循环：** 实现一个标准的训练循环，用于输入数据、计算损失、执行反向传播 (backpropagation)并更新模型参数 (parameter)。

这项实践练习展示了GAT的理论是如何通过常用库转化为代码的。通过理解PyG `MessagePassing`中的`message`和`propagate`机制，您可以有效实现多种GNN架构。请记住，初始化、激活函数、dropout率和头数量等细节是超参数 (hyperparameter)，通常需要针对特定任务进行调整以获得最佳表现。

## 参考资料

- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) — Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)
  介绍了图注意力网络（GAT）架构，阐述了其用于图结构数据的自注意力机制以及本节中实现的理论基础。
- [Graph Representation Learning](https://www.morganclaypool.com/doi/abs/10.2200/S01045ED1V01Y202009AIM046) — William L. Hamilton (2020)
  Journal: Synthesis Lectures on Artificial Intelligence and Machine Learning; Publisher: Morgan & Claypool Publishers; Volume: 46; Pages: 1-159; DOI: [10.2200/S01045ED1V01Y202009AIM046](https://doi.org/10.2200/S01045ED1V01Y202009AIM046)
  一本基础学术书籍，提供了对图表示学习的广泛理解，包括专门关于图神经网络和注意力机制的讨论。

---

[上一节](07-%E6%AF%94%E8%BE%83%E6%9E%B6%E6%9E%84%E9%80%89%E6%8B%A9%E4%B8%8E%E6%9D%83%E8%A1%A1.md) · [下一节](../03-%E5%BA%94%E5%AF%B9GNN%E8%AE%AD%E7%BB%83%E4%B8%AD%E7%9A%84%E5%A4%8D%E6%9D%82%E6%80%A7/01-%E8%BF%87%E5%B9%B3%E6%BB%91%E9%97%AE%E9%A2%98.md)

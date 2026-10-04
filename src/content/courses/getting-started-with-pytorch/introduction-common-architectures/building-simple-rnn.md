---
course: "getting-started-with-pytorch"
chapter: "introduction-common-architectures"
lesson: "building-simple-rnn"
sourceId: 2213
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures/building-simple-rnn"
title: "在PyTorch中构建一个简单的RNN"
description: "使用`nn.RNN`层搭建一个基本的RNN模型。"
order: 5
plots: []
sourceHash: "5d6294592160dfc90474d78178486105ea6aaec78b8ad03d5c44b60b874d4137"
sourceCorrections: []
---

构建一个简单的RNN模型，使用PyTorch的`torch.nn`库，以展示循环神经网络 (neural network) (RNN)如何利用隐藏状态处理序列数据。PyTorch提供了一个便捷的模块`nn.RNN`，它封装了RNN的核心逻辑。

### `nn.RNN` 模块

PyTorch中基本RNN的主要构成部分是`torch.nn.RNN`类。当你创建这个类的一个实例时，你就在创建一个可以处理序列的RNN层（或者可能是多个堆叠层）。

`nn.RNN`层主要接收一个输入序列和一个可选的初始隐藏状态。然后，它会遍历输入序列的每个时间步，根据当前输入和前一个隐藏状态更新其隐藏状态。它会生成一个输出序列（每个时间步的隐藏状态）以及处理完整个序列后的最终隐藏状态。

要初始化一个`nn.RNN`层，你需要指定几个重要参数 (parameter)：

- `input_size`：这定义了在每个时间步输入$x$中期望的特征数量。例如，如果你正在处理维度为300的词嵌入 (embedding)，那么`input_size`将是300。
- `hidden_size`：这决定了隐藏状态$h$中的特征数量。它也定义了每个时间步输出的维度。`hidden_size`的选择是一个超参数 (hyperparameter)，会影响模型的容量。
- `num_layers`：这允许你堆叠多个RNN层。第一个层的输出序列成为第二个层的输入序列，依此类推。默认值为1。堆叠层有时可以帮助模型学习更复杂的时序模式。
- `nonlinearity`：要使用的非线性函数。可以是`'tanh'`（默认）或`'relu'`。
- `batch_first`：一个布尔型参数。如果为`True`，则输入和输出张量以`(batch_size, seq_len, feature_dim)`的形式提供。如果为`False`（默认值），则为`(seq_len, batch_size, feature_dim)`。当使用生成序列批次的数据加载器时，将此参数设为`True`通常更直观。
- `dropout`：如果非零，则在除最后一层之外的每个RNN层的输出上引入一个Dropout层，其Dropout概率等于`dropout`。默认值：0。
- `bidirectional`：如果为`True`，则成为一个双向RNN。默认值：`False`。我们目前将专注于单向RNN。

### 输入和输出的形状

理解输入和输出张量的预期形状对于正确使用`nn.RNN`是必不可少的。为了便于理解，我们假设`batch_first=True`，因为它很常用。

- **输入：** 输入序列应该是一个形状为`(batch_size, seq_len, input_size)`的张量。
  - `batch_size`：批次中的序列数量。
  - `seq_len`：每个序列的长度（时间步数）。
  - `input_size`：每个时间步的特征数量（与`nn.RNN`的`input_size`参数 (parameter)匹配）。
- **初始隐藏状态 (`h_0`)：** （可选）如果你想提供一个初始隐藏状态，它的形状应为`(num_layers, batch_size, hidden_size)`。如果未提供，则默认为全零张量。
- **输出序列 (`output`)：** 该张量包含来自*最后一层*RNN的、*每个*时间步的输出特征（隐藏状态）。其形状为`(batch_size, seq_len, hidden_size)`。
- **最终隐藏状态 (`h_n`)：** 该张量包含处理完整个序列后，*每个*RNN层的最终隐藏状态。其形状为`(num_layers, batch_size, hidden_size)`。你可以将此最终隐藏状态用作后续层（例如用于分类的线性层）的输入。

如果`batch_first=False`（默认值），则输入和输出序列张量中的`batch_size`和`seq_len`维度会互换。隐藏状态张量（`h_0`，`h_n`）始终将`batch_size`作为第二个维度，无论`batch_first`设置如何。

### 实现一个简单的RNN模型

让我们使用`nn.Module`创建一个基本的RNN模型。该模型将包含一个`nn.RNN`层，后面跟着一个`nn.Linear`层，用于将序列的最终隐藏状态映射到输出预测。这种模式在序列分类任务中很常见。

```python
import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_rnn_layers=1):
        """
        初始化SimpleRNNModel。

        Args:
            input_dim (int): 每个时间步输入特征的维度。
            hidden_dim (int): RNN隐藏状态的维度。
            output_dim (int): 最终输出的维度。
            num_rnn_layers (int): 堆叠RNN层的数量。默认值为1。
        """
        super().__init__() # 调用父类 (nn.Module) 的 __init__ 方法
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers

        # 定义RNN层
        # batch_first=True 表示输入/输出张量形状为: (batch, seq, feature)
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True, # 确保输入形状是 (batch, seq_len, input_size)
            nonlinearity='tanh' # 默认激活函数
        )

        # 定义输出层（全连接层）
        # 它以RNN的最终隐藏状态作为输入
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        定义模型的正向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_dim)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_dim)。
        """
        # 用零初始化隐藏状态
        # 形状: (num_layers, batch_size, hidden_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(x.device)

        # 通过RNN层传递数据
        # rnn_out 形状: (batch_size, seq_len, hidden_size)
        # hn 形状: (num_layers, batch_size, hidden_size)
        rnn_out, hn = self.rnn(x, h0)

        # 我们只需要最后一层、最后一个时间步的隐藏状态
        # hn 包含所有层的最终隐藏状态。
        # hn[-1] 获取最后一层的最终隐藏状态。
        # hn[-1] 的形状: (batch_size, hidden_size)
        last_layer_hidden_state = hn[-1]

        # 将最后一个隐藏状态通过全连接层
        # out 形状: (batch_size, output_dim)
        out = self.fc(last_layer_hidden_state)

        return out

# --- 示例用法 ---

# 定义模型参数
INPUT_DIM = 10   # 输入特征维度（例如，嵌入大小）
HIDDEN_DIM = 20  # 隐藏状态维度
OUTPUT_DIM = 5   # 输出维度（例如，类别数量）
NUM_LAYERS = 1   # RNN层数

# 创建模型
model = SimpleRNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
print("模型结构:")
print(model)

# 创建一些虚拟输入数据
BATCH_SIZE = 4
SEQ_LEN = 15
dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM) # 形状: (batch, seq, feature)

# 执行前向传播
output = model(dummy_input)

print(f"\n输入形状: {dummy_input.shape}")
print(f"输出形状: {output.shape}")

# 验证输出形状是否与 (BATCH_SIZE, OUTPUT_DIM) 匹配
assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
```

### 代码分析

1. **初始化 (`__init__`)**：我们定义`nn.RNN`层，指定`input_dim`、`hidden_dim`、`num_rnn_layers`，以及重要的`batch_first=True`。我们还定义了一个标准的`nn.Linear`层（`self.fc`），它将接收来自RNN的最终隐藏状态作为输入并生成模型的最终输出。
2. **正向传播 (`forward`)**：
   - 我们首先从输入张量`x`中确定`batch_size`。
   - 创建一个形状为`(num_layers, batch_size, hidden_dim)`的全零初始隐藏状态`h0`。我们使用`.to(x.device)`确保它与输入`x`位于相同的设备上。
   - 输入`x`和初始隐藏状态`h0`被传递给`self.rnn`层。它返回两个张量：`rnn_out`（来自最后一层的所有时间步的隐藏状态）和`hn`（经过最后一个时间步后所有层的最终隐藏状态）。
   - 由于我们的目标通常是序列分类或摘要，我们通常使用最终隐藏状态。`hn`的形状是`(num_layers, batch_size, hidden_size)`。我们使用`hn[-1]`选择*最后一层*的最终隐藏状态，这会得到一个形状为`(batch_size, hidden_size)`的张量。
   - 这个最终隐藏状态`hn[-1]`通过全连接层`self.fc`，以获得形状为`(batch_size, output_dim)`的最终输出张量。
3. **示例用法**：我们实例化模型并创建随机输入数据，使其符合预期的`(batch_size, seq_len, input_size)`格式（因为我们设置了`batch_first=True`）。运行模型会生成一个输出张量，我们验证其形状是否为`(batch_size, output_dim)`，这适用于多类别分类等任务，其中批次中的每个序列都会被分配一个输出向量 (vector)。

## 参考资料

- [RNN - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) — PyTorch team (2024)
  Publisher: PyTorch Foundation
  PyTorch官方`torch.nn.RNN`模块文档，详细说明其参数、输入/输出形状及用法，与在PyTorch中构建RNN模型直接相关。
- [Finding structure in time](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1) — Jeffrey L. Elman (1990)
  Journal: Cognitive Science; Publisher: Wiley-Blackwell; Volume: 14; Pages: 179-211; DOI: [10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
  一篇开创性论文，介绍了简单循环网络（SRN），也称为Elman网络，这是一种基础的循环神经网络。
- [Deep Learning](https://www.deeplearningbook.org/contents/rnn.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 367
  第10章提供了循环神经网络的详细理论背景，涵盖其架构、训练和各种形式，是一份全面的学术资料。

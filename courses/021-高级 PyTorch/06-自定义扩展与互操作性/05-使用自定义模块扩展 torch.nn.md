# 使用自定义模块扩展 torch.nn

来源：[原文](https://apxml.com/zh/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/custom-nn-modules)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 PyTorch 提供了像 `torch.nn.Linear` 或 `torch.nn.Conv2d` 这样的基本构建块，以及像 `torch.nn.Sequential` 这样的容器，但应用程序通常会通过将更复杂或专门的逻辑封装到可重用组件中来获得好处。扩展 `torch.nn.Module` 是 PyTorch 用于创建这些自定义层或网络部分的标准机制。这种方法提升了模块化、代码组织性和可重用性，使得管理复杂的模型架构变得更加容易。它不仅允许您定义层，还可以定义具体的正向计算逻辑，包括控制流、子组件之间的配合，以及与自定义操作的整合。

### 自定义模块的构成

其核心是，自定义模块是一个继承自 `torch.nn.Module` 的 Python 类。您通常会重写的两个最重要的方法是：

1. **`__init__(self, ...)`:** 构造函数。您在此处定义和初始化模块的组件：

   - **子模块：** 其他 `nn.Module` 类的实例（包括标准 PyTorch 层或其他自定义模块）。
   - **参数 (parameter)：** 可学习的张量，通常是权重 (weight)和偏置 (bias)，使用 `torch.nn.Parameter` 创建。
   - **缓冲区：** 不可学习的状态张量（例如，BatchNorm 中的运行均值/方差），使用 `self.register_buffer()` 注册。
   - 极其重要的一点是，您**必须**在 `__init__` 方法的开头调用 `super().__init__()`。这确保了基类 `nn.Module` 正确初始化，设置了参数跟踪、设备移动和状态保存所需的内部结构。
2. **`forward(self, ...)`:** 此方法定义模块执行的计算。它接收输入张量（以及可能的其他参数）并返回输出张量。您在此方法中使用 `__init__` 中定义的子模块、参数和缓冲区来实现所需的逻辑。PyTorch 的动态计算图是根据 `forward` 中执行的操作构建的。

这是一个基本结构：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomModule(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()  # 必不可少的第一步

        # 定义子模块（层）
        self.layer1 = nn.Linear(input_features, hidden_units)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_units, output_features)

        # 直接定义可学习参数（如果需要）
        # 示例：一个可学习的缩放因子
        self.scale = nn.Parameter(torch.randn(1))

        # 定义不可学习的状态（缓冲区）
        # 示例：一个用于正向传播的计数器（仅作演示）
        self.register_buffer('forward_count', torch.zeros(1, dtype=torch.long))

    def forward(self, x):
        # 使用已初始化的组件定义计算流程
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)

        # 使用自定义参数
        x = x * self.scale

        # 更新缓冲区（如果需要，请确保设备兼容性）
        # 注意：这样的直接修改在标准训练循环中可能不常见，
        # 但它展示了缓冲区的用法。
        self.forward_count += 1

        return x

# 使用示例：
input_dim = 64
output_dim = 10
hidden_dim = 128

model = MyCustomModule(input_dim, output_dim, hidden_dim)
print(model)

# 测试正向传播
dummy_input = torch.randn(4, input_dim) # 批量大小为 4
output = model(dummy_input)
print("输出形状:", output.shape)
print("正向计数:", model.forward_count)

# 参数和缓冲区均被跟踪
for name, param in model.named_parameters():
    print(f"参数: {name}, 形状: {param.shape}")
for name, buf in model.named_buffers():
    print(f"缓冲区: {name}, 值: {buf}")
```

### 实践建议

创建有效的自定义模块不仅仅是继承自 `nn.Module`。请考虑以下做法：

- **初始化用于定义：** `__init__` 主要用于定义组件（子模块、参数 (parameter)、缓冲区）。避免在此处执行大量计算。所有定义为 `nn.Module` 实例或 `nn.Parameter` 实例的属性都会自动注册。这意味着它们会出现在 `model.parameters()` 中，其状态会被保存到 `model.state_dict()` 中，并且 `model.to(device)` 等方法可以正确移动它们。
- **使用 `register_buffer` 管理状态：** 对于模块状态的一部分但不应由优化器更新的张量（如运行统计数据或固定常数），请使用 `self.register_buffer('缓冲区名称', 张量)`。与持有张量的普通 Python 属性不同，缓冲区能被 `state_dict` 和设备放置方法（`.to()`、`.cuda()`、`.cpu()`）正确处理。
- **`forward` 定义计算：** `forward` 方法封装了模块的运行时逻辑。它可以包含任何有效的 Python 代码，包括条件语句（`if`/`else`）和循环（`for`），从而实现动态计算行为。确保梯度计算所需的张量是在 `forward` 中创建或操作的，或作为参数传递的。
- **模块化和专注：** 设计模块以执行特定、明确定义的功能。更小、更专注的模块更易于测试、调试和重用。复杂的网络可以通过组合这些经过良好测试的单元来构建。
- **清晰的输入/输出：** 在模块的文档字符串中记录输入和输出张量的预期形状和类型。这提高了可用性和集成性。
- **可组合性：** 考虑您的模块如何与其他模块结合。`nn.Sequential`、`nn.ModuleList` 和 `nn.ModuleDict` 等标准 PyTorch 容器都可以与自定义模块配合使用。

> 一个使用自定义 `nn.Module` (`MyCustomBlock`) 与标准 PyTorch 层组合网络的视图。自定义块封装了内部层和逻辑。

### 示例：一个简单的缩放点积注意力模块

让我们实现一个基本的缩放点积注意力机制 (attention mechanism)，它是 Transformer 中的一个基本组成部分，作为一个自定义模块。这演示了如何定义参数 (parameter)（隐式地在 `nn.Linear` 内部）以及在 `forward` 中实现特定的数学运算。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleScaledDotProductAttention(nn.Module):
    """ 计算简单的缩放点积注意力。 """
    def __init__(self, d_model, d_k, dropout_p=0.1):
        """
        参数:
            d_model (int): 输入嵌入的维度。
            d_k (int): 键和查询的维度（通常为 d_model // num_heads）。
            dropout_p (float): Dropout 概率。
        """
        super().__init__()
        self.d_k = d_k
        # 用于将输入投影到 Q、K、V 空间的线性层
        self.query_proj = nn.Linear(d_model, d_k)
        self.key_proj = nn.Linear(d_model, d_k)
        self.value_proj = nn.Linear(d_model, d_k) # 通常 d_v = d_k
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, mask=None):
        """
        参数:
            query (torch.Tensor): 查询张量，形状为 (Batch, Seq_len_q, d_model)。
            (torch.Tensor): 张量，形状为 (Batch, Seq_len_k, d_model)。
            value (torch.Tensor): 值张量，形状为 (Batch, Seq_len_v, d_model)。
                                 通常 Seq_len_k == Seq_len_v。
            mask (torch.Tensor, optional): 用于阻止注意力机制关注
                                           某些位置（例如，填充）的掩码张量。
                                           形状为 (Batch, Seq_len_q, Seq_len_k)。
                                           对于被关注的位置，值应为 0；对于被掩码的位置，值应为 -inf。
        返回:
            torch.Tensor: 注意力机制后的输出张量，形状为 (Batch, Seq_len_q, d_k)。
            torch.Tensor: 注意力权重，形状为 (Batch, Seq_len_q, Seq_len_k)。
        """
        # 1. 投影输入
        Q = self.query_proj(query)  # (B, Seq_q, d_k)
        K = self.key_proj(key)      # (B, Seq_k, d_k)
        V = self.value_proj(value)  # (B, Seq_v, d_k)

        # 2. 计算注意力分数 (QK^T / sqrt(d_k))
        # K.transpose(-2, -1) 结果形状为 (B, d_k, Seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 分数形状：(B, Seq_q, Seq_k)

        # 3. 应用掩码（如果提供）
        if mask is not None:
            # 确保掩码具有兼容的维度，可能需要unsqueeze
            # 示例：如果掩码是 (B, Seq_k)，为 Seq_q 广播添加维度
            # mask = mask.unsqueeze(1) # -> (B, 1, Seq_k)
            scores = scores.masked_fill(mask == 0, float('-inf')) # 常见约定：0 表示掩码

        # 4. 应用 softmax 以获得注意力权重
        attn_weights = F.softmax(scores, dim=-1) # (B, Seq_q, Seq_k)

        # 5. 对注意力权重应用 dropout
        attn_weights = self.dropout(attn_weights)

        # 6. 计算值的加权和
        output = torch.matmul(attn_weights, V) # (B, Seq_q, Seq_k) @ (B, Seq_v, d_k) -> (B, Seq_q, d_k)
                                             # 假设 Seq_k == Seq_v

        return output, attn_weights

# 使用示例：
batch_size = 4
seq_len = 10
embed_dim = 128
key_dim = 64

attention_module = SimpleScaledDotProductAttention(d_model=embed_dim, d_k=key_dim)

# 创建虚拟输入（自注意力机制通常使用相同的张量）
q_input = torch.randn(batch_size, seq_len, embed_dim)
k_input = torch.randn(batch_size, seq_len, embed_dim)
v_input = torch.randn(batch_size, seq_len, embed_dim)

output, weights = attention_module(q_input, k_input, v_input)

print("注意力输出形状:", output.shape) # 预期：(4, 10, 64)
print("注意力权重形状:", weights.shape) # 预期：(4, 10, 10)
```

此示例将注意力逻辑封装在一个单一模块中，使其易于集成到像 Transformer 编码器或解码器层这样更大的模型中。

### 进阶考量

- **可变输入大小：** 设计模块以处理不同长度的序列，通常使用填充和掩码，如注意力示例中所示。像 `torch.nn.utils.rnn.pack_padded_sequence` 或自适应池化层这样的技术也可能根据应用而相关。
- **钩子：** `nn.Module` 提供了一个钩子机制（`register_forward_hook`、`register_backward_hook`、`register_forward_pre_hook`），允许您在 `forward` 传递之前或之后，或在 `backward` 传递期间执行自定义代码，而无需修改模块的核心 `forward` 代码。钩子对于调试、可视化或实现某些归一化 (normalization)技术很有用。
- **与自定义操作的接口：** 自定义 `nn.Module` 的 `forward` 方法是调用专门的 C++ 或 CUDA 扩展（本章其他部分介绍）或自定义 `autograd.Function` 实例的自然位置，当性能或特定梯度计算需要它们时。模块结构整齐地封装了标准 PyTorch 组件与这些自定义后端之间的配合。

通过掌握 `torch.nn.Module` 的扩展，您获得了实现几乎任何网络架构或组件的灵活性，能够以清晰、可重用和可维护的方式组织您的代码，这对于应对高级深度学习 (deep learning)项目不可或缺。

## 参考资料

- [\`torch.nn\` - PyTorch documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch Core Team (2024)
  PyTorch神经网络模块基类（`nn.Module`）及其子组件的官方文档，是自定义模块实现的基础参考。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer模型和缩放点积注意力机制的开创性论文，该机制在章节内容中作为自定义`nn.Module`进行演示。
- [Build Custom Layers with \`nn.Module\`](https://pytorch.org/tutorials/beginner/basics/build_model_tutorial.html) — PyTorch Core Team (2024)
  Publisher: PyTorch
  一个PyTorch官方教程，提供通过继承`nn.Module`创建自定义神经网络层的实践指南。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本关于PyTorch深度学习的综合书籍，提供构建、训练和定制神经网络模型的实践指导，包括高级模块开发。

---

[上一节](04-PyTorch%20%E4%B8%8E%20NumPy%20%E7%9A%84%E8%BF%9E%E6%8E%A5.md) · [下一节](06-%E6%89%A9%E5%B1%95%20torch.optim%EF%BC%8C%E4%BD%BF%E7%94%A8%E8%87%AA%E5%AE%9A%E4%B9%89%E4%BC%98%E5%8C%96%E5%99%A8.md)

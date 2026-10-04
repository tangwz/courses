---
course: "advanced-pytorch"
chapter: "pytorch-internals-autograd"
lesson: "gradient-inspection-visualization"
sourceId: 2671
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-1-pytorch-internals-autograd/gradient-inspection-visualization"
title: "梯度查看与图可视化"
description: "用于调试梯度和使用TensorBoard等工具可视化计算图的方法。"
order: 6
plots: []
sourceHash: "883efc68b25c391ce737a110fbc5cf9b0aaaafaed8bb66a7721b143a3565ac20"
sourceCorrections: []
---

了解梯度如何在网络中传递，对于调试和优化必不可少。当模型表现异常或训练停滞时，检查梯度和其对应的计算图通常能提供有用的线索。PyTorch在对需要梯度的张量执行操作时，会动态地构建这个计算图。这里将介绍查看这些梯度和可视化图结构的方法。

### 获取与分析梯度

在调用`loss.backward()`之后，PyTorch会计算损失相对于计算图中所有`requires_grad=True`且参与了损失计算的张量的梯度。这些梯度会累积在对应的叶子张量（通常是模型参数 (parameter)或输入）的`.grad`属性中。

```python
import torch

# 示例设置
w = torch.randn(5, 3, requires_grad=True)
x = torch.randn(3, 2)
y_true = torch.randn(5, 2)

# 前向传播
y_pred = w @ x
loss = torch.nn.functional.mse_loss(y_pred, y_true)

# 反向传播
loss.backward()

# 查看w中累积的梯度
print("Gradient for w:\n", w.grad)

# 非叶子张量或requires_grad=False的张量的梯度通常为None
print("Gradient for x:", x.grad) # 输出: None (默认requires_grad=False)
print("Gradient for y_pred:", y_pred.grad) # 输出: None (非叶子张量，默认不保留梯度)
```

检查`.grad`时常见的几种情况：

1. **梯度为`None`：** 如果张量的`.grad`在调用`.backward()`后为`None`，通常表示：
   - 该张量未设置`requires_grad=True`。
   - 该张量未参与到导致损失的计算图中（例如，它是在`with torch.no_grad():`块中创建或使用`.detach()`进行分离的）。
   - 它是非叶子张量，PyTorch默认不保存中间梯度以节省内存。如果需要查看中间结果的梯度，请使用`tensor.retain_grad()`。
2. **梯度消失：** 梯度变得非常小（例如，$10^{-8}$或更小），通常接近零。这导致图后面（更接近输入端）的权重 (weight)无法有效更新。在深度网络或使用sigmoid等激活函数 (activation function)的网络中很常见，特别是在没有批量归一化 (normalization)或残差连接等方法的情况下。
3. **梯度爆炸：** 梯度变得过大（例如，$10^8$或`NaN`）。这导致训练不稳定、权重更新幅度大，并且损失或权重中常出现`NaN`值。梯度裁剪（第三章介绍）是一种常见的缓解策略。

你可以编程方式检查这些问题：

```python
# 检查None梯度（假设'model'是你的torch.nn.Module实例）
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"Parameter {name} has no gradient.")

# 检查梯度消失/爆炸
max_grad_norm = 0.0
min_grad_norm = float('inf')
nan_detected = False
for param in model.parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if torch.isnan(param.grad).any():
             nan_detected = True
             print(f"NaN gradient detected in parameter: {param.size()}") # 可能需要更具体的识别
        max_grad_norm = max(max_grad_norm, grad_norm)
        min_grad_norm = min(min_grad_norm, grad_norm)

print(f"Max gradient norm: {max_grad_norm:.4e}")
print(f"Min gradient norm: {min_grad_norm:.4e}")
if nan_detected:
    print("Warning: NaN gradients detected!") # 警告：检测到NaN梯度！
```

### 使用Hook进行细粒度检查

为在反向传播 (backpropagation)期间进行更详细的分析，PyTorch提供了**hook（钩子）**。Hook是可以在特定事件发生时（例如张量梯度计算或模块的前向/反向传播）注册执行的函数。

#### 张量Hook (`register_hook`)

你可以直接在张量上注册一个hook。当*该特定张量*的梯度被计算时，这个hook函数将执行。hook函数将梯度作为其唯一参数 (parameter)接收。

```python
import torch

def print_grad_hook(grad):
    print(f"Gradient received: shape={grad.shape}, norm={grad.norm():.4f}")

x = torch.randn(3, 3, requires_grad=True)
y = x.pow(2).sum()

# 在张量x上注册hook
hook_handle = x.register_hook(print_grad_hook)

# 计算梯度
y.backward()

# hook函数（print_grad_hook）会自动调用
# 输出将包含类似以下内容：
# Gradient received: shape=torch.Size([3, 3]), norm=9.5930

# 不再需要时应移除hook以避免内存泄漏
hook_handle.remove()

# 你也可以在hook中修改梯度，但请谨慎使用：
def scale_grad_hook(grad):
  # 示例：将梯度减半
  return grad * 0.5

# x.register_hook(scale_grad_hook)
# y.backward() # 现在存储在x.grad中的梯度将减半
```

Hook对于调试网络的特定部分非常有用。你可以记录梯度统计信息，在`NaN`值出现时精准检查它们，甚至实时修改梯度（尽管修改梯度通常较不常见且需要仔细斟酌）。

#### 模块Hook

你也可以在`torch.nn.Module`实例上注册hook，以便在前向传播期间检查输入和输出，或在反向传播期间检查梯度。

- `register_forward_pre_hook(hook)`: 在模块的`forward`方法之前执行。接收参数`(module, input)`。
- `register_forward_hook(hook)`: 在模块的`forward`方法之后执行。接收参数`(module, input, output)`。
- `register_full_backward_hook(hook)`: 在为模块的输入和输出计算完梯度后执行。接收参数`(module, grad_input, grad_output)`。`grad_input`是一个包含模块输入梯度的元组，`grad_output`是一个包含模块输出梯度的元组。

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = SimpleNet()
input_tensor = torch.randn(4, 10, requires_grad=True)

def backward_hook(module, grad_input, grad_output):
    print(f"\nModule: {module.__class__.__name__}")
    print("  grad_input shapes: ", [g.shape if g is not None else None for g in grad_input])
    print("  grad_output shapes:", [g.shape if g is not None else None for g in grad_output])

# 在linear2层上注册hook
hook_handle_bwd = model.linear2.register_full_backward_hook(backward_hook)

# 前向和反向传播
output = model(input_tensor)
target = torch.randn(4, 1)
loss = nn.functional.mse_loss(output, target)
loss.backward()

# 输出将显示流经linear2的反向梯度形状
# Module: Linear
#   grad_input shapes:  [torch.Size([4, 5]), torch.Size([5]), None] （输入、权重、偏置）如果bias=False，偏置梯度可能为None
#   grad_output shapes: [torch.Size([4, 1])]

hook_handle_bwd.remove() # 清理
```

模块hook对于理解梯度如何逐层传播，或诊断大型网络中特定模块的问题特别有用。

### 可视化计算图

尽管hook能让你以数值方式检查梯度，但可视化计算图可以提供结构性概览。这有助于理解操作与参数 (parameter)之间的依赖关系，确认你的模型架构，或发现意外连接。

#### 使用`torchviz`

一个用于基础图可视化的流行第三方库是`torchviz`。它使用graphviz库来渲染在反向传播 (backpropagation)期间生成的图。

你通常会在输出张量（通常是损失）上调用`torchviz.make_dot`，以可视化其梯度计算图。它返回一个`graphviz.Digraph`对象。

```python
# 要求：pip install torchviz graphviz
import torch
import torchviz

# 简单示例
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)
c = a * b
d = c + a
L = d.mean() # 最终标量输出

# 生成图可视化对象
# params可用于突出显示特定参数
graph = torchviz.make_dot(L, params={'a': a, 'b': b})

# 要查看图，你可以将其渲染到文件或在Jupyter等环境中显示
# graph.render("computation_graph", format="png") # 保存为computation_graph.png
# display(graph) # 在Jupyter环境中

# 为演示目的，我们打印Graphviz源代码
# print(graph.source)
```

> 一个由`torchviz`生成的简单计算图。椭圆代表张量（参数高亮显示），方框代表反向操作（`grad_fn`）。箭头表示反向传播期间的梯度流向。

`torchviz`提供了反向图的清晰高层视图，非常适合理解依赖关系和梯度计算流程。

#### 使用TensorBoard

PyTorch内置支持TensorBoard，这是一个来自TensorFlow的强大可视化工具包。你可以使用`torch.utils.tensorboard.SummaryWriter`记录计算图（以及许多其他内容，如标量、图像、直方图）。

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 再次定义一个简单模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 3)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(3, 1)
    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

model = SimpleNet()
dummy_input = torch.randn(1, 5) # 提供一个示例输入

# 创建一个SummaryWriter实例（默认日志保存到./runs/）
writer = SummaryWriter('runs/graph_demo')

# 将图添加到TensorBoard
# writer需要模型和一个示例输入张量
writer.add_graph(model, dummy_input)
writer.close()

# 要查看图：
# 1. 确保已安装tensorboard (pip install tensorboard)
# 2. 在你的终端运行`tensorboard --logdir=runs/graph_demo`
# 3. 在浏览器中打开提供的URL（通常是http://localhost:6006/）
# 4. 导航到“Graphs”选项卡。
```

TensorBoard直接在你的浏览器中提供了一个交互式图可视化环境。它通常显示一个更详细的图，包括模块范围、参数节点和操作节点。虽然对于非常大的模型可能显得过于复杂，但其交互性允许你展开和折叠图的部分，使其比静态图像更容易浏览复杂的架构。

### 实际考量

- **开销：** 注册许多hook，特别是执行复杂计算或大量I/O（如打印）的hook，会显著减慢训练速度。主要将其用于调试，除非必要，否则不要在生产训练循环中使用。
- **图的复杂性：** 对于非常深或复杂的模型，完整的计算图会变得非常庞大且难以视觉解读。将可视化工作集中在与调试任务相关的特定模块或子图上。
- **动态图：** 请记住PyTorch的图是动态的。可视化的图对应于使用给定输入执行的特定前向传播。如果你的模型具有数据依赖的控制流（例如，影响所用层的`if`语句），图在不同迭代或不同输入之间可能发生变化。

有效检查梯度和可视化计算图是PyTorch高级开发中不可或缺的技能。它们能让你对框架有更深刻的理解，实现有针对性的调试，并做出明智的优化决策。下一章将在此基础上，实现复杂的网络架构。

## 参考资料

- [Automatic differentiation package - torch.autograd](https://pytorch.org/docs/stable/autograd.html) — PyTorch Developers (2024)
  PyTorch 官方文档，详细介绍了自动微分，包括梯度如何计算、存储在 `.grad` 属性中，以及如何使用张量和模块钩子进行检查。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习理论基础的综合性教科书，涵盖反向传播、计算图以及常见的梯度问题，如梯度消失和梯度爆炸。
- [torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html) — PyTorch Developers (2022)
  Publisher: PyTorch Foundation
  PyTorch 官方 TensorBoard 集成文档，包含使用 `SummaryWriter.add_graph` 可视化计算图的方法。
- [PytorchViz GitHub Repository](https://github.com/szagoruyko/pytorchviz) — Sergey Zagoruyko, Adam Paszke, Soumith Chintala, Anton Osokin, Will Price, Alban D (2024)
  `torchviz` 的开源存储库，这是一个广泛用于可视化 PyTorch 计算图的第三方库，展示了操作之间的依赖关系。

---
course: "advanced-pytorch"
chapter: "custom-extensions-interoperability"
lesson: "custom-optimizers"
sourceId: 2786
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/custom-optimizers"
title: "扩展 torch.optim，使用自定义优化器"
description: "通过继承 Optimizer 类来实现自定义优化算法。"
order: 6
plots: []
sourceHash: "f8a521fc6bc0465ef6195c31c2799432f6ea48f04a277f3cc81081c9bd2437c9"
sourceCorrections: []
---

尽管 PyTorch 在 `torch.optim` 中提供了从 SGD 到 Adam 的多种成熟优化算法，但研究和实际用途常能从自定义优化策略中获益。您可能需要实现一篇近期论文中的新算法，调整现有优化器以适应特定限制（例如，仅凭参数 (parameter)组无法处理的分层自适应学习率），或结合不同优化器的步骤。提供了关于如何通过继承 `torch.optim.Optimizer` 来创建自定义优化器的详细指导，这将使您能够全面掌控参数更新过程。

### `torch.optim.Optimizer` 基类

核心来说，任何 PyTorch 优化器都继承自 `torch.optim.Optimizer` 基类。理解其结构对于构建您自己的优化器很必要。主要组成部分包括：

1. **`__init__(self, params, defaults)`**: 构造函数。

   - `params`: 要优化的参数 (parameter)（张量）的可迭代对象，或定义参数组的字典的可迭代对象。参数组允许将不同的超参数 (hyperparameter)（如学习率）应用于模型的不同部分。
   - `defaults`: 一个字典，包含优化器的默认超参数（例如，`{'lr': 0.01, 'momentum': 0.9}`）。这些默认值用于未明确覆盖它们的参数组。
   - 您的自定义优化器 `__init__` 中的第一步应始终是 `super().__init__(params, defaults)`。此调用负责设置 `self.param_groups`，其中存储参数及其相关的超参数。
2. **`step(self, closure=None)`**: 此方法执行一个优化步骤（参数更新）。

   - 通常在 `loss.backward()` 后，每个训练迭代调用一次。
   - 它使用存储在每个参数 `.grad` 属性中的梯度。
   - 可选的 `closure` 参数是一个可调用函数，用于重新评估模型并返回损失。一些优化算法，如 L-BFGS，需要在每一步中多次重新评估损失，因此 `closure` 是必需的。对于大多数常见优化器（SGD、Adam 等），不需要 `closure`。
   - **这是您需要实现的主要方法。**
3. **`zero_grad(self, set_to_none=False)`**: 清除所有优化参数的梯度。

   - 您通常不需要重写此方法。
   - 将 `set_to_none=True` 设置为 `True` 会将 `param.grad` 赋值为 `None`，而不是用零填充。这有时可以通过更快地释放内存并避免内存写入操作来带来微小的性能提升，但如果下游代码期望 `.grad` 始终是 Tensor，则需要仔细处理。
4. **`state`**: 一个字典（通常是 `collections.defaultdict(dict)`），用于保存每个参数的优化器状态。例如，动量优化器在此处存储动量缓冲区，而 Adam 存储梯度和平方梯度的移动平均值。状态通常以参数对象本身作为键（`self.state[param]`）。
5. **`param_groups`**: 字典列表。每个字典表示一组参数，并包含以下键：

   - `'params'`: 属于此组的参数张量列表。
   - 与此组的超参数对应的其他键（例如，`'lr'`、`'momentum'`、`'weight_decay'`）。

### 实现自定义优化器：带有动量的 SGD 示例

让我们从头开始实现带有动量的随机梯度下降 (gradient descent)（SGD），以说明整个过程。带有动量的 SGD 更新规则如下：


$$
v_{t+1} = \mu v_t + g_{t+1}
$$


$$
p_{t+1} = p_t - \alpha v_{t+1}
$$


说明：

- $p_t$ 是时间步 $t$ 的参数 (parameter)。
- $g_{t+1}$ 是时间步 $t+1$ 损失对 $p$ 的梯度。
- $v_t$ 是时间步 $t$ 的动量缓冲区（速度）。
- $\mu$ 是动量系数。
- $\alpha$ 是学习率。

您可以这样实现它：

```python
import torch
from torch.optim import Optimizer
from collections import defaultdict

class CustomSGD(Optimizer):
    """实现带有动量的随机梯度下降。"""

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"无效学习率: {lr}")
        if momentum < 0.0:
            raise ValueError(f"无效动量值: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"无效 weight_decay 值: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # 初始化状态（尽管通常在 step 中惰性初始化）
        # 如果我们在 step 中初始化，这里并非严格需要
        # for group in self.param_groups:
        #     for p in group['params']:
        #         self.state[p] = dict(momentum_buffer=None)

    @torch.no_grad() # 重要：在优化器步骤内禁用梯度跟踪
    def step(self, closure=None):
        """执行一次优化步骤。

        参数:
            closure (callable, optional): 一个可调用函数，用于重新评估模型
                并返回损失。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): # 确保为闭包启用梯度
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue # 跳过没有梯度的参数

                grad = p.grad # 获取梯度张量

                # 如果指定，应用权重衰减（L2 惩罚）
                # 注意：这是标准方法，修改梯度
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # 访问并更新参数状态（动量缓冲区）
                param_state = self.state[p]

                if 'momentum_buffer' not in param_state:
                    # 在第一步中惰性初始化动量缓冲区
                    param_state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    param_state['momentum_buffer'].mul_(momentum).add_(grad) # v = mu*v + grad

                # 获取更新后的动量缓冲区
                momentum_buffer = param_state['momentum_buffer']

                # 执行参数更新步骤
                # p = p - lr * momentum_buffer
                p.add_(momentum_buffer, alpha=-lr)

        return loss
```

**实现要点：**

- **`@torch.no_grad()`**: 使用 `@torch.no_grad()` 装饰 `step` 方法很重要。优化步骤不应成为 autograd 跟踪的计算图的一部分。
- **迭代：** 代码会遍历 `param_groups`，然后遍历每个组内的 `params`。
- **梯度检查：** 它会检查 `if p.grad is None:`，因为模型中的某些参数可能不会接收到梯度（例如，如果它们未在正向传播中使用或与计算图分离）。
- **状态管理：** `momentum_buffer` 存储在 `self.state[p]` 中。它被惰性初始化（参数第一次调用 `step` 时），以避免在参数从未获得梯度时预先分配内存。
- **原地更新：** 参数更新使用 `add_` 和 `mul_` 等原地操作。这会直接修改参数张量，而不会创建新张量，这对于优化器实际更新模型权重 (weight)非常必要。
- **超参数 (hyperparameter)：** 超参数（`lr`、`momentum`、`weight_decay`）从 `group` 字典中获取，允许不同组拥有不同的设置。
- **权重衰减：** 权重衰减（L2 正则化 (regularization)）通常通过在主要更新步骤之前将缩放后的参数值添加到梯度来实现。

### 使用自定义优化器

使用您的自定义优化器就像使用内置优化器一样：

```python
# 假设 'model' 是您的 torch.nn.Module
# 实例化自定义优化器
optimizer = CustomSGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# 在您的训练循环中：
for inputs, targets in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step() # 使用 CustomSGD 逻辑执行更新
```

### 进阶考量

- **参数 (parameter)组：** 您可以在初始化优化器时定义参数组，以应用不同的设置：

  ```python
  optimizer = CustomSGD([
      {'params': model.base.parameters()},
      {'params': model.classifier.parameters(), 'lr': 1e-3} # 分类器使用不同的学习率
  ], lr=1e-4, momentum=0.9) # 其他参数（例如，基础部分）使用默认学习率
  ```
- **闭包：** 如果您的算法需要多次损失评估（如 L-BFGS），您将定义一个 `closure` 函数并将其传递给 `optimizer.step(closure)`。您的 `step` 实现必须适当地调用 `closure()`，可能多次，通常在 `with torch.enable_grad():` 块内。

# 闭包使用示例（具体优化器逻辑不同）

```
def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    return loss
```

# 在 step 方法中（对于类似 L-BFGS 的优化器）

````
# loss = closure() # 内部可能被多次调用
# 使用损失和梯度来更新参数...
```
````

- **与学习率调度器的互动：** 正确管理 `self.param_groups` 的自定义优化器可以与 PyTorch 的学习率调度器 (`torch.optim.lr_scheduler`) 配合使用。调度器会修改每个 `param_group` 中的 `'lr'` 值，然后您的自定义 `step` 函数会读取该值。
- **复杂状态：** 像 Adam 或 AdamW 这样的优化器需要每个参数更多的状态（例如，一阶和二阶矩估计，可能还有步数计数）。您可以通过向 `self.state[p]` 字典添加更多条目来管理这一点。
- **性能：** 尽管 Python 允许快速原型化新的优化器思路，但计算量大的更新规则可能会成为瓶颈。如果分析显示优化器步骤很慢，您可以考虑将核心逻辑实现为自定义 C++ 或 CUDA 扩展以获得最佳性能，正如本章其他部分所讨论的。然而，对于大多数算法，Python 迭代的开销与梯度计算相比可以忽略不计，纯 Python 实现完全足够并且更容易维护。

通过继承 `torch.optim.Optimizer`，您能够实现几乎任何参数更新规则，将新颖的优化研究直接整合到您的 PyTorch 训练流程中，并根据您的特定需求微调 (fine-tuning)学习过程。

## 参考资料

- [\`torch.optim\` - PyTorch documentation](https://pytorch.org/docs/stable/optim.html) — PyTorch Core Team (2025)
  Publisher: PyTorch Foundation
  官方文档提供了PyTorch优化模块的详尽API参考，详细介绍了`torch.optim.Optimizer`基类及其方法，这对于自定义优化器开发至关重要。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: 3rd International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  本文介绍了Adam优化器，这是一个高级优化算法的优秀范例，需要管理更复杂的内部状态，这是在实现除简单SGD之外的自定义优化器时的关键方面。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8: Optimization for Training Deep Models
  这本基础教科书包含一个关于深度学习优化算法的专门章节，解释了SGD、动量和自适应学习率算法等方法的理论基础，为自定义优化器的设计提供了信息。
- [Custom C++ and CUDA Extensions - PyTorch documentation](https://pytorch.org/docs/stable/cpp_extension.html) — PyTorch Core Team (2025)
  Publisher: PyTorch Foundation
  此文档提供了如何使用C++或CUDA实现的自定义操作来扩展PyTorch的说明，这对于优化计算密集型自定义优化器步骤的性能非常有用。

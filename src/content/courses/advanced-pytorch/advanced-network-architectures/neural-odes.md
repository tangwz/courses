---
course: "advanced-pytorch"
chapter: "advanced-network-architectures"
lesson: "neural-odes"
sourceId: 2697
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-2-advanced-network-architectures/neural-odes"
title: "神经常微分方程"
description: "使用神经ODE和合适的求解器实现连续深度模型。"
order: 5
plots: []
sourceHash: "7e6a6e5fb1a285b401376e9dc35cc483e87dcfb07ee684e23512d7f220749f23"
sourceCorrections: []
---

传统深度神经网络 (neural network)，例如残差网络 (ResNets)，通过离散的层序列处理输入。我们可以将ResNet块视为连续变换的欧拉离散化：$h_{t+1} = h_t + f(h_t, \theta_t)$。这种观点自然引出一个问题：我们能否连续地建模这种变换？神经常微分方程 (Neural ODEs) 给出了肯定的答复，它将网络深度定义为连续时间区间，而非层数。

### 连续深度框架

不同于离散变换，神经ODE使用常微分方程 (ODE) 建模隐藏状态 $h(t)$ 随连续时间变量 $t$ 的演变。其主要思想是，使用神经网络 (neural network) $f$（以权重 (weight) $\theta$ 为参数 (parameter)）来定义隐藏状态随时间的变化率：


$$
\frac{dh(t)}{dt} = f(h(t), t, \theta)
$$


在这里，$h(t)$ 表示时间 $t$ 时的隐藏状态，而 $f$ 通常是一个标准神经网络（例如，一个MLP）它以当前状态 $h(t)$、当前时间 $t$ 和参数 $\theta$ 作为输入，输出状态的变化率。

将输入 $z_0$（即 $h(t_0)$）转换为输出 $z_1$（即 $h(t_1)$）的整体过程，是通过在指定时间区间 $[t_0, t_1]$ 上求解此ODE初始值问题得到的：


$$
h(t_1) = h(t_0) + \int_{t_0}^{t_1} f(h(t), t, \theta) dt
$$


这个积分通过ODE求解器进行数值计算。神经网络 $f$ 定义了向量 (vector)场，求解器模拟了隐藏状态通过该向量场从起始时间 $t_0$ 到结束时间 $t_1$ 的路径。

### 神经ODE的优势

这种连续的表述方式具有多项有益的特点：

1. **训练时的内存效率：** 标准反向传播 (backpropagation)需要存储每一层的激活值来计算梯度。对于层数多的网络（或等同于ODE求解器正向传播中的许多步骤），这会消耗大量内存。神经ODE采用*伴随敏感度方法*来计算梯度。该方法涉及逆向求解第二个相关的ODE。重要的是，它计算参数 (parameter) $\theta$ 和初始状态 $h(t_0)$ 所需的梯度时，内存使用量相对于“深度”或积分时间近似为常数。这使得训练具有复杂变换潜力的模型成为可能，而无需承担存储中间状态带来的内存负担。
2. **自适应计算：** 现代ODE求解器在积分过程中会自动调整步长。当动态 $f$ 变化迅速时，它们会采取较小的步长；当动态平滑时，则采取较大的步长。这意味着计算工作量可以适应所学函数的复杂性，与ResNets等固定步长架构相比，可能会带来更高的计算效率。
3. **处理不规则时间序列：** 神经ODE天然适合建模连续过程和在不规则时间点采样的数据。模型可以通过将ODE积分到任意时间 $t$ 来评估该点的隐藏状态。

### 使用PyTorch实现

实现神经ODE通常需要一个提供可微分ODE求解器的外部库。一个常用的选择是 `torchdiffeq`。

通常的工作流程包括：

1. **定义动态函数：** 创建一个标准 `torch.nn.Module` 来表示函数 $f(h(t), t, \theta)$。这个模块以当前状态 `h` 和时间 `t` 作为输入，并返回计算出的导数 `dh/dt`。

   ```python
   import torch
   import torch.nn as nn

   class ODEFunc(nn.Module):
       def __init__(self, hidden_dim):
           super(ODEFunc, self).__init__()
           self.net = nn.Sequential(
               nn.Linear(hidden_dim, hidden_dim),
               nn.Tanh(),
               nn.Linear(hidden_dim, hidden_dim),
           )

       def forward(self, t, h):
           # t：当前时间（标量）
           # h：当前隐藏状态（张量）
           # 返回 dh/dt
           return self.net(h)
   ```
2. **使用ODE求解器：** 使用 `torchdiffeq` 中的 `odeint` 等函数。该函数接收动态函数 `func`、初始状态 `h0`、要评估解的时间点 `t`（例如 `torch.tensor([t0, t1])`）以及可选的求解器参数 (parameter)。它返回在指定时间点计算出的隐藏状态。

   ```python
   # 假设 torchdiffeq 已安装：pip install torchdiffeq
   from torchdiffeq import odeint_adjoint as odeint # 使用伴随方法以节省内存

   # 示例用法：
   func = ODEFunc(hidden_dim=20)
   h0 = torch.randn(batch_size, 20) # 初始状态
   t_span = torch.tensor([0.0, 1.0]) # 从 t=0 积分到 t=1

   # 计算最终状态 h(t1)
   # odeint 通过伴随方法处理数值积分和梯度计算
   h1 = odeint(func, h0, t_span)[-1] # 获取最后一个时间点 (t1) 的状态

   # h1 现在可以在后续层或损失函数中使用
   # 可以通过 h1.backward() 计算 func.parameters() 和 h0 的梯度
   ```

   注意 `odeint_adjoint` 的使用。该版本实现了内存高效的伴随反向传播 (backpropagation)方法。标准的 `odeint` 也可用，但可能占用更多内存。

### 伴随敏感度方法

直接通过ODE求解器的操作进行反向传播 (backpropagation)，可能会耗费大量计算资源和内存，因为它需要存储求解器计算的所有中间状态。伴随方法提供了一种替代方案。

它定义了*伴随状态* $a(t) = \frac{\partial L}{\partial h(t)}$，这表示最终损失 $L$ 对隐藏状态 $h(t)$ 的梯度。该伴随状态的演变由另一个逆时间（从 $t_1$ 到 $t_0$）运行的ODE控制：


$$
\frac{da(t)}{dt} = -a(t)^T \frac{\partial f(h(t), t, \theta)}{\partial h}
$$


损失对参数 (parameter) $\theta$ 的梯度可以通过逆时间积分另一个相关量来计算：


$$
\frac{\partial L}{\partial \theta} = \int_{t_1}^{t_0} a(t)^T \frac{\partial f(h(t), t, \theta)}{\partial \theta} dt
$$


求解这些逆向ODE需要在反向传播过程中获取 $h(t)$ 的值。然而，可以通过再次求解原始正向ODE $\frac{dh(t)}{dt} = f(h(t), t, \theta)$ 来实时重新计算这些值，这次是从 $h(t_1)$ 到 $h(t_0)$ 逆向进行。这种重新计算避免了存储整个正向轨迹，从而大大节省了内存，通常将内存成本从 $O(N_t)$ 降低到 $O(1)$，其中 $N_t$ 是求解器步数。

### 求解器选择与实际考量

`torchdiffeq` 等库提供了多种ODE求解器：

- **显式固定步长求解器：** 欧拉法、中点法、RK4（四阶龙格-库塔法）。简单，但为了稳定性和准确性可能需要较小的步长。
- **自适应步长求解器：** Dormand-Prince (`dopri5`)、Adams方法。自动调整步长，对于平滑问题通常更高效、更准确。`dopri5` 常作为不错的默认选项。
- **隐式求解器：** 对于“刚性”ODE很有用，因为如果步长不大，显式方法会变得不稳定。它们通常在每一步都涉及方程求解，计算开销可能更大。

求解器的选择会影响准确性、稳定性和计算速度。它通常被视为一个需要调整的超参数 (parameter) (hyperparameter)。

**挑战：**

- **计算成本：** 尽管内存高效，但求解ODE可能比固定数量的离散层计算速度慢，特别是使用伴随方法进行反向传播 (backpropagation)时。
- **数值稳定性：** 求解器的选择以及所学动态函数 $f$ 的性质会影响积分过程中的数值稳定性。如果 $f$ 表现不佳，仍然可能出现激活值爆炸或消失的问题。
- **训练动态：** 优化神经ODE有时比优化标准网络更具挑战性。

神经ODE展现了深度学习 (deep learning)与微分方程之间引人入胜的联系。它们提供了一种内存高效的方式来建模复杂、连续的变换，并为涉及连续动态或不规则时间序列数据的问题提供了一种独特的工具，从而扩展了PyTorch中可用的高级网络架构种类。

## 参考资料

- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) — Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud (2018)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2018); Volume: 31; Pages: 6571-6583; DOI: [10.48550/arXiv.1806.07366](https://doi.org/10.48550/arXiv.1806.07366)
  介绍了连续深度神经网络的概念、用于内存高效梯度计算的伴随灵敏度方法以及与残差网络的联系。
- [Numerical Methods for Ordinary Differential Equations](https://doi.org/10.1007/978-0-85729-148-9) — David F. Griffiths, Desmond J. Higham (2010)
  Publisher: Springer; Pages: 271; DOI: [10.1007/978-0-85729-148-9](https://doi.org/10.1007/978-0-85729-148-9)
  一本标准教科书，全面介绍了求解常微分方程的数值方法，涵盖了各种求解器及其特性。

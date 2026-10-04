---
course: "advanced-pytorch"
chapter: "advanced-network-architectures"
lesson: "normalizing-flows"
sourceId: 2694
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-2-advanced-network-architectures/normalizing-flows"
title: "用于生成建模的归一化流"
description: "理解和实现用于密度估计和生成任务的归一化流。"
order: 4
plots: []
sourceHash: "abc7ffaa5c247d1de4204f645bf4bb671b2b3967cccd276b2d89582707374dbe"
sourceCorrections: []
---

与变分自编码器（VAEs）或生成对抗网络 (GAN)（GANs）相比，归一化 (normalization)流为生成建模提供了一种独特的方法。其主要优点在于能够**精确计算数据似然**，同时在数据空间与具有简单分布（如标准高斯分布）的潜在空间之间定义一个**可逆映射**。这使得它们特别适用于需要精确密度估计或可逆性有益的场合。

核心思路是学习一个变换 $f: \mathcal{X} \to \mathcal{Z}$，将复杂的输入数据点 $x \in \mathcal{X}$ 映射到更简单的潜在变量 $z \in \mathcal{Z}$，其中 $p_Z(z)$ 是一个已知、易于处理的概率分布（例如，$\mathcal{N}(0, I)$）。由于 $f$ 被设计为可逆的 ($x = f^{-1}(z)$) 且可微分，我们可以使用概率论中的变量变换定理来关联数据 $p_X(x)$ 的密度与潜在变量 $p_Z(z)$ 的密度。

### 变量变换公式

变量变换公式表明，对于一个将 $x$ 映射到 $z$ 的可逆、可微分函数 $f$，它们概率密度之间的关系是：


$$
p_X(x) = p_Z(f(x)) \left| \det\left(\frac{\partial f(x)}{\partial x^T}\right) \right|
$$


在此，$\frac{\partial f(x)}{\partial x^T}$ 是变换 $f$ 在 $x$ 处计算的雅可比矩阵，而 $\left| \det(\cdot) \right|$ 表示其行列式的绝对值。

这个公式很重要。它告诉我们，如果能计算 $f(x)$ 及其雅可比矩阵的行列式，我们就可以使用已知的 $p_Z$ 密度，计算任何给定数据点 $x$ 的精确概率密度 $p_X(x)$。

对于生成建模，我们通常使用对数似然，这避免了数值下溢并简化了计算：


$$
\log p_X(x) = \log p_Z(f(x)) + \log \left| \det\left(\frac{\partial f(x)}{\partial x^T}\right) \right|
$$


训练归一化 (normalization)流涉及在数据集上最大化此对数似然。这要求变换 $f$ 具备两个性质：

1. 它必须易于可逆（$f^{-1}$ 应该可计算）。
2. 其雅可比行列式必须计算高效。

### 使用双射函数构建流

复杂的变换 $f$ 通常通过组合更简单的可逆函数来构建，这些函数常被称为*双射函数*或*耦合层*：$f = f_L \circ \dots \circ f_2 \circ f_1$。如果每个 $f_i$ 都是可逆的，并且具有易于计算的雅可比行列式，那么复合函数 $f$ 就会继承这些性质。

复合函数 $f$ 的雅可比矩阵是各层雅可比矩阵的乘积：


$$
\frac{\partial f(x)}{\partial x^T} = \frac{\partial f_L(z_{L-1})}{\partial z_{L-1}^T} \dots \frac{\partial f_2(z_1)}{\partial z_1^T} \frac{\partial f_1(x)}{\partial x^T}
$$


其中 $z_i = f_i(z_{i-1})$ 且 $z_0 = x$。

由于性质 $\det(AB) = \det(A)\det(B)$，整体雅可比矩阵的对数行列式变成一个和：


$$
\log \left| \det\left(\frac{\partial f(x)}{\partial x^T}\right) \right| = \sum_{i=1}^L \log \left| \det\left(\frac{\partial f_i(z_{i-1})}{\partial z_{i-1}^T}\right) \right|
$$


这种组合性使我们能够从更简单的模块构建富有表现力的变换，同时保持计算的可处理性。

这是一个归一化 (normalization)流的图示：

> 数据空间 $\mathcal{X}$ 通过一系列可逆函数 $f_1, \dots, f_L$ 变换为简单的潜在空间 $\mathcal{Z}$（例如，高斯分布）。逆变换 $f^{-1}$ 允许通过从 $p_Z(z)$ 中抽取 $z$ 并计算 $x = f^{-1}(z)$ 来从 $p_X(x)$ 采样。

### 常见双射函数架构

设计有效的双射函数是归一化 (normalization)流的核心。一种常用且成功的方法包含**耦合层**。

#### 耦合层（例如，RealNVP，NICE）

Real Non-Volume Preserving (RealNVP) 流，基于NICE (Non-linear Independent Components Estimation) 构建，采用巧妙的掩码策略。它们将输入向量 (vector) $x$ 分成两部分，$x_1$ 和 $x_2$。变换按以下方式进行：

1. 第一部分 $x_1$ 不变通过：$z_1 = x_1$。
2. 第二部分 $x_2$ 使用一个函数（通常是仿射变换）进行变换，其参数 (parameter)由 $x_1$ 决定。例如，一个仿射耦合层计算：
   
   $$
   z_2 = x_2 \odot \exp(s(x_1)) + t(x_1)
   $$
   
   在此，$\odot$ 表示按元素乘法，而 $s(\cdot)$（缩放）和 $t(\cdot)$（平移）是复杂函数，通常作为神经网络 (neural network)实现，以 $x_1$ 为输入。

逆变换也很简单：


$$
x_1 = z_1
$$


$$
x_2 = (z_2 - t(z_1)) \odot \exp(-s(z_1))
$$


此变换的雅可比矩阵是下三角矩阵（或上三角矩阵，取决于哪个部分被变换）：


$$
\frac{\partial z}{\partial x^T} = \begin{pmatrix}
I & 0 \\
\frac{\partial z_2}{\partial x_1^T} & \frac{\partial z_2}{\partial x_2^T}
\end{pmatrix}
$$


值得注意的是，$\frac{\partial z_2}{\partial x_2^T}$ 是一个对角矩阵，其对角元素为 $\exp(s(x_1))$。因此，行列式就是对角元素的乘积：$\prod \exp(s(x_1)) = \exp(\sum s(x_1))$。对数行列式就是缩放网络输出的总和：$\sum s(x_1)$。

这种结构保证了易于求逆和对数行列式的高效计算。为了确保所有维度都被变换，连续的耦合层通常会交换 $x_1$ 和 $x_2$ 的角色或使用不同的掩码。

#### 自回归 (autoregressive)流（例如，MAF，IAF）

另一类重要的模型是**自回归流**。在这些模型中，潜在变量的每个维度 $z_i$ 仅以前面的维度 $z_{1:i-1}$ 为条件（或 $x_{1:i-1}$，取决于方向）。

- **掩码自回归流（MAF）：** 密度估计高效，但采样速度慢，因为采样需要顺序计算。
- **逆自回归流（IAF）：** 采样高效（可并行），但密度估计速度慢。

它们常使用掩码神经网络来强制自回归特性。

### 在PyTorch中实现归一化 (normalization)流

PyTorch的 `torch.distributions` 模块提供了构建归一化流的优秀工具。重要组件包括：

- `torch.distributions.Distribution`：概率分布的基类（如 `Normal`）。
- `torch.distributions.Transform`：可逆变换的基类。提供了如 `AffineTransform`、`ExpTransform` 等实现。通常会继承此基类定义自定义变换。
- `torch.distributions.TransformedDistribution`：通过将一系列 `Transform` 对象应用于基础分布来创建新分布。它自动处理变量变换的计算。
- `torch.distributions.constraints`：用于定义分布的支持域和变换参数 (parameter)的有效性检查。

让我们大致描述一下如何使用耦合层定义一个简单流。您通常会定义一个继承自 `torch.distributions.Transform` 的 `CouplingLayer` 类。

```python
import torch
import torch.nn as nn
import torch.distributions as dist

class CouplingLayer(dist.Transform):
    def __init__(self, input_dim, hidden_dim, mask, base_transform_type='affine'):
        super().__init__()
        self.input_dim = input_dim
        # 确保掩码是二元张量（0和1）
        self.register_buffer('mask', mask) 

        # 定义计算缩放和平移参数的网络 's_t_network'
        self.s_t_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 2) # 输出所有维度的缩放和平移
        )
        
        self.bijective = True
        # 对于仿射变换：定义域 = 实数，值域 = 实数
        self.domain = dist.constraints.real_vector
        self.codomain = dist.constraints.real_vector
        
        # 可选：使用内置变换，如 AffineTransform 
        # if base_transform_type == 'affine' ...（实现细节）

    def _call(self, x):
        """ 应用变换：x -> z """
        x_masked = x * self.mask
        s_t_params = self.s_t_network(x_masked)
        # 将输出分为缩放（s）和平移（t）
        # 确保缩放为正，例如使用 tanh 增加稳定性和缩放
        s = torch.tanh(s_t_params[..., :self.input_dim]) 
        t = s_t_params[..., self.input_dim:]

        # 仅对未掩码的元素应用变换
        # z = x_masked + (x_unmasked * exp(s) + t) * (1 - mask) 
        z = self.mask * x + (1 - self.mask) * (x * torch.exp(s) + t)
        return z

    def _inverse(self, z):
        """ 应用逆变换：z -> x """
        z_masked = z * self.mask
        s_t_params = self.s_t_network(z_masked)
        s = torch.tanh(s_t_params[..., :self.input_dim])
        t = s_t_params[..., self.input_dim:]

        # 仅对未掩码的元素应用逆变换
        # x = z_masked + ((z_unmasked - t) * exp(-s)) * (1 - mask)
        x = self.mask * z + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return x

    def log_abs_det_jacobian(self, x, z):
        """ 计算 log |det J(x)| """
        x_masked = x * self.mask
        s_t_params = self.s_t_network(x_masked)
        s = torch.tanh(s_t_params[..., :self.input_dim])

        # 对数行列式是变换维度上 's' 的总和
        log_det_jacobian = (1 - self.mask) * s 
        # 对变换维度求和
        return log_det_jacobian.sum(-1) 

# 示例用法：
input_dim = 10
hidden_dim = 64
num_flows = 5

# 定义基础分布（例如，标准正态分布）
base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))

# 创建掩码（在步骤之间交替是常见的）
masks = []
for i in range(num_flows):
    mask = torch.zeros(input_dim)
    mask[i % 2::2] = 1 # 简单的交替掩码
    # 或者存在更复杂的掩码策略
    masks.append(mask)
    
# 创建一系列变换（耦合层）
transforms = []
for i in range(num_flows):
    # 交替哪一部分是恒等变换，哪一部分是变换
    current_mask = masks[i] if i % 2 == 0 else (1 - masks[i]) 
    transforms.append(CouplingLayer(input_dim, hidden_dim, current_mask))
    # 可选：在流之间添加置换/激活归一化层

# 构建变换后的分布
flow_dist = dist.TransformedDistribution(base_dist, transforms)

# --- 训练 ---
# optimizer = torch.optim.Adam(flow_dist.parameters(), lr=1e-4)
# 
# for data_batch in dataloader:
#     optimizer.zero_grad()
#     
#     # 计算数据批次的对数概率
#     log_prob = flow_dist.log_prob(data_batch) # 形状：[batch_size]
#     
#     # 最大化对数似然 -> 最小化负对数似然
#     loss = -log_prob.mean() 
#     
#     loss.backward()
#     optimizer.step()

# --- 采样 ---
# n_samples = 64
# samples = flow_dist.sample(torch.Size([n_samples])) # 形状：[n_samples, input_dim]
```

### 优点与考量

**优点：**

- **精确似然：** 允许直接计算和优化数据似然，这与GAN或VAE（它们使用下界）不同。这对于密度估计和模型比较很有益处。
- **可逆性：** 数据与潜在空间之间的映射是明确可逆的，这对于需要在潜在空间中分析的任务很有用。
- **高效采样：** 某些架构（如IAF）支持并行采样。即使是基于耦合的流，通过顺序应用逆变换，通常也能高效地进行采样。

**考量：**

- **架构约束：** 双射函数必须可逆且雅可比矩阵易于处理，这限制了可使用的函数类型。与约束较少的模型相比，这可能会限制表现力，尽管复杂的流仍能对复杂的分布进行建模。
- **计算成本：** 即使雅可比行列式易于处理，计算它们也会在训练期间增加计算开销，特别是对于高维数据或深层流。
- **拓扑结构：** 基本流不能改变空间的拓扑结构；它们是微分同胚。这意味着如果从像高斯分布这样简单的连通基础分布开始，它们可能难以完美地建模具有不连通模式的分布。

### 应用

归一化 (normalization)流已在多个应用场景中得到采用：

- **生成建模：** 生成真实的图像、音频和其他高维数据。
- **密度估计：** 精确建模复杂的概率分布。
- **变分推断：** 在贝叶斯模型中使用流来定义更灵活的近似后验分布，优于像对角高斯这样的简单近似。
- **强化学习 (reinforcement learning)：** 建模复杂的策略或价值函数。
- **表示学习：** 潜在空间 $z$ 可以提供数据 $x$ 的有意义表示。

总之，归一化流为生成建模和密度估计提供了一个数学上优雅且功能强大的框架。凭借可逆变换和变量变换公式，它们允许进行精确的似然计算，为高级深度学习 (deep learning)工具集中特定概率建模任务提供了特有的优点。

## 参考资料

- [Normalizing Flows for Probabilistic Modeling and Inference](https://www.jmlr.org/papers/volume22/19-1028/19-1028.pdf) — George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan (2021)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 22; Pages: 1-70; DOI: [10.5555/3501069.3501170](https://doi.org/10.5555/3501069.3501170)
  对归一化流的全面综述，涵盖理论基础、各种架构和应用。
- [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) — Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1605.08803](https://doi.org/10.48550/arXiv.1605.08803)
  引入了Real NVP，这是一种基于仿射耦合层的广泛使用的归一化流架构。
- [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057) — George Papamakarios, Theo Pavlakou, Iain Murray (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1705.07057](https://doi.org/10.48550/arXiv.1705.07057)
  介绍了掩码自回归流（MAF），这是一种特别适用于密度估计的自回归归一化流。
- [torch.distributions](https://pytorch.org/docs/stable/distributions.html) — PyTorch Developers (2024)
  Publisher: PyTorch
  PyTorch `torch.distributions` 模块的官方文档，对在PyTorch中实现归一化流至关重要。

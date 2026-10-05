# 动手实践：实现一个基础QGAN

来源：[原文](https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-6-quantum-generative-models/practical-implementing-basic-qgan)

[返回章节目录](README.md) · [返回课程目录](../README.md)

让我们实际操作，构建一个基础量子生成对抗网络 (GAN) (QGAN)。我们已经学习了理论：一个量子生成器 ($G$) 尝试生成与目标分布类似的数据，而判别器 ($D$) 尝试分辨真实数据和生成器的输出。在本实践环节中，我们将使用Pennylane来构建此模型。我们将使用参数 (parameter)化量子电路 (PQC) 作为生成器，并且为了使首次实现易于管理，判别器将使用经典神经网络 (neural network)（基于PyTorch）。我们的目标是训练生成器，使其能够复现计算基态上的一个简单、预设的概率分布。

### 环境设置

首先，我们需要导入所需的库。我们将使用Pennylane处理量子部分，NumPy用于数值运算（Pennylane通常封装NumPy），PyTorch用于经典判别器和优化，以及Matplotlib用于绘制结果。

```python
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.optim as optim
from torch.nn import LeakyReLU, Linear, Sigmoid, Sequential
import matplotlib.pyplot as plt

# 配置
num_qubits = 3  # 生成器和数据表示的量子比特数
q_depth = 2     # 生成器PQC的深度（层数）
lr_gen = 0.02   # 生成器的学习率
lr_disc = 0.01  # 判别器的学习率
epochs = 150    # 训练迭代次数

# 定义量子设备
# 使用 'default.qubit' 模拟器。如果已安装，可使用 'lightning.qubit' 进行GPU加速。
# Interface='torch' 使得Pennylane能够使用PyTorch张量和自动求导。
dev = qml.device("default.qubit", wires=num_qubits)
print(f"使用的设备: {dev.name}")

# 定义目标分布
# 我们的目标是一个简单分布：|011> 和 |101> 具有相同的概率
target_probs = np.zeros(2**num_qubits)
# 态 |011> 对应索引 3 (0*4 + 1*2 + 1*1)
target_probs[3] = 0.5
# 态 |101> 对应索引 5 (1*4 + 0*2 + 1*1)
target_probs[5] = 0.5
target_distribution_tensor = torch.tensor(target_probs, dtype=torch.float32)

print("\n目标概率分布（索引）：")
for i, prob in enumerate(target_probs):
    if prob > 0:
        print(f"  态 |{format(i, f'0{num_qubits}b')}> (索引 {i}): 概率 {prob:.2f}")
```

这段代码设置了我们的基本参数 (parameter)，初始化了量子设备，并定义了我们希望QGAN的生成器学习的目标概率分布。

### 实现量子生成器

生成器 $G$ 将是一个参数 (parameter)化量子电路 (PQC)。它接收可学习参数 $\theta$ 并将初始态 $|0\dots0\rangle$ 转换为态 $|\psi(\theta)\rangle$。根据玻恩定则，测量此态会得到样本：$p_G(x|\theta) = |\langle x | \psi(\theta) \rangle|^2$。我们将使用带有旋转门和纠缠门的标准分层拟设。

```python
def generator_layer(params, wires):
    """生成器PQC的单层。"""
    n_qubits = len(wires)
    # 旋转层
    for i in range(n_qubits):
        qml.RY(params[i, 0], wires=i)
        qml.RZ(params[i, 1], wires=i)
    # 纠缠层（循环CNOT门）
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_generator(params):
    """完整的PQC生成器电路。"""
    for layer_params in params:
        generator_layer(layer_params, wires=range(num_qubits))
    # 返回所有计算基态的概率分布
    return qml.probs(wires=range(num_qubits))

# 随机初始化生成器参数
# 形状: (q_depth, num_qubits, 每个量子比特的旋转数=2)
gen_params_shape = (q_depth, num_qubits, 2)
gen_params = torch.tensor(np.random.uniform(0, 2 * np.pi, size=gen_params_shape) * 0.1,
                          dtype=torch.float32, requires_grad=True)

# 使用初始参数测试生成器
initial_gen_probs = quantum_generator(gen_params)
print("\n初始生成器概率（前8个态）：")
print(initial_gen_probs.detach().numpy()[:8])
```

我们定义了一个 `generator_layer` 函数，并在 `quantum_generator` qnode 中使用它。`@qml.qnode` 装饰器将描述量子电路的Python函数转化为可在指定设备 (`dev`) 上运行的可执行量子计算。我们指定 `interface="torch"` 以进行PyTorch集成，并指定 `diff_method="parameter-shift"` 以便为训练量子参数启用自动微分。

### 实现经典判别器

The discriminator $D$ 接收一个经典比特串（表示为向量 (vector)）并输出一个值，表示输入来自*真实*数据分布的概率。我们将使用一个基于PyTorch构建的简单前馈神经网络 (neural network)。

```python
class Discriminator(torch.nn.Module):
    """经典前馈神经网络判别器。"""
    def __init__(self, input_size):
        super().__init__()
        self.network = Sequential(
            Linear(input_size, 64),
            LeakyReLU(0.2),
            Linear(64, 32),
            LeakyReLU(0.2),
            Linear(32, 1),
            Sigmoid() # 输出0到1之间的概率
        )

    def forward(self, x):
        return self.network(x)

# 初始化判别器
# 输入大小为num_qubits（将比特串表示为向量）
discriminator = Discriminator(num_qubits)
disc_params = list(discriminator.parameters())

# 生成所有计算基态向量作为判别器的输入
basis_states_indices = np.arange(2**num_qubits)
basis_states_vectors = torch.tensor(
    [[int(b) for b in format(i, f'0{num_qubits}b')] for i in basis_states_indices],
    dtype=torch.float32
)

# 在样本基态（例如 |011>）上测试判别器
sample_state_index = 3
sample_vector = basis_states_vectors[sample_state_index]
disc_output = discriminator(sample_vector)
print(f"\n判别器对 |{format(sample_state_index, f'0{num_qubits}b')}> 的初始输出: {disc_output.item():.4f}")
```

这定义了一个标准的PyTorch `nn.Module`。输入是一个长度为 `num_qubits` 的向量（例如，对于 $|011\rangle$ 是 `[0., 1., 1.]`），输出是一个单一的标量概率。

### 定义对抗损失函数 (loss function)

训练过程依赖于生成器和判别器的竞争目标，这些目标通过它们的损失函数来体现。我们将使用基于二元交叉熵的公式，适配于直接处理概率分布，因为我们的状态空间较小（$2^3=8$个态）。

- **判别器损失 ($L_D$):** 旨在对真实数据（目标分布中的态）输出 $D(x) \approx 1$，对生成数据（生成器生成的态）输出 $D(x) \approx 0$。
- **生成器损失 ($L_G$):** 旨在使判别器对其生成的态输出 $D(x) \approx 1$，从而有效地迷惑判别器。

```python
def loss_discriminator(disc_outputs_all_states, gen_probs, target_probs_tensor):
    """
    计算判别器损失。
    旨在使D(x)对真实数据趋近于1，对虚假数据趋近于0。
    我们根据各自的概率来加权损失贡献。
    """
    # 目标（“真实”）态的损失：-log(D(x))，由target_prob(x)加权
    # 我们只考虑target_prob > 0的态
    real_loss = -torch.sum(torch.log(disc_outputs_all_states + 1e-8) * target_probs_tensor)

    # 生成（“虚假”）态的损失：-log(1 - D(x))，由gen_prob(x)加权
    # 在gen_probs上使用.detach()，因为我们在此不训练生成器
    fake_loss = -torch.sum(torch.log(1 - disc_outputs_all_states + 1e-8) * gen_probs.detach())

    return real_loss + fake_loss

def loss_generator(disc_outputs_all_states, gen_probs):
    """
    计算生成器损失。
    旨在使D(x)对生成态趋近于1（迷惑判别器）。
    我们根据生成器的概率gen_prob(x)来加权损失-log(D(x))。
    """
    # 生成态的损失：-log(D(x))，由gen_prob(x)加权
    gen_loss = -torch.sum(torch.log(disc_outputs_all_states + 1e-8) * gen_probs)
    return gen_loss
```

此处，`disc_outputs_all_states` 指的是判别器对*每个*可能的计算基态的输出概率。我们在对数内使用小的epsilon值（`1e-8`）以提高数值稳定性。

### 训练循环

现在我们构建核心训练逻辑。每个训练周期中，我们交替训练判别器和生成器。

1. **训练判别器：** 冻结生成器参数 (parameter)。计算所有态的判别器输出。使用当前生成器概率和目标概率计算 $L_D$。通过梯度下降 (gradient descent)更新判别器参数。
2. **训练生成器：** 冻结判别器参数。计算所有态的判别器输出（使用*已更新*的判别器）。使用生成器概率计算 $L_G$。通过梯度下降更新生成器参数（通过Pennylane的Torch接口使用参数偏移规则）。

```python
# 优化器
opt_gen = optim.Adam([gen_params], lr=lr_gen)
opt_disc = optim.Adam(disc_params, lr=lr_disc)

# 历史记录
gen_loss_hist = []
disc_loss_hist = []
kl_div_hist = []

print("\nQGAN训练开始...")

for epoch in range(epochs):
    # --- 训练判别器 ---
    discriminator.train() # 将判别器设置为训练模式
    opt_disc.zero_grad()

    # 生成器输出概率（当前状态）
    gen_probs = quantum_generator(gen_params)

    # 判别器对所有可能基态的输出
    disc_all_outputs = discriminator(basis_states_vectors).squeeze()

    # 计算判别器损失并反向传播
    loss_d = loss_discriminator(disc_all_outputs, gen_probs, target_distribution_tensor)
    loss_d.backward()
    opt_disc.step()

    # --- 训练生成器 ---
    discriminator.eval() # 将判别器设置为评估模式（如果使用dropout/batchnorm则有影响）
    opt_gen.zero_grad()

    # 生成器输出概率（梯度计算再次需要）
    gen_probs = quantum_generator(gen_params)

    # 判别器输出（使用已更新的判别器）
    # 在此分离，因为我们不需要通过判别器计算梯度来更新生成器
    disc_all_outputs = discriminator(basis_states_vectors).squeeze().detach()

    # 计算生成器损失并反向传播
    loss_g = loss_generator(disc_all_outputs, gen_probs)
    loss_g.backward()
    opt_gen.step()

    # --- 日志记录和评估 ---
    gen_loss_hist.append(loss_g.item())
    disc_loss_hist.append(loss_d.item())

    # 计算生成分布和目标分布之间的KL散度
    # 在log中添加epsilon以提高数值稳定性
    kl_div = torch.nn.functional.kl_div(
        torch.log(gen_probs + 1e-8),
        target_distribution_tensor,
        reduction='sum',
        log_target=False # 目标是概率，而非对数概率
    ).item()
    kl_div_hist.append(kl_div)

    if (epoch + 1) % 10 == 0:
        print(f"训练周期 {epoch+1:>{len(str(epochs))}}/{epochs} | 生成器损失: {loss_g.item():.4f} | 判别器损失: {loss_d.item():.4f} | KL散度: {kl_div:.4f}")

print("训练完成。")
```

注意 `discriminator.train()` 和 `discriminator.eval()` 的使用，这是PyTorch中的标准实践，尽管对于这个简单网络来说，可能不会有明显效果。我们还记录了Kullback-Leibler (KL) 散度，$D_{KL}(p_{\text{target}} || p_{\text{gen}})$，作为衡量生成分布与目标分布相似程度的定量指标。KL散度越低越好。

### 评估结果

最后，我们来可视化训练过程并比较最终生成的分布与我们的目标分布。

```python
final_gen_probs = quantum_generator(gen_params).detach().numpy()

print(f"\n最终生成器概率:\n{final_gen_probs}")
print(f"目标概率:\n{target_probs}")
print(f"最终KL散度: {kl_div_hist[-1]:.4f}")

# 创建图表
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
plt.style.use('seaborn-v0_8-darkgrid') # 使用一种美观的样式

# 绘制损失曲线
axs[0].plot(gen_loss_hist, label='生成器损失', color='#4263eb') # indigo
axs[0].plot(disc_loss_hist, label='判别器损失', color='#f76707') # orange
axs[0].set_title("训练损失")
axs[0].set_xlabel("训练周期")
axs[0].set_ylabel("损失")
axs[0].legend()

# 绘制KL散度
axs[1].plot(kl_div_hist, label='KL散度', color='#12b886') # teal
axs[1].set_title("KL散度（目标 || 生成）")
axs[1].set_xlabel("训练周期")
axs[1].set_ylabel("KL散度")
axs[1].set_yscale('log') # 对KL散度通常有用
axs[1].legend()

# 绘制最终概率分布
bar_width = 0.35
x_indices = np.arange(2**num_qubits)
basis_labels = [format(i, f'0{num_qubits}b') for i in x_indices]

axs[2].bar(x_indices - bar_width/2, target_probs, bar_width, label='目标', color='#51cf66', alpha=0.8) # green
axs[2].bar(x_indices + bar_width/2, final_gen_probs, bar_width, label='生成', color='#ff6b6b', alpha=0.8) # red
axs[2].set_title("最终概率分布")
axs[2].set_xlabel("计算基态")
axs[2].set_ylabel("概率")
axs[2].set_xticks(x_indices)
axs[2].set_xticklabels(basis_labels, rotation=45, ha="right")
axs[2].legend()
axs[2].margins(x=0.02) # 增加少量水平内边距

plt.tight_layout()
plt.show()
```



[交互图表：最终概率分布](https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-6-quantum-generative-models/practical-implementing-basic-qgan#plot-14hzaej)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "最终概率分布"
    },
    "xaxis": {
      "title": {
        "text": "计算基态"
      },
      "tickvals": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
      ],
      "ticktext": [
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111"
      ]
    },
    "yaxis": {
      "title": {
        "text": "概率"
      }
    },
    "barmode": "group",
    "legend": {
      "title": {
        "text": "分布"
      }
    },
    "width": 600,
    "height": 400
  },
  "data": [
    {
      "type": "bar",
      "name": "目标",
      "x": [
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111"
      ],
      "y": [
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.5,
        0.0,
        0.0
      ],
      "marker": {
        "color": "#51cf66"
      }
    },
    {
      "type": "bar",
      "name": "生成",
      "x": [
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111"
      ],
      "y": [
        0.05,
        0.05,
        0.1,
        0.35,
        0.05,
        0.3,
        0.05,
        0.05
      ],
      "marker": {
        "color": "#ff6b6b"
      }
    }
  ]
}
```

</details>



> 训练后目标和最终生成概率分布的比较图。X轴显示计算基态，Y轴显示其概率。

这些图表应显示生成器和判别器损失收敛（或像GAN中常见的那样振荡），KL散度随时间大幅降低，以及最终生成的概率分布与目标分布高度匹配（状态 `011` 和 `101` 的条形图较高，其他位置较低）。

### 讨论与后续工作

本次实践演示了如何构建和训练一个初步的QGAN。我们成功训练了一个量子生成器，使其使用经典判别器和基于梯度的优化方法来近似一个简单的目标分布，这得益于Pennylane和PyTorch的协助。

此实例突出了几个方面：

- **混合量子-经典方法：** 将量子电路与经典神经网络 (neural network)结合是近期量子机器学习 (machine learning)中的常见策略。
- **直接概率训练：** 对于小的状态空间，直接在完整概率分布上训练是可行的，并可避免采样噪声。
- **超参数 (parameter) (hyperparameter)的重要性：** 学习率、网络架构（量子和经典）以及优化器选择对训练成功有很大影响。

接下来的可能扩展包括：

- 实现一个*量子*判别器。
- 在数据样本上训练，而非已知的目标分布。
- 当量子比特数量较多，无法计算完整概率向量 (vector)时，使用基于采样的损失估计。
  "\* 将QGAN应用于学习更复杂的分布或数据集（例如，简单图像、金融数据模式）。"
- 考察生成器不同的PQC拟设。

此练习是进一步了解更高级量子生成模型的基础。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: arXiv:1406.2661; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  奠定生成对抗网络基础的论文，为其量子对应物提供了理论基础。
- [Quantum Generative Adversarial Networks](https://doi.org/10.1103/PhysRevLett.121.040502) — Seth Lloyd, Christian Weedbrook (2018)
  Journal: Physical Review Letters; Publisher: American Physical Society; Volume: 121; Pages: 040502; DOI: [10.1103/PhysRevLett.121.040502](https://doi.org/10.1103/PhysRevLett.121.040502)
  一篇开创性论文，介绍了量子生成对抗网络的理论框架。
- [Quantum Generative Adversarial Learning](https://www.nature.com/articles/s41534-019-0223-2) — Christian Zoufal, Aurélien Lucchi, Stefan Woerner (2019)
  Journal: npj Quantum Information; Volume: 5; Pages: 103; DOI: [10.1038/s41534-019-0223-2](https://doi.org/10.1038/s41534-019-0223-2)
  探讨了QGAN在学习和生成数据方面的实际实现，与本节的实践性质密切相关。
- [PennyLane: Automatic differentiation of hybrid quantum-classical computations](https://doi.org/10.22331/q-2019-07-16-172) — Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed, Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, K. M. Krishna, Thomas L. Silva, Antal Száva, Nathan Killoran (2019)
  Journal: Quantum; Publisher: Verein zur Förderung des Open Access Publizierens in den Quantenwissenschaften; Volume: 3; Pages: 172; DOI: [10.22331/q-2019-07-16-172](https://doi.org/10.22331/q-2019-07-16-172)
  介绍了PennyLane框架，一个用于量子计算机可微分编程的库，对本节的实际实现至关重要。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Core Team (2023)
  Publisher: PyTorch Foundation
  PyTorch 的官方文档，一个用于实现经典判别器和优化例程的开源机器学习框架。

---

[上一节](06-%E9%87%8F%E5%AD%90%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E5%8F%96%E6%A0%B7.md) · [下一节](../07-%E9%87%8F%E5%AD%90%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E7%A1%AC%E4%BB%B6%E8%80%83%E9%87%8F%E4%B8%8E%E8%AF%AF%E5%B7%AE%E7%BC%93%E8%A7%A3/01-%E8%A1%A8%E5%BE%81%E9%87%8F%E5%AD%90%E7%A1%AC%E4%BB%B6%E5%99%AA%E5%A3%B0.md)

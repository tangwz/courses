# 动手实践：实现DP-FedAvg

来源：[原文](https://apxml.com/zh/courses/federated-learning/chapter-3-enhancing-privacy-federated-learning/practice-implementing-dp-fedavg)

[返回章节目录](README.md) · [返回课程目录](../README.md)

DP-FedAvg通过修改标准联邦平均算法，将客户端差分隐私纳入其中，从而在联邦学习中应用了差分隐私理论。目标是在训练有用全局模型的同时，保护单个客户端的贡献。

如本章前面所述，在FedAvg中简单平均客户端更新并不能阻止关于客户端本地数据集的潜在信息泄露。DP-FedAvg通过让每个客户端在将更新发送给服务器之前，在本地对其进行扰动来处理此问题。这种扰动包含两个主要步骤：剪裁更新以限制其敏感度，并加入校准过的噪声。

### DP-FedAvg的核心修改

从FedAvg到DP-FedAvg的转变主要涉及本地更新计算和传输阶段客户端的变化。服务器端聚合基本保持不变，仅是简单平均接收到的（现在带噪声的）更新。

以下是一个DP-FedAvg轮次中客户端过程的详细说明：

1. **接收全局模型：** 客户端从服务器接收当前全局模型参数 (parameter)$w_t$。
2. **计算本地更新：** 客户端根据其本地数据计算其本地更新$u_i$，通常通过多个本地SGD步骤完成。此更新可以是梯度$\nabla L_i(w_t)$，也可以是更新后的本地模型与接收到的全局模型$w_{t,i}^{local} - w_t$之间的差异。
3. **剪裁更新范数：** 这是第一个DP特有的步骤。客户端计算其更新的L2范数$||u_i||_2$。如果此范数超出预定义的剪裁阈值$S$（也称为L2敏感度边界），更新将被按比例缩小，使其范数恰好等于$S$。否则，它保持不变。
   
   $$
   u'_i = \frac{u_i}{\max(1, \frac{||u_i||_2}{S})}
   $$
   
   剪裁保证了单个客户端更新对平均值的最大可能影响受到限制，这对于校准噪声来说是必不可少的。$S$的选择是一个代表权衡的超参数 (hyperparameter)：较小的$S$为对抗异常值更新提供了更强的保护，但可能会丢弃那些范数天然较大的更新中的有用信息。较大的$S$保留更多信息，但需要更多噪声以实现同等隐私级别。
4. **添加噪声：** 第二个DP特有的步骤涉及向剪裁后的更新$u'_i$添加噪声。最常用的方法是高斯机制。从高斯分布$\mathcal{N}(0, \sigma^2 I)$中采样的噪声被添加到$u'_i$。
   
   $$
   \tilde{u}_i = u'_i + \mathcal{N}(0, \sigma^2 I)
   $$
   
   此处，$I$是单位矩阵，且$\sigma$是噪声的标准差。噪声的尺度$\sigma$根据剪裁范数$S$、期望的隐私参数$(\epsilon, \delta)$以及可能的参与者数量仔细校准。一个典型关系是$\sigma = z \cdot S$，其中$z$是从隐私参数导出的噪声乘数。对于高斯机制，实现$(\epsilon, \delta)$-DP通常需要设置$\sigma \propto S \frac{\sqrt{\log(1/\delta)}}{\epsilon}$。更高的隐私（更小的$\epsilon$、更小的$\delta$）需要更大的$\sigma$（更多噪声）。
5. **发送带噪声更新：** 客户端将最终的带噪声、已剪裁更新$\tilde{u}_i$发送给中央服务器。

服务器随后聚合这些带噪声的更新，通常通过简单平均：$\Delta w = \frac{1}{N} \sum_{i=1}^{N} \tilde{u}_i$，并更新全局模型$w_{t+1} = w_t + \eta \Delta w$（其中$\eta$是服务器学习率，通常设为1）。

### 代码片段

让我们使用类似Python的伪代码演示客户端剪裁和噪声添加，假设本地更新`local_update`（一个张量）已经计算完成。我们将使用类似PyTorch的语法进行张量操作。

```python
import torch

def dp_client_update(local_update, clipping_norm, noise_multiplier):
    """
    应用剪裁和噪声添加以实现差分隐私。

    Args:
        local_update (torch.Tensor): 计算得到的本地模型更新（例如，梯度）。
        clipping_norm (float): 更新的最大L2范数S。
        noise_multiplier (float): 确定相对于S的噪声标准差的因子z。
                                  通常从(epsilon, delta)导出。

    Returns:
        torch.Tensor: 准备传输的带噪声、已剪裁更新。
    """
    # 计算更新的L2范数
    update_norm = torch.linalg.norm(local_update.flatten(), ord=2)

    # 1. 剪裁
    # 计算缩放因子，避免除以零
    scale_factor = min(1.0, clipping_norm / (update_norm + 1e-9))
    clipped_update = local_update * scale_factor

    # 2. 噪声添加（高斯机制）
    # 计算噪声标准差
    noise_stddev = noise_multiplier * clipping_norm
    # 生成与更新形状相同的高斯噪声
    noise = torch.normal(0, noise_stddev, size=local_update.shape, device=local_update.device)

    noisy_update = clipped_update + noise

    # 记录使用的隐私参数 (epsilon, delta) - 计算取决于具体的机制
    # print(f"Applied DP: Clipping Norm={clipping_norm}, Noise StdDev={noise_stddev}")

    return noisy_update

# --- 客户端训练循环内的示例用法 ---
# 假设：
# model_update = compute_local_update(...) # 计算梯度或模型差值
# S = 1.0 # 剪裁范数超参数
# Z = 1.1 # 噪声乘数超参数（与epsilon, delta相关）

# 应用DP修改
private_update = dp_client_update(model_update, clipping_norm=S, noise_multiplier=Z)

# 将'private_update'发送到服务器，而不是'model_update'
# send_to_server(private_update)
```

此代码片段侧重于核心DP逻辑。一个完整的实现将需要将其集成到一个联邦学习框架中（如TensorFlow Federated、PySyft或Flower），并仔细管理跨轮次的隐私参数 (parameter)和预算累积。

### 对性能的影响

引入噪声不可避免地会影响学习过程。与标准FedAvg相比，实现DP-FedAvg时预期会产生以下影响：

- **收敛速度减慢：** 添加到更新中的噪声可能会模糊真实的梯度方向，可能需要更多通信轮次才能达到目标准确度。
- **最终准确度降低：** 隐私级别($\epsilon, \delta$)与模型能达到的最大效用（准确度）之间通常存在权衡。更强的隐私保障（更多噪声）通常会导致更低的最终准确度上限。
- **超参数 (parameter) (hyperparameter)调优的重要性增加：** DP-FedAvg引入了新的超参数（$S$、$\epsilon$、$\delta$或噪声乘数$z$），需要与标准FL超参数（学习率、本地周期数、每轮客户端数量）一同仔细调优。最佳剪裁范数$S$尤为重要；它通常需要根据训练期间观察到的更新范数分布进行经验性调优。

下图说明了准确度-隐私权衡。



[交互图表：DP噪声对准确度的影响](https://apxml.com/zh/courses/federated-learning/chapter-3-enhancing-privacy-federated-learning/practice-implementing-dp-fedavg#plot-ffjlcr)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100
      ],
      "y": [
        10,
        35,
        55,
        68,
        75,
        80,
        83,
        85,
        86,
        87,
        87.5
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "FedAvg (无DP)",
      "line": {
        "color": "#228be6"
      }
    },
    {
      "x": [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100
      ],
      "y": [
        10,
        32,
        50,
        62,
        69,
        74,
        77,
        79,
        81,
        82,
        82.5
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "DP-FedAvg (低噪声，例如 ε=8)",
      "line": {
        "color": "#fd7e14"
      }
    },
    {
      "x": [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100
      ],
      "y": [
        10,
        25,
        38,
        48,
        55,
        60,
        63,
        65,
        67,
        68,
        68.5
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "DP-FedAvg (高噪声，例如 ε=1)",
      "line": {
        "color": "#fa5252"
      }
    }
  ],
  "layout": {
    "title": {
      "text": "DP噪声对准确度的影响"
    },
    "xaxis": {
      "title": {
        "text": "通信轮次"
      }
    },
    "yaxis": {
      "title": {
        "text": "全局模型准确度 (%)"
      },
      "range": [
        0,
        100
      ]
    },
    "legend": {
      "title": {
        "text": "算法"
      }
    },
    "margin": {
      "l": 60,
      "r": 10,
      "t": 40,
      "b": 40
    }
  }
}
```

</details>



> 标准FedAvg和不同噪声水平的DP-FedAvg在通信轮次中的模型准确度进展（更小的epsilon意味着更高的噪声和更强的隐私保护）。较高的噪声通常会导致收敛速度减慢和最终准确度降低。

### 管理隐私预算

请记住，$(\epsilon, \delta)$参数 (parameter)适用于DP-FedAvg的单个轮次。在训练过程中（多个轮次），总隐私损失会累积。您需要使用组合定理（如高级组合定理）来计算基于每轮参数和总轮次数量的整个训练过程的整体$(\epsilon, \delta)$保障。管理此累积隐私预算是负责任地部署DP-FL系统的一个重要方面。FL框架内的隐私核算库或功能可以帮助自动化此计算。

本实践练习为在联邦学习中实现基本差分隐私奠定了基础。尽管有效，DP-FedAvg只是一种方法。后续章节和部分将介绍更高级的隐私技术和优化。

## 参考资料

- [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) — Cynthia Dwork and Aaron Roth (2014)
  Publisher: Now Publishers
  理解差分隐私理论基础的权威教科书，涵盖高斯机制和隐私组合等内容。
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) — Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang (2016)
  Journal: Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (ACM CCS); Publisher: ACM; Pages: 308-318; DOI: [10.1145/2976749.2978318](https://doi.org/10.1145/2976749.2978318)
  介绍了差分隐私随机梯度下降（DP-SGD）算法，该算法为DP-FedAvg的客户端裁剪和噪声添加机制提供了基础，并涵盖了RDP隐私核算。
- [Differentially Private Federated Averaging Tutorial](https://www.tensorflow.org/federated/tutorials/implementing_dp_fedavg) — TensorFlow Federated Team (2024)
  Journal: TensorFlow Federated Documentation
  TensorFlow Federated框架的实用指南，演示了如何实现差分隐私联邦平均，包括代码示例和配置。

---

[上一节](07-%E9%9A%90%E7%A7%81%E6%94%BB%E5%87%BB%EF%BC%9A%E6%8E%A8%E6%96%AD%E4%B8%8E%E9%87%8D%E5%BB%BA.md) · [下一节](../04-%E5%A4%84%E7%90%86%E5%BC%82%E8%B4%A8%E6%80%A7%E5%92%8C%E4%B8%AA%E6%80%A7%E5%8C%96/01-%E5%BC%82%E6%9E%84%E6%80%A7%E7%9A%84%E6%9D%A5%E6%BA%90%EF%BC%9A%E7%BB%9F%E8%AE%A1%E6%80%A7%E4%B8%8E%E7%B3%BB%E7%BB%9F%E6%80%A7.md)

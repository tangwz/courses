# SCAFFOLD: 联邦优化中的方差降低

来源：[原文](https://apxml.com/zh/courses/federated-learning/chapter-2-advanced-aggregation-algorithms/scaffold-algorithm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然FedAvg为客户端更新聚合提供了一个简单的基础方法，但其性能在存在统计异构性（非独立同分布数据）时常会大幅下降。在差异很大的本地数据集上训练的客户端可能会将全局模型推向冲突的方向，导致收敛缓慢、振荡甚至发散。这种现象常被称为“客户端漂移”，其产生是由于每个客户端所追求的本地优化目标与全局目标之间存在差异。

SCAFFOLD（联邦学习中的随机控制平均）通过引入*控制变量*，提供了一个巧妙的解决办法来减轻这种客户端漂移。主要思路是估算每个客户端在拥有全局数据分布的情况下*本会*采取的更新方向，然后根据此估算值修正实际的本地更新。这可以降低客户端更新之间的方差，使它们与全局目标更一致，从而加速收敛。

### SCAFFOLD的机制

SCAFFOLD通过在服务器和客户端上维护额外的状态信息，修改了标准的联邦学习过程：

1. **服务器控制变量 ($c$):** 表示联邦中*所有*数据平均梯度方向的估算值。
2. **客户端控制变量 ($c_k$):** 每个客户端 $k$ 维护自己的控制变量 $c_k$，表示基于其*本地*数据的梯度方向估算值。

差值 $c - c_k$ 有效地反映了客户端 $k$ 的“漂移”。SCAFFOLD使用这些控制变量来调整本地训练过程和聚合步骤。

### 算法概述

设 $w^t$ 为通信轮次 $t$ 时的全局模型权重 (weight)。

1. **服务器广播:** 服务器将其当前的全局模型 $w^t$ 和控制变量 $c^t$ 发送给选定的客户端。
2. **客户端计算:** 每个参与的客户端 $k$ 执行以下步骤：

   - 初始化其本地模型 $w_k^t = w^t$。
   - 接收服务器控制变量 $c^t$。
   - 执行 $E$ 步本地随机梯度下降 (gradient descent)（SGD）。对于每个本地步骤 $\tau = 0, ..., E-1$，使用本地小批量 $b$：
     - 计算本地梯度：$g_k(w_{k, \tau}^t) = \nabla F_k(w_{k, \tau}^t; b)$，其中 $F_k$ 是客户端 $k$ 的本地损失函数 (loss function)。
     - 使用校正后的梯度更新本地模型：
       
       $$
       w_{k, \tau+1}^t = w_{k, \tau}^t - \eta_l (g_k(w_{k, \tau}^t) - c_k^t + c^t)
       $$
       
       此处，$\eta_l$ 是本地学习率。请注意本地梯度 $g_k$ 是如何通过服务器控制变量 $c^t$ 与客户端控制变量 $c_k^t$ 之间的差值进行调整的。这种调整旨在纠正客户端的本地漂移。
   - 计算总的模型更新方向：$\Delta w_k^t = w_{k, E}^t - w^t$。
   - 更新客户端控制变量 $c_k$。一种常见的方法是：
     
     $$
     c_k^{t+1} = c_k^t - c^t + \frac{1}{E \eta_l} (w^t - w_{k, E}^t)
     $$
     
     此更新反映了在 $E$ 步中观察到的平均本地梯度方向。
   - 计算客户端控制变量的变化：$\Delta c_k^t = c_k^{t+1} - c_k^t$。
   - 将 $\Delta w_k^t$ 和 $\Delta c_k^t$ 发送回服务器。
3. **服务器聚合:** 服务器从参与客户端集合 $S_t$ 接收更新 $(\Delta w_k^t, \Delta c_k^t)$。

   - 聚合模型更新（类似于FedAvg）：
     
     $$
     \Delta w^t = \frac{1}{|S_t|} \sum_{k \in S_t} \Delta w_k^t
     $$
     
   - 更新全局模型：
     
     $$
     w^{t+1} = w^t + \eta_g \Delta w^t
     $$
     
     （$\eta_g$ 是服务器学习率，通常设为1）。
   - 聚合控制变量更新：
     
     $$
     \Delta c^t = \frac{1}{|S_t|} \sum_{k \in S_t} \Delta c_k^t
     $$
     
   - 更新服务器控制变量：
     
     $$
     c^{t+1} = c^t + \Delta c^t
     $$
     

### 方差降低为何有效？

主要原因是，客户端更新中使用的项 $(g_k(w_{k, \tau}^t) - c_k^t + c^t)$ 是对*全局*梯度 $\nabla F(w_{k, \tau}^t)$ 的更好估算，优于原始的本地梯度 $g_k(w_{k, \tau}^t)$。通过减去估算的本地方向 ($c_k^t$) 并加上估算的全局方向 ($c^t$)，SCAFFOLD 有效引导客户端更新趋向全局最小值，减少了由不同本地数据分布引起的方差。

与FedAvg相比，这种方差降低带来了更稳定且通常更快的收敛，特别是在处理明显的统计异构性（非独立同分布数据）时。



[交互图表：收敛性比较（模拟非独立同分布数据）](https://apxml.com/zh/courses/federated-learning/chapter-2-advanced-aggregation-algorithms/scaffold-algorithm#plot-mh8k5l)

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
        0.1,
        0.25,
        0.38,
        0.48,
        0.55,
        0.6,
        0.63,
        0.65,
        0.67,
        0.68,
        0.69
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "FedAvg",
      "line": {
        "color": "#339af0"
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
        0.1,
        0.35,
        0.55,
        0.68,
        0.75,
        0.79,
        0.82,
        0.84,
        0.85,
        0.86,
        0.87
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "FedProx",
      "line": {
        "color": "#ff922b"
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
        0.1,
        0.45,
        0.68,
        0.8,
        0.86,
        0.89,
        0.91,
        0.92,
        0.925,
        0.93,
        0.93
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "SCAFFOLD",
      "line": {
        "color": "#51cf66"
      }
    }
  ],
  "layout": {
    "title": "收敛性比较（模拟非独立同分布数据）",
    "xaxis": {
      "title": "通信轮次"
    },
    "yaxis": {
      "title": "全局模型精度",
      "range": [
        0,
        1
      ]
    },
    "legend": {
      "traceorder": "normal"
    }
  }
}
```

</details>



> FedAvg、FedProx和SCAFFOLD在模拟非独立同分布条件下的收敛行为。SCAFFOLD由于其方差降低原理，通常实现更快的收敛和可能更高的最终精度。

### 实现考量

与FedAvg相比，实现SCAFFOLD会带来一些额外开销：

- **状态:** 服务器和客户端都需要存储各自的控制变量（$c$ 和 $c_k$）。这些变量通常与模型参数 (parameter)的大小相同。
- **通信:** 客户端在每个轮次中需要同时将模型更新（$\Delta w_k$）和控制变量更新（$\Delta c_k$）发送给服务器。服务器除了广播全局模型 $w$ 之外，还需要广播全局控制变量 $c$。这实际上使每轮通信成本比FedAvg增加了一倍，尽管收敛所需的轮次*数量*上可能的减少通常可以弥补这一点。

尽管状态和通信要求有所增加，但SCAFFOLD有效应对客户端漂移的特性，使其成为提升在异构数据上运行的联邦学习系统性能和可靠性的有用的工具。其理论依据在非独立同分布设置下，比FedAvg给出更强的收敛保证。

## 参考资料

- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) — Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh (2020)
  Journal: Proceedings of the 37th International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1910.06378](https://doi.org/10.48550/arXiv.1910.06378)
  介绍了SCAFFOLD算法、其缓解客户端漂移的理论基础以及在非IID数据上的实证性能。
- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) — Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith (2020)
  Journal: Proceedings of Machine Learning and Systems (MLSys); DOI: [10.48550/arXiv.1812.06127](https://doi.org/10.48550/arXiv.1812.06127)
  提出了FedProx，通过在本地目标函数中加入近端项来解决联邦学习中客户端漂移的另一种方法。

---

[上一节](02-FedProx%EF%BC%9A%E5%A4%84%E7%90%86%E7%BB%9F%E8%AE%A1%E5%BC%82%E8%B4%A8%E6%80%A7.md) · [下一节](04-FedNova%EF%BC%9A%E9%92%88%E5%AF%B9%E5%BC%82%E6%9E%84%E7%B3%BB%E7%BB%9F%E7%9A%84%E8%A7%84%E8%8C%83%E5%8C%96%E5%B9%B3%E5%9D%87.md)

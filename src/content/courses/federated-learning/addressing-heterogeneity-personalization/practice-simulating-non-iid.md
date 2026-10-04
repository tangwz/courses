---
course: "federated-learning"
chapter: "addressing-heterogeneity-personalization"
lesson: "practice-simulating-non-iid"
sourceId: 3588
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-4-addressing-heterogeneity-personalization/practice-simulating-non-iid"
title: "实践：模拟非独立同分布数据及缓解方法"
description: "动手模拟非独立同分布数据分布，并应用FedProx或聚类等技术。"
order: 7
plots: ["plots/3588-0.json"]
sourceHash: "86b9f0953de1831acac8a8080c1e22ac07913f74cd424915d6790083fae932da"
sourceCorrections: []
---

模拟联邦学习场景，特别是涉及非独立同分布 (Non-IID) 数据的场景，是算法开发和评估中的主要步骤。它使我们能够在受控但实际的条件下了解不同策略的性能，然后再考虑复杂的部署。

本次实践练习将指导您模拟非独立同分布数据分布，并比较标准联邦平均 (FedAvg) 与一种缓解技术（具体而言，即我们在第2章中介绍的FedProx）的性能。

### 设置模拟

大多数联邦学习模拟都包含三个主要组成部分：一个中心服务器、一组客户端以及在这些客户端之间划分的数据集。您可以使用 TensorFlow Federated (TFF)、PySyft 或 Flower 等框架（如第6章所述）来简化此设置。对于本次练习，我们将侧重于这些步骤，它们适用于不同的框架。

我们通常从适用于分类的标准数据集开始，例如 MNIST 或 CIFAR-10。重要的步骤是如何在模拟客户端之间分配这些数据以创建统计异质性。

### 模拟非独立同分布数据

创建真实的非独立同分布数据对于有意义的模拟非常重要。一种常用且实用的方法是使用**狄利克雷分布**在客户端之间分配类别标签。

假设我们有 $N$ 个客户端和 $C$ 个数据类别。我们可以使用概率向量 (vector) $p_k = (p_{k,1}, ..., p_{k,C})$ 来建模客户端 $k$ 上的类别分布，其中 $p_{k,j}$ 是客户端 $k$ 上属于类别 $j$ 的样本比例。为了生成异质分布，我们可以从带有集中参数 (parameter) $\alpha$ 的狄利克雷分布中抽取每个 $p_k$：

$p_k \sim \text{狄利克雷}(\alpha)$

- $\alpha$ 值较小（例如，$\alpha = 0.1$）会导致高异质性。每个客户端的数据很可能只由一个或少数几个类别主导。
- $\alpha$ 值较大（例如，$\alpha = 100$）会导致分布更接近独立同分布，其中每个客户端的类别分布与全局分布相似。

**实现步骤：**

1. **选择数据集：** 选择一个基准数据集（例如，具有10个类别的 CIFAR-10）。
2. **设置参数：** 定义客户端数量 ($N$) 和狄利克雷参数 ($\alpha$)。
3. **生成比例：** 对于每个客户端 $k=1...N$，采样一个概率向量 $p_k \sim \text{狄利克雷}(\alpha)$。
4. **分配数据：** 划分数据集样本。对于每个类别 $j$，根据比例 $\{p_{1,j}, p_{2,j}, ..., p_{N,j}\}$ 将其样本分配给客户端。如有必要，确保每个客户端收到合理的最小样本数量。

引入异质性的其他方式包括改变每个客户端的数据*量*（数量偏差）或根据数据特征进行划分，但使用狄利克雷的标签分布偏差是一种被普遍接受的基准测试标准。

### 基线：FedAvg 在非独立同分布数据上的性能

首先，使用上面创建的非独立同分布数据分区运行标准 FedAvg 模拟。在多个通信轮次中训练一个合适的模型（例如，用于 CIFAR-10 的简单 CNN）。

FedAvg 轮次中的主要步骤：

1. 服务器选择一部分客户端。
2. 服务器将当前全局模型 $w^t$ 发送给选定的客户端。
3. 每个选定的客户端 $k$ 在其本地数据 $D_k$ 上训练模型 $E$ 个周期，得到一个本地模型 $w_{k}^{t+1}$。
4. 客户端将其更新后的模型 $w_{k}^{t+1}$ 发送回服务器。
5. 服务器聚合模型：
   $w^{t+1} = \sum_{k \in S_t} \frac{n_k}{n} w_{k}^{t+1}$
   此处 $S_t$ 是选定客户端的集合，$n_k = |D_k|$，且 $n = \sum_{k \in S_t} n_k$。

在通信轮次中，监控全局模型在预留的、有代表性的测试集上的准确性。与在独立同分布数据上运行的模拟相比，您可能会观察到收敛速度变慢，最终准确性也可能降低，尤其是在 $\alpha$ 值较小的情况下。这种性能下降突出了非独立同分布数据带来的问题。

### 缓解：实现 FedProx

现在，我们来实现 FedProx，以解决异质性引起的客户端漂移。回顾第2章可知，FedProx 通过添加一个近端项来修改本地客户端目标函数。客户端不再仅仅最小化本地经验损失 $F_k(w) = \frac{1}{n_k} \sum_{i \in D_k} \ell(w; x_i, y_i)$，而是优化：

$\min_w H_k(w) = F_k(w) + \frac{\mu}{2} \|w - w^t\|^2$

此处，$w^t$ 是在轮次 $t$ 开始时从服务器接收到的全局模型，$\mu \ge 0$ 是一个控制近端项强度的超参数 (parameter) (hyperparameter)。此项限制了本地模型 $w$ 在本地训练期间可以偏离初始全局模型 $w^t$ 的程度。

**实现修改：**

1. **修改客户端更新：** 在执行本地训练（例如，使用 SGD）时，梯度更新需要包含近端项的梯度。如果用于最小化 $F_k(w)$ 的本地优化器更新是 $w \leftarrow w - \eta \nabla F_k(w)$，那么 $H_k(w)$ 的更新变为：
   $w \leftarrow w - \eta (\nabla F_k(w) + \mu (w - w^t))$
   这通常可以通过修改损失函数 (loss function)或在所选框架的本地训练循环中直接调整梯度来实现。
2. **调整超参数 $\mu$：** $\mu$ 的值很重要。$\mu=0$ 会恢复为 FedAvg。较大的 $\mu$ 值会强制模型更接近全局模型，这可能会抵消漂移，但也可能在本地数据确实需要不同模型时减慢个性化进程。典型值可能在 0.01 到 1.0 之间，通常需要一些调整。

使用*相同*的非独立同分布数据分区，但使用 FedProx 客户端更新规则，再次运行模拟。对 $\mu$ 使用非零值（例如，$\mu=0.1$ 或 $\mu=1.0$ 作为起始点）。

### 比较 FedAvg 和 FedProx 在非独立同分布数据上的性能

在相同的非独立同分布数据设置上运行两次模拟（FedAvg 和 FedProx）后，比较它们的性能。绘制两个算法在测试集上的全局模型准确性与通信轮次的关系图。



![FedAvg 与 FedProx 在模拟非独立同分布数据上的性能比较](plots/3588-0.json)



> FedAvg 和 FedProx 在合成生成的非独立同分布数据集（例如，使用狄利克雷划分的 CIFAR-10，$\alpha=0.3$）上，测试准确率在通信轮次中的比较。在异质性条件下，FedProx 通常表现出比 FedAvg 更好的稳定性和收敛性。

### 讨论与后续步骤

上述图表中呈现的模拟结果，通常表明在处理显著的统计异质性时，FedProx 能够带来比 FedAvg 更稳定的收敛和更高的最终准确率。近端项有效地限制了客户端在本地训练期间过度偏离，从而缓解了非独立同分布数据对全局模型聚合的负面影响。

本次练习为测试异质性提供了基础：

- **改变异质性：** 更改狄利克雷参数 (parameter) $\alpha$ 以观察其对 FedAvg 和 FedProx 之间性能差距的影响。
- **调整 FedProx：** 尝试 FedProx 中不同 $\mu$ 超参数 (hyperparameter)值。观察收敛速度、最终准确性和稳定性之间的权衡。
- **实现其他方法：** 在相同的非独立同分布设置下，尝试实现并比较本章或第2章中讨论的其他算法，例如 SCAFFOLD 或聚类联邦学习方法。
- **结合异质性：** 模拟统计异质性和系统异质性的组合（例如，在非独立同分布数据中，改变每个客户端的本地周期 $E$）。

"通过系统地模拟这些场景，您将获得对不同联邦学习策略的优点和不足的宝贵了解，为设计更有效、更可靠的系统以供使用奠定基础。"

## 参考资料

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) — H Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas (2017)
  Journal: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics; Publisher: PMLR; Volume: 54; Pages: 1273-1282
  介绍了联邦平均（FedAvg），这是本节中作为联邦学习模拟基线讨论的基础算法。
- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) — Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith (2020)
  Journal: Proceedings of Machine Learning and Systems (MLSys); Volume: 2; Pages: 429-450; DOI: [10.48550/arXiv.1812.06127](https://doi.org/10.48550/arXiv.1812.06127)
  介绍了FedProx，这是一种解决联邦学习中统计异构性和客户端漂移的突出缓解技术，并在练习中进行了实现。
- [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) — Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, Rafael G.L. D'Oliveira, Hubert Eichner, Salim El Rouayheb, David Evans, Josh Gardner, Zachary Garrett, Adrià Gascón, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaid Harchaoui, Chaoyang He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara Javidi, Gauri Joshi, Mikhail Khodak, Jakub Konečný, Aleksandra Korolova, Farinaz Koushanfar, Sanmi Koyejo, Tancrède Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock, Ayfer Özgür, Rasmus Pagh, Mariana Raykova, Hang Qi, Daniel Ramage, Ramesh Raskar, Dawn Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tramèr, Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu, Sen Zhao (2021)
  Journal: Foundations and Trends in Machine Learning; Publisher: Now Publishers Inc.; Volume: 4; Pages: 1-210; DOI: [10.48550/arXiv.1912.04977](https://doi.org/10.48550/arXiv.1912.04977)
  全面概述了联邦学习，包括对统计异构性、使用狄利克雷分布等非IID数据模拟方法以及各种缓解策略的讨论。

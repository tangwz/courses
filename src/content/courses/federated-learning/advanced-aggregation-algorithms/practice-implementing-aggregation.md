---
course: "federated-learning"
chapter: "advanced-aggregation-algorithms"
lesson: "practice-implementing-aggregation"
sourceId: 3543
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-2-advanced-aggregation-algorithms/practice-implementing-aggregation"
title: "实践：实现高级聚合"
description: "在模拟联邦学习环境中动手实现 FedProx 或 SCAFFOLD 等高级聚合算法。"
order: 7
plots: ["plots/3543-0.json"]
sourceHash: "c57a6db0c6c19ddebbd7068ce7d0fbd98279b3138e37e57901507be88d166522"
sourceCorrections: []
---

FedProx 和 SCAFFOLD 等高级聚合算法为联邦学习提供了改进方案。实现 FedProx 的方法将得到演示，说明它如何解决 FedAvg 的一些局限性，特别是在非独立同分布 (non-IID) 的情况下。在模拟环境中，其性能将与标准的 FedAvg 基线进行比较。

## 环境设置：模拟情况

我们假设你已具备基本的联邦学习模拟环境。这通常包括：

1. **服务器：** 协调训练过程，聚合模型更新。
2. **客户端：** 模拟持有本地数据的独立设备。每个客户端执行本地训练。
3. **数据集：** 标准数据集（如 MNIST 或 CIFAR-10），已分区以模拟客户端之间非独立同分布的数据。实现此目的的常见方法是为每个客户端仅分配有限数量的类别。
4. **模型：** 使用 TensorFlow 或 PyTorch 实现的机器学习 (machine learning)模型（例如，用于图像分类的简单 CNN）。

我们的基线将是标准的 FedAvg 实现，其中客户端使用 SGD 在本地训练 $E$ 个周期，并将更新后的模型权重 (weight)发送回服务器进行平均。

## 实现 FedProx 客户端更新

回顾“FedProx：处理统计异质性”一节，FedProx 修改了客户端的本地目标函数。客户端不是简单地最小化客户端 $k$ 数据上的本地损失 $F_k(w)$，而是最小化：


$$
\min_w F_k(w) + \frac{\mu}{2} ||w - w^t||^2
$$


这里，$w^t$ 表示在第 $t$ 轮开始时从服务器接收到的全局模型权重 (weight)，$\mu$ 是一个非负超参数 (parameter) (hyperparameter)，用于控制近端项的强度。此项将本地解 $w$ 拉向全局模型 $w^t$，从而减轻因本地数据分布差异造成的客户端漂移。

实际操作中，实现这一点需要修改客户端的本地训练循环。具体来说，在计算 SGD（或任何优化器）的梯度时，你需要加上近端项的梯度。近端项对 $w$ 的梯度就是 $\mu(w - w^t)$。

以下是一个 Python 代码片段，展示了客户端本地训练步骤中的修改（假设使用 PyTorch）：

```python
# 假设 'model' 是客户端的本地模型实例
# 'global_model_weights' 存储从服务器接收到的权重 w^t
# 'optimizer' 是标准 SGD 优化器
# 'criterion' 是损失函数（例如，CrossEntropyLoss）
# 'local_data_loader' 提供本地数据批次
# 'mu' 是 FedProx 超参数

# 在本地训练前存储初始全局权重
initial_global_weights = [p.clone().detach() for p in model.parameters()]

# 标准本地训练循环
for epoch in range(num_local_epochs):
    for batch_idx, (data, target) in enumerate(local_data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # --- FedProx 修改开始 ---
        proximal_term = 0.0
        # 遍历模型参数和初始全局权重
        for local_param, global_param in zip(model.parameters(), initial_global_weights):
            # 确保包含需要梯度的参数
            if local_param.requires_grad:
                 # 计算 L2 范数平方差
                proximal_term += torch.sum((local_param - global_param.to(local_param.device))**2)

        loss += (mu / 2.0) * proximal_term
        # --- FedProx 修改结束 ---

        loss.backward()
        optimizer.step()

# 本地训练后，'model' 包含更新后的权重
# 将发送回服务器（或增量，取决于实现方式）
```

**实现要点：**

- **存储 $w^t$：** 在开始本地训练迭代之前，客户端必须存储其接收到的全局模型权重 $w^t$ 的副本。
- **梯度计算：** 在优化器步骤之前，近端项的梯度 $\mu(w - w^t)$ 会被添加到从本地损失 $F_k(w)$ 计算出的梯度中。许多深度学习 (deep learning)框架允许你直接将近端项添加到损失函数 (loss function)中，如上所示，自动微分会处理梯度计算。
- **超参数 $\mu$：** $\mu > 0$ 的选择很重要。
  - 如果 $\mu = 0$，FedProx 就会退化为 FedAvg。
  - 较小的 $\mu$ 允许模型更多地偏离全局模型，可能带来更好的个性化效果，但也可能在高度非独立同分布数据上出现类似于 FedAvg 的发散风险。
  - 较大的 $\mu$ 会强制本地模型与全局模型保持非常接近，限制了个性化，但提高了全局模型在异质性下的稳定性和收敛性。找到一个好的值通常需要经验性调优。

FedProx 中的服务器端聚合与 FedAvg 保持一致：它收集更新后的本地模型（或模型增量），并根据每个客户端上的数据点数量计算加权平均值。

## FedAvg 与 FedProx 的比较

为观察 FedProx 的影响，请设置一个具有明显统计异质性的模拟。例如，将 MNIST 数据集分配给 100 个客户端，使每个客户端仅拥有两个数字类别的数据。使用 FedAvg ($\mu = 0$) 和 FedProx（例如，$\mu = 0.01$ 或 $\mu = 0.1$）训练一个简单 CNN，进行固定数量的通信轮次。

在每个通信轮次后，跟踪全局模型在独立、平衡测试集上的准确率。你可能会看到与下述图示类似的结果：



![FedAvg 与 FedProx 收敛性比较（非独立同分布 MNIST）](plots/3543-0.json)



> FedAvg 和 FedProx 在不同 $\mu$ 值下，全局模型在平衡 MNIST 测试集上的准确率随通信轮次的变化，模拟了非独立同分布数据（每个客户端仅拥有两个类别的数据）。

## 分析及后续步骤

该图表显示了常见的成果：

- **FedAvg：** 初始阶段收敛可能更快，但由于非独立同分布数据造成的客户端漂移，最终准确率通常会停滞在较低水平。聚合模型难以泛化到所有数据分布。
- **FedProx ($\mu > 0$)：** 初始阶段收敛可能略慢，尤其是在 $\mu$ 较大时，因为本地训练受到限制。然而，通过减轻客户端漂移，它通常能达到更高的最终全局准确率。$\mu$ 的选择平衡了稳定性和收敛速度。

这项实践练习表明了在异质环境中采用 FedProx 等高级聚合算法的实际优势。

**后续研究：**

1. **尝试不同的 $\mu$ 值：** 尝试不同的 $\mu$ 值。它如何影响收敛速度和最终准确率？它如何影响客户端本地模型之间的性能差异？
2. **实现 SCAFFOLD：** 作为后续一步，尝试实现 SCAFFOLD。这包括修改客户端和服务器逻辑以处理伴随模型更新的控制变量 ($c_i$, $c$)。将其收敛性与 FedAvg 和 FedProx 进行比较。请注意客户端更新（使用控制变量）的差异以及额外的通信开销（发送控制变量增量）。
3. **改变异质性程度：** 改变非独立同分布数据（例如，客户端拥有 1、3 或 5 个类别）的程度，观察 FedAvg 和 FedProx 之间的性能差距如何变化。
4. **组合技术：** 尝试将 FedProx 与后面讨论的其他技术结合，例如通信效率方法（第五章）。

“通过实现和实验这些算法，你将获得对其行为和相关权衡的实际理解，为你在各种应用中构建更有效的联邦学习系统做好准备。”

## 参考资料

- [Federated Optimization in Heterogeneous Networks](https://proceedings.mlsys.org/paper/2020/file/37905f03f7e5242488d752f970e7a2b0-Paper.pdf) — Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith (2020)
  Journal: Proceedings of Machine Learning and Systems (MLSys); Publisher: MLSys Proceedings; Volume: 2; Pages: 429-450
  介绍了 FedProx 的原始论文，解释了其理论动机，并展示了其在联邦学习中解决统计异构性方面的有效性。
- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf) — Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, Ananda Theertha Suresh (2020)
  Journal: Proceedings of the 37th International Conference on Machine Learning (ICML); Volume: 119; Pages: 5157-5166; DOI: [10.48550/PMLR.v119.karimireddy20a](https://doi.org/10.48550/PMLR.v119.karimireddy20a)
  提出了 SCAFFOLD，一种先进的联邦学习算法，它使用控制变量来减轻客户端漂移并改善非独立同分布（non-IID）设置下的收敛性。
- [Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977.pdf) — Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Martin Biester, et al. (2021)
  Journal: Foundations and Trends® in Machine Learning; Volume: 14; Pages: 1-210; DOI: [10.1561/2200000083](https://doi.org/10.1561/2200000083)
  一份全面的综述，概述了联邦学习中的关键挑战，包括统计异构性，并回顾了 FedProx 和 SCAFFOLD 等高级聚合算法。

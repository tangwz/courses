# GNN 中的过拟合与正则化

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-4-training-gnn-models/overfitting-regularization-gnns)

[返回章节目录](README.md) · [返回课程目录](../README.md)

过拟合 (overfitting)是一个常见问题，即模型在训练数据上表现极佳，但无法泛化到未见的新数据。图神经网络 (neural network) (GNN) 旨在学习复杂的关联模式并进行预测，但它们也可能过度学习训练数据。GNN 捕获精细结构的能力使其特别容易产生过拟合。当 GNN 记住了训练图的特定拓扑结构和特征噪声，而不是学习更具普适性的内在模式时，就会发生这种情况。

### GNN 中的过拟合 (overfitting)问题

在图的语境下，过拟合意味着 GNN 生成的节点嵌入 (embedding)专门针对训练节点上的训练任务。因此，当要求模型对验证或测试节点进行预测时，性能会下降。这通常表现为模型的训练损失持续下降，而验证损失却停滞不前或开始上升。



[交互图表：训练损失 vs. 验证损失](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-4-training-gnn-models/overfitting-regularization-gnns#plot-1jjy7kq)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "训练损失 vs. 验证损失"
    },
    "xaxis": {
      "title": {
        "text": "轮次 (Epoch)"
      }
    },
    "yaxis": {
      "title": {
        "text": "损失 (Loss)"
      }
    },
    "legend": {
      "x": 1,
      "y": 1,
      "xanchor": "right"
    },
    "margin": {
      "l": 40,
      "r": 40,
      "t": 60,
      "b": 40
    }
  },
  "data": [
    {
      "x": [
        1,
        5,
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
        1.8,
        1.2,
        0.9,
        0.6,
        0.45,
        0.35,
        0.3,
        0.25,
        0.2,
        0.18,
        0.16,
        0.15
      ],
      "mode": "lines+markers",
      "name": "训练损失",
      "line": {
        "color": "#228be6"
      }
    },
    {
      "x": [
        1,
        5,
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
        1.82,
        1.25,
        0.98,
        0.75,
        0.68,
        0.65,
        0.64,
        0.65,
        0.68,
        0.72,
        0.76,
        0.81
      ],
      "mode": "lines+markers",
      "name": "验证损失",
      "line": {
        "color": "#f03e3e"
      }
    }
  ]
}
```

</details>



> 随着训练的进行，模型在训练集上的表现持续提高，但通过验证损失衡量的泛化能力在第 50 轮左右开始变差。

另一个与 GNN 特有的相关问题是**过度平滑**。随着堆叠更多 GNN 层，消息传递机制有效地扩大了每个节点的感受野。虽然这允许节点从更远的地方收集信息，但也有副作用。随着层数增加，节点的表示变成了其邻居表示的混合体。在经过多层处理后，图中连通分量内所有节点的表示可能会变得几乎完全相同，从而丢失了准确预测所需的特定信息。这种嵌入的同质化会严重降低模型性能。

### 正则化 (regularization)策略

为了对抗过拟合 (overfitting)并提高泛化能力，我们采用正则化技术。这些方法在训练过程中引入限制或添加噪声，以防止模型变得过于复杂并死记硬背训练数据。

#### Dropout

深度学习 (deep learning)中常用的正则化技术是 **Dropout**。在 GNN 中，Dropout 可以应用于节点特征矩阵 `X` 或 GNN 层之间的隐藏嵌入 (embedding)。在每次训练迭代中，它随机将一部分特征维度设为零。这迫使模型学习分布更均匀、更具鲁棒性的表示，防止其过度依赖任何单一特征或一小部分特征。

一种针对 GNN 的变体是 **DropEdge**。DropEdge 不是将节点特征归零，而是在每个训练步中从图的邻接矩阵中随机移除一部分边。这可以看作是一种数据增强方式；模型在每次前向传播中都会接触到略有不同的图结构。通过这种方式，DropEdge 防止模型记住训练图中特定的消息传递路径，迫使其学习对微小结构变化更具适应性的模式。

> DropEdge 在训练期间随机移除边，迫使模型寻找消息传递的替代路径，从而提高其稳定性。

#### 权重 (weight)衰减 (L2 正则化)

权重衰减是另一种标准正则化方法，它惩罚模型中过大的权重。其实现方式是在损失函数 (loss function)中增加一个与模型可学习权重 $w_i$ 平方和成正比的项。修改后的损失函数为：


$$
L_{\text{总}} = L_{\text{原始}} + \lambda \sum_{i} w_i^2
$$


这里，$\lambda$ 是控制正则化强度的超参数 (parameter) (hyperparameter)。通过惩罚大的权重值，权重衰减鼓励模型找到权重较小的简单解。简单的模型通常不易产生过拟合，且往往具有更好的泛化性能。

#### 早停法 (Early Stopping)

早停法是一种实用且有效的技术，它利用验证集来决定何时停止训练。步骤如下：

1. 每轮训练后，监测模型在独立验证集上的表现（如损失或准确率）。
2. 记录到目前为止获得最佳验证表现的模型权重。
3. 如果验证表现连续在预设的轮数（称为“耐心值”）内没有提升，则停止训练过程。
4. 最终用于测试的模型是在验证集上表现最好的那个，而不一定是最后一轮训练的模型。

回顾损失曲线图，早停法会在第 50 轮左右停止训练，此时验证损失处于最低点，从而防止模型进入过拟合阶段。这些正则化技术并不互斥。在实践中，通常会将它们结合使用，例如在采用早停准则训练的模型中同时使用 DropEdge 和权重衰减。

## 参考资料

- [Graph Neural Networks: A Review of Methods and Applications](https://www.sciencedirect.com/science/article/pii/S266654682030010X) — Jie Zhou, Ganqu Cui, Zhengyu Dai, Shuai Sun, Ling Shao, Jianxin Li, Yang You, Zenglin Xu (2020)
  Journal: AI Open; Publisher: Elsevier; Volume: 1; Pages: 57-71; DOI: [10.1016/j.aiopen.2020.11.001](https://doi.org/10.1016/j.aiopen.2020.11.001)
  全面概述图神经网络，包括对过拟合和过平滑等常见挑战的讨论，并提及各种正则化技术。
- [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification](https://openreview.net/forum?id=Bkdq_nEKDr) — Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang (2020)
  Journal: International Conference on Learning Representations (ICLR 2020); Publisher: ICLR; DOI: [10.48550/arXiv.1911.08070](https://doi.org/10.48550/arXiv.1911.08070)
  介绍了DropEdge，一种针对GNN的正则化技术，通过在训练期间随机删除边来防止过拟合并提高泛化能力。
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.jmlr.org/papers/v15/srivastava14a.html) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research (JMLR); Publisher: JMLR; Volume: 15; Pages: 1929−1958
  介绍了Dropout的原始论文，这是一种广泛应用于深度学习（包括GNN）的基本正则化技术。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的深度学习教科书，涵盖了基本的深度学习概念，包括对权重衰减（L2正则化）和提前停止的详细解释。

---

[上一节](05-%E8%8A%82%E7%82%B9%E5%88%86%E7%B1%BB%E7%9A%84%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md) · [下一节](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%E4%BD%A0%E7%9A%84%20GCN.md)

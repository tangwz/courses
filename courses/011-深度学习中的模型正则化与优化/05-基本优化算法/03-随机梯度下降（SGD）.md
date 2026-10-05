# 随机梯度下降（SGD）

来源：[原文](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-5-foundational-optimizers/stochastic-gradient-descent-sgd)

[返回章节目录](README.md) · [返回课程目录](../README.md)

如前所述，使用整个数据集（批量梯度下降 (gradient descent)）计算梯度来更新每一个参数 (parameter)，计算成本会变得非常高昂，特别是在深度学习 (deep learning)中常见的海量数据集下。设想一下，为了迈出微小一步，就需要对数百万张图像计算损失和梯度！这种方法还需要将整个数据集加载到内存中，这通常是不可行的。

随机梯度下降（SGD）提供了一种实用且出人意料地有效的替代方案。SGD不使用完整数据集，而是在每一步仅使用*一个*训练样本来近似梯度。

### SGD的运作机制

核心思想很简单：

1. 从数据集中随机选择一个训练样本 $(x^{(i)}, y^{(i)})$。
2. 计算这个单一样本的损失 $L(f(x^{(i)}; W), y^{(i)})$。
3. 仅使用此样本，计算损失关于参数 (parameter) $W$ 的梯度：$\nabla_W L(f(x^{(i)}; W), y^{(i)})$。
4. 使用此梯度更新参数：
   
   $$
   W \leftarrow W - \eta \nabla_W L(f(x^{(i)}; W), y^{(i)})
   $$
   
   其中 $\eta$ 是学习率。

这个过程会重复进行，通常是每次处理一个样本，遍历整个数据集。遍历整个数据集一次被称为一个*周期*（epoch）。因为每次更新只使用一个样本，所以计算出的梯度是对真实梯度（即在完整数据集上计算出的梯度）的“带噪声”估计。

### 为何使用带噪声的梯度？

虽然使用单个样本会引入噪声，但它带来了显著的优势：

1. **计算效率：** 更新速度极快。你只需要对一个样本进行前向和后向传播，与批量梯度下降 (gradient descent)（Batch GD）相比，每次更新的计算量大幅减少。
2. **频繁更新：** 参数 (parameter)更新频率更高（每个周期N次，N是数据集大小），这可能导致*实际运行时间上*更快的收敛，即使可能需要更多的总步数。
3. **内存效率：** 梯度计算时，一次只需要一个训练样本在内存中。

“随机”一词指的是这种随机抽取单个样本来估计梯度的做法。虽然每一步可能不会直接指向整体损失函数 (loss function)的最小值，但多步的*平均*行为倾向于将参数朝着正确的方向移动。

### SGD的更新路径

梯度估计的噪声特性意味着SGD趋向最小值的路径通常是不规则的，会围绕着批量梯度下降 (gradient descent)（Batch GD）可能采取的最优路径曲折前进。然而，这种噪声并非总是有害的。



[交互图表：SGD 与批量梯度下降的路径对比](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-5-foundational-optimizers/stochastic-gradient-descent-sgd#plot-18npl2p)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "SGD 与批量梯度下降的路径对比",
    "xaxis": {
      "title": "参数 1",
      "range": [
        -1.5,
        2.5
      ]
    },
    "yaxis": {
      "title": "参数 2",
      "range": [
        -1,
        3
      ]
    },
    "showlegend": true,
    "width": 600,
    "height": 450,
    "colorway": [
      "#1c7ed6",
      "#f03e3e"
    ],
    "annotations": [
      {
        "x": 2.0,
        "y": 2.5,
        "text": "最小值",
        "showarrow": false
      }
    ]
  },
  "data": [
    {
      "type": "contour",
      "z": [
        [
          1.25,
          0.5,
          1.25,
          2.75
        ],
        [
          2.5,
          1.0,
          1.5,
          3.0
        ],
        [
          4.25,
          2.5,
          2.25,
          3.75
        ],
        [
          6.5,
          4.5,
          4.25,
          5.75
        ]
      ],
      "x": [
        -1,
        0,
        1,
        2
      ],
      "y": [
        -0.5,
        0.5,
        1.5,
        2.5
      ],
      "colorscale": "Blues",
      "contours": {
        "coloring": "lines"
      },
      "line": {
        "width": 1
      },
      "showscale": false,
      "name": "损失曲面"
    },
    {
      "type": "scatter",
      "x": [
        -1.0,
        -0.7,
        -0.5,
        -0.2,
        0.1,
        0.5,
        0.9,
        1.3,
        1.6,
        1.8,
        2.0
      ],
      "y": [
        -0.5,
        -0.2,
        0.1,
        0.5,
        0.8,
        1.1,
        1.4,
        1.7,
        2.0,
        2.2,
        2.5
      ],
      "mode": "lines+markers",
      "name": "批量梯度下降",
      "line": {
        "width": 2
      }
    },
    {
      "type": "scatter",
      "x": [
        -1.0,
        -0.9,
        -0.6,
        -0.8,
        -0.4,
        -0.1,
        0.3,
        0.1,
        0.5,
        0.8,
        1.0,
        1.3,
        1.1,
        1.5,
        1.7,
        1.9,
        1.8,
        2.1,
        2.0
      ],
      "y": [
        -0.5,
        0.2,
        0.0,
        0.6,
        0.4,
        0.9,
        0.7,
        1.1,
        0.9,
        1.3,
        1.1,
        1.5,
        1.8,
        1.6,
        2.0,
        1.8,
        2.3,
        2.1,
        2.5
      ],
      "mode": "lines+markers",
      "name": "SGD",
      "line": {
        "width": 2
      }
    }
  ]
}
```

</details>



> 简单损失曲面上的优化路径对比。批量梯度下降（Batch GD）沿着更平滑的路径直接趋向最小值，而SGD由于单个样本的梯度估计，会采取一条更带噪声、曲折的路径。

这种噪声有时能帮助优化过程脱离浅层局部最小值或鞍点，这些区域的梯度接近零，但并非真正的最优解。随机波动可能会“推动”参数 (parameter)离开这些区域。

然而，噪声也意味着SGD可能难以精确收敛到最小值。即使在训练后期，参数也经常在最小值附近持续振荡。这就是为什么在使用SGD时，诸如随时间降低学习率等方法很重要。

### 数据洗牌

在使用SGD（及其变体，如小批量梯度下降 (gradient descent)）时，一种常见做法是在每个周期之前打乱训练数据集。如果数据以固定的、有意义的顺序呈现（例如，先是所有A类样本，然后是所有B类样本），那么同一类别批次内的梯度将高度相关，这可能会使更新产生偏差并阻碍收敛。洗牌确保连续的更新基于更多样化的样本，从而随时间产生一个更具代表性（尽管仍然带噪声）的梯度估计。

### 与在线学习的关联

SGD本质上是在执行“在线学习”，即模型随着数据的到来，每次从一个样本中学习。这使得它适用于数据持续流入且无法完全存储的场景。

虽然纯SGD（仅使用一个样本）提供了理论基础，但在实践中，其带噪声的更新可能导致收敛缓慢或不稳定。这引出了小批量梯度下降 (gradient descent)（Mini-batch Gradient Descent），它在批量梯度下降的稳定性和SGD的效率之间提供了一种折衷。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍深度学习基本概念的教科书，其中包含对随机梯度下降及其变体的详细解释。
- [A Stochastic Approximation Method](https://doi.org/10.1214/aoms/1177729586) — Herbert Robbins, Sutton Monro (1951)
  Journal: The Annals of Mathematical Statistics; Publisher: Institute of Mathematical Statistics; Volume: 22; Pages: 400-407; DOI: [10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)
  介绍了随机逼近方法的奠基性论文，该方法为随机梯度下降提供了数学基础。
- [Optimization Methods for Large-Scale Machine Learning](https://doi.org/10.1137/16M1080173) — Léon Bottou, Frank E. Curtis, and Jorge Nocedal (2018)
  Journal: SIAM Review; Publisher: Society for Industrial and Applied Mathematics; Volume: 60; Pages: 223-311; DOI: [10.1137/16M1080173](https://doi.org/10.1137/16M1080173)
  一篇关于大规模机器学习优化方法的综合性综述，重点关注随机梯度下降及其理论特性和实际应用。

---

[上一节](02-%E6%A0%87%E5%87%86%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%9A%84%E9%9A%BE%E9%A2%98.md) · [下一节](04-%E5%B0%8F%E6%89%B9%E9%87%8F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)

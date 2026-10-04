# SGD的挑战：噪声与局部最小值

来源：[原文](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-5-foundational-optimizers/sgd-challenges-noise-minima)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管随机梯度下降 (gradient descent)（SGD）及其小批量变体相比标准批量梯度下降在计算上提供了显著优势，但它们也带来了一系列自身问题，主要源于其梯度估计的噪声特性以及深度学习 (deep learning)中损失曲面的复杂结构。弄清这些问题对于理解为何开发了更高级的优化器非常必要。

### 噪声更新的问题

与批量梯度下降 (gradient descent)不同，后者使用整个数据集计算精确梯度，而SGD每次更新只使用一个示例，小批量GD则使用一小部分数据。这意味着每一步计算出的梯度仅仅是真实梯度的一种*估计*。这种估计可能噪声很大，尤其是在批量大小非常小（或SGD的批量大小为1）时。

想象一下蒙着眼睛试图找到一个多山山谷的底部。批量梯度下降会感受周围整个山谷底部的坡度来迈出一步。小批量梯度下降则感受脚下小块地面的坡度。SGD只感受它所站立的那个微小点上的确切坡度。

这种噪声会带来以下几个结果：

1. **震荡：** 参数 (parameter)更新不会沿着平滑的路径趋向最小值。相反，它们倾向于四处跳动，有时会朝着暂时增加损失的方向移动。这在接近最小值时尤为明显，更新可能会超调并在最佳点附近震荡。
2. **收敛可能变慢：** 虽然每个SGD步骤都很快，但噪声路径意味着与批量梯度下降的更平滑路径相比，它可能需要更多步骤才能达到一个好的最小值。由于每一步的速度，总时间可能仍然更少，但收敛路径本身不那么直接。
3. **更新的方差：** 梯度估计的方向和大小可能因不同小批量（或样本）而显著变化。

下方的图表展示了由于噪声梯度估计，SGD的路径与批量梯度下降可能采取的更平滑路径如何不同。



[交互图表：优化路径](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-5-foundational-optimizers/sgd-challenges-noise-minima#plot-t0w4qa)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "优化路径",
    "xaxis": {
      "title": "参数 1",
      "range": [
        -3,
        3
      ]
    },
    "yaxis": {
      "title": "参数 2",
      "range": [
        -3,
        3
      ]
    },
    "showlegend": true,
    "legend": {
      "x": 0.1,
      "y": 0.9
    }
  },
  "data": [
    {
      "x": [
        2.5,
        2.0,
        1.8,
        1.0,
        0.8,
        0.3,
        0.1
      ],
      "y": [
        2.5,
        2.2,
        1.5,
        1.3,
        0.5,
        0.6,
        0.1
      ],
      "mode": "lines+markers",
      "name": "批量梯度下降",
      "line": {
        "color": "#1c7ed6",
        "width": 2
      },
      "marker": {
        "size": 6
      }
    },
    {
      "x": [
        2.5,
        2.6,
        1.9,
        2.1,
        1.5,
        1.0,
        0.8,
        0.5,
        0.9,
        -0.1,
        0.4,
        0.1
      ],
      "y": [
        2.5,
        1.8,
        2.0,
        1.5,
        1.7,
        1.1,
        0.4,
        0.8,
        0.2,
        0.5,
        -0.2,
        0.0
      ],
      "mode": "lines+markers",
      "name": "SGD/小批量",
      "line": {
        "color": "#f03e3e",
        "dash": "dot",
        "width": 2
      },
      "marker": {
        "size": 6,
        "symbol": "x"
      }
    },
    {
      "type": "contour",
      "z": [
        [
          15.68,
          3.92,
          0.0,
          3.92,
          15.68
        ],
        [
          12.04,
          3.0,
          0.0,
          3.0,
          12.04
        ],
        [
          7.84,
          1.96,
          0.0,
          1.96,
          7.84
        ],
        [
          12.04,
          3.0,
          0.0,
          3.0,
          12.04
        ],
        [
          15.68,
          3.92,
          0.0,
          3.92,
          15.68
        ]
      ],
      "x": [
        -2.8,
        -1.4,
        0,
        1.4,
        2.8
      ],
      "y": [
        -2.8,
        -1.4,
        0,
        1.4,
        2.8
      ],
      "colorscale": "Blues",
      "showscale": false,
      "contours": {
        "coloring": "lines"
      }
    }
  ]
}
```

</details>



> 一个简化的二维损失曲面，显示了平滑路径（如批量梯度下降）与噪声路径（如SGD/小批量梯度下降）向最小值（中心）移动的区别。

尽管通常可以控制，尤其是在有适当学习率的情况下，这种噪声是SGD和小批量方法的一个基本特点。

### 应对复杂损失曲面：局部最小值与鞍点

深度学习 (deep learning)的损失曲面极其复杂且维度很高。它们并非简单的凸形碗状。相反，它们包含许多可能阻碍优化的特征：

- **局部最小值：** 这些点上的损失比所有周围点都低，但并非整个空间中最低的损失（全局最小值）。如果优化器陷入局部最小值，梯度将变为零，标准梯度下降 (gradient descent)方法将停止，可能使模型困在次优状态。

  - 最初，局部最小值被认为是一个主要障碍。然而，研究表明，在深度学习常见的极高维度中，大多数局部最小值的损失值都与全局最小值非常接近，因此它们不像曾经担忧的那样成问题。此外，SGD中的噪声有时可能在此处产生益处，有可能“踢”参数 (parameter)脱离浅层局部最小值。
- **鞍点：** 这些点上的梯度也为零，但它们*不是*最小值。想象一下马鞍：如果你沿着马的脊柱向前或向后移动，你处于最小值，但如果你左右移动（沿着马鞍的侧翼），曲面会向下弯曲。数学上，曲率在某些方向上为正，在其他方向上为负。

  - 现在普遍认为，在高维度空间中，鞍点比局部最小值更普遍且更成问题。问题在于，在鞍点附近，即使在损失可以降低的方向上，梯度也会变得非常小。标准SGD仅依赖当前梯度，可能会显著减慢速度，需要极长时间才能离开鞍点，严重阻碍训练进度。

下方的图表展示了这些点在一个二维损失曲面上的位置。

> 损失曲面的特点：一个全局最小值（最低点），一个局部最小值（低点，但不是最低点），以及一个鞍点（平坦，但在某些方向向下弯曲，在其他方向向上弯曲）。SGD在鞍点附近可能会遇到困难。

总而言之，尽管SGD和小批量梯度下降由于其效率而成为训练深度模型的主力，但它们的噪声更新以及高维度损失曲面中鞍点的普遍存在，对收敛速度和稳定性提出了严峻挑战。这些困难促使了更精巧的优化算法的开发，例如动量法和像Adam这样的自适应方法，我们接下来将讨论。这些算法包含克服噪声和加速通过损失曲面困难区域的机制。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习理论和方法的经典教材，包括优化算法以及非凸损失函数地形挑战的章节。
- [A Stochastic Approximation Method](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/ams/1177729586.full) — Herbert Robbins and Sutton Monro (1951)
  Journal: The Annals of Mathematical Statistics; Publisher: Institute of Mathematical Statistics; Volume: 22; Pages: 400-407; DOI: [10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)
  这篇开创性论文介绍了随机近似方法，为随机梯度下降奠定了数学基础。
- [Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization](https://papers.nips.cc/paper_files/paper/2014/file/17e23e50bedc63b4095e3d8204ce063b-Paper.pdf) — Yann N. Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 27; Pages: 2933-2941; DOI: [10.48550/arXiv.1406.2572](https://doi.org/10.48550/arXiv.1406.2572)
  该论文指出在深度学习高维非凸优化中，鞍点比局部最小值更具挑战性，严重阻碍了优化算法的收敛。
- [The Loss Surfaces of Multilayer Networks](https://proceedings.mlr.press/v38/choromanska15.html) — Anna Choromanska, Mikael Henaff, Michael Mathieu, Gérard Ben Arous, Yann LeCun (2015)
  Journal: Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics; Publisher: PMLR; Volume: 38; Pages: 192-204
  探讨了深度神经网络损失曲面的几何特性，提出对于大型网络而言，大多数局部极小值点的质量与全局极小值点相似。

---

[上一节](04-%E5%B0%8F%E6%89%B9%E9%87%8F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md) · [下一节](06-%E5%B8%A6%E5%8A%A8%E9%87%8F%E7%9A%84%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%EF%BC%9A%E5%8A%A0%E9%80%9F%E6%94%B6%E6%95%9B.md)

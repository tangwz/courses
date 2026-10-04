# WGAN 中的权重剪裁

来源：[原文](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-3-gan-training-stabilization/wgan-weight-clipping)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用 Wasserstein-1 距离 ($W_1$) 作为 GAN 的损失函数 (loss function)，为训练稳定性提供了理论上的优点。Kantorovich-Rubinstein 对偶性将此距离表达为：

$W_1(\mathbb{P}_r, \mathbb{P}_g) = \sup_{||f||_L \le 1} \mathbb{E}_{x \sim \mathbb{P}_r}[f(x)] - \mathbb{E}_{\tilde{x} \sim \mathbb{P}_g}[f(\tilde{x})]$

此处，上确界是针对所有 1-Lipschitz 函数 $f$ 取的。在 WGAN 的情形下，我们的判别器（现在常称为“评论者”）的目标是近似此函数 $f$。因此，WGAN 目标函数要准确近似 Wasserstein 距离的一个重要条件是，评论者函数 $f_w$（由权重 (weight) $w$ 参数 (parameter)化）必须是 1-Lipschitz 的。这意味着其梯度范数在各处应最多为 1：$||\nabla_x f_w(x)||_2 \le 1$。

在训练期间，我们如何才能对神经网络 (neural network)施加此约束？原始 WGAN 论文提出了一种直接但有些粗糙的方法：**权重剪裁**。

### 权重 (weight)剪裁的机制

权重剪裁是一种简单操作，在每次梯度更新后应用于评论者的权重。对于评论者网络中的每个权重 $w_i$，它被限制在一个小的固定范围 $[-c, c]$ 内，其中 $c$ 是一个小的正数常数（例如 0.01）。

权重 $w_i$ 在标准梯度下降 (gradient descent)更新（例如使用 RMSProp 或 SGD，*而不是*像 Adam 那样基于动量的优化器，原始论文中是这样建议的）后的更新步骤如下：

1. **梯度更新**：$w_i \leftarrow w_i - \alpha \cdot \nabla_{w_i} L_{critic}$
2. **剪裁**：$w_i \leftarrow \max(-c, \min(c, w_i))$

此操作有效地将超出 $[-c, c]$ 区间范围的任何权重“剪裁”回边界。

### 为何剪裁？其原理

其原理是，通过保持权重 (weight)较小，我们间接限制了函数 $f_w$ 可能的梯度。如果一个函数 $f_w$ 是 $K$-Lipschitz 的，它满足 $|f_w(x_1) - f_w(x_2)| \le K ||x_1 - x_2||$。权重的量级影响着网络输出相对于其输入的改变速度。将权重限制在一个小范围 $[-c, c]$ 内，希望可以对 Lipschitz 常数 $K$ 施加一个界限，使其理想情况下接近 1。

### 超参数 (parameter) (hyperparameter) $c$

剪裁常数 $c$ 的选择很重要且敏感。

- 如果 $c$ 过小，评论者的能力会受到严重限制。它可能被迫学习非常简单的函数，可能导致传递回生成器的梯度消失，因为评论者无法有效捕获数据分布的复杂性。
- 如果 $c$ 过大，权重 (weight)可能很少触及剪裁边界。这未能有效强制施加 Lipschitz 约束，可能导致 WGAN 旨在防止的相同训练不稳定性（如模式坍塌或梯度爆炸）。

为 $c$ 寻找一个合适的值通常需要针对每个具体问题和架构进行细致调整。

### 权重 (weight)剪裁的主要缺点

虽然易于实现，但权重剪裁引入了几个主要问题：

1. **不正常的权重分布**：根据经验，通常观察到，在使用权重剪裁时，评论者的大部分权重倾向于精确地聚集在边界值 $-c$ 和 $+c$ 处。这表明网络未能有效利用其全部参数 (parameter)空间，并被人为限制。
2. **梯度问题**：硬剪裁操作可能导致梯度问题。对于处于边界的权重，梯度信息可能被有效“剪裁”掉或变为零，阻碍学习。反之，如果 $c$ 过大，梯度仍然可能爆炸。这使得训练对 $c$ 的选择很敏感。
3. **能力降低**：通过强制权重保持较小，我们限制了评论者的表达能力。评论者可能难以学习准确估计 Wasserstein 距离或向生成器提供有益梯度所需的复杂映射。这鼓励评论者学习比它原本可能需要的更简单函数。

考虑对评论者权重的影响。权重剪裁不是产生潜在平滑的分布，而是迫使许多权重聚集在边界 $-c$ 和 $+c$ 处。



[交互图表：评论者权重分布](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-3-gan-training-stabilization/wgan-weight-clipping#plot-139u2xo)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "评论者权重分布",
    "xaxis": {
      "title": "权重值"
    },
    "yaxis": {
      "title": "密度"
    },
    "bargap": 0.1,
    "barmode": "overlay"
  },
  "data": [
    {
      "type": "histogram",
      "x": [
        -0.015,
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.01,
        -0.008,
        -0.005,
        -0.002,
        0,
        0.001,
        0.003,
        0.006,
        0.009,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.012
      ],
      "name": "使用剪裁",
      "marker": {
        "color": "#f06595"
      },
      "opacity": 0.7,
      "xbins": {
        "start": -0.02,
        "end": 0.02,
        "size": 0.001
      }
    },
    {
      "type": "histogram",
      "x": [
        -0.018,
        -0.015,
        -0.012,
        -0.009,
        -0.006,
        -0.003,
        0,
        0.002,
        0.004,
        0.007,
        0.01,
        0.013,
        0.016,
        0.019,
        -0.005,
        0.005,
        -0.008,
        0.008
      ],
      "name": "理想情况（无剪裁）",
      "marker": {
        "color": "#74c0fc"
      },
      "opacity": 0.6,
      "xbins": {
        "start": -0.02,
        "end": 0.02,
        "size": 0.001
      }
    }
  ]
}
```

</details>



> 直方图显示权重剪裁（粉色）如何导致权重堆积在剪裁边界（$\pm c$）处，与没有剪裁的更平滑、更自然的分布（蓝色）相比。

由于这些缺点，特别是 $c$ 值调整的困难以及潜在的梯度流动不良，现代 WGAN 实现中通常避免使用权重剪裁。它曾作为最初的验证手段，但已基本被更具理论依据且更实用的方法所取代，这些方法旨在施加 Lipschitz 约束。最突出的替代方案是梯度惩罚（WGAN-GP），它直接处理梯度范数要求，并将在下一节中阐述。

## 参考资料

- [Wasserstein Generative Adversarial Networks](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) — Martin Arjovsky, Soumith Chintala, and Léon Bottou (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Publisher: Machine Learning Research; Volume: 70; Pages: 214-223; DOI: [10.5555/3305890.3306076](https://doi.org/10.5555/3305890.3306076)
  介绍了 Wasserstein GAN，阐述了其基于地球移动距离的理论基础，并提出了权重裁剪作为强制评论器Lipschitz约束的方法。
- [Improved Training of Wasserstein GANs](https://proceedings.neurips.cc/paper_files/paper/2017/file/892c91e0a653ba19df81a90f89d99f56-Paper.pdf) — Ishaan Gulrajani, Faruk Ahmed, Martín Arjovsky, Vincent Dumoulin, and Aaron Courville (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS 2017); Publisher: Curran Associates, Inc.; Pages: 5767-5777
  针对WGAN中权重裁剪的局限性，提出了梯度惩罚，这是一种更有效、更稳定的正则化方法，用于强制执行Lipschitz约束。
- [Generative Adversarial Networks](https://dl.acm.org/doi/10.1145/3422622) — Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2020)
  Journal: Communications of the ACM; Volume: 63; Pages: 139-144; DOI: [10.1145/3422622](https://doi.org/10.1145/3422622)
  对生成对抗网络进行了综述，讨论了其动机、架构和训练挑战，包括WGAN等方法带来的稳定性改进需求。

---

[上一节](03-%E5%85%B6%E4%BB%96%E6%95%A3%E5%BA%A6%EF%BC%9AWasserstein%20%E8%B7%9D%E7%A6%BB.md) · [下一节](05-%E6%A2%AF%E5%BA%A6%E6%83%A9%E7%BD%9A%20%28WGAN-GP%29.md)

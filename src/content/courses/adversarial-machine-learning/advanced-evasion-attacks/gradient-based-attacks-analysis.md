---
course: "adversarial-machine-learning"
chapter: "advanced-evasion-attacks"
lesson: "gradient-based-attacks-analysis"
sourceId: 4124
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-2-advanced-evasion-attacks/gradient-based-attacks-analysis"
title: "基于梯度的攻击：FGSM、BIM、PGD分析"
description: "细致分析快速梯度符号方法、基本迭代方法和投影梯度下降。"
order: 1
plots: []
sourceHash: "0e1c4067c78a4df179f6815409a2f557164c8675db1f3a7bb83c0f598483d433"
sourceCorrections: []
---

规避攻击的目标是通过在输入中添加精心构造的噪声，在推理 (inference)阶段欺骗模型。基于梯度的攻击是这类攻击中一个基本且被广泛研究的类别。它们依据一个简单而有效的方法：修改输入，使其朝着最大程度增加模型对正确标签损失的方向发展，从而促使模型产生误分类。这要求能够访问模型相对于输入的梯度，因此这些攻击主要属于白盒攻击。我们现在将介绍三种基础的梯度方法：FGSM、BIM和PGD。

### 快速梯度符号方法 (FGSM)

快速梯度符号方法是生成对抗性示例的最早、最简单方法之一。FGSM由Goodfellow等人在2014年提出，它在损失函数 (loss function)上，针对输入特征，执行单步梯度上升。主要思路是，沿着梯度*符号*指示的方向微调 (fine-tuning)输入，是增加损失并可能导致误分类的一种有效途径。

设 $J(\theta, x, y)$ 为用于训练模型的损失函数，其中 $\theta$ 是参数 (parameter)，$x$ 是输入，$y$ 是真实标签。梯度 $\nabla_x J(\theta, x, y)$ 指示了输入空间中损失增长最快的方向。FGSM构造对抗性示例 $x_{adv}$ 如下：


$$
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
$$


这里：

- $\epsilon$ 是一个小的标量值，控制扰动的幅度 ($L_\infty$ 范数)。它确定了单个特征（例如，像素值）允许的最大变化。
- $\text{sign}(\cdot)$ 是符号函数，输入为正时返回+1，为负时返回-1，输入为零时通常返回0。应用符号函数可以确保扰动在梯度非零的每个维度上具有固定的幅度 $\epsilon$，从而在 $L_\infty$ 约束下有效最大化变化。

**FGSM的特点：**

- **速度快：** 它只需要一次梯度计算，因此速度非常快。
- **简单性：** 其思想和实现都很直接。
- **有效性：** 尽管简单，但它对未受防护的模型可能出乎意料地有效。
- **非最优性：** 单一大步可能不是最佳选择。它可能越过决策边界，或落在模型仍然正确的区域。扰动在视觉上可能也过大。

FGSM作为一种重要的基准攻击，也是一些防御策略（如基本对抗训练）的组成部分。

```python
# Python代码片段：FGSM（使用PyTorch等框架）
import torch

def fgsm_attack(model, loss_fn, image, label, epsilon):
    image.requires_grad = True  # 追踪图像的梯度
    output = model(image)
    loss = loss_fn(output, label)
    model.zero_grad() # 清除之前的梯度
    loss.backward()   # 计算损失相对于输入的梯度

    # 获取梯度
    gradient = image.grad.data

    # 获取梯度的符号
    sign_gradient = gradient.sign()

    # 创建受扰动的图像
    perturbed_image = image + epsilon * sign_gradient

    # 可选：将图像值限制在有效范围内（例如，[0, 1]）
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # 将受扰动的图像从计算图中分离
    return perturbed_image.detach()
```

### 基本迭代方法 (BIM) / 迭代FGSM (I-FGSM)

认识到FGSM的单一大步可能并非最优，Kurakin等人在2016年提出了一种迭代版本，通常称为基本迭代方法 (BIM) 或迭代FGSM (I-FGSM)。BIM不是执行一个大小为 $\epsilon$ 的大步，而是使用较小的步长 $\alpha$ 多次应用FGSM步骤，并在每步之后剪裁结果，以确保总扰动保持在原始输入 $x$ 周围的 $\epsilon$-球内。

对于 $N$ 次迭代，过程如下：

1. 初始化：$x^{(0)}_{adv} = x$
2. 迭代 $i$ 从 0 到 $N-1$：
   
   $$
   x^{(i+1)}_{adv} = \text{Clip}_{x, \epsilon} \left( x^{(i)}_{adv} + \alpha \cdot \text{sign}(\nabla_x J(\theta, x^{(i)}_{adv}, y)) \right)
   $$
   

这里：

- $\alpha$ 是每次迭代的步长，通常选择为 $\epsilon / N$ 或一个小的固定值。
- $\text{Clip}_{x, \epsilon}(\cdot)$ 是一个逐元素剪裁函数，它确保生成的 $x^{(i+1)}_{adv}$ 保持在原始输入 $x$ 的 $L_\infty$ 邻域内。具体来说，对于每个元素 $j$：
  $(x^{(i+1)}_{adv})_j = \max(\min((x^{(i+1)}_{adv})_j, x_j + \epsilon), x_j - \epsilon)$
  通常，还需要额外的剪裁步骤来确保像素值保持在有效范围（例如，[0, 1] 或 [0, 255]）内。

**BIM的特点：**

- **有效性：** 对于相同的 $\epsilon$，通常能找到比FGSM更有效的对抗性示例，往往只需更小的扰动即可实现误分类。
- **计算开销：** 需要多次梯度计算（每次迭代一次），使其比FGSM慢。
- **步长调整：** $\alpha$ 和 $N$ 的选择会影响攻击的成功率和速度。

BIM表明迭代优化明显增强了基于梯度攻击的效力。

### 投影梯度下降 (gradient descent) (PGD)

投影梯度下降由Madry等人在2017年提出，可以说是最广泛使用和最强大的一阶迭代攻击。它在BIM的基础上进行构建，并加入了一些改进，使其成为一个更强的对抗者，尤其是在对抗防御方面。PGD被认为是评估模型鲁棒性的标准基准。

主要步骤如下：

1. **随机起始：** 通过向原始输入 $x$ 添加小的随机扰动来初始化对抗性示例 $x^{(0)}_{adv}$，该扰动通常从 $[-\epsilon, \epsilon]$ 均匀抽取，然后投影到以 $x$ 为中心的 $\epsilon$-球上。
   
   $$
   x^{(0)}_{adv} = \Pi_{B(x, \epsilon)} (x + \delta_{\text{random}})
   $$
   
   此处的 $\delta_{\text{random}}$ 是随机噪声，$\Pi_{B(x, \epsilon)}$ 表示投影到以 $x$ 为中心、半径为 $\epsilon$ 的 $L_p$-球上。
2. **迭代梯度上升：** 对于 $i$ 从 0 到 $N-1$：
   a. 计算梯度：$g = \nabla_x J(\theta, x^{(i)}_{adv}, y)$
   b. 使用带符号的梯度步长更新示例：
   
   $$
   x' = x^{(i)}_{adv} + \alpha \cdot \text{sign}(g) \quad \text{（对于 } L_\infty \text{ 范数）}
   $$
   
   （对于 $L_2$ 范数，更新通常是 $x' = x^{(i)}_{adv} + \alpha \cdot g / ||g||_2$）
   c. 投影回 $\epsilon$-球：
   
   $$
   x^{(i+1)}_{adv} = \Pi_{B(x, \epsilon)} (x')
   $$
   
   d. （可选但常见）将值剪裁到有效输入范围（例如，[0, 1]）。

**与BIM的区别：**

- **随机起始：** 这是一个重要区别。从 $\epsilon$-球内的随机点开始，有助于PGD避免陷入原始输入决策边界附近的不良局部最优。这使得攻击对那些可能试图局部平滑损失的防御更为有效。
- **投影：** 投影步骤 $\Pi_{B(x, \epsilon)}$ 被明确规定，确保迭代结果始终保持在所选 $L_p$ 范数定义的允许扰动范围内。对于 $L_\infty$，这种投影等同于BIM中使用的逐元素剪裁。对于 $L_2$，如果扰动向量 (vector)的范数超过 $\epsilon$，则涉及对其重新缩放。

**PGD的特点：**

- **强度：** 被认为是最强的一阶攻击之一。对PGD强大的模型通常对许多其他基于梯度的攻击也强大。
- **基准标准：** 广泛用于评估对抗性防御，并作为PGD对抗训练（PGD-AT）中的攻击方法。
- **灵活性：** 通过改变更新步骤和投影方法，可适应不同的 $L_p$ 范数 ($L_\infty$, $L_2$ )。
- **计算开销：** 与BIM相似，它需要多次迭代和梯度计算。

随机起始和迭代投影使PGD成为评估模型抵御基于梯度规避尝试能力的一个强大基准。

> FGSM 的单步与 BIM 和 PGD 中迭代更新及投影/剪裁循环的比较。PGD 通常在循环前添加随机初始化。

### 分析和比较

| 特性 | FGSM | BIM / I-FGSM | PGD |
| --- | --- | --- | --- |
| **迭代次数** | 1 | 多次 (N) | 多次 (N) |
| **步长** | 大 ($\epsilon$) | 小 ($\alpha < \epsilon$) | 小 ($\alpha < \epsilon$) |
| **约束** | 通过符号和 $\epsilon$ 隐式 | 每步显式剪裁 | 每步显式投影 |
| **起始点** | 原始输入 $x$ | 原始输入 $x$ | 围绕 $x$ 的随机扰动 |
| **速度** | 非常快 | 中等 | 中等 |
| **强度** | 基准 | 强于 FGSM | 非常强 / 基准标准 |
| **复杂度** | 低 | 中等 | 中等 |

尽管FGSM提供了快速评估，但BIM通过迭代提高了有效性。PGD通过增加随机起始和正式投影进一步增强了这一点，使其成为寻找对抗性示例和评估防御的更可靠方法。$L_\infty$ 范数和 $L_2$ 范数之间的选择也影响攻击：$L_\infty$ PGD 倾向于在许多像素上产生微小变化，而 $L_2$ PGD 可能产生更局部但可能更大的变化，从而导致视觉上不同的扰动。在考察更复杂的基于优化或基于查询的攻击之前，理解这些基于梯度的方法是根本。

## 参考资料

- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) — Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (2014)
  Journal: arXiv; Publisher: arXiv; DOI: [10.48550/arXiv.1412.6572](https://doi.org/10.48550/arXiv.1412.6572)
  介绍快速梯度符号法 (FGSM)，这是一种基础的单步梯度攻击方法。
- [Adversarial Examples in the Physical World](https://arxiv.org/abs/1607.02533) — Alexey Kurakin, Ian Goodfellow, Samy Bengio (2016)
  Journal: arXiv preprint arXiv:1607.02533; DOI: [10.48550/arXiv.1607.02533](https://doi.org/10.48550/arXiv.1607.02533)
  提出基本迭代法 (BIM) 或迭代FGSM (I-FGSM)，通过多次小步迭代和裁剪扩展了FGSM。
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://doi.org/10.48550/arXiv.1706.06083) — Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu (2017)
  Journal: International Conference on Learning Representations (ICLR'18); DOI: [10.48550/arXiv.1706.06083](https://doi.org/10.48550/arXiv.1706.06083)
  引入投影梯度下降 (PGD)，作为一种强大且普遍认可的对抗性攻击，也是衡量模型鲁棒性的基准。

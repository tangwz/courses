# Sigmoid 函数

来源：[原文](https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-4-supervised-learning-classification/sigmoid-function)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在前一节中，我们介绍了逻辑回归作为一种二分类方法，目标是预测输入属于两个类别中的哪一个（通常标记 (token)为0或1）。但是，一个看起来有些像线性回归的算法，最终是如何预测类别的呢？线性回归产生的输出值可以是任意实数，这对于区分两个离散类别没有直接用处。

我们需要一种方法来转换线性方程的输出，我们称之为 $z = w \cdot x + b$ (此处 $w$ 代表权重 (weight)，$x$ 代表输入特征，$b$ 代表偏差)，转换为一个表示概率的值。概率很方便地限制在0和1之间，使它们成为分类任务的理想选择。如果概率高（接近1），我们就预测为类别1；如果概率低（接近0），我们就预测为类别0。

这就是 **Sigmoid 函数**，也称为**逻辑函数**，发挥作用的地方。它是一个数学函数，接受任意实数作为输入，并将其压缩到0和1之间的输出值。

### Sigmoid 公式

Sigmoid 函数，通常用希腊字母西格玛 $\sigma$ 表示，定义为：


$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$


在这里，$z$ 是函数的输入 (在逻辑回归中，它是线性部分的输出，$w \cdot x + b$)，并且 $e$ 是自然对数的底（约2.718）。

### 特性与形状

让我们看看这个函数的作用是什么：

1. **输出范围：** 无论 $z$ 取什么值（无论是-1000、0还是500），输出 $\sigma(z)$ 将始终严格介于0和1之间。它永远不会真正达到0或1，但会非常接近。
2. **大输入值的表现：** 当 $z$ 变得非常大且为正时，$e^{-z}$ 趋近于0。因此，$\sigma(z)$ 趋近于 $\frac{1}{1 + 0} = 1$。
3. **大负输入值的表现：** 当 $z$ 变得非常大且为负时，$e^{-z}$ 变得非常大且为正。因此，$\sigma(z)$ 趋近于 $\frac{1}{1 + \text{一个很大的数}}$，这使得它接近0。
4. **零值时的表现：** 当 $z = 0$ 时，$e^{-0} = 1$。所以，$\sigma(0) = \frac{1}{1 + 1} = 0.5$。

绘制时，该函数呈现出独特的“S”形：



[交互图表：Sigmoid 函数](https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-4-supervised-learning-classification/sigmoid-function#plot-141bddj)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        0.0067,
        0.018,
        0.0474,
        0.1192,
        0.2689,
        0.5,
        0.7311,
        0.8808,
        0.9526,
        0.982,
        0.9933
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#228be6",
        "width": 3
      }
    }
  ],
  "layout": {
    "title": "Sigmoid 函数",
    "xaxis": {
      "title": "z (输入)"
    },
    "yaxis": {
      "title": "σ(z) (输出概率)",
      "range": [
        -0.05,
        1.05
      ]
    },
    "height": 350,
    "margin": {
      "l": 50,
      "r": 20,
      "t": 40,
      "b": 40
    }
  }
}
```

</details>



> Sigmoid 函数 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 将任意实数输入 $z$ 平滑地映射到0和1之间的输出值。

### 与逻辑回归的关联

在逻辑回归中，模型像线性回归一样计算线性组合 $z = w \cdot x + b$。然而，模型不直接将 $z$ 作为预测值，而是将 $z$ 输入到Sigmoid函数中：


$$
h_\theta(x) = \sigma(z) = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}
$$


输出 $h_\theta(x)$ (这里的 $\theta$ 指代模型的参数 (parameter)，$w$ 和 $b$) 现在被解释为输入 $x$ 属于正类别（类别1）的**估计概率**。

例如，如果对于给定的输入 $x$，模型计算出 $z=2$，那么输出概率就是 $\sigma(2) \approx 0.88$。这意味着模型估计这个输入有88%的可能性属于类别1。如果另一个输入导致 $z=-1$，输出概率是 $\sigma(-1) \approx 0.27$，表示有27%的可能性属于类别1（或者反过来说，有73%的可能性属于类别0）。

这种概率输出对逻辑回归来说非常重要。通常，我们设置一个**决策阈值**（通常是0.5）来将此概率转换为明确的类别预测。如果 $h_\theta(x) \ge 0.5$，我们预测为类别1；否则，我们预测为类别0。 $z=0$ 且 $\sigma(z)=0.5$ 的点通常对应于分隔预测类别的边界，我们将在讨论决策边界时后续讨论这一点。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](http://www-stat.stanford.edu/~tibs/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: 119-122
  本书是统计学习的综合参考资料，涵盖了逻辑回归及其基础数学函数。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 205-212 (for Logistic Regression and Sigmoid); DOI: [10.1007/9780387310732](https://doi.org/10.1007/9780387310732)
  一本标准教科书，介绍了机器学习模型的概率基础，包含对逻辑回归和S型函数严谨的阐述。
- [Lecture Notes on Logistic Regression (CS229 Machine Learning)](http://cs229.stanford.edu/notes/cs229-notes2.pdf) — Andrew Ng (2021)
  Publisher: Stanford University; Pages: 1-5
  来自顶级机器学习课程的官方讲义，清晰地解释了S型函数在逻辑回归中的作用。

---

[上一节](02-%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E4%BB%8B%E7%BB%8D.md) · [下一节](04-%E5%86%B3%E7%AD%96%E8%BE%B9%E7%95%8C.md)

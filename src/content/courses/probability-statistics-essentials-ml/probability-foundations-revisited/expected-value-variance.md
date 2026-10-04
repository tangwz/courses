---
course: "probability-statistics-essentials-ml"
chapter: "probability-foundations-revisited"
lesson: "expected-value-variance"
sourceId: 1355
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-1-probability-foundations-revisited/expected-value-variance"
title: "期望值与方差"
description: "计算并理解随机变量的期望值和方差。"
order: 5
plots: []
sourceHash: "8249d6025c133afead85025d5dd7ed68536ea3feecf690b6407c4478a210e9db"
sourceCorrections: []
---

既然我们理解了随机变量是如何将样本空间中的结果映射到数值的，我们通常希望对这些变量的特征进行概括。只知道可能的值是不够的；我们需要方法来描述这些值分布的“中心”和“离散程度”。期望值和方差是提供这种概括的基本度量。

### 期望值：分布的中心

**期望值**，记为 $E[X]$ 或有时为 $\mu_X$（或简写为 $\mu$），表示随机变量 $X$ 可能取值的加权平均，权重 (weight)即为这些值的概率。直观地说，如果您多次重复涉及 $X$ 的实验并计算结果的平均值，该平均值将趋近于期望值 $E[X]$。它就像概率分布的“质量中心”。

#### 期望值的计算

离散型和连续型随机变量的计算方式略有不同：

- **对于离散型随机变量 $X$：** 如果 $X$ 可以取值 $x_1, x_2, ..., x_n$，对应概率分别为 $P(X=x_1), P(X=x_2), ..., P(X=x_n)$，则期望值为：
  $E[X] = \sum_{i} x_i P(X=x_i)$
  求和范围涵盖所有可能的 $x_i$ 值。
- **对于连续型随机变量 $X$：** 如果 $X$ 具有概率密度函数 (PDF) $f(x)$，则期望值通过对 $x$ 与 $f(x)$ 的乘积在 $X$ 的整个定义域上积分来计算：
  $E[X] = \int_{-\infty}^{\infty} x f(x) dx$

**示例：公平的六面骰子**

令 $X$ 为表示投掷公平六面骰子结果的随机变量。可能的值为 $\{1, 2, 3, 4, 5, 6\}$，每个值的概率为 $1/6$。

期望值为：
$E[X] = 1 \cdot \frac{1}{6} + 2 \cdot \frac{1}{6} + 3 \cdot \frac{1}{6} + 4 \cdot \frac{1}{6} + 5 \cdot \frac{1}{6} + 6 \cdot \frac{1}{6}$
$E[X] = \frac{1+2+3+4+5+6}{6} = \frac{21}{6} = 3.5$
请注意，期望值 (3.5) 并非骰子实际可能掷出的值。它是多次投掷后得到的长期平均结果。

#### 期望值的性质

期望值具有一些有用的线性性质：

1. **常数：** 如果 $c$ 是常数，则 $E[c] = c$。
2. **缩放与平移：** 对于常数 $a$ 和 $b$，有 $E[aX + b] = aE[X] + b$。
3. **随机变量的和：** 对于任意两个随机变量 $X$ 和 $Y$，有 $E[X + Y] = E[X] + E[Y]$。无论 $X$ 和 $Y$ 是否独立，此性质都成立。

这些性质在简化涉及随机变量组合的计算时非常有用。

### 方差与标准差：度量离散程度

期望值告诉我们分布的中心位置，但它并没有说明数值的离散程度。这些值是紧密地聚集在均值附近，还是广泛地分散开来？**方差**度量这种离散程度。

随机变量 $X$ 的**方差**，记为 $Var(X)$ 或 $\sigma^2_X$（或简写为 $\sigma^2$），是随机变量与其期望值 $\mu = E[X]$ 之间平方差的期望值。

$Var(X) = E[(X - \mu)^2]$

较高的方差意味着 $X$ 的值平均而言更远离均值。较低的方差意味着它们更接近均值。

#### 方差的计算

与期望值类似，计算取决于变量是离散型还是连续型：

- **对于离散型随机变量 $X$：** 使用定义 $\mu = E[X]$：
  $Var(X) = \sum_{i} (x_i - \mu)^2 P(X=x_i)$
- **对于连续型随机变量 $X$：** 使用定义 $\mu = E[X]$ 和 PDF $f(x)$：
  $Var(X) = \int_{-\infty}^{\infty} (x - \mu)^2 f(x) dx$

通常有一个更方便的计算公式源自定义：
$Var(X) = E[X^2] - (E[X])^2$
要使用此公式，您首先计算 $E[X]$（均值）和 $E[X^2]$（$X$ 平方的期望值），然后将它们代入公式。请记住，对于离散变量，$E[X^2] = \sum_i x_i^2 P(X=x_i)$；对于连续变量，$E[X^2] = \int_{-\infty}^{\infty} x^2 f(x) dx$。

#### 标准差

方差的单位是原始随机变量单位的平方（例如，如果 $X$ 以米为单位，则 $Var(X)$ 以平方米为单位）。这可能难以直接理解。**标准差**，记为 $\sigma_X$ 或 $SD(X)$（或简写为 $\sigma$），是方差的正平方根：

$\sigma = \sqrt{Var(X)}$

标准差的单位与原始随机变量 $X$ 相同，使其能更直观地理解与均值的典型偏差。

**示例：公平的六面骰子（续）**

我们发现 $E[X] = 3.5$。让我们使用定义来计算方差：


$$
\begin{aligned}
Var(X) = & (1 - 3.5)^2 \frac{1}{6} + (2 - 3.5)^2 \frac{1}{6} + (3 - 3.5)^2 \frac{1}{6} \\
& + (4 - 3.5)^2 \frac{1}{6} + (5 - 3.5)^2 \frac{1}{6} + (6 - 3.5)^2 \frac{1}{6} \\
= & \frac{1}{6} [(-2.5)^2 + (-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2 + (2.5)^2] \\
= & \frac{1}{6} [6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25] \\
= & \frac{1}{6} [17.5] \approx 2.917
\end{aligned}
$$


或者，使用计算公式 $Var(X) = E[X^2] - (E[X])^2$：
首先，计算 $E[X^2]$：


$$
\begin{aligned}
E[X^2] &= 1^2 \frac{1}{6} + 2^2 \frac{1}{6} + 3^2 \frac{1}{6} + 4^2 \frac{1}{6} + 5^2 \frac{1}{6} + 6^2 \frac{1}{6} \\
&= \frac{1+4+9+16+25+36}{6} = \frac{91}{6} \approx 15.167
\end{aligned}
$$


现在，计算方差：
$Var(X) = E[X^2] - (E[X])^2 = \frac{91}{6} - (3.5)^2 = \frac{91}{6} - 12.25 = \frac{91}{6} - \frac{73.5}{6} = \frac{17.5}{6} \approx 2.917$
两种方法得到相同的结果。

标准差为：
$\sigma = \sqrt{Var(X)} = \sqrt{\frac{17.5}{6}} \approx \sqrt{2.917} \approx 1.708$
因此，对于公平的骰子投掷，期望结果是 3.5，并且结果通常与此均值偏差约 1.708。

#### 方差的性质

方差也具有重要的性质：

1. **非负性：** $Var(X) \ge 0$。仅当 $X$ 是常数时，方差才为零。
2. **常数：** 如果 $c$ 是常数，则 $Var(c) = 0$。
3. **缩放与平移：** 对于常数 $a$ 和 $b$，有 $Var(aX + b) = a^2 Var(X)$。请注意，添加常数 $b$ 会平移分布，但不会改变其离散程度，因此 $b$ 不会影响方差。通过 $a$ 缩放会使方差按 $a^2$ 缩放。
4. **独立随机变量的和：** 如果 $X$ 和 $Y$ 是*独立*随机变量，则 $Var(X + Y) = Var(X) + Var(Y)$。如果它们不独立，公式将涉及协方差，这是一个我们可能稍后会涉及的主题。

理解期望值和方差非常重要。它们提供了概率分布中心趋势和离散程度的简明概括，构成了统计学和机器学习 (machine learning)中许多思想的根本，从评估估计量到理解预测中的不确定性。在后续章节中，我们将看到 NumPy 和 SciPy 等 Python 库如何使这些值的计算对于各种分布变得直接。

## 参考资料

- [Probability and Statistics for Engineering and the Sciences](https://www.cengage.com/c/probability-and-statistics-for-engineering-and-the-sciences-9e-devore/9781305251809/) — Jay L. Devore (2016)
  Publisher: Cengage Learning
  一本广泛使用的教科书，提供了概率和统计的平衡处理及实用示例，深入涵盖了期望值和方差及其应用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: 37-59; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  本书提供了坚实的统计基础，早期章节涵盖了期望值和方差等基本概率论知识，对理解统计机器学习至关重要。

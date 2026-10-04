---
course: "bayesian-machine-learning"
chapter: "gaussian-processes"
lesson: "gp-kernels-properties-selection"
sourceId: 3590
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-4-gaussian-processes/gp-kernels-properties-selection"
title: "协方差函数（核）：性质与选择"
description: "分析常见的核函数（RBF、马特恩、周期）及核函数构建方法。"
order: 3
plots: ["plots/3590-0.json", "plots/3590-1.json"]
sourceHash: "31d5ed1403333e91eb4ea0aafdc54069d696c9ca430c12003132beef2bfac1d6"
sourceCorrections: []
---

高斯过程完全由其均值函数 $m(\mathbf{x})$ 和协方差函数 $k(\mathbf{x}, \mathbf{x}')$ 确定。均值函数定义了函数在任意点 $\mathbf{x}$ 的平均期望值，而协方差函数（常称为**核函数**）则真正决定了能建模的函数的*形状*和*结构*。核函数包含了对正尝试学习的函数的设想，例如它的平滑性、周期性或平稳性。

### 协方差函数（核函数）的作用

核函数 $k(\mathbf{x}, \mathbf{x}')$ 量化 (quantization)了函数在两个不同输入点 $\mathbf{x}$ 和 $\mathbf{x}'$ 处的函数值之间的关系。具体来说，它定义了随机变量 $f(\mathbf{x})$ 和 $f(\mathbf{x}')$ 之间的协方差：


$$
k(\mathbf{x}, \mathbf{x}') = \text{协方差}(f(\mathbf{x}), f(\mathbf{x}')) = \mathbb{E}[(f(\mathbf{x}) - m(\mathbf{x}))(f(\mathbf{x}') - m(\mathbf{x}'))]
$$


如果点 $\mathbf{x}$ 和 $\mathbf{x}'$ 被核函数认为是“相似”的（即 $k(\mathbf{x}, \mathbf{x}')$ 很大），那么我们期望它们对应的输出值 $f(\mathbf{x})$ 和 $f(\mathbf{x}')$ 接近。反之，如果核函数认为它们不相似（$k(\mathbf{x}, \mathbf{x}')$ 很小或为零），它们的输出值关联性则较弱。

对于任何有限的 $N$ 个输入点集合 $\mathbf{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$，核函数定义了 $N \times N$ 的协方差矩阵 $\mathbf{K}$，其中每个元素 $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$。该矩阵是高斯过程计算的核心。

### 核函数的基本性质

为了使函数 $k(\cdot, \cdot)$ 成为有效的协方差函数，它必须满足两个主要性质：

1. **对称性：** $f(\mathbf{x})$ 和 $f(\mathbf{x}')$ 之间的协方差必须与 $f(\mathbf{x}')$ 和 $f(\mathbf{x})$ 之间的协方差相同。
   
   $$
   k(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}', \mathbf{x})
   $$
   
2. **正半定（PSD）：** 对于任何有限点集 $\{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ 和任何实值向量 (vector) $\mathbf{a} = [a_1, \dots, a_N]^T$，结果协方差矩阵 $\mathbf{K}$ 必须满足：
   
   $$
   \mathbf{a}^T \mathbf{K} \mathbf{a} \ge 0
   $$
   
   这保证了函数值的任何线性组合的方差 $\text{方差}(\sum_i a_i f(\mathbf{x}_i)) = \mathbf{a}^T \mathbf{K} \mathbf{a}$ 是非负的，这是任何有效协方差矩阵的基本要求。默瑟定理提供了一种更深层的关联，指出任何连续、对称、正半定核函数都可以看作是某个（可能是无限维的）特征空间中的内积。

### 理解核超参数 (parameter) (hyperparameter)

大多数有用的核函数都有参数，常称为**超参数**，它们控制着核函数的行为。理解这些参数对有效的高斯过程建模来说很要紧。两个常见的超参数是：

- **长度尺度 ($l$)：** 该参数控制着随着输入点之间距离的增加，函数值之间的关联衰减的速度。*小*的长度尺度意味着关联迅速下降，导致函数变化快（波动大）。*大*的长度尺度表明距离较远的点仍保持关联，从而产生更平滑的函数。
- **输出方差 / 信号方差 ($\sigma_f^2$)：** 该参数缩放了函数值的整体方差。它控制从高斯过程先验中抽取的函数的典型幅度或垂直范围。 $k(\mathbf{x}, \mathbf{x}) = \sigma_f^2$ 通常表示任何单个点 $\mathbf{x}$ 处的先验方差（假设均值函数为零）。

长度尺度的影响在下面展示，显示了从使用平方指数核但长度尺度不同的高斯过程抽取的样本。



![长度尺度对高斯过程样本的影响（平方指数核）](plots/3590-0.json)



> 从零均值高斯过程先验（平方指数核，$\sigma_f^2=1$）抽取的函数样本。紫色线使用了较短的长度尺度（$l=0.5$），产生了快速变化的函数。蓝色线使用了较长的长度尺度（$l=2.0$），生成了更平滑的函数。

### 常用核函数示例

选择合适的核函数通常由对函数预期特性的先验知识指导。这里列出一些广泛使用的核函数：

#### 平方指数（SE）核 / 径向基函数（RBF）核

这可谓是最常见的核函数，通常是一个好的起点。


$$
k_{SE}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{||\mathbf{x} - \mathbf{x}'||^2}{2l^2}\right)
$$


- **性质：** 它生成非常平滑的函数（无限可微）。它是**平稳的**，意味着其值仅取决于位移 $||\mathbf{x} - \mathbf{x}'||$，而不是点的绝对位置。
- **超参数 (parameter) (hyperparameter)：** 长度尺度 $l$，输出方差 $\sigma_f^2$。
- **ARD：** 对于多维输入 $\mathbf{x} \in \mathbb{R}^D$，你可以使用自动相关性判定（ARD），通过为每个维度分配一个单独的长度尺度 $l_d$：
  
  $$
  k_{SE-ARD}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{1}{2}\sum_{d=1}^D \frac{(x_d - x'_d)^2}{l_d^2}\right)
  $$
  
  优化后，$l_d$ 值大的维度被有效忽略，表明其对预测的相关性较低。

#### 马特恩核

马特恩核家族在控制函数的平滑性方面提供了更多灵活性。


$$
k_{Mat\acute{e}rn}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu}\frac{r}{l}\right)^\nu K_\nu\left(\sqrt{2\nu}\frac{r}{l}\right)
$$


其中 $r = ||\mathbf{x} - \mathbf{x}'||$，$\Gamma(\cdot)$ 是伽马函数，而 $K_\nu(\cdot)$ 是第二类修正贝塞尔函数。

- **性质：** 参数 $\nu > 0$ 控制了函数样本的可微性。从使用马特恩核的高斯过程抽取的函数，若 $\nu > k$，则可微 $k$ 次。常见选择是 $\nu = 3/2$（一次可微）和 $\nu = 5/2$（两次可微），这对物理过程来说通常比平方指数核的无限可微性更实际。当 $\nu \to \infty$ 时，马特恩核收敛于平方指数核。当 $\nu = 1/2$ 时，产生**指数核**，它生成连续但不可微的函数（粗糙路径，类似于奥恩斯坦-乌伦贝克过程）。马特恩核是平稳的。
- **超参数：** 长度尺度 $l$，输出方差 $\sigma_f^2$，平滑参数 $\nu$。通常 $\nu$ 是固定值（例如 3/2 或 5/2），而不是进行优化。

#### 周期核

对建模表现出重复模式的函数很合适。


$$
k_{Per}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{2 \sin^2(\pi ||\mathbf{x} - \mathbf{x}'|| / p)}{l^2}\right)
$$


- **性质：** 建模周期为 $p$ 的重复函数。这里的长度尺度 $l$ 控制周期内的平滑性。它对季节性数据或已知频率的信号有用。平稳的。
- **超参数：** 长度尺度 $l$，输出方差 $\sigma_f^2$，周期 $p$。

#### 线性核

建模简单的线性关系。


$$
k_{Lin}(\mathbf{x}, \mathbf{x}') = \sigma_b^2 + \sigma_v^2 (\mathbf{x} - c)(\mathbf{x}' - c)^T
$$


（存在其他变体，例如 $k_{Lin}(\mathbf{x}, \mathbf{x}') = \sigma_b^2 + \sigma_v^2 \mathbf{x}^T \mathbf{x}'$）。

- **性质：** 生成线性趋势函数。它是**非平稳的**，因为协方差取决于绝对位置 $\mathbf{x}$ 和 $\mathbf{x}'$，而不仅仅是它们的差值。参数 $c$ 代表一个偏移量。$\sigma_b^2$ 是一个常数偏移方差，而 $\sigma_v^2$ 控制斜率的方差。
- **超参数：** 偏移方差 $\sigma_b^2$，斜率方差 $\sigma_v^2$，偏移量 $c$。

下面是不同核函数类型样本的比较。



![不同核函数的高斯过程样本](plots/3590-1.json)



> 从零均值高斯过程先验（$\sigma_f^2=1$）中使用不同核函数抽取的函数样本。平方指数（SE）核产生了最平滑的样本。马特恩 3/2 样本较不平滑（一次可微）。周期样本清楚地显示了重复模式。

### 核函数构建：组合核函数

核函数真正的效用通常来自组合基本核函数，以捕捉数据中更复杂的结构。如果 $k_1$ 和 $k_2$ 是有效核函数，那么以下组合也是有效的核函数：

- **加法：** $k_{sum}(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') + k_2(\mathbf{x}, \mathbf{x}')$
  这对应于将函数 $f(\mathbf{x})$ 建模为两个独立高斯过程的和，$f(\mathbf{x}) = f_1(\mathbf{x}) + f_2(\mathbf{x})$，其中 $f_1 \sim \mathcal{GP}(0, k_1)$ 且 $f_2 \sim \mathcal{GP}(0, k_2)$。这对建模加性效应非常有用，比如长期趋势加上季节性成分（$k_{Lin} + k_{Per}$）。
- **乘法：** $k_{prod}(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') \times k_2(\mathbf{x}, \mathbf{x}')$
  乘法可以产生有趣的相互作用。例如，将平滑核（如平方指数核）与周期核相乘（$k_{SE} \times k_{Per}$）可以建模一个幅度随时间平滑变化的周期函数。即使基本核函数是平稳的，核函数的乘法也可以产生非平稳效应。

通过仔细组合核函数，你可以将关于函数结构的复杂先验设想直接编码到模型中。

### 核函数选择策略

选择合适的核函数是高斯过程建模中一个重要的环节。以下是一般方法：

1. **利用领域知识：** 你对生成数据的潜在过程了解多少？预期它是平滑的（平方指数核、马特恩核）吗？它有周期（周期核）吗？有基线趋势（线性核）吗？从反映这些设想的核函数开始。
2. **目视检查：** 绘制数据常能显示趋势、季节性或突变等模式，这些能提示合适的核函数结构或组合。
3. **从简单开始：** 从标准核函数（如平方指数核或马特恩核）开始（例如马特恩 3/2 或 5/2）。这些对许多应用来说通常足够灵活。
4. **考虑组合：** 如果简单核函数未能捕捉结构，请考虑加性或乘性组合（例如：趋势 + 季节性，平滑变化的幅度）。
5. **超参数 (parameter) (hyperparameter)优化：** 无论最初选择如何，核函数的超参数（$l, \sigma_f^2, p$ 等）*必须*根据数据调整。这通常通过最大化边际似然来完成，我们将在下一节讨论这一点。
6. **模型比较：** 如果你有几个候选核函数或核函数组合，请使用模型比较技术（例如比较边际似然或使用交叉验证）来选择最能解释数据的那个。

核函数的选择根本上塑造了高斯过程模型。它定义了高斯过程在看到数据*之前*认为可信的函数空间。选择得当的核函数，结合优化过的超参数，能够让高斯过程从数据中有效学习，并做出准确预测以及有意义的不确定性估计。接下来，我们将考察如何从数据中学习这些重要的超参数。

## 参考资料

- [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf) — Carl Edward Rasmussen and Christopher K. I. Williams (2006)
  Publisher: The MIT Press
  关于高斯过程的权威教科书，提供了对协方差函数、其性质、常见类型和构建规则的全面解释。对于全面理解至关重要。
- [Machine Learning: A Probabilistic Perspective](https://mitpress.mit.edu/books/machine-learning-probabilistic-perspective) — Kevin P. Murphy (2012)
  Publisher: The MIT Press
  一本广泛而深入的概率机器学习教科书。在机器学习的更广阔背景下，包括高斯过程和支持向量机，提供了对核函数的论述。
- [Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond](https://mitpress.mit.edu/9780262700436/learning-with-kernels/) — Bernhard Schölkopf, Alexander J. Smola (2002)
  Publisher: The MIT Press
  这本书是机器学习中核方法的基础教材，详细讨论了Mercer定理、正定核及其在高斯过程之外的应用。

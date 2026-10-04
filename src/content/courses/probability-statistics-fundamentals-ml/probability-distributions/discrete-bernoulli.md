---
course: "probability-statistics-fundamentals-ml"
chapter: "probability-distributions"
lesson: "discrete-bernoulli"
sourceId: 2446
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-4-probability-distributions/discrete-bernoulli"
title: "离散分布：伯努利"
description: "了解伯努利分布，用于描述只有两种结果的单次试验。"
order: 3
plots: ["plots/2446-0.json"]
sourceHash: "04db546c4236eda363a217b461bcc5643b2b53c2436a02b590adf765ad6a9ec5"
sourceCorrections: []
---

伯努利分布是最简单的离散概率分布。它由离散随机变量的概率质量函数（PMF）来描述。

设想一个只有两种可能结果的实验。例如抛掷硬币（正面或反面），检查电子邮件是否为垃圾邮件（垃圾邮件或非垃圾邮件），或者用户点击广告（点击或未点击）。这些都是**伯努利试验**的例子。

伯努利试验是只有两个互斥结果的单次随机实验，通常标记 (token)为“成功”和“失败”。

**伯努利分布**用于描述单次伯努利试验结果的概率。它只依赖于一个参数 (parameter)：

- $p$：定义为“成功”的结果的概率。

因为只有两种结果，所以“失败”的概率必然是 $1-p$。

我们定义一个随机变量 $X$ 来表示伯努利试验的结果。通常，我们设定：

- 如果结果是“成功”，则 $X = 1$
- 如果结果是“失败”，则 $X = 0$

伯努利随机变量 $X$ 的概率质量函数（PMF）很简单：

$P(X=1) = p$
$P(X=0) = 1-p$

这指明了 $X$ 可以取的两个可能值的概率。有时，你可能会看到它用一个更紧凑的公式表示：

$P(X=k) = p^k (1-p)^{1-k} \quad \text{对于 } k \in \{0, 1\}$

我们来验证这个紧凑公式是否适用。
如果 $k=1$（成功），公式得出 $p^1 (1-p)^{1-1} = p^1 (1-p)^0 = p \times 1 = p$。正确。
如果 $k=0$（失败），公式得出 $p^0 (1-p)^{1-0} = p^0 (1-p)^1 = 1 \times (1-p) = 1-p$。正确。

### 伯努利试验的例子

- **抛掷硬币：** 单次抛掷一枚均匀硬币。设“正面”为成功。则 $p=0.5$。$P(X=1) = 0.5$ 且 $P(X=0) = 0.5$。如果硬币是偏的，70% 的情况是正面朝上，那么 $p=0.7$。$P(X=1) = 0.7$ 且 $P(X=0) = 0.3$。
- **质量控制：** 检测单个生产部件。设“有缺陷”为成功（这是一种不寻常的成功定义，但有可能！）。如果5%的部件有缺陷，则 $p=0.05$。$P(X=1) = 0.05$（有缺陷的概率）且 $P(X=0) = 0.95$（无缺陷的概率）。
- **医学检测：** 患者接受针对特定病症的检测。设“检测结果为阳性”为成功。概率 $p$ 将取决于检测的特性以及患者是否患有该病症。

### 伯努利分布的可视化

因为只有两种结果，所以可视化很简单。它是一个只有两根条形图的柱状图。我们来可视化一个成功概率 ($p$) 为 0.7 的伯努利分布：



![伯努利分布PMF (p=0.7)](plots/2446-0.json)



> 伯努利分布的PMF，其中 $p=0.7$。失败的概率 ($X=0$) 是 $1-p=0.3$，成功的概率 ($X=1$) 是 $p=0.7$。

伯努利分布之所以重要，是因为它是处理多次试验的更复杂分布（例如我们接下来要看的二项分布）的构成部分。在机器学习 (machine learning)中，它常用于建模二元结果，例如在逻辑回归中，模型预测一个正类（“成功”）的概率。掌握这个简单的分布为理解更深层的概率思想打下了坚实基础。

## 参考资料

- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 37-43
  涵盖机器学习相关概率分布的基础著作，为二元结果建模中的伯努利试验提供了背景。
- [Introduction to Probability and Statistics](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+18.05r_10+2022_Summer/about) — Dr. Jeremy Orloff, Dr. Jennifer French Kamrin (2023)
  Journal: MIT OpenCourseWare; Publisher: Massachusetts Institute of Technology
  大学入门级概率与统计课程材料，涵盖伯努利分布等基础概念。
- [Probability and Statistics for Engineering and the Sciences](https://www.cengage.com/c/probability-and-statistics-for-engineering-and-the-sciences-10e-devore/9781305251809/) — Jay L. Devore (2016)
  Publisher: Cengage Learning; Pages: 100-102
  采用实用方法讲解概率论和统计方法，包括离散概率分布的教材。

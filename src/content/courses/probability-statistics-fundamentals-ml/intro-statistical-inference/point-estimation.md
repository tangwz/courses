---
course: "probability-statistics-fundamentals-ml"
chapter: "intro-statistical-inference"
lesson: "point-estimation"
sourceId: 2463
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-5-intro-statistical-inference/point-estimation"
title: "点估计"
description: "了解用于估计总体参数的点估计值（如样本均值）。"
order: 2
plots: ["plots/2463-0.json"]
sourceHash: "7ae2eadcc9349ad74235365de7df8f234c7c2d1032b3084c2166775082e95c51"
sourceCorrections: []
---

正如本章引言中所述，统计推断使我们能够仅凭较小样本的数据，对较大的总体进行有根据的推测。其中一种最直接的方法是进行**点估计**。

其主要思想很简单：我们想要估计总体的一个未知特征，这个特征称为**参数 (parameter)**。由于我们通常无法测量整个总体，因此我们会从样本中计算出一个相似的特征，这称为**统计量**。这个样本统计量就是我们对未知总体参数的单一“最佳猜测”，或者说点估计。

### 总体参数 (parameter)与样本统计量

区分这两个观念很重要：

- **总体参数：** 概括整个总体某个特征的数值。这些值通常未知，也正是我们想要估计的。例子包括：
  - 加拿大所有成年女性的*真实*平均身高（总体均值，通常表示为 $\mu$）。
  - 所有网站访问者中点击特定广告的*真实*比例（总体比例，通常表示为 $p$）。
  - 机器生产的零件直径的*真实*变异性（总体方差，$\sigma^2$，或标准差，$\sigma$）。
- **样本统计量：** 从样本数据中计算出的数值。我们使用这个数值来估计对应的总体参数。例子包括：
  - 从500名加拿大女性样本中计算出的平均身高（样本均值，表示为 $\bar{x}$）。
  - 从10,000名网站访问者样本中观察到的点击比例（样本比例，表示为 $\hat{p}$）。
  - 从100个零件样本中计算出的变异性（样本方差，$s^2$，或标准差，$s$）。

可以将总体参数看作是我们想要达到的固定真实值，而样本统计量则是我们根据能够收集到的数据计算出的近似值。

### 常用点估计量

我们用来计算样本统计量的公式或规则称为**估计量**。将估计量应用于样本数据后得到的具体数值称为**估计值**。

这里有一些常见例子：

1. **估计总体均值 ($\mu$)：** 总体均值 $\mu$ 最常用的估计量是样本均值 $\bar{X}$。公式（即估计量）是：

   
   $$
   \bar{X} = \frac{\sum_{i=1}^{n} X_i}{n} = \frac{X_1 + X_2 + ... + X_n}{n}
   $$
   

   如果我们收集一个样本并计算样本均值，例如 $\bar{x} = 165.2$ 厘米，那么 $165.2$ 厘米就是我们对总体平均身高 $\mu$ 的点*估计值*。
2. **估计总体比例 ($p$)：** 要估计总体比例 $p$（例如，缺陷品的比例），我们使用样本比例 $\hat{P}$。估计量是：

   
   $$
   \hat{P} = \frac{\text{样本中成功的数量}}{\text{样本大小}} = \frac{x}{n}
   $$
   

   如果我们测试200个物品（n=200）并发现10个有缺陷（x=10），那么我们对真实缺陷品比例的点*估计值*是 $\hat{p} = 10 / 200 = 0.05$ 或 5%。
3. **估计总体方差 ($\sigma^2$)：** 总体方差 $\sigma^2$ 的一个常用估计量是样本方差 $S^2$。公式是：

   
   $$
   S^2 = \frac{\sum_{i=1}^{n} (X_i - \bar{X})^2}{n-1}
   $$
   

   请注意分母中的 $n-1$。使用 $n-1$ 而不是 $n$ 会使 $S^2$ 成为 $\sigma^2$ 的“无偏”估计量，这意味着它平均而言不会系统地低估或高估真实总体方差。为我们的样本计算此值即可得到点*估计值* $s^2$。

### 为什么称为“点”估计？

之所以称为点估计，是因为它为总体参数 (parameter)提供了一个单一的数值作为估计值。想象一个表示参数所有可能值（例如平均身高）的数轴；我们的点估计就是该数轴上的一个点，代表我们基于样本的最佳猜测。

例如，如果我们估计客户平均花费为 \$45.50，那么这个单一值就是我们的点估计值。尽管这是我们最佳的猜测，但我们知道它不太可能*精确地*等于真实的总体平均花费。样本会有所不同，不同的样本可能会产生略有差异的估计值。这种固有的不确定性引出了*区间估计*（置信区间）的观念，我们将在接下来讨论。区间估计为参数提供一个合理的值范围，承认其中存在的不确定性。

### 例子：估计平均订单价值

假设我们运营一个电子商务网站，并想估计*所有*客户的平均订单价值（AOV）（即总体参数 (parameter) $\mu$）。我们无法从所有订单中计算，因此我们抽取50个近期订单作为样本：

```python
import numpy as np

# 50个近期订单价值的样本（美元）
order_values = np.array([
    55.2, 34.1, 89.0, 120.5, 42.0, 65.8, 22.1, 98.7, 77.3, 105.6,
    30.9, 48.2, 71.5, 60.0, 112.8, 58.4, 88.2, 45.1, 99.9, 75.0,
    135.2, 28.6, 50.0, 68.9, 91.3, 102.1, 40.5, 79.8, 66.2, 84.7,
    115.0, 33.5, 52.8, 70.1, 95.4, 108.3, 47.9, 62.7, 81.6, 100.2,
    36.8, 56.7, 73.9, 64.3, 89.1, 110.5, 49.6, 67.0, 83.3, 97.4
])

# 计算样本均值（我们对mu的点估计值）
sample_mean_aov = np.mean(order_values)

# 计算样本标准差（对sigma的点估计值）
# 使用ddof=1表示无偏样本标准差（估计总体标准差）
sample_std_dev = np.std(order_values, ddof=1)

print(f"Sample Size (n): {len(order_values)}")
print(f"Point Estimate for Average Order Value (μ): ${sample_mean_aov:.2f}")
print(f"Point Estimate for Standard Deviation (σ): ${sample_std_dev:.2f}")
```

```text
样本大小 (n): 50
平均订单价值 (μ) 的点估计值: $74.29
标准差 (σ) 的点估计值: $27.68
```

基于这个样本，我们对所有客户的真实平均订单价值 ($\mu$) 的点估计值是 $74.29。我们对总体标准差 ($\sigma$) 的点估计值是$27.68。



![样本订单价值分布](plots/2463-0.json)



> 50个样本订单价值的直方图。虚线红色线条表示计算出的样本均值（\$74.29），它作为所有客户未知平均订单价值的点估计值。

点估计构成了许多统计推断过程的基础，并常用于机器学习 (machine learning)，例如在基于测试集估计模型的错误率或确定假设用于建模数据的概率分布参数时。然而，记住它们是基于有限数据的单一猜测很重要。接下来，我们将查看如何量化 (quantization)这些估计值周围的不确定性。

## 参考资料

- [Statistical Inference](https://www.taylorfrancis.com/books/9781003456285) — George Casella, Roger L. Berger (2002)
  Publisher: Duxbury Thomson Learning; Pages: Chapter 7; DOI: [10.1201/9781003456285](https://doi.org/10.1201/9781003456285)
  一本关于统计推断的综合教材，涵盖了点估计的理论基础、估计量的性质以及寻找良好估计量的方法。
- [Probability and Statistics](https://www.pearson.com/us/higher-education/program/DeGroot-Probability-and-Statistics-Classic-Version-4th-edition/PGM296683.html) — Morris H. DeGroot and Mark J. Schervish (2012)
  Publisher: Pearson; Pages: Chapter 7
  清晰阐释了点估计，区分了总体参数和样本统计量，并展示了常见估计量。
- [Practical Statistics for Data Scientists: 50 Essential Concepts](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, Peter Gedeck (2020)
  Publisher: O'Reilly Media; Pages: Chapter 1
  讨论了位置和变异性的估计，为点估计在数据科学和机器学习应用中的使用提供了实践背景。
- [Introduction to Probability and Statistics](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/) — Jeremy Orloff, Jonathan Bloom (2014)
  Publisher: MIT OpenCourseWare; Pages: Unit III, Lectures 12-14
  麻省理工学院的免费在线课程资料，涵盖了统计推断的基础概念，包括参数、统计量和点估计。

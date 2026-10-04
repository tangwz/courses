---
course: "calculus-fundamentals-machine-learning"
chapter: "optimization-with-derivatives"
lesson: "cost-functions-ml"
sourceId: 2273
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-3-optimization-with-derivatives/cost-functions-ml"
title: "机器学习中的代价函数"
description: "介绍代价函数（损失函数）作为衡量模型误差的方法。"
order: 3
plots: ["plots/2273-0.json"]
sourceHash: "2c217e60cca7f999f69eab9161fd0cf62c01b43f910f1c4d8bbc73f4a4fc2936"
sourceCorrections: []
---

在上一节中，我们讨论了优化的基本思想，即找到函数的最低点或最高点。在机器学习 (machine learning)中，这不仅仅是一个抽象练习。我们优化特定的函数，以使模型能够有效地从数据中学习。但我们究竟要最小化或最大化什么呢？

“想象一下训练一个机器学习模型。目标通常是使预测尽可能接近实际结果。例如，如果我们预测房价，我们希望模型的房价预测与房屋实际售价非常接近。如果我们分类图像，我们希望模型分配正确的标签（例如“猫”或“狗”）。”

我们需要一种方法来衡量模型预测与真实值之间“偏差”的程度。这种衡量方式由**代价函数**体现，它也常被称为**损失函数 (loss function)**或有时称为**目标函数**。

### 什么是代价函数？

代价函数将模型的预测值和实际目标值作为输入，并输出一个数值。这个数值表示模型在训练数据上产生误差的*代价*或*惩罚*。

- **代价高：** 表示模型的预测值远离真实值。模型表现不佳。
- **代价低：** 表示预测值接近真实值。模型表现良好。

想象一下你在学习玩飞镖。你的代价函数可以是你的飞镖落在靶心外的平均距离。平均距离越低（代价越低）意味着你进步了。类似地，在机器学习 (machine learning)中，我们希望调整模型以实现尽可能低的代价。

### 示例：均方误差 (MSE)

最常见的代价函数之一，特别适用于回归问题（例如预测价格），是**均方误差 (MSE)**。

1. **计算误差：** 对于每个数据点 $i$，计算模型的预测值 ($y_{pred, i}$) 与实际值 ($y_{actual, i}$) 之间的差值。这个差值就是误差：$e_i = y_{pred, i} - y_{actual, i}$。
2. **误差平方：** 对每个单独的误差进行平方：$e_i^2 = (y_{pred, i} - y_{actual, i})^2$。平方运算有两个有用的作用：
   - 它使所有误差都为正值，这样低估和高估就不会相互抵消。
   - 它对较大误差的惩罚远重于较小误差（例如，误差为4时贡献16，而误差为2时仅贡献4）。
3. **计算平均值：** 对训练集中的所有 $N$ 个数据点的这些平方误差求平均。

MSE的公式是：
$\text{均方误差} = \frac{1}{N} \sum_{i=1}^{N} (y_{pred, i} - y_{actual, i})^2$

这里，$\sum$（西格玛）表示“求和”，而 $i=1$ 到 $N$ 告诉我们要对从第一个 ($i=1$) 到最后一个 ($i=N$) 的所有数据点求和。$\frac{1}{N}$ 用于求平均值。

我们使用MSE训练模型的目标是找到模型参数 (parameter)（例如线性模型 $y = mx + b$ 中的斜率 $m$ 和截距 $b$），使这个MSE值尽可能小。

### 代价的可视化

将代价函数视为定义一个曲面会很有帮助。这个曲面上的“位置”由为模型参数 (parameter)（例如 $m$ 和 $b$）选择的特定值决定。曲面上任何位置的“高度”表示这些参数对应的代价函数值。

对于一个试图拟合某些数据的简单线性模型 $y = mx + b$，MSE代价取决于 $m$ 和 $b$ 的选择值。如果我们简化并假设固定 $b$ 而只改变 $m$，我们可以绘制代价（MSE）与 $m$ 的不同值之间的关系图。这通常看起来像一个碗状。



![代价（均方误差）与模型参数（m）的关系](plots/2273-0.json)



> 代价函数（如均方误差）与单个模型参数绘制时的一个典型形状。最低点表示使误差最小化的参数值。

当有多个参数时（例如 $m$ *和* $b$），这会变得更高维度（像两个参数的3D碗状），但核心思想保持不变：我们正在这个代价空间中寻找最低点。该最低点的坐标为我们提供了模型的最佳参数值。

这让我们回到了优化。代价函数为我们提供了我们想要最小化的具体量。我们本章中学习的、使用导数的技术，正是我们用来有效找到那个最小点的工具。

## 参考资料

- [Machine Learning](https://www.coursera.org/learn/machine-learning) — Andrew Ng (2022)
  Publisher: DeepLearning.AI and Stanford Online
  广受好评的入门课程，清晰地解释了线性回归和梯度下降背景下的成本函数，尤其是均方误差。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本基础教材（第二版），对损失函数（包括均方误差）及其在模型拟合和优化中的应用进行了严谨的统计和数学处理。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本权威教材为机器学习中各种损失函数和优化技术提供了全面的理论基础，适合寻求更深层次数学理解的读者。

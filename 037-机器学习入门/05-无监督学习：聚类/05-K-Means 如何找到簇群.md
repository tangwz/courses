# K-Means 如何找到簇群

来源：[原文](https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-5-unsupervised-learning-clustering/how-k-means-finds-clusters)

[返回章节目录](README.md) · [返回课程目录](../README.md)

K-Means 算法是一种将无标签数据分组为 $K$ 个不同簇的方法。但它实际是如何做到这一点的呢？K-Means 并非凭空知道簇在哪里；它通过一个迭代过程找到它们，本质上是通过几个步骤逐步调整其估计，直到形成一个合理的分组。

把它想象成在一个社区中建立 $K$ 个服务中心，以最小化居民的平均出行距离。你可能首先随机放置这些中心，然后确定每个居民离哪个中心最近。接下来，你会将每个中心移动到分配给它的居民的平均位置。你会重复这个过程，很可能，这些中心点会自然地移向实际人口密集区域的中心。K-Means 的工作方式与此非常相似。

这个过程包括两个主要步骤，这些步骤会重复进行直到簇稳定下来：

1. **分配步骤：** 将每个数据点分配给离它最近的中心点。
2. **更新步骤：** 根据分配给它的点重新计算每个中心点的位置。

我们来更详细地讲解这些步骤。

### K-Means 迭代过程

1. **初始化：** 首先，我们需要为 $K$ 个中心点设置起始位置。一种常见的方法是简单地从数据集中随机选择 $K$ 个数据点，并将它们指定为初始中心点。或者，它们也可以随机放置在数据空间中。起始位置可能会影响最终的簇，但现在，我们只需设想我们有 $K$ 个初始中心点位置。
2. **分配步骤：** 现在，遍历数据集中的每个数据点。对于每个数据点，计算它到 $K$ 个中心点中 *每个* 的距离。这里最常见的距离测量方法是标准欧几里得距离（可以把它想象成图上两点之间的直线距离）。将该数据点分配给由 *最近* 中心点代表的簇。对所有数据点执行此操作后，你将形成 $K$ 个初始分组或簇。
3. **更新步骤：** 初始中心点只是猜测。我们可以改进它们。对于每个簇，计算当前分配给该簇的所有数据点的 *均值*（平均位置）。将该簇的中心点移动到这个新计算的均值位置。这个新位置实际上成为当前在该簇中的数据点的重心。
4. **重复：** 现在，随着中心点位置的更新，分配可能会改变。原本离中心点 A 最近的数据点，在中心点 B 移动后，现在可能离中心点 B 更近了。因此，我们重复这个过程：

   - 返回**分配步骤**：将 *所有* 数据点重新分配给它们最近的 *新* 中心点。
   - 执行**更新步骤**：根据新的分配重新计算 $K$ 个中心点的位置。
5. **收敛：** 不断重复分配和更新步骤。当分配在迭代之间不再明显变化，即中心点不再大量移动时，或者当达到预设的最大迭代次数时，算法就已收敛（完成）。此时，中心点代表了找到的簇的中心，分配则指示每个数据点属于哪个簇。

> K-Means 算法的迭代过程：初始化中心点，然后重复分配点和更新中心点直到收敛。

这种迭代式调整是 K-Means 如何“找到”数据中隐藏的分组的方式。通过不断根据当前成员调整簇中心，然后根据新的中心重新分配成员，算法会逐步将中心点吸引到数据点的密集区域，最终稳定下来，从而定义最终的簇。

## 参考资料

- [Some methods for classification and analysis of multivariate observations](https://www.stat.berkeley.edu/users/census/macq.pdf) — J. B. MacQueen (1967)
  Journal: Proceedings of the fifth Berkeley symposium on mathematical statistics and probability; Publisher: University of California Press; Volume: 1; Pages: 281-297
  这篇基础论文对K-Means算法进行了早期的正式描述。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2013)
  Publisher: Springer
  这本教科书提供了K-Means及相关统计学习方法的易懂入门介绍。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本全面的教科书，涵盖了理论基础和算法，包括对K-Means的详细讲解。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) — Christopher Bishop (2006)
  Publisher: Springer
  对K-Means进行了全面解释，常将其视为期望最大化(EM)算法的一个实例。

---

[上一节](04-%E9%80%89%E6%8B%A9%E8%81%9A%E7%B1%BB%E6%95%B0%E9%87%8F%20%28K%29.md) · [下一节](06-K-%E5%9D%87%E5%80%BC%E7%AE%97%E6%B3%95%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)

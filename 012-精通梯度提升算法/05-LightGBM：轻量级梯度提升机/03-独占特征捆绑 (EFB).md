# 独占特征捆绑 (EFB)

来源：[原文](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-5-lightgbm-light-gradient-boosting/lightgbm-efb)

[返回章节目录](README.md) · [返回课程目录](../README.md)

高维数据集，尤其是那些包含稀疏特征（如独热编码的类别变量）的数据集，对梯度提升算法构成了较大的难题。随着特征数量的增加，在每个节点分裂时为所有特征构建直方图会变得计算成本非常高。LightGBM 使用一种称为独占特征捆绑 (EFB) 的高效方法来缓解这一维度瓶颈。

EFB 背后的主要发现是，稀疏数据集通常包含许多*互斥*特征。这意味着这些特征对于同一数据实例很少或从不同时取非零值。一个常见的例子来源于独热编码：如果一个类别变量如“城市”被编码，那么在任何给定样本中，只有其中一个生成的二元特征（例如，`is_City_London`、`is_City_Paris`、`is_City_Tokyo`）可以为1；其余的必须为0。这些特征是互斥的。

EFB 运用这一特性，将这些互斥特征捆绑成一个单一、更密集的特征表示，从而减少了算法在直方图构建和分裂点查找过程中需要处理的有效特征数量。这种捆绑过程保留了分裂点查找的必要信息，同时大幅提升了训练速度。

### 识别特征捆绑

EFB 的第一步是确定哪些特征可以安全地捆绑在一起。LightGBM 将此建模为一个图问题：

1. **图构建：** 创建一个图，其中每个特征都是一个节点。
2. **边定义：** 如果两个特征节点*不*互斥，则在它们之间添加一条边。也就是说，如果这两个特征在至少一定数量的数据实例中同时取非零值，则存在一条边。通常允许少量冲突（非零共现），这由参数 (parameter)控制，因为完美的互斥性可能过于苛刻，从而限制了捆绑机会。从不同时取非零值的特征之间将没有边。
3. **图着色：** 对此冲突图应用贪婪图着色算法。目标是为每个节点（特征）分配一个“颜色”（代表一个捆绑ID），使得任何通过边连接的两个节点都不能共享相同的颜色。目的是最小化使用的不同颜色数量，因为每种颜色对应一个最终的捆绑特征。被分配相同颜色的特征被认为足够互斥，可以捆绑在一起。

尽管最优图着色是NP-难问题，但一个简单的贪婪方法（遍历特征并分配未被冲突邻居使用的第一个可用颜色）在实践中效果良好且计算高效。

> 五个稀疏特征的冲突图。边连接相互冲突（同时取非零值）的特征。贪婪着色分配颜色（捆绑），使得连接的节点具有不同的颜色。在这里，(F1, F2) 构成捆绑 1（蓝色），(F3, F5) 构成捆绑 2（绿色），F4 构成捆绑 3（橙色）。

### 构建捆绑特征

一旦捆绑被识别（具有相同颜色的特征），LightGBM 会为每个捆绑创建一个新的单一特征。为了确保捆绑内的原始特征保持可区分，它们的取值范围在新捆绑特征的直方图箱中被仔细偏移。

考虑两个互斥特征，特征 A（取值范围 [0, $k_A$]）和特征 B（取值范围 [0, $k_B$]），它们被分配到同一个捆绑。一个新的捆绑特征，特征捆绑 AB，被创建。特征 A 的非零值被直接映射到特征捆绑 AB 的直方图中的箱 [1, $k_A$]。特征 B 的非零值被映射到箱 [$k_A+1$, $k_A+k_B$]。两个原始特征的零值都映射到捆绑特征直方图中的箱 0。

这种偏移机制确保在捆绑特征上找到最佳分裂点等同于在原始组成特征中找到最佳分裂点。例如，在箱 $j$ 处找到一个分裂点，其中 $1 \le j \le k_A$ 对应于特征 A 上的分裂，而在箱 $j$ 处找到一个分裂点，其中 $k_A+1 \le j \le k_A+k_B$ 对应于特征 B 上的分裂。

### 优点与权衡

独占特征捆绑提供了显著优点：

- **降低特征维度：** 用于直方图构建和分裂点查找的特征数量减少，对于非常稀疏的高维数据通常会大幅减少。
- **提升训练速度：** 需要构建和扫描的直方图更少，从而实现更快的训练迭代。
- **内存效率：** 存储特征直方图所需的内存更少。

主要权衡在于，如果捆绑的特征不是*完全*互斥（即，如果冲突容忍度允许捆绑偶尔与非零值共现的特征），则可能引入少量信息损失。然而，通过控制允许的冲突率（例如，通过 LightGBM 中的 `max_conflict_rate` 参数 (parameter)，尽管这通常是自动处理的），可以有效管理这种损失。在实践中，EFB 带来的计算增益通常远远超过其对适用数据集的准确性造成的任何轻微影响。

EFB 与 PCA 等传统降维技术相比，是一种独特方法。PCA 创建原始特征的密集线性组合，而 EFB 则专门针对稀疏的独占特征，并以一种保留其在捆绑表示中原始分裂潜力的方式将它们捆绑起来，纯粹专注于树构建的计算效率。这是一种针对现代机器学习 (machine learning)数据集中常见结构而设计的巧妙优化。

## 参考资料

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS 2017); Publisher: Curran Associates, Inc.; Pages: 3146-3154; DOI: [10.48550/arXiv.1706.09516](https://doi.org/10.48550/arXiv.1706.09516)
  介绍LightGBM并详细阐述独占特征捆绑算法的基础论文。
- [Exclusive Feature Bundling (EFB)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGhPw8HztmnyPvg6tU2czGGhJHf9lv7xiKIfhOrb3MzPK_zFXwLGwhGWwGzScNZ4u7jayFOZgi2Zoqfl-RxNh8wjPs-k_hxmV23qY_B9y6COWsElGhWKwjODlmES-ZoIICK99h8-4HhADVoN56t2VqnFnL1jxqyFcB_p4KZ-3BM0F01lEIPLxRhfd39hRE2DWO7r7tYFbxbn4qggQzT_IS7O7ZMUlErs66LVuT6m8dQOL3_) — LightGBM Contributors (2024)
  LightGBM官方文档，说明EFB算法、参数及其在LightGBM库中的实际应用。
- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: The MIT Press
  一本经典算法教材，涵盖图着色和贪心算法，这些是EFB捆绑识别过程的算法基础。

---

[上一节](02-%E5%9F%BA%E4%BA%8E%E6%A2%AF%E5%BA%A6%E7%9A%84%E5%8D%95%E4%BE%A7%E9%87%87%E6%A0%B7%20%28GOSS%29.md) · [下一节](04-%E5%9F%BA%E4%BA%8E%E7%9B%B4%E6%96%B9%E5%9B%BE%E7%9A%84%E5%88%86%E8%A3%82%E7%82%B9%E6%9F%A5%E6%89%BE.md)

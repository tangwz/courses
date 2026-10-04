---
course: "introduction-autoencoders-feature-learning"
chapter: "autoencoders-learning-features"
lesson: "visualizing-learned-representations"
sourceId: 6426
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-4-autoencoders-learning-features/visualizing-learned-representations"
title: "学习到的表示的简单可视化"
description: "了解可视化自编码器瓶颈层学习到的特征或表示的简单方法。"
order: 6
plots: ["plots/6426-0.json"]
sourceHash: "7db58f690281bdfbfb43460eb819717441d7dd8a02781f235514d16004931807"
sourceCorrections: []
---

了解自编码器的瓶颈层包含输入数据的压缩、学习到的表示后，一个自然的问题出现：这些学得的特征到底是什么样子的？我们能否了解自编码器的“思维”，看看它认为什么很重要？可视化这些学习到的表示能提供有益的看法，了解自编码器如何处理信息以及它是否捕获了有意义的模式。

### 为什么要可视化学习到的特征？

可视化自编码器学习到的特征有几个用途：

1. **获得直觉**：它可以帮助我们建立直觉，理解自编码器认为数据哪些方面是重要的。如果我们给它输入猫和狗的图像，潜在空间（瓶颈表示所在的那个空间）是否显示出与这些类别相关的明确区分或组织？
2. **调试和模型评估**：如果可视化显示潜在空间中点随机散布，没有可辨识的结构，这可能说明自编码器没有有效学习，或者瓶颈层过大或过小。
3. **理解降维**：当自编码器将一个784维图像（例如28x28像素的MNIST数字）压缩到瓶颈层的2或3个维度时，可视化这些2D或3D点可以直接显示数据是如何被压缩和组织的。
4. **数据结构分析**：有时，这些可视化能够展现数据中以前不明显的底层结构，比如对应不同输入类别的独立聚类，即便自编码器是在无标签（无监督）条件下训练的。

### 瓶颈层激活（潜在空间）的可视化

可视化学习到的特征最直接的方式之一是查看瓶颈层本身的激活。如果你的瓶颈层设计为非常低的维度，比如2或3维，你可以直接绘制这些激活。

手写数字的MNIST数据集，每张图片为28x28像素，构成784维数据。当训练一个带有2维瓶颈层的自编码器时，即可：

1. 将每个MNIST图像通过自编码器中已训练的编码器部分。
2. 收集每个图像在瓶颈层产生的2D输出。
3. 创建一个散点图，其中每个点对应一张图像，其x和y坐标是瓶颈层的两个激活值。

如果自编码器学习得很好，你可能会看到相同数字的图像（例如，所有“0”或所有“1”）在这个2D潜在空间中倾向于聚在一起。这表明自编码器已学会用相似的紧凑编码表示相似的数字。

设想一个场景。假设我们有三种不同类型的数据项。将它们通过一个将其压缩成2维表示的编码器后，我们可能会得到一个像这样的散点图：



![2D潜在空间可视化](plots/6426-0.json)



> 一个散点图，显示了来自三个不同类别的数据点被映射到2维潜在空间。类别A和B形成不同的聚类，而类别C可能更分散或混合，这表明了自编码器如何区分或归组这些数据类型。

如果你的瓶颈层有3个维度，你可以创建一个3D散点图。对于维度超过3的瓶颈层，直接绘图不可行。在这种情况下，你可以采用t-SNE（t分布随机邻居嵌入 (embedding)）或PCA（主成分分析）等进一步的降维技术专门用于可视化，将高维潜在空间投影到2D或3D。然而，这些是独立的技巧，为了基本理解，关注本身低维的瓶颈层是一个好的起点。

### 潜在空间的重构可视化

另一种有用的方法是从潜在空间中取点，并*仅*通过自编码器的解码器部分。这显示了自编码器将其学习到的特征空间中的不同区域或特定点与哪种类似原始数据关联起来。

例如，对于MNIST数字的2D潜在空间：

1. 在你的潜在空间占据的2D平面中选择一个点$(x, y)$。这个点不必是实际输入图像生成的点；你可以选择任意坐标。
2. 将这个$(x, y)$对作为你训练好的解码器网络的输入。
3. 解码器的输出将是一张图像（例如，28x28像素）。这就是自编码器从其学习到的特征空间中的那个特定点“想象”或重构出来的。

通过在潜在空间中以网格状采样点并为每个点生成重构，你可以创建流形可视化。你可能会观察到平滑的过渡：当你从编码“1”的区域移动到编码“7”的区域时，重构的图像可能会逐渐从看起来像“1”的变成像“7”的。

这个技术非常有用，因为它突出了自编码器如何学会将潜在空间中的连续变化映射到数据空间中的连续变化。它让你了解自编码器为其压缩表示中不同坐标赋予的“含义”。

例如，如果你用面部图像训练了一个自编码器，然后在潜在空间中沿着一条线采样点，解码的图像可能会显示一张脸平滑地改变某个属性，比如从微笑过渡到中性表情，或者从看向左边到看向右边。这表明自编码器已将这些变化捕获为其学习到的特征之一。

### 可视化结果的解释

当你查看这些可视化时，有几点需要考虑：

- **聚类**：对应相似输入（例如，相同数字、相同对象类别）的点是否聚在一起？明确的聚类是自编码器正在学习区分不同类型数据的好迹象。
- **分离**：这些聚类分离得有多好？如果不同类别的聚类显著重叠，那么这种表示对那些类别来说可能区分度不高。
- **连续性（对于重构）**：潜在空间中相邻的点解码出的结果在视觉上是否相似？沿着潜在空间的一个方向移动是否对应着重构输出中有意义的变化？这表明潜在空间结构良好。
- **空白区域**：潜在空间中是否存在“空白”区域？如果你尝试从这些区域进行解码，会发生什么？有时，解码器会从远离实际数据映射的区域生成嘈杂或无意义的输出。

记住，自编码器学习到的特征通常是抽象的。虽然2D潜在空间中的一个轴可能大致对应数字的“粗细”，另一个轴对应“倾斜”，但情况很少如此简单直接。自编码器学习能最好帮助它重构数据的特征，这些特征可能不总是与人类可理解的属性完全吻合。然而，这些可视化为我们观察这些学习到的内部表示提供了一个有价值的窗口，有助于我们理解和信任模型构建的表示。在我们开始构建第一个自编码器时，我们将了解到如何实际生成其中一些可视化。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍了自编码器及其架构和学习表示的属性。它涉及本节中关于潜在空间和特征学习的核心概念。
- [Reducing the Dimensionality of Data with Neural Networks](https://www.cs.toronto.edu/~hinton/absps/science.pdf) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  介绍了用于降维的深度自编码器概念，这是本节中讨论的生成低维潜在空间的核心机制。
- [Visualizing Data using t-SNE](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) — Laurens van der Maaten, Geoffrey Hinton (2008)
  Journal: Journal of Machine Learning Research; Volume: 9; Pages: 2579-2605
  介绍了 t-SNE，一种可视化高维数据的技术。该方法直接适用于将高维自编码器潜在空间投影到2D或3D进行检查。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGWF23WDgk611TQOYurNVGn6_DtsFcR7DJomhi-_8ZTwOWe1_duj0gLfdFUNXLduIbbWhUDkmFxTLtEhjM4TIehCxWJ3MyqjFTmMZLJ1eXR4WTMt_cATGAEjzw7qMizffqVFMMak3jPdof6DPTCx1WFnCsUYjPrQYFPIRN6yn8XulAFH0aOb5qi1NlHa-SY3hw3L51ZXa1IywDx4rqNPmOQgUHOjA==) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的指南，包含自编码器的实现示例，以及潜在空间投影和重建的演示。

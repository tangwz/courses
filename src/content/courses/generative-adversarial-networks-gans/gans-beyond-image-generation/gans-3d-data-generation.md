---
course: "generative-adversarial-networks-gans"
chapter: "gans-beyond-image-generation"
lesson: "gans-3d-data-generation"
sourceId: 2762
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-6-gans-beyond-image-generation/gans-3d-data-generation"
title: "三维数据生成 (点云, 网格)"
description: "使GANs适应于生成表示为点云或网格的三维形状。"
order: 6
plots: []
sourceHash: "10dbd843e8560f8efe0c3f3e6fd79358e3ecff0e81e575f217da95e31c2f338a"
sourceCorrections: []
---

生成标准图像二维网格结构之外的数据，给GANs带来了独特的难题。将对抗训练用于三维形状时，需要认真考量数据表示、网络结构以及合适的损失函数 (loss function)。三维形状的两种常见表示是点云和多边形网格，每种都需采用不同的方法。

### 生成点云

点云可能是最简单的三维表示：一个无序的点集 $S = \{p_i \in \mathbb{R}^3\}_{i=1}^N$，这个集中的 $N$ 是点的数量。每个点 $p_i$ 指定 $(x, y, z)$ 坐标，并可能带有颜色或表面法线等其他属性。主要难题源于该点集的*无序性*。为规则网格设计的标准卷积层不能直接适用，因为它们假设了固定的空间邻域结构和顺序。

**结构考量：**
为处理点云，GAN生成器和判别器常借鉴PointNet和PointNet++等结构。这些网络实现了*排列不变性*，即输出不受输入集中点序的影响。这通常是通过对每个点独立应用共享变换（如MLPs）后，使用对称函数（如最大池化）来完成的。

点云生成器可能接收一个潜在向量 (vector) $z$，并常通过全连接层将其变换为 $N \times 3$ 坐标集。确保生成的点能准确代表三维表面，需要精心的网络设计。判别器反之，接收点云（真实的或生成的）作为输入，并输出一个表示真实性的单一标量。它需要从无序点集中学习表示真实三维结构的特征，并再次运用排列不变层。

**点集损失函数 (loss function)：**
标准的像素级损失（如L1或L2）不适合用于比较点云，因为两个不同点集之间缺乏对应关系。因此，会使用专用损失函数，这些函数常整合到GAN的目标中，或在某些情况下直接用于生成器训练。两个重要例子是倒角距离（Chamfer Distance, CD）和地球移动距离（Earth Mover's Distance, EMD），点集EMD也称为Wasserstein-1距离。

两个点集 $S_1$ 和 $S_2$ 之间的倒角距离定义为：


$$
d_{CD}(S_1, S_2) = \sum_{x \in S_1} \min_{y \in S_2} \|x-y\|_2^2 + \sum_{y \in S_2} \min_{x \in S_1} \|x-y\|_2^2
$$


它衡量的是一个集合中每个点到另一个集合中最近邻点的平均平方距离。它的计算效率相对较高，但有时可能倾向于生成松散覆盖目标形状的点，而不是精确匹配其密度。

地球移动距离在两个大小相等点集之间寻找最佳匹配（双射 $\phi$），并汇总匹配点之间的距离：


$$
d_{EMD}(S_1, S_2) = \min_{\phi: S_1 \to S_2} \sum_{x \in S_1} \|x - \phi(x)\|_2
$$


EMD常被认为是衡量点云之间差异的更好指标，因为它反映了将一个分布转换为另一个分布的“成本”。然而，它的计算成本高于CD，特别是对于大型点云，并且通常要求点集具有相同的基数。

这些距离可在GAN框架内使用，常用于引导生成器或评估真实与生成点分布之间的相似性。例如，判别器可以使用Wasserstein目标进行训练，隐式地最小化真实与生成分布之间EMD的近似值。

### 生成网格

多边形网格使用顶点（三维空间中的点）和面（通常是三角形或四边形）来表示表面，这些面定义了顶点之间的连接。这种显式连接信息捕获了点云所没有的表面拓扑。然而，这种额外结构给生成模型带来了新的复杂性。

**难题：**

1. **可变拓扑：** 不同对象具有不同数量的顶点和面，以及复杂的连接模式。直接生成这种可变结构很困难。
2. **流形性：** 真实的网格通常应是流形（局部类似于二维平面）。确保生成的网格满足此类几何约束并非易事。
3. **可微性：** 网格连接的离散性质对通过生成器的基于梯度的优化构成了难题。

**网格生成方法：**

1. **基于体素的生成：** 一种间接方法是首先生成三维体素网格表示。体素网格是一个三维数组，其中每个单元表示占据（在物体内部或外部）。使用三维卷积神经网络 (neural network) (CNN)（3D CNNs）的GANs可以生成这些网格。然后可以使用行进立方体（Marching Cubes）等算法从体素网格中提取网格。尽管更简单，但体素表示存在内存消耗高和离散化伪影的问题，限制了可实现的精度。
2. **基于变形的生成：** 这些方法从一个具有固定拓扑的模板网格（例如球体）开始。生成器网络学习预测顶点位移，将模板变形为目标形状。这简化了拓扑处理，但将生成器限制在拓扑上等同于模板的形状，从而限制了表达能力。
3. **基于图的生成：** 网格可以视为图，其中顶点是节点，边定义连接。图神经网络（GNNs）非常适合处理这种不规则结构。GNNs可以整合到生成器和判别器中。生成器可能输出顶点位置，并可能预测边连接或面信息。设计能有效生成几何形状和合理连接的GNNs仍是一个活跃的研究方向。

> 点云将形状表示为独立点的集合，而网格通过边连接的顶点定义表面，形成面（由边连接隐含）。

4. **隐式表示：** 一种强有力的方法是生成一个*隐式函数*，而非显式几何形状。生成器输出函数 $f(x, y, z)$ 的参数 (parameter)，使得形状表面由等值面定义，通常是零水平集 $f(x, y, z) = 0$。$f$ 的常见选择包括符号距离函数（SDFs）或占据函数（预测一个点是在形状内部还是外部）。
   - **结构：** 生成器通常是一个多层感知器（MLP），它将潜在代码 $z$ 和三维查询点 $(x, y, z)$ 作为输入，输出函数值 $f(x, y, z)$。
   - **优点：** 隐式表示可以建模任意拓扑，并获得高有效精度，因为表面是连续的。生成函数通常比直接生成复杂网格结构更容易。
   - **提取：** 在推断时，可以使用行进立方体等算法从学习到的隐式函数中提取网格。
   - **判别器：** 判别器也可以在隐式场上操作，或者通过在预测的零水平面上采样点并评估其分布（通过SDFs的球体追踪或求根，或占据网络中预测决策边界附近的采样获得）来工作。

### 三维数据判别器设计

判别器的作用保持不变：区分真实三维形状与生成的形状。其结构必须与所选表示匹配：

- **点云：** 常用基于PointNet的结构。
- **体素：** 通常使用3D CNNs。
- **网格 (基于图)：** GNNs处理网格结构。
- **网格 (隐式)：** 判别器可能评估隐式函数本身，或者更常见的是，它从生成的表面（通过SDFs的球体追踪或求根，或占据网络中预测决策边界附近的采样获得）采样点，并使用类似PointNet的结构处理这些采样点。

### 评估指标

评估生成的三维形状与图像评估有共同难题，即评估保真度（质量）和多样性。常见的定量指标包括：

- **倒角距离 (CD) / 地球移动距离 (EMD)：** 在大量生成的形状与真实形状（表示为点云）之间计算。值越低表示相似性越好。
- **交并比 (IoU) / 体素精度：** 主要用于基于体素的表示。
- **光场距离 (LFD)：** 基于从多个视角渲染的二维投影来比较形状，将三维评估与基于图像的指标联系起来。
- **分布指标 (MMD, 覆盖率)：** 最小匹配距离 (MMD) 使用预训练 (pre-training)的三维形状分类器提取的特征，衡量生成形状与真实形状分布之间的距离（类似于图像的FID）。覆盖率评估生成器捕获的真实数据分布的比例。

定性目视检查对于判断生成的3D模型的精细细节、表面质量和合理性仍然不可或缺。

总之，将GANs应用于三维数据生成需要比标准CNNs更进一步。通过点云排列不变性技术、针对网格的专用图网络或隐式函数表示，结合合适的损失函数 (loss function)（如CD或EMD）和评估指标，使得对抗训练能够合成复杂的三维结构。

## 参考资料

- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://proceedings.neurips.cc/paper_files/paper/2017/file/99c5e07b4d5de9d18c350dda14ef6366-Paper.pdf) — Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 30
  介绍了一种用于处理点云的层次化神经网络架构，解决了置换不变性并学习多尺度特征，这对于3D感知和生成具有基础性作用。
- [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf) — Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, Steven Lovegrove (2019)
  Journal: IEEE Conference on Computer Vision and Pattern Recognition; Pages: 165-174
  介绍了一种利用深度神经网络学习连续符号距离函数（SDFs）来表示3D形状的方法，能够实现高分辨率和拓扑复杂的几何体。
- [Learning Deformable Mesh Generation using Graph Convolutional Networks](https://openaccess.thecvf.com/content/CVPR2020/html/Wang_Learning_Deformable_Mesh_Generation_Using_Graph_Convolutional_Networks_CVPR_2020_paper.html) — Shaonan Wang, Wenzheng Chen, Xiaojuan Qi, Kwok-Ho Mak, Ka-Wing To, Wen-Ping Wang (2020)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE / CVF; Pages: 14217-14226; DOI: [10.1109/CVPR42600.2020.01423](https://doi.org/10.1109/CVPR42600.2020.01423)
  提出了一种通过使用图卷积网络（GCNs）从模板预测顶点形变来生成3D可变形网格的方法，解决了网格拓扑的挑战。

---
course: "cnns-for-computer-vision"
chapter: "object-detection-algorithms"
lesson: "non-maximum-suppression-variants"
sourceId: 2563
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-3-object-detection-algorithms/non-maximum-suppression-variants"
title: "非极大值抑制的变体"
description: "了解标准非极大值抑制 (NMS) 及其在目标检测后处理中的多种形式。"
order: 6
plots: []
sourceHash: "45336bf88d0a9d2698e06ad17bdbf8c0f4a88f3e79b83108f7ae09364c88550e"
sourceCorrections: []
---

目标检测器提出潜在的边界框后，通常会在同一物体周围生成多个高度重叠的框，每个框都带有不同的置信度分数。非极大值抑制 (NMS) 的作用是清除这些多余的检测结果，只保留置信度最高、互不干扰的框。虽然标准贪婪NMS算法得到普遍使用，但它可能效果不理想，特别是在物体密集排列的场景中。将对旨在解决这些不足的NMS变体进行分析。

### 标准NMS算法及其不足

回顾标准贪婪NMS的步骤：

1. 按置信度分数降序排列所有检测框。
2. 选择置信度分数最高的框 $B_{max}$ 并将其添加到最终检测结果列表。
3. 从框列表中移除 $B_{max}$。
4. 对于所有剩余的框 $B_i$，计算它们与 $B_{max}$ 的交并比 (IoU)。
5. 移除所有 $IoU(B_{max}, B_i)$ 大于预设阈值 $N_{threshold}$ 的框 $B_i$。
6. 重复步骤 2-5，直到框列表为空。

这种方法简单且计算效率高。然而，其主要不足在于步骤5。如果一个正确的检测框 $B_j$ 与 $B_{max}$ 存在明显重叠（例如，两个不同的物体彼此非常接近），即使它代表一个不同的物体实例，如果其IoU超过 $N_{threshold}$，它也可能被完全移除。这可能导致召回率下降，特别是在拥挤的图像中。

> 标准NMS与Soft-NMS在处理两个重叠框（A和B）时的比较，其中A的初始分数更高。标准NMS会移除框B，而Soft-NMS则保留框B但降低其置信度分数。

### Soft-NMS

Soft-NMS 并非完全消除重叠框，而是提出根据与得分更高的已选框的重叠程度来降低其置信度分数。这样做的理由是，如果一个框与一个置信度很高的检测结果存在明显重叠，其本身是正确检测的可能性较小，但它不应完全丢弃，特别是当它可能代表一个独立、邻近的物体时。

框 $B_i$ 的分数 $s_i$ 根据其与已选择的最高分框 $B_{max}$ 的IoU进行更新：


$$
s_i = s_i \times f(IoU(B_{max}, B_i))
$$


函数 $f(\cdot)$ 是一个惩罚函数，它随着IoU的增加而减小。两种常见形式是：

1. **线性惩罚：**
   
   $$
   f(iou) = \begin{cases} 1 & \text{如果 } iou < N_{threshold} \\ 1 - iou & \text{如果 } iou \ge N_{threshold} \end{cases}
   $$
   
2. **高斯惩罚：**
   
   $$
   f(iou) = e^{-\frac{iou^2}{\sigma}}
   $$
   

这里，$N_{threshold}$ 是标准NMS阈值（用于决定何时在线性情况下开始应用惩罚），而 $\sigma$ 是一个控制高斯衰减陡度的参数 (parameter)。

通过衰减分数而非完全消除框，Soft-NMS通常会提高平均精度 (AP)，特别适用于物体遮挡或密度大的数据集。代价是与标准NMS相比计算成本略有增加，并且引入了一个可能需要调整的新参数（高斯变体的 $\sigma$）。

### DIoU-NMS (距离-IoU NMS)

标准NMS和Soft-NMS仅依赖IoU指标。然而，IoU不考虑边界框中心之间的距离。考虑两种情况：两个IoU高的框，因为它们紧密包围同一物体；或两个IoU相同但代表两个独立、相邻物体，其框恰好明显重叠。标准NMS对这两种情况一视同仁。

DIoU-NMS将两个框中心点之间的归一化 (normalization)距离纳入抑制判据。其主要依据是，当根据 $B_{max}$ 抑制框 $B_i$ 时，不仅要考虑IoU，还要考虑它们中心之间的距离。如果中心相距很远，即使它们的IoU很高，$B_i$ 也不太可能是与 $B_{max}$ 相同的物体所产生的多余检测。

DIoU指标本身对中心之间的距离进行惩罚。在DIoU-NMS中，抑制条件被修改。DIoU-NMS可能使用包含DIoU值的判据，而不是仅仅检查 $IoU(B_{max}, B_i) > N_{threshold}$，其定义为：


$$
DIoU = IoU - \frac{\rho^2(b, b_{gt})}{c^2}
$$


其中 $\rho(\cdot)$ 是欧几里得距离，$b$ 和 $b_{gt}$ 是两个框的中心点，$c$ 是覆盖两个框的最小外接框的对角线长度。当在NMS中使用时，惩罚项 $\frac{\rho^2(b, b_{gt})}{c^2}$ 在抑制期间被添加到IoU计算中。具有高IoU但中心相距较远的框受到的惩罚较少，被抑制的可能性也较小。

这使得DIoU-NMS在保留对独立但邻近物体的正确检测方面特别有效，与标准NMS相比，在拥挤场景中带来更好的性能。

### 其他变体

- **矩阵NMS：** 用于一些实时实例分割模型，如YOLACT，矩阵NMS并行化抑制过程。它不是顺序迭代，而是并行计算成对IoU并根据重叠情况衰减分数，使其在GPU上非常快速。它的公式通常是根据其所用到的特定架构而定制的。
- **自适应NMS：** 用于点云目标检测方法，如Point RCNN，自适应NMS根据物体密度或检测分数调整抑制阈值，在稀疏区域变得更严格，在密集区域更宽松。

### NMS的选择与调整

标准NMS与其变体之间的选择取决于具体的应用和数据集特性：

- **标准NMS：** 最快，通常足以应对物体分离良好的简单场景。除非出现特定问题（如拥挤区域的漏检），否则它是默认选择。
- **Soft-NMS：** 在最大化召回率很重要且物体通常紧密排列或被遮挡时，是一个不错的选择。对标准NMS的实现只需极少改动。
- **DIoU-NMS：** 当标准NMS因边界框高度重叠而错误抑制独立邻近物体时，推荐使用。它提供了比单独IoU更精确的判据。

无论哪种变体，IoU阈值 ($N_{threshold}$) 仍是一个重要的超参数 (parameter) (hyperparameter)。较低的阈值会导致更激进的抑制（框更少、更独立），而较高的阈值则更宽松（可能导致更多重叠的框和更多误报）。同样，在NMS*之前*用于过滤框的初始置信度分数阈值对NMS算法的输入影响很大。这些阈值的最佳值通常通过在验证数据集上评估性能来凭经验确定。

总而言之，虽然贪婪NMS为完善目标检测提供了基本方法，但Soft-NMS和DIoU-NMS等变体提供了更精细的策略来处理重叠物体的复杂性，通常会带来检测准确性的显著提升，特别是在具有挑战性的密集场景中。了解这些替代方案可以更好地根据您的目标检测任务的特定需求调整后处理阶段。

## 参考资料

- [Soft-NMS -- Improving Object Detection With One Line of Code](https://arxiv.org/abs/1704.04503) — Navaneeth Bodla, Bharat Singh, Rama Chellappa, Larry S. Davis (2017)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Pages: 2788-2796; DOI: [10.48550/arXiv.1704.04503](https://doi.org/10.48550/arXiv.1704.04503)
  介绍了Soft-NMS算法，该算法通过衰减重叠框的得分而非直接移除，提高了密集场景中的召回率。
- [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287) — Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren (2020)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence (AAAI); Volume: 34; Pages: 11449-11456; DOI: [10.48550/arXiv.1911.08287](https://doi.org/10.48550/arXiv.1911.08287)
  提出了DIoU和CIoU损失，并引入了DIoU-NMS，它在抑制标准中考虑了中心点距离。
- [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689) — Daniel Bolya, Chong Zhou, Fanyi Xiao, Yong Jae Lee (2019)
  Journal: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV); Pages: 9015-9024; DOI: [10.1109/ICCV.2019.00913](https://doi.org/10.1109/ICCV.2019.00913)
  介绍了Matrix NMS，这是一种在YOLACT实例分割模型中用于加速后处理的并行化NMS方法。

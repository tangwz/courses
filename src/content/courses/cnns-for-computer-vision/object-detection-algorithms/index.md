---
course: "cnns-for-computer-vision"
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-3-object-detection-algorithms"
sourceId: 609
chapter: "object-detection-algorithms"
title: "目标检测算法"
order: 3
description: "学习高级目标检测算法，包括两阶段（Faster R-CNN）和单阶段（YOLO、SSD）检测器。"
hasQuiz: false
---

在图像分类的基础上，本节讨论如何在图像中定位 *and* 识别多个对象。这就是目标检测的目的。本节中，您将学习有影响力的深度学习目标检测模型的设计与工作方式。

我们将分析两种主要方法的架构差异：以R-CNN系列（包含带有区域提议网络的Faster R-CNN）为代表的两阶段检测器，以及YOLO和SSD等计算更快的单阶段检测器。您将实际理解重要组件，例如用于提议潜在目标位置的锚框，以及用于过滤重叠检测结果的非极大值抑制 ($NMS$) 等技术。我们还将介绍标准评估协议，重点关注平均精度均值 ($mAP$) 等指标，并以实践实现练习作结。

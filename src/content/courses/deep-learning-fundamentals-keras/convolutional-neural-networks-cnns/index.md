---
course: "deep-learning-fundamentals-keras"
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns"
sourceId: 932
chapter: "convolutional-neural-networks-cnns"
title: "卷积神经网络 (CNN)"
order: 4
description: "学习CNN的基本知识以及如何使用Keras构建它们来完成图像处理任务。"
hasQuiz: true
---

尽管标准的完全连接网络独立处理输入特征，但它们在处理图像等数据中存在的空间层级结构时表现不佳。本章介绍**卷积神经网络 (CNN)**，这是一种专门的架构，旨在高效地从网格状数据中学习，通过保留和分析空间关系。

您将学习CNN的基本构成部分以及如何使用Keras实现它们：

*   **卷积层 ($Conv2D$)**：了解滤波器（核）如何在输入数据上滑动以检测边缘、纹理和形状等模式，并生成特征图。
*   **池化层 ($MaxPooling2D$)**：掌握池化如何降低特征图的空间维度，使模型对物体位置的变化更具适应性，并减少计算负担。
*   **构建CNN架构**：学习如何堆叠卷积层和池化层，然后接上`Flatten`和`Dense`层，以构建能够完成图像分类等任务的模型。
*   **处理图像数据**：学习使用Keras工具来加载、预处理和增强图像数据集。
*   **实际操作**：通过在一个标准图像数据集上构建和训练一个CNN模型来应用这些知识。

在本章结束时，您将能够为图像相关任务构建、训练和理解CNN的基本运作方式。

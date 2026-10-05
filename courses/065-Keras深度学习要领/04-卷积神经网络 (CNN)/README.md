# 第 4 章：卷积神经网络 (CNN)

来源：[原章节](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns)

[返回课程目录](../README.md)

尽管标准的完全连接网络独立处理输入特征，但它们在处理图像等数据中存在的空间层级结构时表现不佳。本章介绍**卷积神经网络 (CNN)**，这是一种专门的架构，旨在高效地从网格状数据中学习，通过保留和分析空间关系。

您将学习CNN的基本构成部分以及如何使用Keras实现它们：

*   **卷积层 ($Conv2D$)**：了解滤波器（核）如何在输入数据上滑动以检测边缘、纹理和形状等模式，并生成特征图。
*   **池化层 ($MaxPooling2D$)**：掌握池化如何降低特征图的空间维度，使模型对物体位置的变化更具适应性，并减少计算负担。
*   **构建CNN架构**：学习如何堆叠卷积层和池化层，然后接上`Flatten`和`Dense`层，以构建能够完成图像分类等任务的模型。
*   **处理图像数据**：学习使用Keras工具来加载、预处理和增强图像数据集。
*   **实际操作**：通过在一个标准图像数据集上构建和训练一个CNN模型来应用这些知识。

在本章结束时，您将能够为图像相关任务构建、训练和理解CNN的基本运作方式。

## 小节

- 1. [卷积网络简介](01-%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C%E7%AE%80%E4%BB%8B.md)
- 2. [卷积层 (Conv2D)](02-%E5%8D%B7%E7%A7%AF%E5%B1%82%20%28Conv2D%29.md)
- 3. [池化层 (MaxPooling2D)](03-%E6%B1%A0%E5%8C%96%E5%B1%82%20%28MaxPooling2D%29.md)
- 4. [构建一个简单的CNN架构](04-%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84CNN%E6%9E%B6%E6%9E%84.md)
- 5. [CNN中的展平层和全连接层](05-CNN%E4%B8%AD%E7%9A%84%E5%B1%95%E5%B9%B3%E5%B1%82%E5%92%8C%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82.md)
- 6. [在Keras中处理图像数据](06-%E5%9C%A8Keras%E4%B8%AD%E5%A4%84%E7%90%86%E5%9B%BE%E5%83%8F%E6%95%B0%E6%8D%AE.md)
- 7. [了解特征图](07-%E4%BA%86%E8%A7%A3%E7%89%B9%E5%BE%81%E5%9B%BE.md)
- 8. [实践：构建图像分类CNN](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BBCNN.md)

章节测验：[在线测验](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns/quiz)

# 第 3 章：构建你的第一个用于特征提取的自动编码器

来源：[原章节](https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-3-building-first-autoencoder-feature-extraction)

[返回课程目录](../README.md)

自动编码器的核心要点已介绍完毕，本章着重介绍构建第一个用于特征提取模型所需采取的实际步骤。我们将逐步完成整个工作流程，从初始数据准备到提取和理解学习到的特征。

本章将指导您完成以下内容：
*   准备和预处理数据，包括归一化和缩放，以便有效训练自动编码器。
*   设计编码器和解码器网络，考虑诸如层数、每层神经元数量以及激活函数的选择。
*   为潜在空间（瓶颈层）选择合适的维度。
*   选择合适的损失函数，例如均方误差（$MSE$）或二元交叉熵（$BCE$），并配置优化器（例如Adam, SGD），设定恰当的学习率。
*   监控训练阶段，然后使用训练好的模型从瓶颈层提取特征表示。
*   潜在空间的可视化方法，特别是针对低维表示。
一个动手实践环节将演示构建一个自动编码器，用于从表格数据集提取特征。

## 小节

- 1. [自编码器训练的数据准备](01-%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E8%AE%AD%E7%BB%83%E7%9A%84%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87.md)
- 2. [编码器网络设计考量](02-%E7%BC%96%E7%A0%81%E5%99%A8%E7%BD%91%E7%BB%9C%E8%AE%BE%E8%AE%A1%E8%80%83%E9%87%8F.md)
- 3. [确定潜在空间维度](03-%E7%A1%AE%E5%AE%9A%E6%BD%9C%E5%9C%A8%E7%A9%BA%E9%97%B4%E7%BB%B4%E5%BA%A6.md)
- 4. [解码器网络设计策略](04-%E8%A7%A3%E7%A0%81%E5%99%A8%E7%BD%91%E7%BB%9C%E8%AE%BE%E8%AE%A1%E7%AD%96%E7%95%A5.md)
- 5. [为自编码器选择合适的损失函数](05-%E4%B8%BA%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E9%80%89%E6%8B%A9%E5%90%88%E9%80%82%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 6. [优化器选择和学习率配置](06-%E4%BC%98%E5%8C%96%E5%99%A8%E9%80%89%E6%8B%A9%E5%92%8C%E5%AD%A6%E4%B9%A0%E7%8E%87%E9%85%8D%E7%BD%AE.md)
- 7. [监控自动编码器训练进程](07-%E7%9B%91%E6%8E%A7%E8%87%AA%E5%8A%A8%E7%BC%96%E7%A0%81%E5%99%A8%E8%AE%AD%E7%BB%83%E8%BF%9B%E7%A8%8B.md)
- 8. [从瓶颈层提取特征的方法](08-%E4%BB%8E%E7%93%B6%E9%A2%88%E5%B1%82%E6%8F%90%E5%8F%96%E7%89%B9%E5%BE%81%E7%9A%84%E6%96%B9%E6%B3%95.md)
- 9. [潜在空间可视化（适用时）](09-%E6%BD%9C%E5%9C%A8%E7%A9%BA%E9%97%B4%E5%8F%AF%E8%A7%86%E5%8C%96%EF%BC%88%E9%80%82%E7%94%A8%E6%97%B6%EF%BC%89.md)
- 10. [实践：从表格数据中获取特征](10-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BB%8E%E8%A1%A8%E6%A0%BC%E6%95%B0%E6%8D%AE%E4%B8%AD%E8%8E%B7%E5%8F%96%E7%89%B9%E5%BE%81.md)

章节测验：[在线测验](https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-3-building-first-autoencoder-feature-extraction/quiz)

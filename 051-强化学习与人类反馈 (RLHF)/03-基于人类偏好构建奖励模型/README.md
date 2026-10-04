# 第 3 章：基于人类偏好构建奖励模型

来源：[原章节](https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-3-reward-modeling-human-preferences)

[返回课程目录](../README.md)

监督微调（SFT）提供了基线，但实现对齐需要更直接的方法来纳入人类对响应质量的判断。本章将重心转向构建奖励模型（$RM$），这是一个重要部分，它学习预测人类偏好哪些AI生成的响应。

您将学习创建这个$RM$的流程，从直接从成对比较中学习的思路开始。我们将介绍收集人类偏好数据的方法、组织这些数据集的方式以及选择合适的模型架构。本章详细介绍常见的训练目标，例如基于Bradley-Terry模型的那些，其公式如下：

$$ P(\text{response}_1 \succ \text{response}_2 | \text{prompt}) = \sigma(RM(\text{prompt}, \text{response}_1) - RM(\text{prompt}, \text{response}_2)) $$

$\sigma$代表sigmoid函数，$\succ$表示偏好。

此外，我们将研究校准奖励模型分数以更好地反映偏好强度的方法，并讨论奖励建模过程中可能遇到的问题，包括数据质量问题以及模型发现意外捷径（奖励欺骗）的风险。在本章结束时，您将明白如何训练一个能量化人类偏好的模型，为强化学习优化做好准备。

## 小节

- 1. [偏好学习的思路](01-%E5%81%8F%E5%A5%BD%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%80%9D%E8%B7%AF.md)
- 2. [人类偏好数据收集](02-%E4%BA%BA%E7%B1%BB%E5%81%8F%E5%A5%BD%E6%95%B0%E6%8D%AE%E6%94%B6%E9%9B%86.md)
- 3. [偏好数据集的格式与结构](03-%E5%81%8F%E5%A5%BD%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84%E6%A0%BC%E5%BC%8F%E4%B8%8E%E7%BB%93%E6%9E%84.md)
- 4. [奖励模型架构](04-%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.md)
- 5. [奖励模型训练目标](05-%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87.md)
- 6. [奖励模型校准](06-%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B%E6%A0%A1%E5%87%86.md)
- 7. [奖励模型中可能出现的问题](07-%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B%E4%B8%AD%E5%8F%AF%E8%83%BD%E5%87%BA%E7%8E%B0%E7%9A%84%E9%97%AE%E9%A2%98.md)
- 8. [动手实践：训练奖励模型](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B.md)

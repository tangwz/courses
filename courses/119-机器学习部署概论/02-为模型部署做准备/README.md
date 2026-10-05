# 第 2 章：为模型部署做准备

来源：[原章节](https://apxml.com/zh/courses/basics-ml-deployment/chapter-2-preparing-model-deployment)

[返回课程目录](../README.md)

训练完机器学习模型后，当务之急是保存其学到的状态。不保存的话，模型只存在于内存中，一旦训练脚本结束就会丢失。本章将讲述如何让模型持久保存，并为在不同环境或稍后使用做好准备。

你将了解模型序列化是什么，它是将训练好的模型对象转换为可存储（如文件）的格式，以便之后加载回内存的过程。我们将介绍为此目的的两个常用Python库：`pickle` 和 `joblib`，说明它们的用法以及何时更适合使用其中一个而非另一个。

此外，准备模型不仅仅是保存算法的权重。我们将讨论追踪和管理模型所依赖的软件库（及其版本）的必要性。你还将学习如何保存数据预处理步骤（如缩放器或编码器），这些步骤必须在新输入数据进行预测前保持一致地应用。本章包含实际示例，引导你完成简单模型的保存和加载。

## 小节

- 1. [保存训练好的模型](01-%E4%BF%9D%E5%AD%98%E8%AE%AD%E7%BB%83%E5%A5%BD%E7%9A%84%E6%A8%A1%E5%9E%8B.md)
- 2. [模型序列化简介](02-%E6%A8%A1%E5%9E%8B%E5%BA%8F%E5%88%97%E5%8C%96%E7%AE%80%E4%BB%8B.md)
- 3. [使用 Pickle 保存模型](03-%E4%BD%BF%E7%94%A8%20Pickle%20%E4%BF%9D%E5%AD%98%E6%A8%A1%E5%9E%8B.md)
- 4. [使用 Joblib 保存模型](04-%E4%BD%BF%E7%94%A8%20Joblib%20%E4%BF%9D%E5%AD%98%E6%A8%A1%E5%9E%8B.md)
- 5. [处理模型依赖](05-%E5%A4%84%E7%90%86%E6%A8%A1%E5%9E%8B%E4%BE%9D%E8%B5%96.md)
- 6. [保存预处理步骤](06-%E4%BF%9D%E5%AD%98%E9%A2%84%E5%A4%84%E7%90%86%E6%AD%A5%E9%AA%A4.md)
- 7. [动手实践：保存和加载简单模型](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E7%AE%80%E5%8D%95%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/basics-ml-deployment/chapter-2-preparing-model-deployment/quiz)

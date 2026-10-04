# 第 2 章：处理缺失数据

来源：[原章节](https://apxml.com/zh/courses/intro-feature-engineering/chapter-2-handling-missing-data)

[返回课程目录](../README.md)

实际数据集经常不完整。机器学习算法通常无法直接处理缺失值，因此数据预处理是必需的第一步。本章介绍管理缺失数据的方法。

您将学习使用 Pandas 识别数据中缺失项的方法。我们将考察导致数据缺失的常见原因，例如完全随机缺失 (MCAR)、随机缺失 (MAR) 和非随机缺失 (MNAR)。然后着重介绍实际的填补策略，从均值、中位数和众数填补等简单方法开始。您还将学习创建指示特征，以保留缺失值原始位置的信息。更高级的方法，包括 Scikit-learn 中的 K-近邻 (KNN) 填补器和迭代填补器，将用于多元填补。最后，我们将比较这些不同的方法，以帮助您选择适用于您具体情况的策略。本章包含应用这些方法的实践练习。

## 小节

- 1. [识别缺失值](01-%E8%AF%86%E5%88%AB%E7%BC%BA%E5%A4%B1%E5%80%BC.md)
- 2. [缺失数据机制 (MCAR, MAR, MNAR)](02-%E7%BC%BA%E5%A4%B1%E6%95%B0%E6%8D%AE%E6%9C%BA%E5%88%B6%20%28MCAR%2C%20MAR%2C%20MNAR%29.md)
- 3. [简单填充策略：均值、中位数、众数](03-%E7%AE%80%E5%8D%95%E5%A1%AB%E5%85%85%E7%AD%96%E7%95%A5%EF%BC%9A%E5%9D%87%E5%80%BC%E3%80%81%E4%B8%AD%E4%BD%8D%E6%95%B0%E3%80%81%E4%BC%97%E6%95%B0.md)
- 4. [创建缺失值指示器](04-%E5%88%9B%E5%BB%BA%E7%BC%BA%E5%A4%B1%E5%80%BC%E6%8C%87%E7%A4%BA%E5%99%A8.md)
- 5. [多变量填充：KNN填充器](05-%E5%A4%9A%E5%8F%98%E9%87%8F%E5%A1%AB%E5%85%85%EF%BC%9AKNN%E5%A1%AB%E5%85%85%E5%99%A8.md)
- 6. [多元插补：迭代插补器](06-%E5%A4%9A%E5%85%83%E6%8F%92%E8%A1%A5%EF%BC%9A%E8%BF%AD%E4%BB%A3%E6%8F%92%E8%A1%A5%E5%99%A8.md)
- 7. [比较插补方法](07-%E6%AF%94%E8%BE%83%E6%8F%92%E8%A1%A5%E6%96%B9%E6%B3%95.md)
- 8. [动手实践：填充缺失数据](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%A1%AB%E5%85%85%E7%BC%BA%E5%A4%B1%E6%95%B0%E6%8D%AE.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-feature-engineering/chapter-2-handling-missing-data/quiz)

# 第 4 章：XGBoost：极限梯度提升

来源：[原章节](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-4-xgboost-extreme-gradient-boosting)

[返回课程目录](../README.md)

本章介绍XGBoost（极限梯度提升），直接基于前面讨论的梯度提升基本思想和正则化方法。XGBoost是一种常用且有效的梯度提升实现方式，以多项重要改进为特点，旨在提升性能和准确性。

我们将分析使XGBoost高效的核心构成部分：
*   **正则化学习目标**：了解XGBoost如何将$L_1$和$L_2$惩罚项直接纳入目标函数中，以控制模型复杂度。
*   **分枝寻找算法**：学习寻找分枝的精确贪心算法，以及处理大规模数据集的近似算法。
*   **稀疏性处理**：了解其内置机制，以高效处理缺失值。
*   **系统优化**：回顾并行处理和缓存优化等可提高训练速度的方法。

在本章结束时，您将掌握XGBoost背后的技术细节，并准备好使用其Python库进行实现，为实际应用配置其主要参数。

## 小节

- 1. [GBM 的原理及优化](01-GBM%20%E7%9A%84%E5%8E%9F%E7%90%86%E5%8F%8A%E4%BC%98%E5%8C%96.md)
- 2. [正则化学习目标](02-%E6%AD%A3%E5%88%99%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%9B%AE%E6%A0%87.md)
- 3. [分裂查找算法：精确贪心](03-%E5%88%86%E8%A3%82%E6%9F%A5%E6%89%BE%E7%AE%97%E6%B3%95%EF%BC%9A%E7%B2%BE%E7%A1%AE%E8%B4%AA%E5%BF%83.md)
- 4. [分割查找算法：近似贪婪算法](04-%E5%88%86%E5%89%B2%E6%9F%A5%E6%89%BE%E7%AE%97%E6%B3%95%EF%BC%9A%E8%BF%91%E4%BC%BC%E8%B4%AA%E5%A9%AA%E7%AE%97%E6%B3%95.md)
- 5. [稀疏感知分裂查找](05-%E7%A8%80%E7%96%8F%E6%84%9F%E7%9F%A5%E5%88%86%E8%A3%82%E6%9F%A5%E6%89%BE.md)
- 6. [系统优化：缓存感知与并行处理](06-%E7%B3%BB%E7%BB%9F%E4%BC%98%E5%8C%96%EF%BC%9A%E7%BC%93%E5%AD%98%E6%84%9F%E7%9F%A5%E4%B8%8E%E5%B9%B6%E8%A1%8C%E5%A4%84%E7%90%86.md)
- 7. [XGBoost API: 参数与配置](07-XGBoost%20API-%20%E5%8F%82%E6%95%B0%E4%B8%8E%E9%85%8D%E7%BD%AE.md)
- 8. [实践操作：实现XGBoost](08-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E5%AE%9E%E7%8E%B0XGBoost.md)

---
course: "intro-feature-engineering"
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-3-encoding-categorical-features"
sourceId: 445
chapter: "encoding-categorical-features"
title: "类别特征编码"
order: 3
description: "学习将类别数据转换为数值表示的多种技术，以用于机器学习模型。"
hasQuiz: true
---

许多数据集包含类别特征，表示产品类型、地理位置或用户群等信息。这些特征通常包含有价值的预测信息，但大多数机器学习算法需要数值输入。本章侧重于解决这一问题，介绍将类别数据转换为合适数值表示的方法。

你将学会：

*   区分标称型和序数型类别数据，并理解这种差异对编码的重要性。
*   实现常见的编码策略，例如独热编码、序数编码、目标编码、二进制编码和哈希编码。
*   处理针对具有大量独特类别（高基数）的特征进行编码的特定难题。
*   比较不同编码技术之间的权衡。

我们将使用 Pandas 和 Scikit-learn 等 Python 库应用这些技术，为你的类别数据在机器学习模型中的有效使用做好准备。

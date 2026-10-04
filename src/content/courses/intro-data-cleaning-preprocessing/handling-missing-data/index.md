---
course: "intro-data-cleaning-preprocessing"
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-2-handling-missing-data"
sourceId: 796
chapter: "handling-missing-data"
title: "识别和处理缺失数据"
order: 2
description: "学习如何使用简单方法，查找和处理数据集中的缺失值（NaN、空值）。"
hasQuiz: true
---

实际数据集通常不完整。数据条目可能因数据收集过程中的错误、传输问题，或者仅仅是信息当时无法获取而缺失。这些空缺，通常表现为 `NaN`（非数字）、`NULL` 或其他占位符，会严重影响数据分析和机器学习模型的表现。许多算法在遇到缺失值时无法正常运行。

本章主要介绍处理这一常见问题的基本方法。您将学到：

*   如何使用编程方法和基础可视化手段，识别数据集中缺失数据的存在、模式和范围。
*   处理这些空缺的不同方式，包括：
    *   删除包含缺失值的行或列（删除法）。
    *   用计算得出的替代值（例如均值、中位数或众数）填充缺失条目（填充法）。
*   在为您的具体情况和数据选择合适策略时，需要考量的因素。

我们将通过实际例子来说明这些方法，为准备更干净、更可靠的数据打下基础。

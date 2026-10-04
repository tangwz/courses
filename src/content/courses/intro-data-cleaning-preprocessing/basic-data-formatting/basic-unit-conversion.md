---
course: "intro-data-cleaning-preprocessing"
chapter: "basic-data-formatting"
lesson: "basic-unit-conversion"
sourceId: 4034
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-5-basic-data-formatting/basic-unit-conversion"
title: "基本单位转换示例"
description: "阐述如何在列内通过简单算术统一单位。"
order: 5
plots: []
sourceHash: "c5db6763aec4a181132c40a5df48a203334c106d9bd91734818c2994f9cfba4f"
sourceCorrections: []
---

当数据中存在大小写不一致或多余空格时，问题便会产生。同样，同一列中包含不同单位的测量值会使分析变得困难或导致结果不准确。试想一个包含产品重量的数据集，其中有些以千克（kg）列出，有些则以磅（lbs）列出。若不先对单位进行统一，直接计算平均重量或比较产品将是误导性的。

让我们看一个简单例子。假设我们有一个跟踪不同物品重量的小型数据集：

| 物品编号 | 重量 | 单位 |
| --- | --- | --- |
| A101 | 2.5 | kg |
| B203 | 5.5 | lbs |
| C305 | 1.2 | kg |
| D407 | 11.0 | lbs |
| E509 | 3.0 | kg |

我们的目标是使“重量”列保持一致。一种常见做法是选择一个标准单位，例如千克，并将所有其他测量值转换为该单位。

### 转换过程

为了将磅转换为千克，我们使用转换系数：$1 \text{ 磅} = 0.453592 \text{ 千克}$。

该过程包括以下一般步骤：

1. **识别**单位为“磅”的行。
2. **选择**与这些行对应的“重量”值。
3. **应用**转换公式：将磅的重量乘以0.453592，得到千克的重量。
   
   $$
   \text{以千克计的重量} = \text{以磅计的重量} \times 0.453592
   $$
   
4. **更新**“重量”列，将转换后的行的计算千克值填入。
5. 可选地，将所有行的“单位”列更新为“千克”，或者，如果所有重量现在都默认以千克计，则移除该列。

### 应用转换

让我们将此应用于示例数据。

- 物品B203：重量为5.5磅。
  $5.5 \text{ 磅} \times 0.453592 \approx 2.49 \text{ 千克}$
- 物品D407：重量为11.0磅。
  $11.0 \text{ 磅} \times 0.453592 \approx 4.99 \text{ 千克}$

在执行这些计算并更新数据集（可能创建一个新的标准化重量列或修改现有列）后，我们的数据如下所示：

| 物品编号 | 重量 | 单位 | 重量 (千克) |
| --- | --- | --- | --- |
| A101 | 2.5 | kg | 2.50 |
| B203 | 5.5 | lbs | 2.49 |
| C305 | 1.2 | kg | 1.20 |
| D407 | 11.0 | lbs | 4.99 |
| E509 | 3.0 | kg | 3.00 |

> 原始重量与标准化千克重量的比较。

现在，所有重量都以千克表示（在“重量 (千克)”列中），可以直接进行比较和准确的计算，例如求平均重量。

“本示例使用重量，但同样的原理也适用于其他测量值，例如长度（英寸与厘米）、温度（华氏度与摄氏度）或货币，只要存在明确的转换系数。这种基本的单位统一是准备数据进行有效分析的常见且必要的步骤。虽然本示例简单直接，但数据可能需要更复杂的逻辑来识别单位或处理单位记录方式的变化（例如，‘kg’、‘kilo’、‘kilogram’）。不过，其核心思想始终一致：识别、转换和统一。”

## 参考资料

- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  使用 Python 和 pandas 进行数据处理的基础指南，阐释了实用的数据清洗和转换技术，包括处理不一致的单位。（第3版）
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供了数据预处理步骤的实践示例，包括特征缩放和数据转换，这对于为机器学习模型准备数据和确保单位一致性至关重要。（第3版）

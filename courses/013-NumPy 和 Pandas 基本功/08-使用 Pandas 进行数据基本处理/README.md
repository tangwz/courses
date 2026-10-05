# 第 8 章：使用 Pandas 进行数据基本处理

来源：[原章节](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-8-basic-data-manipulation-pandas)

[返回课程目录](../README.md)

实际数据在有效分析前，通常需要进行清理和重塑。

在学习了如何使用 Pandas Series 和 DataFrame 创建、加载和选择数据之后，我们现在转向数据处理这项重要工作。原始数据集经常包含缺失值、无关信息或不方便处理的列名。

本章介绍在 Pandas DataFrames 中整理数据的基本方法。您将学习如何：

*   识别缺失数据点（通常表示为 $NaN$）。
*   应用处理这些缺失值的方法，可以通过删除或用合适的值填充。
*   添加新列，这些列通常基于现有列的计算生成。
*   移除不需要的行或列。
*   重命名列，以提高清晰度或保持一致性。
*   根据索引或特定列的值对 DataFrame 进行排序，以便更好地整理信息。

掌握这些操作为您后续的数据分析或建模工作打下了扎实功底。

## 小节

- 1. [识别缺失数据](01-%E8%AF%86%E5%88%AB%E7%BC%BA%E5%A4%B1%E6%95%B0%E6%8D%AE.md)
- 2. [处理缺失数据：删除](02-%E5%A4%84%E7%90%86%E7%BC%BA%E5%A4%B1%E6%95%B0%E6%8D%AE%EF%BC%9A%E5%88%A0%E9%99%A4.md)
- 3. [处理缺失数据：填充](03-%E5%A4%84%E7%90%86%E7%BC%BA%E5%A4%B1%E6%95%B0%E6%8D%AE%EF%BC%9A%E5%A1%AB%E5%85%85.md)
- 4. [删除列和行](04-%E5%88%A0%E9%99%A4%E5%88%97%E5%92%8C%E8%A1%8C.md)
- 5. [添加新列](05-%E6%B7%BB%E5%8A%A0%E6%96%B0%E5%88%97.md)
- 6. [修改现有列](06-%E4%BF%AE%E6%94%B9%E7%8E%B0%E6%9C%89%E5%88%97.md)
- 7. [重命名列](07-%E9%87%8D%E5%91%BD%E5%90%8D%E5%88%97.md)
- 8. [数据排序](08-%E6%95%B0%E6%8D%AE%E6%8E%92%E5%BA%8F.md)
- 9. [实践练习：清理和修改数据框](09-%E5%AE%9E%E8%B7%B5%E7%BB%83%E4%B9%A0%EF%BC%9A%E6%B8%85%E7%90%86%E5%92%8C%E4%BF%AE%E6%94%B9%E6%95%B0%E6%8D%AE%E6%A1%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-8-basic-data-manipulation-pandas/quiz)

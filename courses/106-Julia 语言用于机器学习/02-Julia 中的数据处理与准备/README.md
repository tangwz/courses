# 第 2 章：Julia 中的数据处理与准备

来源：[原章节](https://apxml.com/zh/courses/julia-for-machine-learning/chapter-2-julia-data-manipulation-preparation)

[返回课程目录](../README.md)

输入数据的质量直接影响任何机器学习模型的性能。本章侧重于使用 Julia 准备数据以进行分析和模型训练的必要步骤。您将学习如何将各种来源的数据加载到 `DataFrames.jl` 中，这是一个重要的 Julia 表格数据处理包。我们将介绍数据清洗的方法，例如处理缺失值和识别异常值。此外，您将了解到数据转换方法，包括数值特征的缩放、分类变量的编码以及连续数据的分箱。我们还将阐述特征工程的原理，并演示如何在 Julia 环境中从您现有数据中创建新的、有用的特征。最后，您将看到如何使用 Julia 的绘图库，如 `Plots.jl` 和 `Makie.jl`，进行数据可视化，这有助于数据理解和预处理。完成本章后，您将掌握在 Julia 中处理和准备数据集以进行机器学习任务的实用技能。

## 小节

- 1. [使用 DataFrames.jl 加载和保存数据](01-%E4%BD%BF%E7%94%A8%20DataFrames.jl%20%E5%8A%A0%E8%BD%BD%E5%92%8C%E4%BF%9D%E5%AD%98%E6%95%B0%E6%8D%AE.md)
- 2. [数据清洗：处理缺失值和异常值](02-%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97%EF%BC%9A%E5%A4%84%E7%90%86%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%92%8C%E5%BC%82%E5%B8%B8%E5%80%BC.md)
- 3. [数据转换：缩放、编码和分箱](03-%E6%95%B0%E6%8D%AE%E8%BD%AC%E6%8D%A2%EF%BC%9A%E7%BC%A9%E6%94%BE%E3%80%81%E7%BC%96%E7%A0%81%E5%92%8C%E5%88%86%E7%AE%B1.md)
- 4. [特征工程原则](04-%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B%E5%8E%9F%E5%88%99.md)
- 5. [在 Julia 中应用特征工程](05-%E5%9C%A8%20Julia%20%E4%B8%AD%E5%BA%94%E7%94%A8%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B.md)
- 6. [使用 Plots.jl 和 Makie.jl 进行数据可视化](06-%E4%BD%BF%E7%94%A8%20Plots.jl%20%E5%92%8C%20Makie.jl%20%E8%BF%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- 7. [动手实践：数据清洗与特征创建](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97%E4%B8%8E%E7%89%B9%E5%BE%81%E5%88%9B%E5%BB%BA.md)

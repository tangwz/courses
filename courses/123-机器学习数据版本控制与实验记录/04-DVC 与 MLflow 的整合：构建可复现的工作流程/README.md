# 第 4 章：DVC 与 MLflow 的整合：构建可复现的工作流程

来源：[原章节](https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-4-integrating-dvc-mlflow)

[返回课程目录](../README.md)

在前几章中，我们学习了使用 DVC 进行数据版本管理和使用 MLflow 进行实验追踪的基本方法，现在我们将把这些工具结合起来使用。高效整合数据管理与实验记录，是构建真正可复现机器学习工作流程的**核心部分**。

本章将提供关于以下方面的**实践指导**：

*   将 DVC 追踪的特定数据版本与 MLflow 记录的相应实验运行**连接起来**。
*   组织您的机器学习项目结构，以**方便**同时使用 DVC 和 MLflow。
*   在 MLflow 运行中自动记录 DVC 元数据的**方法**。
*   使用 `dvc run` 和 `dvc repro` 等 DVC 命令来**建立**和重现自动化管道。
*   将 MLflow 追踪直接**引入** DVC 管道阶段。
*   **制定**最佳实践，以在同时使用 DVC 和 MLflow 时保持**一致且可重现**的工作流程。

在本章结束时，您将**学会**如何构建**一体化系统**，在这些系统中，数据、代码、参数和结果的变化都能**得到持续的追踪和管理**。

## 小节

- 1. [关联数据版本与实验](01-%E5%85%B3%E8%81%94%E6%95%B0%E6%8D%AE%E7%89%88%E6%9C%AC%E4%B8%8E%E5%AE%9E%E9%AA%8C.md)
- 2. [为集成构建项目结构](02-%E4%B8%BA%E9%9B%86%E6%88%90%E6%9E%84%E5%BB%BA%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84.md)
- 3. [在 MLflow 中记录 DVC 元数据](03-%E5%9C%A8%20MLflow%20%E4%B8%AD%E8%AE%B0%E5%BD%95%20DVC%20%E5%85%83%E6%95%B0%E6%8D%AE.md)
- 4. [构建 DVC 流水线](04-%E6%9E%84%E5%BB%BA%20DVC%20%E6%B5%81%E6%B0%B4%E7%BA%BF.md)
- 5. [复现 DVC 流水线](05-%E5%A4%8D%E7%8E%B0%20DVC%20%E6%B5%81%E6%B0%B4%E7%BA%BF.md)
- 6. [追踪 DVC 流水线指标](06-%E8%BF%BD%E8%B8%AA%20DVC%20%E6%B5%81%E6%B0%B4%E7%BA%BF%E6%8C%87%E6%A0%87.md)
- 7. [结合 DVC 流水线与 MLflow 追踪](07-%E7%BB%93%E5%90%88%20DVC%20%E6%B5%81%E6%B0%B4%E7%BA%BF%E4%B8%8E%20MLflow%20%E8%BF%BD%E8%B8%AA.md)
- 8. [集成工作流程的最佳实践](08-%E9%9B%86%E6%88%90%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B%E7%9A%84%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md)
- 9. [动手实践：构建集成式流程](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E9%9B%86%E6%88%90%E5%BC%8F%E6%B5%81%E7%A8%8B.md)

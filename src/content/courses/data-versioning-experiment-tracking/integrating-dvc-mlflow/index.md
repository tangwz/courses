---
course: "data-versioning-experiment-tracking"
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-4-integrating-dvc-mlflow"
sourceId: 812
chapter: "integrating-dvc-mlflow"
title: "DVC 与 MLflow 的整合：构建可复现的工作流程"
order: 4
description: "结合数据版本管理与实验追踪，构建完全可重现的机器学习管道。"
hasQuiz: false
---

在前几章中，我们学习了使用 DVC 进行数据版本管理和使用 MLflow 进行实验追踪的基本方法，现在我们将把这些工具结合起来使用。高效整合数据管理与实验记录，是构建真正可复现机器学习工作流程的**核心部分**。

本章将提供关于以下方面的**实践指导**：

*   将 DVC 追踪的特定数据版本与 MLflow 记录的相应实验运行**连接起来**。
*   组织您的机器学习项目结构，以**方便**同时使用 DVC 和 MLflow。
*   在 MLflow 运行中自动记录 DVC 元数据的**方法**。
*   使用 `dvc run` 和 `dvc repro` 等 DVC 命令来**建立**和重现自动化管道。
*   将 MLflow 追踪直接**引入** DVC 管道阶段。
*   **制定**最佳实践，以在同时使用 DVC 和 MLflow 时保持**一致且可重现**的工作流程。

在本章结束时，您将**学会**如何构建**一体化系统**，在这些系统中，数据、代码、参数和结果的变化都能**得到持续的追踪和管理**。

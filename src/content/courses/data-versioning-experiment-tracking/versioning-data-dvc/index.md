---
course: "data-versioning-experiment-tracking"
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-2-versioning-data-dvc"
sourceId: 809
chapter: "versioning-data-dvc"
title: "使用 DVC 管理数据版本"
order: 2
description: "学习数据版本控制 (DVC) 的原理和实际使用方法，以在机器学习中管理数据集。"
hasQuiz: true
---

第一章指出了管理机器学习项目的难题，尤其是在处理不适合标准 Git 工作流程的大型数据集时。本章介绍数据版本控制 (DVC)，这是一个开源工具，专门设计用于与代码一起管理数据版本，从而帮助解决这些问题。

我们将首先查看不同的数据版本管理方法，然后专注于 DVC 的运作方式以及它如何与 Git 结合。你将学习如何：

*   在现有 Git 仓库中初始化 DVC。
*   使用 `dvc add` 开始追踪数据文件和目录。
*   配置远程存储（例如 AWS S3、Google Cloud Storage 或 Azure Blob Storage）。
*   使用 `dvc push` 和 `dvc pull` 在本地机器和远程存储之间同步数据。
*   切换到与特定 Git 提交对应的数据的不同版本。

本章包含实际步骤，并以一个实践练习作为结尾，你将在其中运用这些命令来管理一个样本数据集的版本。在本章结束时，你将能够使用 DVC 在你的机器学习项目中实施有效的数据版本管理。

# 第 2 章：使用 DVC 管理数据版本

来源：[原章节](https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-2-versioning-data-dvc)

[返回课程目录](../README.md)

第一章指出了管理机器学习项目的难题，尤其是在处理不适合标准 Git 工作流程的大型数据集时。本章介绍数据版本控制 (DVC)，这是一个开源工具，专门设计用于与代码一起管理数据版本，从而帮助解决这些问题。

我们将首先查看不同的数据版本管理方法，然后专注于 DVC 的运作方式以及它如何与 Git 结合。你将学习如何：

*   在现有 Git 仓库中初始化 DVC。
*   使用 `dvc add` 开始追踪数据文件和目录。
*   配置远程存储（例如 AWS S3、Google Cloud Storage 或 Azure Blob Storage）。
*   使用 `dvc push` 和 `dvc pull` 在本地机器和远程存储之间同步数据。
*   切换到与特定 Git 提交对应的数据的不同版本。

本章包含实际步骤，并以一个实践练习作为结尾，你将在其中运用这些命令来管理一个样本数据集的版本。在本章结束时，你将能够使用 DVC 在你的机器学习项目中实施有效的数据版本管理。

## 小节

- 1. [数据版本控制方法](01-%E6%95%B0%E6%8D%AE%E7%89%88%E6%9C%AC%E6%8E%A7%E5%88%B6%E6%96%B9%E6%B3%95.md)
- 2. [介绍数据版本控制 (DVC)](02-%E4%BB%8B%E7%BB%8D%E6%95%B0%E6%8D%AE%E7%89%88%E6%9C%AC%E6%8E%A7%E5%88%B6%20%28DVC%29.md)
- 3. [在项目中设置DVC](03-%E5%9C%A8%E9%A1%B9%E7%9B%AE%E4%B8%AD%E8%AE%BE%E7%BD%AEDVC.md)
- 4. [跟踪数据文件和目录](04-%E8%B7%9F%E8%B8%AA%E6%95%B0%E6%8D%AE%E6%96%87%E4%BB%B6%E5%92%8C%E7%9B%AE%E5%BD%95.md)
- 5. [数据版本的存储与获取](05-%E6%95%B0%E6%8D%AE%E7%89%88%E6%9C%AC%E7%9A%84%E5%AD%98%E5%82%A8%E4%B8%8E%E8%8E%B7%E5%8F%96.md)
- 6. [将 DVC 连接到远程存储 (S3、GCS、Azure Blob)](06-%E5%B0%86%20DVC%20%E8%BF%9E%E6%8E%A5%E5%88%B0%E8%BF%9C%E7%A8%8B%E5%AD%98%E5%82%A8%20%28S3%E3%80%81GCS%E3%80%81Azure%20Blob%29.md)
- 7. [在不同数据版本间切换](07-%E5%9C%A8%E4%B8%8D%E5%90%8C%E6%95%B0%E6%8D%AE%E7%89%88%E6%9C%AC%E9%97%B4%E5%88%87%E6%8D%A2.md)
- 8. [动手实践：数据集版本管理](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E7%89%88%E6%9C%AC%E7%AE%A1%E7%90%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-2-versioning-data-dvc/quiz)

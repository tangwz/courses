---
course: "data-versioning-experiment-tracking"
chapter: "versioning-data-dvc"
lesson: "introducing-dvc"
sourceId: 4062
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-2-versioning-data-dvc/introducing-dvc"
title: "介绍数据版本控制 (DVC)"
description: "DVC 功能、架构及其与 Git 配合方式的概述。"
order: 2
plots: []
sourceHash: "844dba893de47efd46059467f90b2cfdeedf58c3164ff4acbdc109d05b4e26c5"
sourceCorrections: []
---

管理机器学习 (machine learning)项目会遇到特有难题，特别是当数据集和模型过大，无法放入标准Git仓库时。Git擅长代码版本控制，但跟踪数GB大小的文件会影响其性能并导致仓库膨胀。数据版本控制 (DVC) 解决了这些挑战。

DVC是一款开源工具，专门设计用于为数据和模型提供版本控制能力，它与Git协同工作而非取代Git。可以将其视为Git能力的延伸，以应对机器学习中常见的庞大文件。

### DVC 的工作方式：主要思路

DVC背后的基本思路很简单：DVC不会将大文件直接存储在Git仓库中，而是存储轻量级的元数据文件（称为`.dvc`文件），这些文件充当指向实际数据的指针。这些`.dvc`文件*由*Git跟踪。

具体情况如下：

1. **跟踪数据：** 您使用 `dvc add` 命令告诉DVC需要跟踪哪些大文件或目录。
2. **创建元数据：** DVC计算数据内容的哈希值（一个独特的指纹，通常是MD5），并创建一个小的`.dvc`文件。该文件包含哈希值和定位实际数据所需的其他信息。
3. **存储数据：** DVC将原始数据文件移入一个隐藏的本地缓存（通常在项目目录的`.dvc/cache`中）。此缓存使用内容寻址存储，即文件根据其哈希值存储。这能防止数据重复；如果同一文件有多个副本（即使名称或位置不同），DVC在缓存中只存储一份。
4. **版本元数据：** 您将小的`.dvc`文件提交到Git，就像处理任何其他代码变更一样。您的Git仓库现在包含代码和这些指针，但没有大文件本身。
5. **共享数据（可选）：** 您可以配置DVC，使用 `dvc push` 命令将本地缓存中的实际数据推送到远程存储（如云存储桶或网络驱动器）。这样可以方便协作和备份，而不会给Git服务器增加负担。

### DVC 架构组成

了解DVC需要认识其主要组成部分以及它们如何协同作用：

- **DVC 命令行界面 (CLI)：** 这是您与DVC交互的主要界面。`dvc init`、`dvc add`、`dvc push`、`dvc pull` 和 `dvc checkout` 等命令用于管理数据版本控制过程。
- **`.dvc` 文件：** 这些小型、易读的文本文件（通常为YAML格式）位于您的Git仓库中。它们包含元数据，包括数据文件的哈希值、大小，以及其在DVC缓存或远程存储配置中的潜在路径。它们充当Git历史记录与特定数据版本之间的链接。
- **DVC 缓存：** 一个本地目录（例如 `.dvc/cache`），DVC在此处存储实际数据文件，并按其内容哈希值进行组织。此机制通过重复数据删除来保证数据完整性和高效存储。默认情况下，DVC会尝试使用优化的文件链接（如reflinks或硬链接）来避免在您的工作区和缓存之间重复数据，从而节省磁盘空间。
- **远程存储：** DVC可以存储和获取数据版本的本地机器之外的位置。这可以是云存储（AWS S3、Google Cloud Storage、Azure Blob Storage）、SSH服务器、网络驱动器，甚至是您本地文件系统上的另一个目录。远程存储可以按项目或全局配置。

### 与 Git 紧密配合

DVC旨在补充Git，为代码和数据版本控制创建统一的工作流程：

- **Git 跟踪：** 您的源代码（`.py`文件、脚本等）和`.dvc`元数据文件。
- **DVC 跟踪：** `.dvc`文件引用的那些大文件和模型成果。

当您在Git中切换分支或检出某个历史提交时，您会得到与该历史点对应的代码和`.dvc`文件。然而，您工作区中的大文件可能不会自动与这些`.dvc`文件匹配。为了使您的工作区数据与当前`.dvc`文件指定的版本保持一致，您通常需要运行 `dvc checkout` 或 `dvc pull`。DVC随后会根据`.dvc`文件中的信息，从缓存或远程存储中获取正确的数据版本，并将其放入您的工作目录。

下图描绘了这种关系：

> Git、工作区、DVC 缓存和远程存储之间的关系。Git对代码和`.dvc`文件进行版本控制，而DVC则根据`.dvc`文件中的信息，管理工作区、缓存和远程存储之间的数据流动。

### 为何使用 DVC？

采用DVC能为您的机器学习 (machine learning)项目带来多方面好处：

- **轻量化 (quantization)Git仓库：** 由于大文件单独处理，Git历史记录保持整洁，仓库体积小且速度快。
- **数据和模型版本控制：** 提供一种系统化的方法，随着时间推移跟踪数据集和模型的变化，类似于Git跟踪代码变化。
- **可复现性：** 将特定版本的代码（通过Git提交）与所用数据和模型的精确版本（通过`.dvc`文件）关联起来，使实验更容易复现。
- **协作：** 通过共享远程存储，简化团队成员之间共享大型数据集和模型的过程。
- **存储无关：** 可与各种后端存储选项配合使用，提供部署上的灵活性。
- **框架无关：** 独立于特定的机器学习库或框架运行。

既然您已了解DVC是什么以及它如何与Git配合使用，接下来的部分将指导您完成在项目中设置DVC、跟踪数据、配置远程存储以及管理不同数据版本的实际步骤。

## 参考资料

- [Data Version Control (DVC) Documentation](https://dvc.org/doc) — Iterative.ai (2024)
  了解DVC功能、命令和架构组件的主要和最权威的来源。
- [Practical MLOps: How to Get Your Machine Learning Models Into Production](https://www.oreilly.com/library/view/practical-mlops/9781492080342/) — Mark Treveil, Dejan Simic, Nick O'Leary, Adam Kawa, Nikita Smirnov, Dmytro Katler, and Pascal van Kooten (2022)
  Publisher: O'Reilly Media; Pages: Chapter 6: Data and Model Versioning
  提供MLOps的实用指南，其中有一个专门的章节，深入讨论了数据和模型版本控制，包括DVC的作用和实现。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  从全面视角探讨如何设计稳健的机器学习系统，涵盖数据管理以及版本控制在生产环境中的重要性。

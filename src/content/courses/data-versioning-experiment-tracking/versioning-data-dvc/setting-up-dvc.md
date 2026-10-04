---
course: "data-versioning-experiment-tracking"
chapter: "versioning-data-dvc"
lesson: "setting-up-dvc"
sourceId: 4064
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-2-versioning-data-dvc/setting-up-dvc"
title: "在项目中设置DVC"
description: "关于在Git仓库中初始化DVC并配置远程存储的指南。"
order: 3
plots: []
sourceHash: "ebee09dd9d303c9ee6ceacc4ee6244a2c2fba77dea8ef4455762fcfa61c3ce55"
sourceCorrections: []
---

将数据版本控制（DVC）引入您的机器学习 (machine learning)项目，首先需要在项目目录下对其进行初始化。因为DVC设计为与Git配合使用，所以其主要条件是您的项目已经是一个Git仓库。如果您尚未初始化Git，请先进行此操作：

```bash
# 进入您的项目根目录
cd path/to/your-ml-project

# 初始化Git（如果尚未完成）
git init
# 添加并提交您现有的项目文件
git add .
git commit -m "Initial project commit"
```

Git设置好后，您现在可以将DVC引入项目结构中。

### `dvc init` 命令

初始化DVC的命令很简单：`dvc init`。从您的Git仓库根目录运行此命令。

```bash
# 确保您位于Git仓库的根目录
cd path/to/your-ml-project

# 初始化DVC
dvc init
```

执行 `dvc init` 会进行几项操作：

1. **创建 `.dvc` 目录：** 此目录类似于Git的 `.git` 目录。DVC在此处存储其内部信息，包括配置文件、缓存目录结构（即使缓存本身位于别处），以及与被追踪数据相关的元文件。通常情况下，除非您了解其影响，否则不应手动修改 `.dvc` 内部的文件。
2. **创建 `.dvc/config` 文件：** 这是您的DVC项目的主要配置文件。它用于定义远程存储位置、缓存设置以及其他DVC行为。初始时，它可能非常简洁。
3. **创建 `.dvcignore` 文件：** 类似于 `.gitignore`，此文件告诉DVC要忽略哪些文件或模式。这有助于防止DVC追踪不应作为数据工件进行版本控制的临时文件、日志或输出。例如，您可能将虚拟环境目录或IDE配置文件夹添加到 `.dvcignore`。
4. **打印指导信息：** 该命令通常会输出确认初始化成功的消息，并提示后续步骤，通常涉及将新创建的DVC文件添加到Git。

以下是初始化后DVC如何融入项目结构的视图：

> 运行 `dvc init` 后，`.dvc` 目录和 `.dvcignore` 文件会在您的Git仓库中创建。DVC使用这些文件来管理元数据，而大型数据文件则存储在单独的缓存位置。

### 将DVC配置提交到Git

DVC设计的一个重要方面是，其配置文件和元数据文件（如 `.dvc` 目录内容和我们稍后会讲到的指针文件）*就是*要由Git追踪的。这确保了项目的状态，包括哪些数据版本与哪些代码版本关联，都能完全记录在Git历史中。

运行 `dvc init` 后，您应该立即将创建的DVC文件添加到Git暂存区并提交它们：

```bash
# 查看状态以查看新的DVC文件
git status

# 添加 .dvc 目录和 .dvcignore 文件
git add .dvc .dvcignore

# 提交这些更改
git commit -m "在项目中初始化DVC"
```

提交这些文件可确保任何克隆您的仓库并安装了DVC的人都可以立即开始使用DVC命令，例如 `dvc pull`，以获取与特定Git提交对应的正确数据版本。

### 初始配置概述

由 `dvc init` 创建的 `.dvc/config` 文件包含项目特定的设置。虽然我们稍后会详细介绍远程存储的配置（第2.6节：连接DVC到远程存储），但查看初始文件是什么样子会很有用：

```ini
# 全新执行 'dvc init' 后 .dvc/config 的内容
[core]
    remote = # 尚未配置默认远程存储
[cache]
    # 默认缓存设置可能会出现在这里，例如，缓存类型
    # 类型 = 硬链接、符号链接、复制（DVC根据操作系统/文件系统确定默认值）
```

初始时，没有配置远程存储。您需要明确设置一个远程位置（例如S3存储桶、GCS存储桶、Azure Blob存储容器，甚至只是文件系统上的另一个目录），DVC将把实际数据文件推送到那里，以便共享和备份。此配置步骤对于协作以及将数据存储在本地存储之外非常重要。

DVC初始化完毕并且其配置文件已提交到Git，您的项目现在已准备好进行下一步：告诉DVC它应该开始追踪哪些具体的数据文件或目录。

## 参考资料

- [DVC Documentation: Get Started](https://dvc.org/doc/start) — Iterative.ai (2024)
  DVC 安装和初始项目设置的官方指南，包括 `dvc init` 命令和生成的项目结构。
- [Git - Book: Getting Started - Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics) — Scott Chacon, Ben Straub (2014)
  Publisher: Apress
  理解 Git 初始化和基本版本控制的基础资源，这是 DVC 的先决条件。
- [Practical MLOps: How to Take Machine Learning Models from Prototype to Production](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH0sKCpVt6AoOw_48fDDRDx3S2Pp0toW97m4e4vQ8EG63xn2XgbXrkpcmmqD4cMDJj5HaUQkf_8Z1hugFw1fuakAUSoQgztJC3FNe9qeaxJBfyVx9LYk5M8XE9Mgy-1b5DBzNiS9tOa4G1yNm44O5kwaGD5KOAFgaF8gqMqBIE==) — Noah Gift, Alfredo Deza (2021)
  Publisher: O'Reilly Media; Pages: 458
  提供 MLOps 工作流程中数据版本控制的更广泛背景，阐明了 DVC 等工具对机器学习项目的重要性。

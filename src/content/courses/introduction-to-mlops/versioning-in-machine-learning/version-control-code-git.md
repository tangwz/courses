---
course: "introduction-to-mlops"
chapter: "versioning-in-machine-learning"
lesson: "version-control-code-git"
sourceId: 7210
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-3-versioning-in-machine-learning/version-control-code-git"
title: "使用 Git 进行代码版本控制"
description: "关于使用 Git 进行机器学习代码版本控制、分支管理以及团队协作的基础指南。"
order: 2
plots: []
sourceHash: "bd9eea22bc67314cc7418c005502034952c204572c70bfe5fed2fdfebb11c4d9"
sourceCorrections: []
---

虽然机器学习 (machine learning)模型是核心，但负责预处理数据、定义架构和运行训练过程的源代码则是其必不可少的辅助。就像电影剧本会经过多次修改一样，你的代码也会不断演进。为了有序地管理这种演进，我们使用版本控制，而 Git 就是版本控制的行业标准。

可以把 Git 想象成代码的详尽实验笔记。Git 并不是用 `train_v1.py`、`train_v2_fixed.py` 和 `train_final_for_real.py` 这种名称来保存文件的不同版本，而是提供了一个结构化系统，在任何时间点为整个项目拍摄“快照”。这些快照被称为提交（commits），它们记录了每一次更改的完整历史，让你能够随时回退到代码库的任何过去状态。

对于机器学习项目来说，这不仅是为了方便，更是必须的。它为可复现性奠定了基础，能够回答诸如“目前生产环境中运行的模型是使用哪个版本的代码训练的？”等重要问题。

### 为什么 Git 对机器学习 (machine learning)项目非常重要

使用像 Git 这样的版本控制系统会带来几个直接的好处，这在机器学习生命周期中尤为重要。

- **项目完整历史记录：** 每次保存快照（提交）时，你都需要编写一条消息，说明你更改了内容以及原因。这为你的项目开发创建了一个详尽且可搜索的日志。如果某次更改引入了错误或降低了模型性能，你可以轻松地找到问题所在，并恢复到之前的正常版本。
- **使用分支安全地进行实验：** 机器学习本质上是实验性的。你可能想尝试一种新的特征工程方法、不同的算法或新的超参数 (parameter) (hyperparameter)值。Git 允许你创建一个“分支”，它是代码库的一个独立副本。你可以在这个分支上自由尝试，而不会给主代码带来风险。如果实验成功，你可以将更改合并回主项目。如果失败了，直接删除该分支即可。
- **简化协作：** 团队合作时，Git 允许多人同时处理同一套代码。它提供了合并不同开发者代码更改的机制，并在两人修改了文件的同一部分时管理冲突。这通过托管在 GitHub、GitLab 或 Bitbucket 等平台上的远程仓库来实现。

### Git 核心概念

要有效使用 Git，需要理解几个核心思路。虽然 Git 是一个功能强大的工具，但只要掌握其中几个组成部分，就能应对大部分场景。

#### 仓库：项目的家

Git **仓库**（或简称 “repo”）是一个包含项目代码和一个名为 `.git` 的隐藏子文件夹的目录。这个 `.git` 文件夹是 Git 存储项目完整历史记录的地方，包括所有的提交和分支。你可以在电脑上拥有一个本地仓库，也可以在服务器（如 GitHub）上存储一个远程仓库，以便备份和协作。

#### 提交：时间点的快照

**提交**（commit）是仓库中所有文件在特定时间点的快照。每个提交都有一个唯一的标识符（哈希值），并与之前的提交相连，形成一个历史链。提交过程分为两步：首先，选择要包含的更改（这称为“暂存”，staging），然后将其保存为带有说明消息的提交。这个过程鼓励你将相关的更改整理成逻辑单元。

#### 分支：并行的开发线路

A **分支**（branch）代表一条独立的开发线路。默认情况下，仓库从一个分支开始，通常命名为 `main` 或 `master`。这个分支通常存放稳定且可用于生产的代码。当你想要开发新功能或进行实验时，可以创建一个从 `main` 分出的新分支。这样你就可以独立工作。一旦工作完成并经过测试，你可以将功能分支**合并**（merge）回 `main` 分支，整合你的新代码。

> 一个典型的 Git 分支工作流。新功能的开发在独立的分支 (`feature/add-metric`) 上进行，随后再合并回稳定的 `main` 分支。

### 机器学习 (machine learning)的基本 Git 工作流

以下是在个人工作流中为代码进行版本控制时会用到的基本命令。

1. **初始化仓库：** 在终端进入项目文件夹并运行此命令。这会创建隐藏的 `.git` 目录，并将项目转变为 Git 仓库。

   ```bash
   git init
   ```
2. **暂存更改：** 创建或修改某些文件（如 `preprocess.py`）后，你需要告诉 Git 你想跟踪这些更改。这被称为暂存。

   ```bash
   # 暂存特定文件
   git add preprocess.py

   # 或暂存当前目录下的所有已更改文件
   git add .
   ```
3. **提交更改：** 暂存更改后，通过提交将其保存到项目历史中。`-m` 标志允许你提供说明消息。

   ```bash
   git commit -m "添加初始数据预处理脚本"
   ```
4. **创建并切换到新分支：** 在开始新实验之前，创建一个新分支。`checkout -b` 命令会创建一个新分支并立即切换到该分支。

   ```bash
   git checkout -b experiment/new-feature-scaling
   ```

   现在，你进行的任何提交都将在这个新分支上，而不会影响 `main` 分支。
5. **合并分支：** 实验成功后，切换回主分支并合并实验分支的更改。

   ```bash
   # 切换回主分支
   git checkout main

   # 将实验分支的更改合并到主分支
   git merge experiment/new-feature-scaling
   ```

### 不应该用 Git 跟踪的内容

Git 擅长对基于文本的文件进行版本控制，如 Python 脚本（`.py`）、配置文件（`.yaml`）和文本文件（`.md`）。然而，它并不适合跟踪大型二进制文件。这类文件包括：

- **大型数据集**（CSV、Parquet 文件、图像）
- **训练好的模型文件**（`.pkl`、`.h5`、`.pt`）
- **临时文件**和特定环境的文件夹（例如 `__pycache__`）

尝试在 Git 中存储这些大文件会使仓库迅速膨胀，导致下载和管理变慢。为了告诉 Git 忽略特定的文件和文件夹，可以在项目的根目录下创建一个名为 `.gitignore` 的特殊文件。该文件中的每一行都指定了要忽略的文件或文件夹模式。

一个典型的 Python 机器学习 (machine learning)项目的 `.gitignore` 如下所示：

```
# 忽略 Python 缓存
__pycache__/

# 忽略虚拟环境文件夹
.venv/
env/

# 忽略大型数据文件
data/
*.csv

# 忽略训练好的模型
models/
*.pkl
```

通过明确地忽略数据和模型，你实际上是承认了它们需要不同的版本控制策略。Git 负责可复现性中的“代码”部分，为后续章节中使用其他工具处理“数据”和“模型”部分做好了准备。

## 参考资料

- [Git - Documentation](https://git-scm.com/doc) — Git Community (N/A)
  提供所有 Git 命令和功能的官方参考材料和指南。
- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  一本全面的 Git 指南，从基础操作到高级技术，适合所有技能水平。
- [Designing Machine Learning Systems: An Iterative Approach to Development, Deployment, and Maintenance](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  在构建和维护可靠机器学习系统的更广阔背景下，讨论代码版本控制。

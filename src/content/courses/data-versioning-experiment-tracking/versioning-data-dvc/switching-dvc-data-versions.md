---
course: "data-versioning-experiment-tracking"
chapter: "versioning-data-dvc"
lesson: "switching-dvc-data-versions"
sourceId: 4072
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-2-versioning-data-dvc/switching-dvc-data-versions"
title: "在不同数据版本间切换"
description: "了解 Git 提交和 DVC 如何配合使用以检出特定版本的数据。"
order: 7
plots: []
sourceHash: "41881578aea37c39a2bd63856d69f2b18befb14df1eed388c906052c22dd9f68"
sourceCorrections: []
---

DVC 使用 `.dvc` 元数据文件跟踪数据文件，并将这些元数据文件提交到 Git。数据内容通过 `dvc push` 和 `dvc pull` 在本地机器和远程存储之间同步。DVC 与 Git 的这种集成使得切换不同版本的数据与代码一同进行成为可能，这是其主要功能之一。

机器学习 (machine learning)项目经常涉及使用不同数据集或同一数据集的不同版本（例如，清理前和清理后，或使用不同特征集）进行试验。能够可靠地将项目数据恢复到特定的历史状态，对于再现性、调试和结果比较非常重要。DVC 与 Git 配合使用，使这个过程变得简单明了。

### Git 与 DVC 在版本管理上的配合

核心思想是：**Git 跟踪您的代码演变*以及*指向数据的指示，而 DVC 管理与这些指示关联的实际数据文件。**

当您运行 `dvc add data.csv` 然后 `git add data.csv.dvc` 时，您是在告诉 Git 跟踪*元数据文件*（`data.csv.dvc`）。这个文件包含诸如 `data.csv` 实际内容的哈希值等信息，但不是内容本身。当您进行 Git 提交时，您就捕获了代码的状态以及当时 `.dvc` 文件的特定版本。

可以这样理解：

- **Git 提交：** 您的代码库的快照，以及一份指令列表（`.dvc` 文件），指定哪个版本的数据与该代码快照对应。
- **DVC 缓存/远程存储：** 您的所有不同数据文件的实际存储位置，通过它们的内容哈希值进行索引。

### 数据版本切换流程

在项目历史中关联到特定时间点的数据版本之间切换，涉及一个两步过程：使用标准 Git 命令，然后使用一个 DVC 命令：

1. **使用 `git checkout`：** 检出对应所需项目状态的 Git 提交、分支或标签。此操作会更新 Git 在您的工作目录中跟踪的文件，包括 `.dvc` 文件。您的代码将恢复到该提交的状态，而且重要的一点是，`.dvc` 文件现在将指向与该提交关联的数据版本。
2. **使用 `dvc checkout`：** 在 Git 更新 `.dvc` 文件后，运行 `dvc checkout`。此命令指示 DVC 读取您工作目录中*当前*的 `.dvc` 文件并同步实际数据文件。DVC 将在其本地缓存中找到对应的数据内容（由 `.dvc` 文件中的哈希值识别），并将其放入您的工作空间，如果需要则覆盖以前的版本。

让我们用一个例子来说明。假设您的 Git 历史中有两个重要的提交：

- `commit-A`：使用了 `data/raw_images_v1.zip`。此提交中的 `data/raw_images_v1.zip.dvc` 文件指向此特定数据集版本的哈希值。
- `commit-B`：将数据集更新到 `data/raw_images_v2.zip`（可能在一些清理或添加之后）。此提交中的 `data/raw_images_v1.zip.dvc` 文件指向这个*新*数据集版本的哈希值。

假设您当前正在 `commit-B` 对应的状态下工作，但您想回溯并检查结果或使用 `commit-A` 中的数据重新运行分析。

**步骤 1：切换 Git 历史**

```bash
# 确保您当前的工作已保存或已提交
git status

# 检出上一个提交
git checkout commit-A
```

此时，Git 会更新您的代码库并恢复 `commit-A` 中存在的 `data/raw_images_v1.zip.dvc` 文件。然而，您工作空间中实际的 `data/raw_images_v1.zip` 文件可能仍然是 `v2` 版本（或者如果您刚刚克隆了仓库，甚至可能缺失）。您的工作空间数据会暂时与 `.dvc` 元数据文件不同步。

**步骤 2：使用 DVC 同步数据**

```bash
# 告诉 DVC 根据当前的 .dvc 文件更新工作空间数据
dvc checkout data/raw_images_v1.zip.dvc
# 或者，更新仓库中所有由 DVC 跟踪的文件：
# dvc checkout
```

DVC 读取 `data/raw_images_v1.zip.dvc`（它现在指向 `v1` 哈希），在其缓存中找到对应的 `v1` 数据，并将其放入您的 `data/` 目录，替换 `v2` 内容。您的工作空间现在准确反映了项目在 `commit-A` 时的状态，包括代码和数据。

要返回 `commit-B` 的状态，您只需反向操作：

```bash
git checkout commit-B
dvc checkout
```

### 如果数据不在本地缓存中怎么办？

`dvc checkout` 命令依赖于所需的DVC数据版本存在于本地缓存中（通常位于 `.dvc/cache`）。如果您检出的 Git 提交所关联的数据尚未从远程存储下载（例如，在克隆仓库或切换到非常旧的分支之后），`dvc checkout` 可能会报告数据缺失。

在这种情况下，您首先需要使用 `dvc pull` 从您配置的 DVC 远程存储中获取所需数据：

```bash
# 在 git checkout commit-A 之后
# 尝试同步工作空间数据
dvc checkout

# 如果由于缓存缺失而失败，则从远程拉取：
dvc pull data/raw_images_v1.zip.dvc
# 或者拉取当前提交的所有数据：
# dvc pull

# 现在数据已本地缓存，因此 checkout 将会成功
# (注意：dvc pull 通常会隐式执行 checkout，
# 但运行 dvc checkout 可确保工作空间与缓存同步)
dvc checkout
```

`dvc pull` 从远程获取与*当前* `.dvc` 文件对应的数据到本地缓存，并且通常也会直接更新工作空间文件。之后运行 `dvc checkout` 是一个好习惯，可以确保工作空间根据 Git 检出的 `.dvc` 文件与缓存正确同步。

通过将 `git checkout` 用于管理代码和元数据历史，与 `dvc checkout`（以及在需要时使用 `dvc pull`）用于同步关联的大型数据文件结合起来，您获得了一种强大而可靠的方法来浏览机器学习 (machine learning)项目的完整历史。这种精确恢复过去状态的能力对于再现性和协作开发而言非常根本。

## 参考资料

- [DVC Documentation](https://dvc.org/doc) — Iterative, Inc. (2024)
  数据版本控制（DVC）命令和概念的官方参考，包括用于将数据版本与Git同步的`dvc checkout`和`dvc pull`。
- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  Git的权威指南，涵盖了`git checkout`等核心概念和命令，这对于代码和元数据版本切换至关重要。
- [Designing Machine Learning Systems: An Iterative Approach to Development, Deployment, and Maintenance](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  讨论构建稳健机器学习系统的最佳实践，提供了数据管理和重现性挑战的背景，而数据版本控制正是解决这些挑战的关键。

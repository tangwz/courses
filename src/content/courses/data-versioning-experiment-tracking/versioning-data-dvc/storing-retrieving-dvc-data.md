---
course: "data-versioning-experiment-tracking"
chapter: "versioning-data-dvc"
lesson: "storing-retrieving-dvc-data"
sourceId: 4068
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-2-versioning-data-dvc/storing-retrieving-dvc-data"
title: "数据版本的存储与获取"
description: "了解用于在远程存储中管理数据的 'dvc push' 和 'dvc pull' 命令。"
order: 5
plots: []
sourceHash: "c7f7f834d4e17df47cad61c2f0ede9436e081a148bb4b7003cf923af695ae8fe"
sourceCorrections: []
---

DVC 追踪数据文件和目录，并为每个文件或目录创建一个对应的 `.dvc` 文件。这个小的文本文件包含元数据，例如数据的哈希值，随后被提交到 Git。然而，实际的大数据文件不存储在 Git 中；它存储在 DVC 的缓存中（通常位于项目目录下的 `.dvc/cache` 目录中）。Git 会通过 `.gitignore` 文件忽略这些数据文件，该文件在 DVC 初始化时会自动更新。

此时，数据仅存在于您的本地机器上。为了与协作者共享此数据，或从不同环境（例如用于训练的云服务器）中获取，您需要将其上传到一个共享位置，也就是 DVC 远程存储。

### 使用 `dvc push` 上传数据

`dvc push` 命令用于将数据从您的本地 DVC 缓存上传到已配置的远程存储。它检查由*当前* Git 提交（或如果更改尚未提交，则为当前工作目录）追踪的 `.dvc` 文件，如果远程存储中尚不存在，则从本地缓存（`.dvc/cache`）上传相应的数据文件到远程。

这样理解：

1. 您修改或添加数据。
2. 您运行 `dvc add <your_data>` 以使用新的数据哈希更新 `.dvc` 文件。
3. 您运行 `git add <your_data.dvc>` 和 `git commit -m "Update dataset"` 以在 Git 中记录指向此数据版本的*指针*。
4. 您运行 `dvc push` 以将与该哈希关联的*实际数据内容*上传到远程存储。

```bash
# 添加数据后的工作流程示例
git add data/images.dvc
git commit -m "Add processed images v1.1"

# 现在，将实际数据文件推送到已配置的远程
dvc push
```

此命令通常会输出正在推送的文件数量。DVC 效率很高；它只上传哈希值在远程存储中尚不存在的文件，避免重复上传。如果您在没有更改的情况下再次运行 `dvc push`，它会迅速判断所有内容都已同步。

### 使用 `dvc pull` 获取数据

相反地，`dvc pull` 命令从远程存储下载数据文件到您的本地 DVC 缓存中，并根据当前存在的 `.dvc` 文件将其正确放置到您的工作区中。

这在以下情况中很实用：

- 协作者推送了新版本的数据，您需要获取它。
- 您切换到使用了不同数据版本的 Git 分支或提交。
- 您正在新机器上设置项目。

获取与特定代码版本关联的数据的常见工作流程是：

1. 使用 Git 切换到所需的提交或分支：`git checkout <branch_name_or_commit_hash>`。这会将您工作区中的 `.dvc` 文件更新为与历史记录中的该特定点匹配。
2. 运行 `dvc pull`。DVC 读取更新后的 `.dvc` 文件，确定所需的数据哈希，检查本地缓存中是否存在相应数据，并从远程存储下载所有缺失的文件。然后它会将这些文件链接到您的工作区中。

```bash
# 切换到可能使用不同数据的分支
git checkout experiment-new-feature

# 拉取此分支上 .dvc 文件对应的数据
dvc pull
```

`dvc pull` 确保您的工作区包含该 Git 提交发生时使用的精确数据版本。与 `dvc push` 类似，它只下载本地缺失的文件，使其效率很高。

### Git-DVC 同步流程

命令 `dvc push` 和 `dvc pull` 是在本地缓存和远程存储之间同步 DVC 管理的大数据文件的主要方式。它们与 Git 协同工作：Git 管理轻量级的 `.dvc` 指针文件，确定*哪个*数据版本属于特定的代码提交，而 DVC 则负责存储和传输这些指针引用的*实际*数据内容。

> 展示了本地工作区、Git 仓库、DVC 缓存、远程 Git 仓库和 DVC 远程存储之间使用 Git 和 DVC 命令的交互流程。

通过配置远程存储（下一节会介绍）并在提交 `.dvc` 文件更改后持续使用 `dvc push`，以及在切换不同代码版本后使用 `dvc pull`，您便能建立一个可靠的流程，以便与代码一起管理数据版本。这确保了任何切换到特定提交的人都可以获取与之关联的精确数据，大大提升了机器学习 (machine learning)项目的可复现性。

## 参考资料

- [DVC User Guide](https://dvc.org/doc/user-guide) — Iterative, Inc. (2024)
  介绍DVC核心命令和概念，如数据缓存、远程存储和Git集成。
- [MLOps Engineering at Scale](https://www.oreilly.com/library/view/mlops-engineering-at/9781098132845/) — Carl Osipov (2022)
  Publisher: O'Reilly Media
  提供构建和运行机器学习系统的指南，包括数据管理和版本控制实践。

---
course: "getting-started-with-git"
chapter: "branching-merging-basics"
lesson: "making-commits-on-branch"
sourceId: 1017
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-4-branching-merging-basics/making-commits-on-branch"
title: "在分支上进行提交"
description: "了解提交是如何添加到当前活跃分支的，不影响其他分支。"
order: 5
plots: []
sourceHash: "cab5c98604f2697a9ba3dcc41ff790678b1c2bb2f042e10c66ce2fbb3a1a10dd"
sourceCorrections: []
---

使用 `git branch <branch-name>` 创建分支，以及使用 `git switch <branch-name>`（或旧的 `git checkout <branch-name>`）切换您的工作焦点到另一个分支，是 Git 中的基本操作。切换分支时，特殊的 `HEAD` 指针会改变其指向的提交，并更新您工作目录中的文件以匹配该提交的快照。

那么，当您切换到一个新分支后进行更改时会发生什么？流程与您在第2章学到的标准 Git 工作流完全相同：您修改文件，使用 `git add` 暂存这些更改，然后使用 `git commit` 记录一个快照。

这里重要的一点是*上下文 (context)*。当您运行 `git commit` 时，Git 会创建一个新的提交对象，其父提交是 `HEAD` 在此提交*之前*所指向的提交。由于 `HEAD` 现在指向您当前分支的最新提交，新提交只会延长*该分支*的历史。其他分支完全不会察觉到这些新提交，也不会受到影响，直到您决定合并它们（我们很快就会讲到）。

可以这样理解：创建并切换到一个新分支，就为您的项目创建了一条并行的历史线。在此新分支上进行的提交会将其事件添加到其特定的历史线上，而原始的历史线（例如 `main` 分支）则保持不变。

让我们来看一个示意图。假设您的 `main` 分支上有几个提交。然后，您创建一个名为 `feature` 的新分支并切换到它。

> 最初，`main` 和新 `feature` 分支都指向同一个提交 (C1)。`HEAD` 指向 `feature`，因为这是您切换到的分支。

现在，如果您在 `feature` 分支上进行更改、添加并提交：

```bash
# (假设您当前在 'feature' 分支上)
# 编辑一个文件，例如，向 'feature_details.txt' 添加内容
echo "正在开发新功能" > feature_details.txt
git add feature_details.txt
git commit -m "开始开发功能 X"
```

Git 创建了一个新提交（我们称之为 C2），其父提交是 C1。`feature` 分支指针向前移动，指向这个新提交 C2。`main` 分支指针*没有移动*。

> `feature` 分支已推进到提交 C2，而 `main` 仍停留在 C1。`HEAD` 仍然跟随 `feature`。

如果您切换回 `main` 分支 (`git switch main`) 并进行一个不同的提交（例如 C3，修复一个错误），那个提交将会在 `main` 分支的历史线上添加到 C1 *之后*，与 `feature` 分支上的 C2 完全独立。

```bash
git switch main
# 已切换到 'main' 分支
# 编辑一个不同的文件，例如，修复 'main_code.py' 中的问题
echo "print('bug fixed!')" > main_code.py
git add main_code.py
git commit -m "修复 main 上的严重错误"
# 现在 HEAD 指向 main，而 main 指向 C3
```

现在的历史记录看起来是这样，在 C1 之后有两条独立的发展线：

> 提交历史已分叉。C2 仅存在于 `feature` 分支上，而 C3 仅存在于 `main` 分支上。`HEAD` 指向 `main`。

您可以使用 `git log` 加上一些有用的选项来查看这种分支结构：

```bash
git log --oneline --graph --all
```

- `--oneline`：将每个提交显示为一行（提交哈希值和消息）。
- `--graph`：绘制一个基于文本的图表，显示分支和合并历史。
- `--all`：显示所有分支的提交，而不仅仅是当前分支的提交。

输出将以图形方式显示分叉情况，类似于上面的示意图。

这种隔离是 Git 分支强大之处的根本。您可以在自己的分支上开发新功能或修复错误，而无需担心会破坏主代码库的稳定性或干扰其他分支上的工作。每个分支都记录了一系列独立的提交，这些提交都基于一个共同的起点。稍后，您将学习如何使用 `git merge` 将这些独立的历史记录重新组合。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon, Ben Straub (2014)
  Publisher: Apress
  涵盖了Git分支的基本机制、HEAD指针的作用以及新提交如何扩展当前分支的历史，尤其是在第3.1和3.2章中，包括本节中展示的可视化内容。
- [Git - git-commit Documentation](https://git-scm.com/docs/git-commit) — The Git Development Community (2024)
  解释`git commit`命令、其选项以及它如何创建新的提交对象并推进当前分支指针的官方文档。
- [Git - git-branch Documentation](https://git-scm.com/docs/git-branch) — The Git Development Community (2024)
  详细说明`git branch`命令的官方文档，该命令用于创建、列出和删除分支，为分支概念提供上下文。

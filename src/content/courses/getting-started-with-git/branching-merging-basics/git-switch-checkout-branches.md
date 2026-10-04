---
course: "getting-started-with-git"
chapter: "branching-merging-basics"
lesson: "git-switch-checkout-branches"
sourceId: 1015
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-4-branching-merging-basics/git-switch-checkout-branches"
title: "在分支间切换 (git checkout 或 git switch)"
description: "了解如何使用 `git checkout` 或更新的 `git switch` 命令将你的工作目录切换到不同分支。"
order: 3
plots: []
sourceHash: "d4a9fb5c2def10eca436fcb7c7e60e32f29b314912464485372ea2f5a6d26ac6"
sourceCorrections: []
---

切换分支是 Git 中的一项基本操作，它允许开发者将工作环境移动到特定的开发线上。例如，使用 `git branch <branch-name>` 创建新分支只是建立一个新指针，它不会自动将你的开发环境移动到新的工作线上。你的工作目录始终反映你当前所在分支的状态，例如在切换之前通常默认为 `main` 分支。

要将当前工作环境更改到另一个分支，你需要“切换”到该分支。这个操作告诉 Git：“我现在想在*这个*分支上工作。”当你切换分支时，Git 会执行两项主要工作：

1. 它会更新一个名为 `HEAD` 的特殊指针，使其指向你所切换到的分支的最新提交。`HEAD` 通常指向你当前所在分支的*名称*。
2. 它会修改你的工作目录中的文件，使其与目标分支最新提交中存储的快照匹配。

可以这样想：就像把多个项目草稿放在不同的桌子上。创建分支就像设置一张新桌子 (`git branch new-feature`)。切换分支就像实际地移动过去在新桌子上工作 (`git checkout new-feature` 或 `git switch new-feature`)，而桌子上的文件（纸张）也会随之改变，以反映*那个*草稿的状态。

### 使用 `git checkout`

过去，切换分支的主要命令一直是 `git checkout`。它的用法很简单：

```sh
git checkout <branch-name>
```

例如，如果你之前使用 `git branch fix-login-bug` 创建了一个名为 `fix-login-bug` 的分支，你会这样切换到它：

```sh
# 假设你当前在 'main' 分支上
git checkout fix-login-bug
```

运行此命令后，Git 会提示：

```
切换到分支 'fix-login-bug'
```

现在，你进行的任何新提交都将添加到 `fix-login-bug` 分支，独立于 `main` 分支或任何其他分支推进其历史。你的工作目录文件将反映它们在 `fix-login-bug` 最新提交时的状态（如果你是刚创建它，那么将与你从中分支出来的提交相同）。

### 介绍 `git switch` (现代方式)

`git checkout` 命令功能很全面。除了切换分支，它还可以用来从不同提交或暂存区（索引）恢复工作目录中的文件。这种功能重叠有时会让初学者感到困惑。

为解决此问题，Git 的新版本（从 2.23 开始）引入了两个更明确的命令：`git switch` 用于更改分支，`git restore` 用于更改文件。`git switch` 命令专门用于分支操作，使其目的更明确。

要使用 `git switch` 切换到我们的 `fix-login-bug` 分支，你会运行：

```sh
# 假设你当前在 'main' 分支上
git switch fix-login-bug
```

结果与使用 `git checkout fix-login-bug` 完全相同：

```
切换到分支 'fix-login-bug'
```

你的 `HEAD` 现在指向 `fix-login-bug`，并且你的工作目录也相应地更新了。

### 切换的可视化

设想一下，你的仓库历史最初是这样的，`HEAD` 指向 `main` 分支：

> `HEAD`指向`main`，`main`指向提交C3。

现在，你从 `main` 创建一个新分支 `feature-x`，使用 `git branch feature-x`。`HEAD` 指针尚未移动。

> `feature-x`已创建，两个分支都指向C3，`HEAD`仍停留在`main`。

当你运行 `git switch feature-x` (或 `git checkout feature-x`) 时，Git 会更新 `HEAD`：

> `HEAD`现在指向`feature-x`。你的工作目录反映了提交C3的状态（最初）。后续提交将推进`feature-x`。

### 重要事项

- **未提交的更改：** Git 会尽力保护你的工作。如果你的工作目录或暂存区中有修改与你尝试切换到的分支冲突（意味着切换会覆盖你的更改），Git 通常会阻止切换并警告你。你通常需要提交你的更改、暂存它们（稍后介绍的内容），或者丢弃它们，然后才能切换分支。
- **创建并切换：** `git checkout` 和 `git switch` 都提供选项，可以在一步之内创建新分支并立即切换到它。
  - `git checkout -b <new-branch-name>`
  - `git switch -c <new-branch-name>`
    这是一个常用快捷方式，它结合了 `git branch <name>` 和 `git switch <name>`。

### 建议

尽管 `git checkout` 在切换分支方面运作良好，并且在较旧的教程和项目中广泛使用，但如果你的 Git 版本支持 `git switch`，通常建议使用它以提高清晰度。它使你的命令意图明确：你正在将你的工作环境切换到另一个分支。

能够快速切换于不同的开发线之间，对于发挥 Git 的分支能力很基本。你现在可以在一个分支上开始一个新功能，切换到另一个分支修复紧急错误，然后再切换回你的功能分支，保持工作隔离，直到你准备好合并它。

## 参考资料

- [git-switch(1) Manual Page](https://git-scm.com/docs/git-switch) — Git Project (2024)
  描述了用于切换分支的专用命令，该命令在Git 2.23中引入，详细说明了其目的和分支管理选项。
- [git-checkout(1) Manual Page](https://git-scm.com/docs/git-checkout) — Git Project (2024)
  详细介绍了多功能的 `checkout` 命令，解释了其在切换分支、创建新分支和恢复文件方面的功能，突出了其多种用途。
- [3.2 Git Branching - Basic Branching and Merging](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) — Scott Chacon, Ben Straub (2014)
  Publisher: Apress
  提供了Git分支模型、`HEAD`指针、`checkout`操作以及分支如何在仓库中演变的基本解释。

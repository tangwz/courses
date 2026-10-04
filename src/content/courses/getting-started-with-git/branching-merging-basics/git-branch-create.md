---
course: "getting-started-with-git"
chapter: "branching-merging-basics"
lesson: "git-branch-create"
sourceId: 1014
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-4-branching-merging-basics/git-branch-create"
title: "创建新分支 (git branch)"
description: "学习如何使用 `git branch <branch-name>` 命令创建新分支。"
order: 2
plots: []
sourceHash: "af06f3128a9ab8f7f14ac5e3272c06f43a53a778986d8838aba81fb6accdde09"
sourceCorrections: []
---

同时处理不同任务，比如开发一个新功能的同时修复一个独立的错误，需要一种方法来隔离这些工作流程。Git通过分支提供了这种隔离。可以将分支看作项目历史中可移动的书签或指针。当你想开始一条新的开发线时，就创建一个新分支。

创建新分支的命令是 `git branch` 后面跟着你想给分支起的名称。

```bash
git branch <branch-name>
```

例如，如果你要开始为一个新网站的头部内容工作，你可能会创建一个名为 `update-header` 的分支：

```bash
git branch update-header
```

运行这个命令会发生什么？Git会创建一个新的指针（本例中名为 `update-header`），它指向你当前所在的*完全相同的提交*（即 `HEAD` 指针所在的位置）。理解这一点很重要：**这个命令只创建分支；它不会切换到该分支**。你的工作目录文件保持不变，并且你仍然在你运行命令之前的那个分支上。

设想你的项目历史看起来像这样，其中 `main` 是你的主要分支，`HEAD` 表示你当前的位置：

> 创建新分支前的仓库状态。`HEAD` 指向 `main`，而 `main` 指向最新的提交 `C3`。

现在，如果你运行 `git branch update-header`，状态会变成：

> 运行 `git branch update-header` 后。一个新的指针 `update-header` 被创建，它指向与 `main` 相同的提交 `C3`。`HEAD` 仍指向 `main`。

请注意，现在存在一个新指针 `update-header`，并指向提交 `C3`。然而，`HEAD` 仍指向 `main`，这表明 `main` 仍然是你当前活跃的分支。

### 分支命名规范

虽然Git允许分支使用几乎任何名称（除了包含空格或某些特殊字符的名称），但使用有意义的名称是个好习惯。常见的规范包括：

- 使用连字符分隔单词（例如：`fix-login-bug`）。
- 使用 `feature/`、`bugfix/` 或 `hotfix/` 等前缀来对分支进行分类（例如：`feature/user-profiles`，`bugfix/issue-451`）。

这些规范让你和你的合作者仅通过查看分支名称就能更容易地理解它们的用途。

创建分支为独立开发创造了条件。下一步，我们将在下一节中介绍，是实际切换到你的新分支并开始在其上工作。

## 参考资料

- [Git Reference Manual - git-branch](https://git-scm.com/docs/git-branch) — Git Development Community (Current)
  `git branch` 命令的官方参考资料，详述其用法和各种选项。
- [Pro Git, Chapter 3: Git Branching - Branches in a Nutshell](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  一个基础章节，解释Git分支作为指针以及它们的创建方式，配有实用的图表。
- [Git Branching](https://www.atlassian.com/git/tutorials/git-branch) — Atlassian (Current)
  Publisher: Atlassian
  一份关于理解和创建Git分支的实用指南，提供清晰的解释，适合新用户。

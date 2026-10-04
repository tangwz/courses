---
course: "getting-started-with-git"
chapter: "working-with-remote-repositories"
lesson: "git-pull-command"
sourceId: 1030
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories/git-pull-command"
title: "从远程拉取更改 (git pull)"
description: "了解 `git pull` 如何将从远程拉取更改和将其合并到当前本地分支结合起来。"
order: 8
plots: []
sourceHash: "2877fe94fd0e386a97ae5523055af8d70aa4e15ab7ab4c601afda53c2a939639"
sourceCorrections: []
---

`git fetch` 命令从远程仓库下载新数据，但不会自动将任何数据合并到你的工作文件中。它会更新你的远程跟踪分支（如 `origin/main`），让你能看到远程仓库中有什么改变，在决定如何纳入它们之前。

虽然先拉取再手动合并能提供精细的控制，但一个非常常见的工作流程是简单地使用其对应的远程分支的最新更改来更新你当前的本地分支。为此，Git 提供了 `git pull` 命令。

可以把 `git pull` 看作一个方便的快捷方式。在最基本的形式中，它连续执行两个操作：

1. `git fetch`: 它从远程仓库下载最新更改，更新你本地仓库中的远程跟踪分支（例如，`origin/main`）。
2. `git merge`: 它立即尝试将对应的已下载的远程跟踪分支合并到你当前的本地分支中。

### 基本用法

如果你的当前本地分支被设置来跟踪一个远程分支（这在使用 `git clone` 时会自动完成，也可以手动配置），那么只需运行 `git pull` 通常就足够了：

```bash
# 确保你位于想要更新的分支上，例如 main
git switch main

# 从跟踪的远程分支（通常是 origin/main）拉取更改
git pull
```

Git 会根据与你当前本地分支（本例中是 `main`）关联的跟踪信息，推断要从哪个远程仓库拉取（通常是 `origin`）以及要合并哪个远程分支。

如果你需要明确指定，或者你的分支没有跟踪远程分支，你可以指定远程名称和分支名称：

```bash
# 从远程仓库 'origin' 的 'main' 分支拉取更改
git pull origin main
```

这个命令明确告诉 Git：“从名为 `origin` 的远程仓库拉取 `main` 分支，然后将 `origin/main` 合并到我当前的本地分支。”

### 了解底层的拉取与合并操作

想象一下 `git pull` 的作用会有帮助。想象你的本地 `main` 分支和远程 `origin/main` 分支略有分歧：

> 你的本地 `main` 分支有提交 `C`，而远程 `origin/main` 分支有提交 `D`。两者都从提交 `B` 分出。

当你在本地 `main` 分支上运行 `git pull origin main` 时：

1. **拉取 (Fetch):** Git 联系 `origin`，下载提交 `D`，并更新你的本地 `origin/main` 指针使其指向 `D`。此时你的本地 `main` 分支保持不变。
2. **合并 (Merge):** Git 接着运行 `git merge origin/main`。由于你的本地 `main` (在 `C`) 和已拉取的 `origin/main` (在 `D`) 从它们的共同祖先 `B` 分出，Git 执行合并。这会在你的本地 `main` 分支上创建一个新的合并提交（我们称之为 `E`），它结合了 `C` 和 `D` 的更改。

> 在 `git pull` 之后，你的本地 `main` 分支现在指向新的合并提交 `E`，它包含了你之前的本地提交 `C` 和已拉取的远程提交 `D` 的更改。`origin/main` 指针也反映了已拉取的状态 (`D`)。

如果你的本地分支自上次同步以来没有新的提交，并且远程分支只是顺序添加了新的提交，`git pull` 将会执行 *快进合并*。这只是将你的本地分支指针向前移动以匹配远程跟踪分支，而不创建合并提交。

### 处理合并冲突

因为 `git pull` 执行合并操作，如果你的本地分支和远程分支在历史分歧后修改了同一文件的相同行，你可能会遇到合并冲突。

如果发生冲突，`git pull` 将在创建合并提交之前停止，并报告冲突文件。解决这些冲突的过程与第4章中合并本地分支时描述的完全相同：

1. Git 会在受影响的文件中标记 (token)冲突。
2. 你手动编辑文件来解决差异，选择保留哪些更改。
3. 你使用 `git add <file>` 暂存已解决的文件。
4. 你通过运行 `git commit` 完成合并（Git 通常会提供默认的合并提交消息）。

### `pull` 与 `fetch` + `merge` 的比较

那么，你应该使用 `git pull` 还是先 `git fetch` 再 `git merge` 这两步操作呢？

- `git pull`: 当你想快速用远程更改更新当前分支，并且有充分理由相信立即合并是合适的时候，可以使用此命令。它方便日常同步。
- `git fetch` + `git merge`: 当你需要更多控制时使用此方法。先拉取可以让你在决定合并之前检查传入的更改。你可以使用 `git log origin/main..main` 等命令查看远程分支上有哪些提交但本地分支还没有，或者使用 `git diff origin/main` 查看实际的代码差异。如果你有本地更改，或者想在合并远程更新之前仔细审查，这种方法通常更安全。

对于初学者，`git pull` 通常更简单，适合直接更新。随着你变得更熟练，尤其是在复杂的协作项目中，你可能会更喜欢单独拉取和合并所提供的控制。

### 最佳实践

- **干净的工作目录：** 在运行 `git pull` 之前，通常最好确保你的工作目录是干净的（没有未提交的更改）。运行 `git status` 检查。在有未提交更改的情况下拉取有时会使合并过程复杂化。如果你有尚未准备好提交的本地更改，可以考虑在拉取之前使用 `git stash`（这是一个更高级的学习主题）。
- **了解你的分支：** 在拉取之前，请确保你位于正确的本地分支上（`git status` 或 `git branch` 会告诉你）。拉取会将更改合并到你的*当前*分支中。

总之，`git pull` 是用于从远程仓库获取更改并立即合并到当前本地分支的命令。它简化了与协作者保持同步的过程，使其成为处理远程仓库的基本命令。

## 参考资料

- [git-pull Documentation](https://git-scm.com/docs/git-pull) — Git Community (2024)
  《git pull》命令的官方文档，详细说明其用法、选项以及与《git fetch》和《git merge》的交互。
- [git-merge Documentation](https://git-scm.com/docs/git-merge) — Git Community (2024)
  《git merge》命令的官方文档，解释了其不同的合并策略、冲突解决及其作为《git pull》一部分的作用。
- [Pro Git (2nd Edition)](https://git-scm.com/book/en/v2/Git-Branching-Remote-Branches) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  Git官方书籍，提供了Git概念的全面解释，包括《git pull》在远程工作流和分支策略中的作用。链接指向“远程分支”章节。

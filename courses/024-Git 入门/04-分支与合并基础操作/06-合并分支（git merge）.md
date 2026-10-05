# 合并分支（git merge）

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-4-branching-merging-basics/git-merge-command)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将独立开发的代码线，通常位于不同分支上，合并回主分支（例如 `main` 或 `master`），是协作项目中一个常见的目标。这种组合不同分支的历史和更改的过程称为**合并**。Git 提供了 `git merge` 命令，专门用于此目的。

合并会将源分支（例如您的功能分支）的更改与目标分支（例如 `main` 分支）的历史结合起来。Git 在自动组合这些更改方面表现出色。

### `git merge` 命令

要执行合并，您首先需要切换到您想要将其他分支合并*进来*的分支。这通常是您的主要开发线，例如 `main` 分支。一旦您位于目标分支上，就可以使用 `git merge` 命令，并指定您想要从中合并*出来*的分支的名称。

例如，如果您已经完成了 `new-feature` 分支上的工作，并希望将这些更改整合到您的 `main` 分支中，您可以运行以下命令：

1. 切换到目标分支（`main`）：

   ```bash
   git switch main
   ```

   或者，使用旧命令：

   ```bash
   git checkout main
   ```
2. 将源分支（`new-feature`）合并到当前分支（`main`）：

   ```bash
   git merge new-feature
   ```

### 合并的工作方式

当您执行 `git merge` 时，Git 在后台执行几个步骤：

1. **确定基础：** Git 会回溯两个分支（当前分支和要合并的分支）的历史，以找到最佳共同祖先提交。这是两个分支最初分歧的点。
2. **计算更改：** Git 会确定自该共同祖先以来每个分支上所做的更改。
3. **组合更改：** 它会尝试组合这些更改集。

如果更改发生在文件的不同部分或完全不同的文件中，Git 通常可以自动组合它们而没有任何问题。

### 合并提交

通常，当自两个分支分歧以来，*两个*分支都发生了更改时，Git 会创建一个新的特殊提交，称为**合并提交**。此提交充当项目历史中的一个统一衔接点。它的标志性特征是它有*多个父提交*：一个指向您所在分支（例如 `main`）的最新提交，另一个指向您合并进来的分支（例如 `new-feature`）的最新提交。

此合并提交本身不引入新的文件更改（除非需要解决冲突），而是作为一个记录，表明两个分支的历史在此处合并。

> 合并提交 (M) 组合了来自两个父提交（`main` 分支上的 B，`new-feature` 分支上的 D）的工作，整合了分叉的历史。`main` 标签现在指向 M。

在 `main` 分支上运行 `git merge new-feature` 后，Git 可能会输出消息，表示哪些文件已更改并确认合并提交的创建（如果有必要）。您的 `main` 分支现在包含了 `new-feature` 分支上完成的所有工作。

合并是 Git 中整合工作的一项基本操作。虽然它通常运行顺畅，但有时 Git 会遇到合并分支上的更改相互冲突的情况。我们将在后续章节中讨论 Git 如何处理更简单的合并场景（快进合并）以及如何解决冲突。

## 参考资料

- [git-merge Documentation](https://git-scm.com/docs/git-merge) — Git Community (2023)
  Git `merge` 命令的官方手册页，描述其用法、选项和行为。
- [Pro Git](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress; Pages: Chapter 3.2: Basic Branching and Merging
  一本关于 Git 的全面书籍，其中有一个专门章节解释了基本分支和合并过程。
- [Git Merge Tutorial](https://www.atlassian.com/git/tutorials/using-branches/git-merge) — Atlassian (2023)
  一个清晰实用的教程，通过图示和命令示例解释了 `git merge` 的工作原理。

---

[上一节](05-%E5%9C%A8%E5%88%86%E6%94%AF%E4%B8%8A%E8%BF%9B%E8%A1%8C%E6%8F%90%E4%BA%A4.md) · [下一节](07-%E7%90%86%E8%A7%A3%E5%BF%AB%E8%BF%9B%E5%90%88%E5%B9%B6.md)

---
course: "getting-started-with-git"
chapter: "viewing-history-undoing-changes"
lesson: "git-diff-command"
sourceId: 1004
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-3-viewing-history-undoing-changes/git-diff-command"
title: "比较更改 (git diff)"
description: "了解如何使用 `git diff` 查看工作目录、暂存区和提交之间的差异。"
order: 1
plots: []
sourceHash: "4de513226c69c004192efb4383adc03a8f0b7bfe43892a89d62648e53d954255"
sourceCorrections: []
---

在修改文件后，甚至在用 `git add` 将它们暂存或用 `git commit` 永久保存之前，你通常会想清楚知道自己更改了什么。你是不是不小心删除了某一行？你添加的配置设置正确吗？Git 为此提供了一个专门的命令：`git diff`。这个命令是你查看项目不同状态之间差异的主要工具：你的工作目录、暂存区和提交历史。

### 理解 `git diff` 输出

在我们查看具体的比较之前，我们先来了解 `git diff` 生成的输出。它通常以“补丁”格式显示差异，这是一种表示文本文件之间更改的标准方式。以下是 `git diff` 典型输出的细分说明：

```diff
diff --git a/config.txt b/config.txt
index e69de29..fb8f733 100644
--- a/config.txt
+++ b/config.txt
@@ -1,0 +1,2 @@
+verbose = true
+theme = dark
```

1. `diff --git a/config.txt b/config.txt`: 此标题行表示正在使用 Git diff 格式运行差异比较，比较 `config.txt` 文件的两个版本。`a/` 代表源文件（旧版本），`b/` 代表目标文件（新版本）。
2. `index e69de29..fb8f733 100644`: 此行提供 Git 内部使用的元数据，包括被比较文件版本的哈希标识符和文件模式信息。在进行基本的差异查看时，通常可以忽略此行。
3. `--- a/config.txt`: 此行表示源文件（`a/` 版本）。
4. `+++ b/config.txt`: 此行表示目标文件（`b/` 版本）。
5. `@@ -1,0 +1,2 @@`: 这是“块头”（hunk header）。它总结了文件中特定块（或“hunk”）内的更改。
   - `-1,0` 表示更改从 `a/` 版本（源文件）的第 1 行开始，该块涵盖源文件中的 0 行（因为它为空）。
   - `+1,2` 表示更改从 `b/` 版本（目标文件）的第 1 行开始，该块包含目标文件中的 2 行。
   - 有时你会看到周围未更改的行被包含进来，以便提供上下文 (context)。
6. 以 `+` 开头的行：这些行已在 `b/` 版本（较新状态）中添加。
7. 以 `-` 开头的行：这些行存在于 `a/` 版本中，但在 `b/` 版本中已被移除。
8. 以空格开头的行：这些行未更改，并为实际的修改提供了上下文。

### 比较你的工作目录与暂存区

也许 `git diff` 最常见的用法是查看你在工作目录中所做的但尚未暂存的更改。不带任何参数 (parameter)运行该命令会显示这些未暂存的更改：

```bash
git diff
```

如果你在上次提交之后，*或*在上次使用 `git add` 暂存文件之后，编辑了一个文件（比如 `README.md`），`git diff` 将显示你添加或删除的行，并与当前暂存区中的版本（如果暂存区与该文件的上次提交匹配，则与上次提交的版本）进行比较。

例如，如果你修改了 `README.md` 并运行 `git diff`，你可能会看到如下输出：

```diff
diff --git a/README.md b/README.md
index 1a2b3c4..5d6e7f8 100644
--- a/README.md
+++ b/README.md
@@ -1,4 +1,5 @@
 # My Project
 This is a sample project to demonstrate Git.

-It currently includes basic setup.
+It currently includes basic setup.
+Added a new feature description.
```

这告诉你，行“Added a new feature description.”已添加到你工作目录中的 `README.md`，但尚未暂存。如果你现在运行 `git add README.md`，此更改将移动到暂存区。添加后立即再次运行 `git diff` 将不显示任何输出，因为你的工作目录现在与该文件的暂存区匹配。

### 查看为下次提交暂存的内容

在使用 `git add` 暂存一些更改后，你可能想仔细查看下次提交将包含哪些内容。你可以使用 `--staged`（或 `--cached`）选项将暂存区中的文件与上次提交进行比较：

```bash
git diff --staged
```

或者等效地：

```bash
git diff --cached
```

这个命令在运行 `git commit` 之前进行最终审查时非常有用。它只显示当前已暂存的更改。如果你添加了前面示例中的 `README.md` 更改，`git diff --staged` 现在将显示该添加，而 `git diff`（不带 `--staged`）将为空（假设没有进一步的编辑）。

`git diff` 是一个不可或缺的工具，用于随时了解项目状态。它帮助你在提交前发现错误，核实你的更改，并通常让你了解项目在不同快照之间是如何变化的。在使用 Git 时，你会经常用到它。

## 参考资料

- [git-diff Command Documentation](https://git-scm.com/docs/git-diff) — The Git Community (2024)
  Publisher: The Git Community
  `git diff` 命令及其参数和输出格式的官方完整参考资料。
- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  一本关于 Git 的权威书籍，提供 Git 概念的全面解释，包括对 `git diff` 的详细介绍。

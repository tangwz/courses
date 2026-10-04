---
course: "getting-started-with-git"
chapter: "first-git-repository"
lesson: "git-add-command"
sourceId: 998
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-2-first-git-repository/git-add-command"
title: "将文件添加到暂存区 (git add)"
description: "了解如何使用 `git add` 暂存文件中的更改，为下次提交做准备。"
order: 4
plots: []
sourceHash: "2ae7e3ae8a788cd86e5464f714da35b2649a838d350af2f4666b990821444113"
sourceCorrections: []
---

将工作目录中的文件修改或创建新文件后，Git 需要知道你想在下一次快照（或提交）中包含哪些特定更改。这时就需要使用**暂存区**（有时也称为索引）。可以把暂存区看作一个准备空间，你可以在其中收集所有计划在下一次提交中一同保存的更改。

用于将更改添加到此暂存区的命令是 `git add`。它告诉 Git：“将此文件（或这些文件）的当前状态准备好，以供下次提交。” 这一步很重要，因为它允许你选择性地决定哪些修改将成为项目历史的一部分。你可能做了多次修改，但只想将相关的更改组合成一个单独的、有逻辑的提交。

### 暂存单个文件

使用 `git add` 最常见的方式是指定要暂存的确切文件。如果你修改了名为 `README.md` 的文件并创建了一个新文件 `main.py`，可以这样暂存 `README.md` 中的更改：

```bash
git add README.md
```

如果现在运行 `git status`，你会看到 `README.md` 已经从“未暂存的更改”部分（如果它是新文件且刚刚添加，则为“未跟踪的文件”）移动到“待提交的更改”部分。文件 `main.py` 仍将列为未暂存或未跟踪。

### 暂存多个文件

你可以在 `git add` 命令后面列出多个具体文件来暂存它们：

```bash
git add README.md main.py styles.css
```

此命令会暂存所有这三个列出文件的当前状态。

### 暂存所有更改

通常，你会希望暂存当前工作目录及其子目录中的所有修改和新文件。你可以使用句点 (`.`) 来实现这一点，它代表当前目录：

```bash
git add .
```

此命令会查看从当前目录向下（包括其子目录）的所有文件。它会暂存：

- 对已跟踪文件的修改。
- 新创建的文件。

使用 `git add .` 时要小心。它会添加当前路径中的*所有内容*，因此请确保你没有意外地暂存临时文件、构建输出或本应列在 `.gitignore` 文件（我们之前讨论过）中的敏感信息。

另一个常用选项是 `git add -A`（或 `git add --all`）：

```bash
git add -A
```

此命令会暂存整个仓库中的*所有*更改，无论你当前在哪个目录。这包括修改、新文件和*已删除*的文件。虽然功能强大，但使用时要像使用 `git add .` 一样小心，以避免暂存不想要的更改。

### 流程图示

`git add` 命令是工作目录和暂存区之间的桥梁，它为将更改保存到仓库历史（通过 `git commit` 完成）的最后一步做准备。

> `git add` 命令将更改从工作目录移动到暂存区。随后的 `git commit` 操作会将暂存区的内容保存到仓库。

请记住，`git add` 不会将更改永久保存到历史记录中；它只是做准备。你仍然需要使用 `git commit`（接下来会讲到）来将暂存的快照记录到仓库的时间线中。在一次提交前多次运行 `git add` 是完全正常的；每次执行都会将指定文件的*当前*状态添加到暂存区，覆盖该文件之前暂存的任何版本。

## 参考资料

- [git-add(1) Manual Page](https://git-scm.com/docs/git-add) — Git Developers (2024)
  `git add` 命令的官方文档，详细说明了其用法和选项。
- [Pro Git](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository) — Scott Chacon and Ben Straub (2024)
  Publisher: Apress
  一本权威的 Git 基础书籍，涵盖了暂存区和 `git add` 命令。
- [Git Glossary](https://git-scm.com/docs/gitglossary) — Git Developers (2024)
  官方 Git 术语表，提供了“索引”和“暂存区”等核心 Git 概念的定义。

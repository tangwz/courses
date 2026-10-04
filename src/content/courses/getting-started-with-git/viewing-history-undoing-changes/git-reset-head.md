---
course: "getting-started-with-git"
chapter: "viewing-history-undoing-changes"
lesson: "git-reset-head"
sourceId: 1007
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-3-viewing-history-undoing-changes/git-reset-head"
title: "取消暂存文件 (git reset HEAD <file>)"
description: "学习如何使用 `git reset HEAD <file...>` 从暂存区移除文件。"
order: 4
plots: []
sourceHash: "b0836543706adf8d8dd8f941ebf6db68dbd24e3a009ea134f3dbd766baa7c54e"
sourceCorrections: []
---

Git 工作流程通常涉及在工作目录修改文件，然后使用 `git add` 将特定更改放入暂存区，为下一次提交做准备。但是，有时您可能将文件添加到暂存区后又改变主意。或许您不小心添加了它，或者这些更改还没完全准备好永久记录到项目历史中。

Git 提供了一种“取消暂存”文件的方法，它可以有效地将文件从暂存区移除，同时不丢弃您在工作目录中对实际文件所做的更改。为此的命令是 `git reset HEAD <file>`。

### 理解 `git reset HEAD <file>`

我们来分析这个命令：

- `git reset`：这是一个多功能命令，用于撤销更改。它有不同的模式，但当与 `HEAD` 和文件路径一起使用时，它会专门影响暂存区。
- `HEAD`：在 Git 中，`HEAD` 通常指向当前分支上最新的提交。可以把它看作代表着项目在仓库中最后一次保存的状态。
- `<file>`：这是您想从暂存区移除的文件（或多个文件）的名称。

当您运行 `git reset HEAD <file>` 时，您是在告诉 Git：“对于这个特定的文件，将其在*暂存区*的状态重置为与 `HEAD` 提交中存储的版本一致。”

这里重要的一点是，此命令**只更改暂存区（也称索引）**。它**不会**更改工作目录中文件的内容。如果您在暂存 `<file>` 时对其进行了修改，取消暂存后这些修改仍然会保留；Git 只是将其从计划用于下次提交的更改列表中移除。

### 如何使用

基本语法是：

```bash
git reset HEAD <path/to/your/file>
```

您可以指定多个文件：

```bash
git reset HEAD file1.txt path/to/another/file2.js
```

如果您想取消暂存所有当前已暂存的文件，可以省略文件路径：

```bash
git reset HEAD
```

如果您使用 `git add .` 暂存了多个文件，然后意识到只想用其中一部分更改来创建更集中的提交，这会很有用。

### 示例情况

假设您正在处理一个项目。您修改了两个文件：`index.html` 和 `style.css`。

1. 您决定将两者都暂存以待提交：

   ```bash
   # 检查状态 - 两者都已修改但未暂存
   git status

   # 暂存这两个文件
   git add index.html style.css

   # 再次检查状态 - 两者都已暂存
   git status
   ```

   您的 `git status` 输出现在会在“待提交的更改”下显示 `index.html` 和 `style.css`。
2. 您检查这些更改，并发现 `style.css` 中的修改是实验性的，不应包含在此次提交中。您想取消暂存它，但保留对文件所做的更改。
3. 对特定文件使用 `git reset HEAD`：

   ```bash
   git reset HEAD style.css
   ```
4. 再次检查状态：

   ```bash
   git status
   ```

   现在，`git status` 将显示：

   - `index.html` 在“待提交的更改”下（它仍然已暂存）。
   - `style.css` 在“未暂存的更改”下（它已取消暂存，但 Git 识别出它仍与上次提交的版本不同）。

您的工作目录中的 `style.css` 文件保持在您上次编辑时的样子。您只是将其从暂存区移除了。您现在可以只提交 `index.html`，或者继续处理 `style.css` 并在它准备好后再次暂存。

> 该图示说明了此过程：更改从工作目录添加到暂存区。`git reset HEAD style.css` 将 `style.css` 从暂存区移出，使其在工作目录中保持已修改状态，而 `index.html` 仍保持暂存。随后的提交将只包含 `index.html`。

在完善您的提交时，使用 `git reset HEAD <file>` 是一种常见操作。它让您可以精确控制项目历史中下一次快照包含的内容，使您能够撤销暂存步骤而不会丢失您的工作。

## 参考资料

- [git-reset Documentation](https://git-scm.com/docs/git-reset) — Git contributors (2023)
  为 `git reset` 命令提供权威且全面的参考，详细介绍了其各种模式和选项，包括如何取消暂存文件。
- [Pro Git](https://git-scm.com/book/en/v2/Git-Tools-Reset-Demystified) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  这本免费在线书籍是学习 Git 概念（从基础到高级）的必备资源。该特定章节清楚地解释了 `git reset` 及其不同用途，包括从索引中取消暂存更改。
- [Git reset](https://www.atlassian.com/git/tutorials/undoing-changes/git-reset) — Atlassian (2023)
  Publisher: Atlassian
  一篇实用的教程，通过清晰的示例和图表解释了 `git reset` 命令，适合初学者学习管理暂存区。

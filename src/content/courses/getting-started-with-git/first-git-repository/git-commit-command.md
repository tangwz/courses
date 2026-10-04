---
course: "getting-started-with-git"
chapter: "first-git-repository"
lesson: "git-commit-command"
sourceId: 999
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-2-first-git-repository/git-commit-command"
title: "提交修改 (git commit)"
description: "使用 `git commit` 命令将已暂存修改的快照保存到仓库的历史记录中。"
order: 5
plots: []
sourceHash: "6e23cfc70c4b7486b6a8c03dd01aed75d80004c088ebda4259ecfd0765b96cc8"
sourceCorrections: []
---

`git commit` 命令是 Git 的核心操作之一，用于将项目修改的快照永久保存到仓库的历史记录中。它会获取已添加到暂存区的修改，并将其记录为项目的一个新版本。可以将一次提交想象成在游戏中保存进度；它会创建一个安全点，你以后可以回到这个点。每次提交都代表了项目在特定时间点的一个版本。

### 保存已暂存的修改

基本命令很简单：

```bash
git commit
```

当你运行此命令不带任何额外选项时，Git 会打开你在设置时配置的文本编辑器（或默认编辑器，如 Vim 或 Nano）。该编辑器会提示你输入一条*提交信息*。

### 提交信息的重要性

提交信息是你在此次提交中所做修改的描述。它是给你未来的自己和协作者的留言，说明了进行这些修改的*原因*。虽然我们将在下一节介绍如何编写有用的信息，但基本结构通常包括：

1. 简短、描述性的总结行（通常不超过50个字符）。
2. 一个空行。
3. 如有必要，更详细的说明，提供背景、原因或实现细节。

编写完信息后，保存文件并关闭编辑器。Git 随后会创建提交，包含当前暂存区中的修改。

### `-m` 快捷方式

对于只需简短总结行的简单修改，你可以使用 `-m` 标记 (token)直接在命令行提供信息，从而绕过文本编辑器：

```bash
git commit -m "Add initial project structure with README"
```

此命令会立即执行提交，使用所提供的字符串作为信息。这对于小型、直接的修改来说很方便，但对于更复杂的更新，花时间在编辑器中编写详细信息通常是更好的做法。

### 幕后发生了什么？

当你执行 `git commit` 时，Git 会执行几项操作：

1. 它获取暂存区（索引）中文件所表示的精确状态。
2. 它创建一个新的提交对象。该对象包含：
   - 一个指向项目内容快照的引用（具体来说，是代表项目根目录的树对象）。
   - 元数据：作者的姓名和电子邮件（来自你的配置）、时间戳和你的提交信息。
   - 一个或多个指向此提交之前紧邻提交的引用。这些被称为父提交。大多数提交有一个父级，代表线性历史。我们稍后会看到的合并提交则有多个父级。
3. 它为这个提交对象计算一个唯一的40字符SHA-1校验和（哈希值）。这个哈希值作为提交的唯一标识符。
4. 它更新当前分支（例如，`main`）以指向这个新创建的提交。

这个过程确保每次提交都是项目的一个完整、可识别的快照，并按时间顺序链接到以前的版本。

> `git commit` 命令使用已暂存的修改创建一个新的提交对象（例如，`9e1d5`）。指示你当前位置的 `HEAD` 引用，以及分支指针（`main`），会向前移动以指向这个新提交。

### 示例流程

让我们演练一个典型流程：

1. **修改文件：** 编辑你的 `README.md` 文件。
2. **检查状态：** 运行 `git status`。Git 将报告 `README.md` 已修改但未暂存。

   ```bash
   $ git status
   On branch main
   Changes not staged for commit:
     (use "git add <file>..." to update what will be committed)
     (use "git restore <file>..." to discard changes in working directory)
           modified:   README.md

   no changes added to commit (use "git add" and/or "git commit -a")
   ```
3. **暂存文件：** 运行 `git add README.md`。
4. **再次检查状态：** 运行 `git status`。Git 现在显示文件已暂存并准备好提交。

   ```bash
   $ git status
   On branch main
   Changes to be committed:
     (use "git restore --staged <file>..." to unstage)
           modified:   README.md
   ```
5. **提交修改：** 运行 `git commit -m "Update README with project description"`。

   ```bash
   $ git commit -m "Update README with project description"
   [main 9e1d5f0] Update README with project description
    1 file changed, 2 insertions(+)
   ```
6. **最后状态检查：** 运行 `git status`。Git 现在报告工作目录是干净的，表示所有跟踪的修改都已提交。

   ```bash
   $ git status
   On branch main
   nothing to commit, working tree clean
   ```

你已成功记录了项目的一个快照！该提交（由其唯一的 SHA-1 哈希值标识，例如输出中的 `9e1d5f0`，尽管你的会不同）现在是项目历史记录的一部分。你可以使用 `git log` 命令查看这段历史，我们将在很快了解到这个命令。

请记住，`git commit` 只记录那些已经通过 `git add` *暂存*的修改。在你上次 `git add` *之后*对工作目录中文件所做的任何修改将不会包含在提交中，除非你先暂存它们。这个两步过程（先暂存再提交）让你能精确控制项目历史记录中每个快照所包含的内容。

## 参考资料

- [git-commit(1) Manual Page](https://git-scm.com/docs/git-commit) — The Git Community (2023)
  这是 `git commit` 命令的官方且权威来源，详细说明了其所有选项和行为。
- [Pro Git (Chapter 2: Git Basics - Recording Changes to the Repository)](https://git-scm.com/book/en/v2/Git-Basics/Recording-Changes-to-the-Repository) — Scott Chacon, Ben Straub (2014)
  Publisher: Apress; DOI: [10.1007/978-1-4842-0076-6](https://doi.org/10.1007/978-1-4842-0076-6)
  这本书中关于Git工作流的章节提供了全面解释，包括暂存和提交更改，并附有实用示例。
- [Pro Git (Chapter 10: Git Internals - Git Objects)](https://git-scm.com/book/en/v2/Git-Internals/Git-Objects) — Scott Chacon, Ben Straub (2014)
  Publisher: Apress
  这部分内容说明了 Git 提交的底层机制，详细描述了提交对象、树对象和数据对象如何创建和链接以构成项目历史。

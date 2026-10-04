# 检查文件状态 (git status)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-2-first-git-repository/git-status-command)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在使用 Git 仓库时，了解其当前状态非常重要。你经常会想知道：你更改了哪些文件？哪些文件已准备好提交？是否有 Git 尚不知道的新文件？回答这些问题的命令是 `git status`。它可能是 Git 中最常用的命令，因为它会报告你工作目录和暂存区的当前状态。可以把它当作你仓库当前状况的仪表盘。

运行 `git status` 总是安全的；它不会更改你的任何文件或仓库中的任何内容。它只提供信息。

我们来看看你可能会看到的不同类型的输出。

### 清洁工作目录

如果你在 `git commit` 后立即运行 `git status`，或在克隆仓库后立即运行，并且你尚未进行任何更改，你可能会看到类似于以下内容的输出：

```bash
$ git status
On branch main
nothing to commit, working tree clean
```

此消息告诉你一些事项：

- `On branch main`：你当前在名为 `main` 的分支上。分支是开发线，而 `main` 是 Git 中的默认分支名。我们将在稍后详细介绍分支。
- `nothing to commit, working tree clean`：这是主要部分。这意味着 Git 没有跟踪任何已修改的文件，并且没有更改暂存以供下次提交。你的工作目录与仓库中的上一个快照（提交）完全匹配。

### 未跟踪文件

现在，假设你在项目目录中创建了一个新文件，例如 `README.md`。如果你再次运行 `git status`，输出将改变：

```bash
$ touch README.md # 创建一个名为 README.md 的空文件
$ git status
On branch main
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        README.md

nothing added to commit but untracked files present (use "git add" to track)
```

下面是具体说明：

- `Untracked files`：此部分列出了存在于你的工作目录中但尚未成为 Git 仓库一部分的文件。Git 看到了该文件（`README.md`），但没有跟踪它的历史记录。
- `(use "git add <file>..." ...)`：Git 贴心地建议下一步：使用 `git add` 开始跟踪新文件并将其暂存以供下次提交。

### 未暂存的更改（已修改文件）

假设你有一个 Git 正在跟踪的文件（这意味着它是先前提交的一部分），并且你对其进行了修改。假设 `report.txt` 已经提交。如果你编辑 `report.txt`，然后运行 `git status`：

```bash
$ git status
On branch main
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   report.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

- `Changes not staged for commit`：此部分列出了 Git 知道的文件（它们在上次提交中），但自那以后在你的工作目录中已被修改。然而，这些更改尚未标记 (token)为包含在下次提交中。
- `modified: report.txt`：清楚地表明 `report.txt` 已被更改。
- Git 再次提供了有用的提示：使用 `git add` 暂存更改，或使用 `git restore`（在旧版 Git 中是 `git checkout -- <file>`）放弃你在工作目录中进行的更改。

### 待提交的更改（已暂存文件）

你通过使用 `git add` 将更改添加到暂存区来告诉 Git 你希望在下次提交中包含哪些更改。如果你添加已修改的 `report.txt` 和新的 `README.md` 文件：

```bash
$ git add report.txt README.md
$ git status
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   report.txt
        new file:   README.md
```

- `Changes to be committed`：此部分很重要。它列出了当前暂存区中的所有更改。如果你现在运行 `git commit`，这些更改将包含在提交中。
- `modified: report.txt`：对 `report.txt` 的修改现在已暂存。
- `new file: README.md`：新创建的 `README.md` 现在已暂存。
- 如果你错误地添加了文件，Git 还会告诉你如何取消暂存文件（`git restore --staged <file>...` 或在旧版本中是 `git reset HEAD <file>...`）。

### 混合状态

文件同时处于多种状态是很常见的。例如，你可能会暂存 `report.txt`，但在提交之前再次修改它。或者你可能暂存一个文件，而另一个文件仍处于已修改但未暂存的状态，第三个文件是新创建但未被跟踪的。`git status` 清楚地处理了这种情况：

```bash
$ git status
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   report.txt

Changes not staged for commit:
  (use "git restore <file>..." to discard changes in working directory)
        modified:   report.txt # 暂存后再次修改
        modified:   utils.py   # 另一个已修改但未暂存的文件

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        config.yaml
```

此输出显示：

1. `report.txt` 已暂存（你使用 `git add` 添加的版本）。
2. `report.txt` 自暂存后也在工作目录中被修改了。如果你现在提交，只有暂存的版本会进入提交。你需要再次运行 `git add report.txt` 来暂存最新的更改。
3. `utils.py` 已被修改但未暂存。
4. `config.yaml` 是一个 Git 尚未跟踪的新文件。

> 此图示说明了文件相对于工作目录和暂存区可能处于的不同状态，正如 `git status` 所报告的。

### 简短状态输出

为了获得更紧凑的视图，你可以使用 `-s` 或 `--short` 标志：

```bash
$ git status -s
 M report.txt  # M 在第一列：已暂存
 M report.txt  # M 在第二列：已修改但未暂存
 M utils.py    # M 在第二列：已修改但未暂存
?? config.yaml # ?? 表示未跟踪
```

这些字母指示了状态：

- `M`（第一列，绿色）：已修改并已暂存
- `A`（第一列，绿色）：已添加（新文件）并已暂存
- `M`（第二列，红色）：已修改但未暂存
- `??`（红色）：未跟踪

一旦你熟悉了状态码，这种简短格式会很有用。

经常使用 `git status` 是个好习惯。它能帮助你跟踪你的工作，确保你只提交你打算提交的更改，并为其他 Git 命令（如 `git add` 和 `git commit`）提供上下文 (context)。

## 参考资料

- [\`git-status\` Documentation](https://git-scm.com/docs/git-status) — Git Community (2024)
  提供 `git status` 命令及其各种输出和可用选项的权威解释，对于理解 Git 中的文件状态至关重要。
- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  一本 Git 基础知识的全面指南，详细解释了 Git 工作流程、工作目录、暂存区和仓库，并清晰阐述了 `git status` 的输出。在线版本持续更新。
- [Git Status Tutorial](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE9fURrWpnUV4pnyE3hm5PFTD_g-9aTVrgbsPOIMqVgznCYzrjKOiMknFkeVRl2NCyrJeAJIcyiMimpvL9CQBw4bHpi11B5uMRQCFz-JDNT0H-E6dhTKRheOzhY1Mg2naJIrD343fnGB-gNxr-8185bhh7aOcRHO54c) — Atlassian (2024)
  Publisher: Atlassian
  提供清晰实用的 `git status` 输出解释和示例，通过广受欢迎的在线资源帮助用户理解文件状态和 Git 工作流程。

---

[上一节](02-Git%20%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B%EF%BC%9A%E5%B7%B2%E4%BF%AE%E6%94%B9%E3%80%81%E5%B7%B2%E6%9A%82%E5%AD%98%E3%80%81%E5%B7%B2%E6%8F%90%E4%BA%A4.md) · [下一节](04-%E5%B0%86%E6%96%87%E4%BB%B6%E6%B7%BB%E5%8A%A0%E5%88%B0%E6%9A%82%E5%AD%98%E5%8C%BA%20%28git%20add%29.md)

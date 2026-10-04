---
course: "getting-started-with-git"
chapter: "viewing-history-undoing-changes"
lesson: "git-commit-amend"
sourceId: 1008
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-3-viewing-history-undoing-changes/git-commit-amend"
title: "修改最后一次提交 (git commit --amend)"
description: "了解如何使用 `git commit --amend` 修改最近的提交消息或添加遗漏的更改。"
order: 5
plots: []
sourceHash: "844c7c7fef2275ec56c77139b3a7b637e3e3e8f03382a53585ceec3e72c01ff8"
sourceCorrections: []
---

这种情况每个人都会遇到：你提交了一次更改，然后立即发现提交消息里有拼写错误，或者你忘记包含一个小改动或一个相关文件。Git 提供了一种便捷的方式来修改你最近一次提交，而不是创建一个新的提交来修复前一个提交：那就是 `git commit --amend` 命令。

可以把 `git commit --amend` 看作是在说：“实际上，我上次提交的意思是 *这个*。” 它并非真正意义上*修改*了前一个提交（Git 中的提交通常被视为不可更改的快照）。相反，它会将你当前暂存的更改，与你正在修改的提交中的更改合并，然后创建一个全新的提交，替代原有的最后一次提交。

### `git commit --amend` 的运行方式

当你运行 `git commit --amend` 时，Git 会执行以下操作：

1. 它会查看你当前暂存区中的更改（即你用 `git add` 添加的那些）。
2. 它会获取你正在修改的提交*之前*的提交（即其父提交）的内容（文件快照）。
3. 它会创建一个新的提交，其中包含父提交内容与当前暂存内容的合并结果。
4. 这个新提交会替换掉你当前分支上原先的最新提交。旧提交会被丢弃（除非它在其他地方被引用，但对于典型的本地工作而言，它会从你的分支历史中实际消失）。

### 常见用法

#### 修正最后一次提交的消息

这可能是最常用的情况。如果你刚刚提交了一次更改，发现消息中有拼写错误：

```bash
# 没有暂存任何更改，只是修改消息
git commit --amend
```

这个命令会打开你配置的文本编辑器（如 Vim、Nano、VS Code 等），显示之前的提交消息。你只需编辑消息，保存文件并关闭编辑器。Git 就会用修正后的消息创建新的提交。

另外，你可以直接在命令行中使用 `-m` 选项提供新的消息：

```bash
git commit --amend -m "Fix: Correct variable name in calculation"
```

#### 添加遗漏的文件或更改

另一种常见情况是，你发现忘记暂存某个文件或某个本来应该属于上次提交的具体更改。

1. 首先，对你的文件进行必要的更改。
2. 使用 `git add` 暂存这些新更改：

   ```bash
   # 修正 style.css
   git add style.css
   ```
3. 现在，运行 `git commit --amend`。Git 会将新暂存的 `style.css` 更改与原先最后一次提交中已有的更改合并。

   ```bash
   git commit --amend
   ```

这通常会再次打开你的编辑器，允许你在需要时（因为你添加了更多更改）修改提交消息。如果你想完全保留原始提交消息，可以添加 `--no-edit` 标志：

```bash
# 修正 style.css
git add style.css
# 修改最后一次提交，添加 style.css，但保留原始消息
git commit --amend --no-edit
```

### 修改操作的可视化

将修改操作如何改变提交历史进行可视化，可能会有所帮助。

> 修改前，`main` 分支指向提交 B，其中存在错误。运行 `git commit --amend` 后（可能是在修正消息或添加文件之后），`main` 分支现在指向一个全新的提交 B'，其中包含了所需的更改。原来的提交 B 不再是 `main` 分支直接历史的一部分。

### 重要警告：修改已推送的提交

使用 `git commit --amend` 有一个重要的注意事项：**通常应避免修改已推送到共享远程仓库的提交。**

请记住，`git commit --amend` 通过替换最后一次提交来*重写*你的分支历史。如果其他人已经拉取了你后来用修改版替换的原始提交，他们的历史将与你的历史产生分歧。当他们尝试拉取新更改，或者你尝试推送你修改后的提交时，Git 会发现历史冲突，导致混乱和潜在的复杂合并情况。推送修改后的历史通常需要“强制推送”（`git push --force` 或 `git push --force-with-lease`），这可能会覆盖他人的工作，并且在协作工作流程中通常不被建议。

**经验法则：** 仅当提交只存在于你的*本地*仓库时，可以随意使用 `git commit --amend`。一旦提交已被共享（已推送），通常更好的做法是创建一个*新*提交来解决问题（使用 `git revert`，我们稍后会讨论，或者只是添加一个修正提交），而不是修改共享历史。

修改是一个强大的工具，可以在与他人共享之前，保持你的本地提交历史清晰和有条理。使用它来修正拼写错误，或者将小的、相关的更改打包到紧邻的前一个提交中。

## 参考资料

- [git-commit man page](https://git-scm.com/docs/git-commit) — Git contributors (2024)
  Git `git commit` 命令的官方文档，详细说明了 `--amend` 选项及其行为。
- [Pro Git](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History) — Scott Chacon and Ben Straub (2014)
  Publisher: Git SCM
  Git 的综合指南，在重写历史的背景下，提供了 `git commit --amend` 的详细说明和实际示例。
- [Git commit --amend Tutorial](https://www.atlassian.com/git/tutorials/rewriting-history/git-commit--amend) — Atlassian (2024)
  Publisher: Atlassian
  一份实用的教程，提供了关于 `git commit --amend` 的清晰说明、使用场景和重要警告。

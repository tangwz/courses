---
course: "getting-started-with-git"
chapter: "viewing-history-undoing-changes"
lesson: "git-revert-command"
sourceId: 1009
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-3-viewing-history-undoing-changes/git-revert-command"
title: "撤销提交 (git revert)"
description: "学习如何通过使用 `git revert` 创建新的撤销提交来安全地撤销之前的提交。"
order: 6
plots: []
sourceHash: "3e877c401a761ffe42efdcbedc134aa36cbf2721b79374d8c411282af93c29a2"
sourceCorrections: []
---

有时，你创建了一个提交，后来发现它引入了错误或包含了你不想保留的更改。虽然 `git commit --amend` 命令可以用于修改*最近一个*提交，但如果问题提交在你的历史记录中更靠前怎么办？删除或重写可能已与他人共享的历史记录会引发问题。Git 提供了一种更安全的方式来撤销之前提交的更改：`git revert`。

与修改现有历史记录的命令（例如我们将在别处讨论的 `git reset`，或 `git commit --amend`）不同，`git revert` 不会删除任何内容。相反，它会计算出特定提交所引入的更改，并创建一个执行*完全相反操作*的*新*提交。这个新的“撤销提交”能够有效地撤销原始提交的更改，同时保留项目的历史记录。由于它添加了一个新提交而不是重写历史，因此即使在已与协作者共享的分支上使用，也是安全的。

### git revert 的工作方式

可以把它想象成在会计账本中添加一个条目来修正之前的错误。你不会抹去原始的错误条目；而是添加一个新条目来反转其效果。

当你运行 `git revert <commit-hash>` 时，Git 会执行以下步骤：

1. 它检查指定的提交 (`<commit-hash>`)，以确定它相对于其父提交引入的更改。
2. 它计算撤销原始提交效果所需的逆向更改。
3. 它将这些逆向更改应用到你的当前工作目录并暂存它们。
4. 它用这些逆向更改创建一个新提交。默认情况下，Git 会自动生成一个类似“Revert '[原始提交主题]'”的提交消息，并打开你配置的文本编辑器，允许你在撤销提交最终确定前修改该消息。

### 执行撤销操作

假设你的提交历史记录如下（简化版）：

- `a1b2c3d` 添加功能 Y
- `e4f5g6h` 修复功能 X 中的错误
- `i7j8k9l` 项目初始设置

你意识到提交 `e4f5g6h`（“修复功能 X 中的错误”）实际上引入了另一个不明显的错误。你想撤销该特定提交所做的更改。

你会使用以下命令：

```bash
git revert e4f5g6h
```

接着，Git 会：

1. 计算如何反转 `e4f5g6h` 中所做的更改。
2. 应用这些反转。
3. 打开你的编辑器，并预填充一条消息，可能如下：

   ```
   Revert "Fix bug in feature X"

   This reverts commit e4f5g6h1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8.
   ```
4. 在你保存并关闭编辑器后，Git 会创建一个新提交。

你的历史记录现在看起来像这样：

> 该图显示了 `git revert` 如何添加一个新提交（红色）来撤销提交 `e4f5g6h` 的更改，同时保持原始历史记录的完整。

如果你检查你的项目文件，`e4f5g6h` 引入的更改将不再存在。原始提交 `e4f5g6h` 仍然存在于历史记录中，但其效果已被新的撤销提交抵消。

### Revert 与 Reset 的对比

理解 `git revert` 和 `git reset`（后者修改历史记录）之间的区别是很重要的。

- `git revert`：创建一个*新*提交来撤销更改。对共享历史记录安全。保留了原始更改及其撤销的记录。
- `git reset`：移动分支指针，可能丢弃提交。重写历史记录。对于已与他人共享的提交通常不安全。

对于撤销属于你的公共或共享历史记录的更改，`git revert` 几乎总是首选方法。它清晰地表明之前的更改已被撤销，而不会抹去历史上下文 (context)。

### 潜在冲突

就像合并分支一样，撤销一个提交有时也会导致冲突。如果后续提交修改了你正在撤销的提交也涉及到的相同代码行，就会发生这种情况。如果 Git 在撤销过程中遇到冲突，它会停止并告诉你哪些文件存在冲突。你需要手动编辑这些文件来解决冲突的更改，然后使用 `git add` 将它们标记 (token)为已解决，最后运行 `git revert --continue` 来完成该过程。(或者使用 `git revert --abort` 来取消撤销操作)。

使用 `git revert` 提供了一种安全的机制，可以在不中断协作或丢失有价值上下文 (context)的情况下，纠正项目历史记录中的错误。它是维护清晰易懂提交历史记录的重要工具。

## 参考资料

- [git-revert(1) Manual Page](https://git-scm.com/docs/git-revert) — Git Documentation (2024)
  提供 `git revert` 命令的官方详细技术规范，包括所有可用选项及其用法。
- [Reverting a Commit (from "Pro Git" book)](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History#_revert) — Scott Chacon and Ben Straub (2024)
  Publisher: Apress
  提供了对 `git revert` 的全面解释，将其作为一种不修改现有历史记录的安全撤销更改的方法，并与其它历史修改工具进行了区分。

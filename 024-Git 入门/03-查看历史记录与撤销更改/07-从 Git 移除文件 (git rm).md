# 从 Git 移除文件 (git rm)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-3-viewing-history-undoing-changes/git-rm-command)

[返回章节目录](README.md) · [返回课程目录](../README.md)

有时，Git 跟踪的某些文件可能不再需要或被错误地添加到了你的项目中。仅仅使用操作系统文件管理器或命令行（Linux/macOS 上的 `rm`，Windows 上的 `del`）从项目文件夹中删除文件是不够的。Git 会注意到文件已从你的工作目录中消失，但它不会自动为下次提交暂存此删除。你需要一个特定的 Git 命令来正确记录从项目历史中移除文件。

### 使用 `git rm` 移除文件

为此任务设计的命令是 `git rm`。它一步执行两个主要操作：

1. **从你的工作目录中删除文件**（就像标准的 `rm` 或 `del` 命令一样）。
2. 在 Git 索引（暂存区）中**暂存文件的删除**，准备将其记录到下次提交中。

要删除名为 `obsolete_feature.js` 的文件，你将运行：

```bash
git rm obsolete_feature.js
```

运行此命令后，如果你检查状态，Git 将显示删除已暂存：

```bash
git status
```

```
在分支 main 上
要提交的更改：
  （使用 "git restore --staged <文件>..." 取消暂存）
        已删除：    obsolete_feature.js
```

该文件现已从你的工作目录中消失，其删除已暂存。下次 `git commit` 将在项目历史中记录此删除。

```bash
git commit -m "移除废弃功能代码"
```

这确保该文件已从 Git 跟踪的你项目的未来版本中正确移除。

### 在本地保留文件，取消 Git 跟踪

有时，你可能希望 Git 停止跟踪某个文件，但仍想在你的工作目录中保留该文件。这通常发生在你意外提交了一个不应被跟踪的文件时，例如包含敏感信息的配置文件、大型数据文件或编辑器特定设置。

对于这种情况，你使用 `git rm` 的 `--cached` 选项：

```bash
git rm --cached config.local
```

此命令将 `config.local` 从 Git 索引（暂存区）中移除，有效地告诉 Git 停止跟踪它。但是，它会将 `config.local` 文件留在你的工作目录中，不作改动。

运行 `git rm --cached` 后，`git status` 将显示删除已暂存，如果该文件之前未被忽略，它很可能会显示为“未跟踪文件”：

```bash
git status
```

```
在分支 main 上
要提交的更改：
  （使用 "git restore --staged <文件>...” 取消暂存）
        已删除：    config.local

未跟踪文件：
  （使用 "git add <文件>...” 将其包含在将要提交的内容中）
        config.local
```

使用 `git rm --cached` 后，强烈建议立即将文件名（本例中为 `config.local`）添加到你的 `.gitignore` 文件中。这可以防止你或协作者在未来的提交中意外地将其重新添加到 Git 跟踪中。

提交暂存的删除：

```bash
git commit -m "停止跟踪本地配置文件"
```

现在，Git 不再跟踪 `config.local`，但该文件仍在你的本地项目文件夹中可用。

> 说明了文件在工作目录和 Git 索引（暂存区）中在不同删除操作（操作系统删除（`rm`）、`git rm` 和 `git rm --cached`）后的状态。

总结来说，`git rm` 是用于移除 Git 跟踪的文件的标准命令，它同时处理工作目录的删除和暂存更改。当你需要 Git 停止关注某个文件但又想在本地保留它时，请使用 `git rm --cached`，并且记住之后更新你的 `.gitignore`。这两个命令都需要后续的 `git commit` 来在项目历史中最终确定删除。

## 参考资料

- [git-rm Documentation](https://git-scm.com/docs/git-rm) — Git Community (2023)
  这是 `git rm` 命令的官方参考文献，阐述了其用法和选项，包括 `--cached`。
- [Pro Git, Chapter 2.2: Git Basics - Recording Changes to the Repository - Removing Files](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository#_removing_files) — Scott Chacon and Ben Straub (2024)
  Publisher: Apress
  这本备受推崇的 Pro Git 书籍的这一章节，在基础 Git 操作中解释了如何使用 `git rm` 和 `git rm --cached` 来删除文件。
- [gitignore Documentation](https://git-scm.com/docs/gitignore) — Git Community (2023)
  提供了关于如何使用 `.gitignore` 文件有效排除 Git 跟踪的文件的说明，这在 `git rm --cached` 后很重要。

---

[上一节](06-%E6%92%A4%E9%94%80%E6%8F%90%E4%BA%A4%20%28git%20revert%29.md) · [下一节](08-%E7%A7%BB%E5%8A%A8%E6%88%96%E9%87%8D%E5%91%BD%E5%90%8D%E6%96%87%E4%BB%B6%20%28git%20mv%29.md)

# 删除分支 (git branch -d)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-4-branching-merging-basics/git-delete-branches)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当你在某个分支上的工作成功并入主开发线（通常是合并到 `main` 或 `master` 等分支）后，该功能分支本身往往就变得多余了。保留过多的旧分支会使你的仓库视图变得杂乱，并增加查找的难度。Git 提供了一个简单的命令来清理这些已完成的分支。

### 删除本地分支

删除本地分支的主要命令是 `git branch` 加上 `-d` 选项（`-d` 是 `--delete` 的简写形式）：

```bash
$ git branch -d <branch-name>
```

将 `<branch-name>` 替换为你想要移除的分支的实际名称。

举例来说，如果你创建了一个名为 `fix-login-bug` 的分支，提交了修改，然后切换回 `main`，并使用 `git merge fix-login-bug` 将 `fix-login-bug` 成功合并到 `main` 中，那么你就可以安全地删除 `fix-login-bug` 分支了：

```bash
# 确保你当前不在要删除的分支上
$ git switch main
Switched to branch 'main'

# 删除已合并的分支
$ git branch -d fix-login-bug
Deleted branch fix-login-bug (was a1b2c3d).
```

输出中显示的哈希值 `a1b2c3d` 是被删除分支上最后一次提交的 SHA-1 标识符。这确认了删除操作已成功。

### 安全考虑：-d 标志

使用小写 `-d` 标志是推荐的删除分支方式，因为它包含一个安全检查。如果分支包含尚未合并到你当前所在分支（或者更普遍地，尚未合并到 HEAD 的任何祖先分支）中的工作（提交），Git 会阻止你删除该分支。

如果你尝试使用 `-d` 删除一个未合并的分支，Git 将拒绝并显示如下错误信息：

```bash
$ git branch -d experimental-feature
error: The branch 'experimental-feature' is not fully merged.
If you are sure you want to delete it, run 'git branch -D experimental-feature'.
```

这个安全功能有助于避免意外丢失工作。它会促使你再次确认该分支特有的提交是否真的不再需要，或者是否应该先合并它们。

### 强制删除：-D 标志

有时，即使分支尚未合并，你可能也想删除它。这可能发生在你开始了一个实验性功能，后来决定不再继续下去，并想完全放弃这些工作时。

在这种情况下，你可以使用大写 `-D` 标志：

```bash
$ git branch -D <branch-name>
```

这个命令会强制删除分支，无论其合并状态如何。

```bash
$ git branch -D experimental-feature
Deleted branch experimental-feature (was e4f5g6h).
```

**谨慎使用 `-D`。** 由于它会绕过安全检查，如果提交只存在于被删除的分支上且尚未合并到其他地方，你可能会丢失它们。在使用 `git branch -D` 之前，请确保你确实不再需要该分支上的任何修改。

### 验证删除

删除分支后，你可以通过再次列出本地分支来验证它是否已被移除：

```bash
$ git branch
* main
  another-feature
```

请注意，被删除的分支（在我们示例中的 `fix-login-bug` 或 `experimental-feature`）不再出现在列表中。

### 删除远程分支

记住，`git branch -d` 和 `git branch -D` 只会删除你*本地*仓库中的分支。它们不会影响任何远程仓库（例如 GitHub 或 GitLab 上托管的仓库）。删除远程分支涉及到 `git push` 命令，通常会在更详细地讨论远程仓库交互时介绍。目前，请先关注如何通过在分支达到目的且其工作已安全并入后，移除它们来保持本地仓库的整洁。

## 参考资料

- [git-branch(1) Manual Page](https://git-scm.com/docs/git-branch) — Git Project (2024)
  Git `git branch` 命令的官方文档，说明其用于创建、列出和删除分支的用法，包括 `-d` 和 `-D` 选项。
- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  一本广受推崇的Git书籍，提供了Git分支模型和常见操作（如删除本地分支）的详细解释。
- [Deleting a Branch](https://www.atlassian.com/git/tutorials/undoing-changes/git-branch-delete) — Atlassian (2024)
  Publisher: Atlassian
  一份实用指南，解释了删除本地和远程Git分支的过程，适用于理解常见工作流程。

---

[上一节](08-%E5%A4%84%E7%90%86%E5%90%88%E5%B9%B6%E5%86%B2%E7%AA%81.md) · [下一节](10-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%88%86%E6%94%AF%E4%B8%8E%E5%90%88%E5%B9%B6%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B.md)

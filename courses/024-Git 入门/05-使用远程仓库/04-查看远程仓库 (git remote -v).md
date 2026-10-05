# 查看远程仓库 (git remote -v)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories/git-remote-view)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将本地仓库连接到一个或多个远程仓库后，或者克隆一个已包含远程连接的项目时，你经常需要查看配置了哪些远程仓库以及它们指向何处。Git 为此提供了一个简单命令。

### 列出你的远程仓库

要查看本地仓库已知晓的、分配给远程仓库的短名称列表，你可以使用 `git remote` 命令：

```bash
$ git remote
origin
```

这个命令会列出你指定的每个远程句柄的短名称。如果你克隆了一个仓库，你至少会看到 `origin`。这是 Git 为你克隆的服务器设定的默认名称。如果你使用 `git remote add` 手动添加了远程仓库，它们的短名称也会在这里出现。

### 查看远程 URL

尽管知道短名称很有用，但它并没有告诉你这些远程仓库的位置。要查看与每个短名称关联的 URL，你可以在 `git remote` 命令中添加 `-v` (或 `--verbose`) 选项：

```bash
$ git remote -v
origin  https://github.com/user/project.git (fetch)
origin  https://github.com/user/project.git (push)
upstream	https://github.com/original-maintainer/project.git (fetch)
upstream	https://github.com/original-maintainer/project.git (push)
```

此输出提供更多细节。每个远程连接都列出两次：一次用于拉取（下载数据），一次用于推送（上传数据）。

- **短名称：** 这是你用于指代远程仓库的别名（例如，`origin`、`upstream`）。
- **URL：** 这是远程仓库托管的完整地址。它可以是 HTTPS URL、SSH URL（如 `git@github.com:user/project.git`）或另一种协议。
- **(fetch)：** 表明当你运行 `git fetch` 或 `git pull` 等命令以从远程仓库获取更改时使用的 URL。
- **(push)：** 表明当你运行 `git push` 以将更改发送到远程仓库时使用的 URL。

在许多常见情况下，给定远程短名称的拉取和推送 URL 将相同，如上例中 `origin` 和 `upstream` 所示。然而，Git 允许它们不同，这在特定设置中可能很有用（例如，拉取时只读访问，推送到不同位置时写入访问，尽管这对于初学者来说不太常见）。

运行 `git remote -v` 是处理远程仓库时的基本步骤。它可以帮助你：

1. **验证连接：** 确认你的本地仓库已正确连接到预期的远程仓库。
2. **检查 URL：** 确保 URL 正确，特别是在你遇到连接问题时。
3. **识别短名称：** 提醒自己可用于 `git fetch origin` 或 `git push upstream my-feature-branch` 等命令的短名称。

知道如何查看你配置的远程仓库对于管理你的连接和有效协作是不可或缺的。它提供关于你的本地项目连接到何处的清晰信息，并让你能够自信地与远程仓库交互。

## 参考资料

- [git-remote(1) Manual Page](https://git-scm.com/docs/git-remote) — Git Project (2024)
  `git remote` 命令的官方和全面文档，解释其选项和用法。
- [Pro Git](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  权威书籍章节，详细介绍了如何在 Git 中管理和交互远程仓库，包括查看远程仓库。
- [Managing remote repositories](https://docs.github.com/en/get-started/getting-started-with-git/managing-remote-repositories) — GitHub Docs (2024)
  来自领先 Git 托管平台的实用指南，提供管理远程连接的实际示例。

---

[上一节](03-%E6%B7%BB%E5%8A%A0%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%20%28git%20remote%20add%29.md) · [下一节](05-%E5%85%8B%E9%9A%86%E7%8E%B0%E6%9C%89%E4%BB%93%E5%BA%93%20%28git%20clone%29.md)

# 从远程仓库获取更改 (git fetch)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories/git-fetch-command)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然像 `git clone` 这样的命令最初会获取远程仓库的完整副本，`git push` 会将你的本地更改发送*到*远程仓库，但你还需要一种方法将他人从远程仓库所做的更改拉取到你的本地副本中，*而无需*立即将它们合并到你的工作文件中。这正是 `git fetch` 命令的功能。

你可以把 `git fetch` 想象成检查你的项目邮箱。它会连接到指定的远程仓库（通常名为 `origin`），并下载你本地没有的任何数据。这包括自你上次通信以来在远程仓库上发生的新提交、新分支以及对现有分支的更新。

然而，这是一个重要的区别：`git fetch` *只下载*这些新数据。它将更新放入你的本地仓库数据库，但并*不*会自动将其合并到你的本地工作分支（如 `main`）或修改你的工作目录。你当前的工作不受影响。

### `git fetch` 如何更新你的本地仓库

当你运行 `git fetch` 时，Git 执行以下主要操作：

1. **连接：** 它连接到你指定的远程仓库（如果未指定，则默认为 `origin`）。
2. **下载数据：** 它识别远程仓库中存在但你的本地仓库中缺少的任何提交和相关文件，并下载它们。
3. **更新远程跟踪分支：** 这是 `git fetch` 使下载的信息可访问的主要方式。对于远程仓库上的每个分支（例如 `main`），Git 会在你的本地仓库中更新一个对应的*远程跟踪分支*。它们就像书签一样，记住你上次拉取时远程分支的状态。它们遵循 `<remote_name>/<branch_name>` 的命名约定，例如 `origin/main`、`origin/feature-x` 等。这些分支是本地引用，但你不能直接编辑它们；当你执行 fetch 时，Git 会自动移动它们。

### 运行命令

使用 fetch 最常用方式是：

```bash
git fetch <remote_name>
```

例如，要从名为 `origin` 的默认远程仓库拉取所有更新，你可以运行：

```bash
git fetch origin
```

如果你配置了多个远程仓库，你需要指定要从哪个仓库拉取。如果你只是运行 `git fetch` 而不指定远程名称，如果 `origin` 存在且被配置为默认，Git 通常会从 `origin` 拉取。

### 查看拉取的更改

由于 `git fetch` 不修改你的工作分支（如 `main`），你如何查看拉取了哪些更改？你将本地分支与更新后的远程跟踪分支进行比较。

例如，在运行 `git fetch origin` 后，你可以使用 `git log` 查看 `origin/main` 上有但你本地 `main` 尚未有的提交：

```bash
# 显示 origin/main 上有但你本地 main 尚未有的提交
git log main..origin/main

# 你也可以查看更改摘要
git log --oneline --graph --decorate main..origin/main
```

要查看你的本地 `main` 与拉取的 `origin/main` 之间的合并差异（如补丁）：

```bash
# 显示两个分支之间的实际代码更改
git diff main..origin/main
```

> 此图说明了 `git fetch` 如何更新本地 `origin/main` 引用以匹配远程 `main` 的状态。将这些更改集成到本地 `main` 分支需要一个单独的 `git merge` 命令。

### 为什么使用 `git fetch` 而不是 `git pull`？

你可能想知道，如果 `git pull`（我们接下来会讲到）通常一步完成拉取和合并，为什么还要使用 `git fetch`。使用 `git fetch` 提供更安全的工作流程，尤其是在复杂的项目中：

- **合并前检查：** 它允许你在将传入的更改集成到你的本地工作之前对其进行检查（`git log origin/main`，`git diff origin/main`）。这有助于你理解更改并预见潜在冲突。
- **控制集成：** 你决定*何时*以及*如何*集成拉取的更改（例如，使用 `git merge` 或 `git rebase`）。
- **更新多个分支：** Fetch 会从远程仓库拉取*所有*分支的更新，使你无需在本地检出即可查看各种特性分支的状态。

总之，`git fetch` 是你用来从远程仓库下载最新信息到本地 Git 数据库的命令，它会更新你的远程跟踪分支。它是与远程协作者保持同步而不立即更改你自己的本地开发分支的一个基本命令。拉取（fetch）后的下一步通常是集成这些更改，通常使用 `git merge`。

## 参考资料

- [Pro Git](https://git-scm.com/book) — Scott Chacon, Ben Straub (2014)
  Publisher: Apress
  深入介绍 `git fetch`，解释其从远程仓库获取更改而不合并的功能，以及远程跟踪分支的概念。这是 Git 用户的标准资源。
- [git-fetch(1) Manual Page](https://git-scm.com/docs/git-fetch) — Git Project (2024)
  `git fetch` 命令的官方文档，详细说明所有可用选项、用法及其更新远程跟踪分支的具体行为。这是一份权威参考资料。

---

[上一节](06-%E5%90%91%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%E6%8E%A8%E9%80%81%E6%9B%B4%E6%94%B9%20%28git%20push%29.md) · [下一节](08-%E4%BB%8E%E8%BF%9C%E7%A8%8B%E6%8B%89%E5%8F%96%E6%9B%B4%E6%94%B9%20%28git%20pull%29.md)

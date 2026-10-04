---
course: "getting-started-with-git"
chapter: "working-with-remote-repositories"
lesson: "git-push-command"
sourceId: 1028
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories/git-push-command"
title: "向远程仓库推送更改 (git push)"
description: "使用 `git push` 将你的本地提交和分支上传到已配置的远程仓库。"
order: 6
plots: []
sourceHash: "161366218c48f80490886b2aae862721b723505f9ba8ef9d17f82a46025e09e5"
sourceCorrections: []
---

当你在本地进行了一些修改，并使用 `git commit` 将它们保存为提交后，通常会希望将这些更新分享给其他人，或者只是将它们备份到远程仓库（例如 GitHub 或 GitLab）。此时就用到 `git push` 命令了。它是将已提交的更改*从*本地仓库*发送到*远程仓库的主要方式。

把它想象成将你保存在计算机上的文档上传到共享在线文件夹。你的一系列提交就是文档的更新，而远程仓库就是大家可以看到最新版本的共享文件夹。

### `git push` 的工作原理

该命令的基本结构很简单：

```bash
git push <remote-name> <branch-name>
```

我们来分解一下：

- `git push`：这是命令本身，告诉 Git 你打算将已提交的更改*发送*到远程位置。
- `<remote-name>`：这指定了你要将更改发送到*哪个*远程仓库。记得在之前的章节中提到过，你可能使用 `git remote add` 添加了一个远程仓库，或者克隆时会自动设置一个吗？通常，你克隆或手动添加的默认远程仓库名为 `origin`。因此，你在这里会经常使用 `origin`。你可以随时使用 `git remote -v` 列出你配置的远程仓库及其 URL，以确保正确。
- `<branch-name>`：这指定了你要发送*哪个*本地分支的提交。如果你一直在主开发线上直接进行提交，这通常是 `main`（在一些旧项目中可能是 `master`）。如果你创建了一个功能分支来处理某个具体事项（如分支章节中所述），你会在这个地方使用该分支的名称，例如 `feature-login`。

### 一个常见示例

最常见的使用情况是，将你在本地 `main` 分支上所做的提交推送到 `origin` 远程仓库：

```bash
git push origin main
```

当你执行此命令时，Git 会执行以下几个步骤：

1. 它查看你本地 `main` 分支的历史。
2. 它与 `origin` 远程仓库通信，以查看其 `main` 分支的历史。
3. 它确定哪些提交存在于你的本地 `main` 分支上，但 `origin/main` 分支上没有。
4. 它将那些缺失的提交和相关数据传输（推送）到 `origin` 远程仓库。
5. 最后，它更新 `origin` 远程仓库上的 `main` 分支指针，使其指向你刚刚推送的最新提交。

> 此图展示了 `git push` 命令如何从你的本地仓库获取提交并将其发送以更新远程仓库。

### 首次推送新分支

如果你在本地创建了一个全新的分支，比如 `fix-typos`，并在上面做了一些提交，现在想把它分享到远程仓库怎么办？第一次推送这个新分支时，远程仓库（`origin`）甚至还不知道它的存在。你需要明确地告诉 Git 在远程仓库上创建这个分支，而且重要的是，将你的本地分支与这个新的远程分支关联起来。

你可以使用 `-u` 选项来完成此操作（它是 `--set-upstream` 的简写）：

```bash
git push -u origin fix-typos
```

这个命令会执行两个重要操作：

1. 它将你的本地 `fix-typos` 分支，连同远程仓库没有的所有可从该分支到达的提交，推送到 `origin` 远程仓库。这会在远程仓库上创建 `fix-typos` 分支。
2. 它在你的本地 `fix-typos` 分支和远程仓库上的 `origin/fix-typos` 分支之间建立了一个“跟踪关系”。

这种跟踪关系很方便。一旦建立好（通过首次推送时使用 `-u`），对于你在 `fix-typos` 分支上进行的任何后续推送，你通常只需输入更简单的命令：

```bash
git push
```

Git 会自动知道你打算将当前分支（`fix-typos`）推送到其对应的已跟踪远程分支（`origin/fix-typos`）。这种跟踪关系也简化了拉取更改的操作，我们将在 `git pull` 部分详细介绍。

### 如果推送失败怎么办？（非快进错误）

有时，你的推送尝试可能会被远程仓库拒绝。Git 通常会显示包含 `rejected`、`failed to push some refs` 等字样以及经常出现的 `non-fast-forward` 的错误信息。

此错误通常表示远程分支在你上次与它同步本地仓库（使用 `git fetch` 或 `git pull` 等命令）后，接收了新的提交。设想你基于提交 `B` 开始工作，然后添加了提交 `C`。同时，其他人也从提交 `B` 开始，添加了提交 `D`，并在你尝试推送 `C` 之前成功地将 `D` 推送到了远程仓库。

> 这种情况发生在远程分支在你上次更新本地副本后，有了新的提交。推送你的更改会覆盖远程仓库上现有但不同的历史。

Git 会阻止这种“非快进”推送，因为接受你的提交 `C` 会导致远程仓库丢失提交 `D`。历史已发生分歧。

标准的解决办法是，在再次推送*之前*，先将远程更改合并到你的本地分支。你需要获取最新更改（包括提交 `D`）并将其与你的本地工作（提交 `C`）合并。`git pull` 命令就是为这种获取和合并操作而设计的，我们将在其专属章节中详细介绍。目前，主要的要点是，“非快进”错误通常意味着在 `push` 之前你需要 `pull`。

### 认证注意事项

推送更改涉及向远程仓库写入数据，因此几乎总是需要认证。你需要向远程服务器（如 GitHub、GitLab 或 Bitbucket）证明你拥有修改项目的必要权限。

认证方法取决于你连接远程仓库时使用的协议（通常是 HTTPS 或 SSH）：

- \*\*HTTPS：**你可能会被要求输入用户名和密码。现在更常见的是，出于安全考虑，托管平台会要求你使用**个人访问令牌（PAT）\*\*而非普通密码。你可以在该平台的网站上生成此令牌，并在 Git 提示时将其用作密码。
- \*\*SSH：\*\*此方法使用加密密钥对。你将你的公共 SSH 密钥添加到托管平台上的账户，Git 会使用你的私钥（安全地存储在你的计算机上）自动进行认证，无需为每次推送输入密码或令牌。

如果 `git push` 因认证错误而失败，Git 的信息通常会指引你找到解决办法。你可能需要检查你存储的凭据、生成 PAT，或确保你的 SSH 密钥已正确配置并添加到远程平台。请查阅你的特定 Git 托管服务的文档以获取详细说明。

有效地使用 `git push` 可以让你分享你的贡献，与团队成员协作，并在远程服务器上保留项目历史的备份。它是与远程 Git 仓库协作的基本命令。

## 参考资料

- [git-push(1) Manual Page](https://git-scm.com/docs/git-push) — Git Developers (2024)
  `git push` 命令的官方和最权威指南，详细说明其语法、选项和操作细节。
- [Working with Remotes (from Pro Git, 2nd Edition)](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes) — Scott Chacon and Ben Straub (2024)
  Publisher: Apress
  《Pro Git》一书中的一个综合章节，解释了远程仓库的基本概念、如何管理它们以及 `git push` 在共享更改中的作用。
- [Creating a personal access token (classic)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) — GitHub Docs (2024)
  Publisher: GitHub
  GitHub 官方文档，详细说明了生成和使用个人访问令牌（PAT）的过程，以实现在通过 HTTPS 推送时的安全认证，这是一项重要的安全实践。

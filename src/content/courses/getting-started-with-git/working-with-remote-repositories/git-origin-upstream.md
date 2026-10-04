---
course: "getting-started-with-git"
chapter: "working-with-remote-repositories"
lesson: "git-origin-upstream"
sourceId: 1031
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories/git-origin-upstream"
title: "理解 `origin` 和 `upstream`"
description: "了解远程仓库命名的常见约定，特别是关于 `origin` 和 `upstream` 的。"
order: 9
plots: []
sourceHash: "07707b56c766288ed9b9b2cd8ca7215cf727b3079b683c640019c612a2a19c28"
sourceCorrections: []
---

当你开始使用 `git clone`、`git fetch`、`git pull` 和 `git push` 等命令与远程仓库交互时，你会经常遇到 `origin` 和有时会遇到 `upstream` 这两个名称。它们并非 Git 的特殊关键字，而是远程仓库命名上广泛采用的惯例。知道它们通常代表什么，会让协作工作流程变得更加清晰。

### 默认远程仓库：`origin`

回想一下 `git clone` 命令。当你从 GitHub 或 GitLab 等托管平台克隆一个仓库时，使用如下命令：

```bash
git clone https://github.com/someuser/someproject.git
```

Git 主要做两件事：

1. 它将所有项目数据（提交、分支等）复制到你的本地机器，创建一个名为 `someproject` 的新目录。
2. 它会自动配置一个指向你克隆来源 URL 的*远程*连接。默认情况下，Git 将此远程连接命名为 `origin`。

因此，`origin` 只是 Git 分配给你克隆来源远程仓库的默认简称。这是一个表明“此本地仓库源自何处”的惯例。你可以在克隆后立即运行 `git remote -v` 来验证这一点：

```bash
cd someproject
git remote -v
# Output will likely show:
# origin  https://github.com/someuser/someproject.git (fetch)
# origin  https://github.com/someuser/someproject.git (push)
```

这表明 `origin` 这个名称关联到原始 URL，用于抓取（下载）和推送（上传）操作。因为它是默认名称，许多与远程仓库交互的 Git 命令（如 `git fetch`、`git pull`、`git push`）在你没有明确指定其他远程名称时，都会使用 `origin`。例如，`git push` 通常等同于 `git push origin`。

尽管 `origin` 是默认名称，但需要记住它只是一个名称。你可以使用 `git remote rename origin destination` 命令移除或重命名它，不过，除非有特殊原因，否则偏离此惯例的情况并不常见。

### 源项目：`upstream`

使用 `upstream` 的惯例通常在协作场景中发挥作用，特别是在开源开发或使用派生仓库（forks）时。常见的工作流程包括：

1. **派生（Forking）：** 你在 GitHub 等平台找到一个你想要贡献或尝试使用的项目（我们称之为“原始”项目）。你在这个平台创建自己的仓库副本。这个副本被称为“派生仓库”（fork）。
2. **克隆：** 然后你将*你的派生仓库*（而不是直接原始项目）克隆到你的本地机器。正如我们刚刚了解的，Git 会自动设置一个名为 `origin` 的远程仓库，指向你在 GitHub 上的*派生仓库*。
3. **贡献/更新：** 你在本地进行更改，然后将它们推送到你的 `origin`（你的派生仓库）。要贡献回原始项目，你通常会从你的派生仓库创建一个拉取请求（Pull Request）。很重要的一点是，在你工作期间，原始项目可能会收到其他贡献者的更新。你会希望将这些更新拉取到你的本地仓库和你的派生仓库中，以保持同步。

这就是 `upstream` 的用武之地。为了跟踪*原始*项目仓库（你最初派生出来的那个），通常会添加另一个远程连接并将其命名为 `upstream`。你可以使用 `git remote add` 手动完成此操作：

```bash
# 在你本地克隆的派生仓库目录下
git remote add upstream https://github.com/original-owner/original-project.git

# 验证远程仓库
git remote -v
# 输出将显示 origin 和 upstream：
# origin    https://github.com/your-username/original-project.git (fetch)
# origin    https://github.com/your-username/original-project.git (push)
# upstream  https://github.com/original-owner/original-project.git (fetch)
# upstream  https://github.com/original-owner/original-project.git (push)
```

现在你有两个已命名的远程仓库：

- `origin`：指向你在托管平台上的个人派生仓库。你通常在此处拥有推送（push）权限。
- `upstream`：指向你派生来源的原始项目仓库。你通常只在此处拥有抓取（fetch）权限（除非你是原始项目的直接协作者）。

你可以使用 `upstream` 从原始项目抓取更改并将其合并到你的本地工作中：

```bash
# 从原始项目抓取分支和提交
git fetch upstream

# 切换到你的本地 main 分支（或等效分支）
git switch main

# 将 upstream main 分支的更改合并到你的本地 main 分支
git merge upstream/main

# 将更新后的 main 分支（包括来自 upstream 的更改）推送到你的派生仓库
git push origin main
```

> 本地仓库、`origin`（派生仓库）和 `upstream`（原始项目）仓库之间的关系图。实线代表直接的 Git 操作，虚线代表通常通过托管平台界面执行的操作。

总结：

- `origin` 是你克隆来源的远程仓库的默认名称，通常是你的派生仓库或你的主要远程副本。
- `upstream` 是你派生来源的原始远程仓库的惯用名称，主要用于抓取更新。

虽然你可以随意命名你的远程仓库，但遵循 `origin` 和 `upstream` 的命名惯例，能让熟悉 Git 和协作开发实践的其他开发者立即理解你的工作流程。这些名称提供了关于每个远程连接预期用途的背景信息。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  一本全面的 Git 指南，涵盖了 `origin` 作为默认远程仓库以及 `upstream` 在协作工作流和分支管理中的作用。在线版本持续更新。
- [git-remote Documentation](https://git-scm.com/docs/git-remote) — The Git Community (2024)
  官方文档，解释了如何管理远程仓库，包括默认分配的 `origin` 名称以及如何添加新的远程仓库（如 `upstream`）。
- [git-clone Documentation](https://git-scm.com/docs/git-clone) — The Git Community (2023)
  官方文档，详细说明了 `git clone` 命令及其如何自动配置 `origin` 远程仓库以指向源仓库。

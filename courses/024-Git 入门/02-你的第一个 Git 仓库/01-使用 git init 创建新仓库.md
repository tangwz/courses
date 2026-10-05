# 使用 git init 创建新仓库

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-2-first-git-repository/git-init-command)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要使用 Git 跟踪项目，系统需已安装并配置好 Git。此过程允许你将任何项目置于版本控制之下。无论是处理现有项目，还是启动一个全新项目，在项目主目录中的基本第一步都是初始化一个 Git 仓库。

执行此操作的命令是 `git init`。

这个命令会在你的项目文件夹中创建一个名为 `.git` 的新子目录。这个 `.git` 目录是你的仓库的核心；它包含 Git 跟踪更改、管理历史记录、存储配置设置以及更多功能所需的所有文件和元数据。基本上，所有使你的项目成为 Git 仓库的东西都存在于这个隐藏目录中。你的实际项目文件，通常被称为“工作目录”或“工作树”，则与其并存。

让我们来看一个例子。假设你有一个名为 `my-website` 的新网站项目目录。

首先，使用命令行进入你的项目目录：

```bash
cd path/to/my-website
```

请确保将 `path/to/my-website` 替换为你的项目文件夹的实际路径。

现在，运行 `git init` 命令：

```bash
git init
```

Git 会回复一条消息，确认初始化成功：

```
Initialized empty Git repository in /path/to/my-website/.git/
```

就这样！你的 `my-website` 目录现在就是一个 Git 仓库了。尽管表面看起来变化不大，但 Git 现在已经准备好开始跟踪你在此文件夹中的工作了。

`.git` 目录在大多数操作系统上默认是隐藏的。要查看它，你可以在 Linux 或 macOS 上使用 `ls -a` 命令，或在 Windows 命令提示符中使用 `dir /a`：

```bash
# 在 Linux 或 macOS 上
ls -a

# 在 Windows 命令提示符中
dir /a
```

你应该能看到 `.git` 列在你的项目文件和目录中（如果已有的话）。

> 运行 `git init` 前后的目录结构对比。该命令会添加隐藏的 `.git` 目录，其中包含仓库的元数据。

重要的是要明白 `git init` 创建的是一个*本地*仓库。它不会自动连接到任何远程托管服务，例如 GitHub。它只是在你的电脑上设置了必要的配置，以便 Git 开始管理你的项目版本。每个项目你只需要运行一次 `git init`。

仓库初始化完成后，你现在可以开始 Git 的核心工作流程了：即将文件添加到暂存区并提交项目快照，这些我们将在接下来介绍。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress; Pages: Chapter 1: Getting Started
  全面的Git指南，在早期章节中详细介绍了仓库创建和内部结构。
- [git-init Documentation](https://git-scm.com/docs/git-init) — Git Project (2024)
  `git init` 命令的官方文档，解释了其目的、用法和可用选项。
- [How to initialize a Git repository](https://www.atlassian.com/git/tutorials/setting-up-a-repository/git-init) — Atlassian (2024)
  一个关于使用 `git init` 设置新本地Git仓库的实用教程，附有清晰的示例。

---

[上一节](../01-%E7%89%88%E6%9C%AC%E6%8E%A7%E5%88%B6%E4%B8%8EGit%E7%AE%80%E4%BB%8B/10-%E8%8E%B7%E5%8F%96Git%E5%91%BD%E4%BB%A4%E7%9A%84%E5%B8%AE%E5%8A%A9.md) · [下一节](02-Git%20%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B%EF%BC%9A%E5%B7%B2%E4%BF%AE%E6%94%B9%E3%80%81%E5%B7%B2%E6%9A%82%E5%AD%98%E3%80%81%E5%B7%B2%E6%8F%90%E4%BA%A4.md)

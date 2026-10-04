---
course: "getting-started-with-git"
chapter: "working-with-remote-repositories"
lesson: "git-remote-add"
sourceId: 1025
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories/git-remote-add"
title: "添加远程仓库 (git remote add)"
description: "了解如何使用 `git remote add` 命令将本地仓库连接到远程仓库。"
order: 3
plots: []
sourceHash: "f3b19a50d67759524b8eca97abb044f941f2ff284e15f1e3e30e7f2b655db697"
sourceCorrections: []
---

连接本地Git仓库与远程仓库是实现协作和项目共享的重要环节。如果你已经使用 `git init` 在本地初始化了一个仓库，并希望将其与托管在GitHub、GitLab或私有服务器等服务上的项目关联起来，你需要明确地告知本地Git该远程仓库的地址。`git remote add` 命令用于完成此操作。

此命令会将远程仓库的位置注册到一个特定名称下，创建一个书签或别名，你可以在 `git push` 或 `git pull` 等其他Git命令中使用它。它不传输任何数据；它只是设置连接信息。

### 命令语法

该命令的基本结构是：

```bash
git remote add <shortname> <url>
```

我们来逐一分析这些组成部分：

- `<短名称>`：这是你希望在本地用来指代远程仓库的别名或昵称。它是一种简写方式，可以让你每次都省去输入完整URL的麻烦。按照惯例，`origin` 名称通常用于你进行交互的主要远程仓库（你克隆自的仓库，或者你推送更改的主要仓库）。但是，如果你需要管理多个远程仓库（例如，用于你派生原始项目的 `upstream`，或用于次要服务器的 `backup`），你可以选择其他名称。使用清晰、有辨识度的短名称很有帮助。
- `<URL>`：这是指向远程仓库的实际URL或路径。Git支持多种URL格式，但最常见的两种是：
  - **HTTPS：** 格式类似于 `https://github.com/username/repository-name.git`。这些在最初设置时通常更容易，特别是对于公共仓库，并且通常能很好地穿透防火墙。与远程仓库交互时，你可能会被要求输入用户名和密码（或个人访问令牌）。
  - **SSH：** 格式类似于 `git@github.com:username/repository-name.git`。此格式使用Secure Shell协议，通常依赖预先配置的SSH密钥进行身份验证，避免了重复输入凭据的麻烦。这在专业开发环境中非常常见。

### 实际案例

设想你在本地创建了一个新项目：

```bash
# 进入你的项目目录
cd my-new-project

# 如果尚未初始化，请初始化一个Git仓库
git init

# 进行一些初始提交...
echo "Project setup" > README.md
git add README.md
git commit -m "Initial commit"
```

现在，假设你已经在GitHub上创建了一个对应的空仓库，地址是 `https://github.com/your-username/my-new-project.git`。要将你的本地仓库与这个远程仓库关联起来，你可以运行：

```bash
git remote add origin https://github.com/your-username/my-new-project.git
```

此命令会告诉你的本地Git：“在 `https://github.com/your-username/my-new-project.git` 有一个远程仓库，我想用短名称 `origin` 来指代它。”

运行此命令后，你不会立即看到任何确认成功的输出，但Git已将此信息存储在其配置中。你可以使用 `git remote -v` 命令验证远程仓库是否正确添加，我们将在下一节中介绍这个命令。它会列出所有已注册的远程仓库及其URL。

请记住，`git remote add` 只是建立链接。你的本地提交仍然只存在于你的机器上。要将你的提交发送到名为 `origin` 的远程仓库，你随后会使用 `git push origin main` 这样的命令（假设 `main` 是你的默认分支）。

## 参考资料

- [git-remote Documentation](https://git-scm.com/docs/git-remote) — Git Community (2024)
  Git `git remote` 命令的官方参考资料，详细说明其语法、选项和远程连接的管理方式。
- [Pro Git](https://git-scm.com/book/en/v2/Git-on-the-Server) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress; Pages: Chapter 4: Git on the Server
  关于Git的全面资源，其中专门有一节介绍远程仓库和 `git remote add` 命令。
- [Git Remote Tutorial](https://www.atlassian.com/git/tutorials/git-remote) — Atlassian (2024)
  Publisher: Atlassian
  提供关于使用 `git remote` 管理远程仓库的实用指南，包括 `add` 子命令的教程。

---
course: "intro-data-engineering"
chapter: "essential-tools-data-engineers"
lesson: "practice-basic-git-commands"
sourceId: 5602
sourceUrl: "https://apxml.com/zh/courses/intro-data-engineering/chapter-6-essential-tools-data-engineers/practice-basic-git-commands"
title: "实践：Git 基本命令"
description: "动手操作练习使用版本控制的基本 Git 命令。"
order: 6
plots: []
sourceHash: "7dbd61e7f8b36f6705a7a1149ca9b1681934d6a5b59767b006f9d4fe23e345f3"
sourceCorrections: []
---

Git 基本命令是管理数据工程项目所必需的。这些动手操作练习涵盖了常用 Git 命令。持续使用版本控制是专业软件开发的一个标志，对于数据工程而言，它在追踪代码、配置乃至有时是文档的变动方面同样不可或缺。

若要跟随操作，您的系统上需要安装 Git。如果您尚未安装，可以在 Git 官方网站 (<https://git-scm.com/downloads>) 上找到安装说明。安装完成后，打开您偏好的命令行界面（macOS/Linux 上的终端，Windows 上的 Git Bash 或命令提示符/PowerShell）。

### 初始化仓库

使用 Git 追踪项目的第一步是初始化一个仓库。Git 仓库本质上是项目文件夹中一个隐藏目录（`.git`），Git 会在此处存储项目的所有历史和元数据。

1. **创建一个项目目录：** 我们为本次练习创建一个新目录。

   ```bash
   mkdir my-data-project
   ```
2. **进入该目录：**

   ```bash
   cd my-data-project
   ```
3. **初始化 Git：** 指示 Git 开始追踪此目录。

   ```bash
   git init
   ```

   您应该看到类似于以下内容的输出：

   ```
   Initialized empty Git repository in /path/to/your/my-data-project/.git/
   ```

现在，`my-data-project` 已是一个 Git 仓库。您在此处添加的任何文件都可以被追踪。

### 查看状态、暂存和提交变动

核心 Git 工作流程包括对文件进行变动，将这些变动添加到“暂存区”，然后（附带消息）将它们永久提交到仓库的历史记录中。

1. **创建文件：** 我们创建一个简单的文本文件来模拟项目资源，例如脚本或笔记。

   ```bash
   # 在 Linux/macOS 上
   echo "Initial configuration for data pipeline" > config.txt

   # 在 Windows (命令提示符) 上
   echo Initial configuration for data pipeline > config.txt

   # 在 Windows (PowerShell) 上
   "Initial configuration for data pipeline" | Out-File -Encoding UTF8 config.txt
   ```
2. **检查状态：** 使用 `git status` 查看 Git 对您项目文件的了解情况。

   ```bash
   git status
   ```

   输出会类似于这样：

   ```
   On branch main

   No commits yet

   Untracked files:
     (use "git add <file>..." to include in what will be committed)
           config.txt

   nothing added to commit but untracked files present (use "git add" to track)
   ```

   Git 看到了 `config.txt` 但告诉您它处于“未追踪”状态。这意味着 Git 尚未监测它的变动。
3. **暂存文件：** 使用 `git add` 将文件移至暂存区。这会告诉 Git 您希望将此文件的当前版本包含在*下一次*提交中。

   ```bash
   git add config.txt
   ```

   > **提示：** 您可以使用 `git add .`（注意句点）暂存当前目录及子目录中所有已修改和新增的文件。使用此命令时请注意，确保您没有意外暂存不打算追踪的文件。
4. **再次检查状态：**

   ```bash
   git status
   ```

   输出现在变了：

   ```
   On branch main

   No commits yet

   Changes to be committed:
     (use "git rm --cached <file>..." to unstage)
           new file:   config.txt
   ```

   Git 现在在“要提交的变动”下显示 `config.txt`。它已暂存。
5. **提交变动：** 使用 `git commit` 将暂存的变动保存到仓库历史记录中。`-m` 标志允许您直接提供描述性的提交消息。

   ```bash
   git commit -m "Add initial pipeline configuration file"
   ```

   您会看到确认提交的输出：

   ```
   [main (root-commit) abc1234] Add initial pipeline configuration file
    1 file changed, 1 insertion(+)
    create mode 100644 config.txt
   ```

   `abc1234` 部分是此次提交的唯一标识符（哈希）的开头。
6. **最后一次检查状态：**

   ```bash
   git status
   ```

   输出应该指示一个干净的工作目录：

   ```
   On branch main
   nothing to commit, working tree clean
   ```

### 查看提交历史

要查看您所做提交的历史记录，请使用 `git log` 命令。

```bash
git log
```

这会显示一个详细的提交列表，从最近的提交开始：

```
commit abc1234567890defabcdef1234567890abcdef (HEAD -> main)
Author: Your Name <your.email@example.com>
Date:   Tue Sep 17 10:30:00 2023 -0700

    Add initial pipeline configuration file
```

> **提示：** 如需更紧凑的视图，请尝试 `git log --oneline`：
>
> ```bash
> git log --oneline
> ```
>
> 输出：
>
> ```
> abc1234 (HEAD -> main) Add initial pipeline configuration file
> ```

### 忽略文件

项目通常会生成您*不*希望 Git 追踪的文件，例如日志文件、临时文件、大型数据文件或敏感信息（如 API 密钥）。您可以通过在仓库的根目录中创建 `.gitignore` 文件，指示 Git 忽略特定的文件或模式。

1. **创建 `.gitignore` 文件：**

   ```bash
   # 在 Linux/macOS 上
   touch .gitignore

   # 在 Windows (命令提示符) 上 - 创建一个空文件
   type nul > .gitignore

   # 在 Windows (PowerShell) 上
   New-Item .gitignore -ItemType File
   ```
2. **编辑 `.gitignore` 文件：** 添加要忽略的文件/目录模式。在文本编辑器中打开 `.gitignore` 并添加以下行：

   ```
   # 忽略日志文件
   *.log

   # 忽略临时目录中的文件
   temp/

   # 忽略敏感凭证
   credentials.json
   ```

   每行指定一个模式。`*` 是通配符。末尾的 `/` 表示一个目录。
3. **暂存并提交 `.gitignore`：** 由于 `.gitignore` 是项目配置的一部分，您应该像处理其他项目文件一样添加并提交它。

   ```bash
   git add .gitignore
   git commit -m "Add .gitignore file"
   ```

   现在，如果您创建一个名为 `app.log` 的文件或一个名为 `temp` 的目录，`git status` 将不会将其列为未追踪文件。

### 克隆现有仓库

您通常不是从新项目开始，而是从 GitHub、GitLab 或 Bitbucket 等远程托管服务上已有的项目开始操作。您可以使用 `git clone` 获取仓库的本地副本。

1. **查找仓库 URL：** 前往 GitHub 等服务，找到您感兴趣的公共仓库。查找“克隆”或“代码”按钮，它将提供一个 URL（通常以 `.git` 结尾）。
2. **克隆仓库：** 导航到 `my-data-project` 文件夹*之外*的目录，您希望将此新项目放置在此处。然后运行克隆命令：

   ```bash
   # 使用 URL 的示例
   git clone https://github.com/some-user/some-repo.git
   ```

   Git 将下载整个项目历史记录并创建一个名为 `some-repo`（或仓库名称）的新目录。然后您可以 `cd some-repo` 并使用相同的 `status`、`add`、`commit` 和 `log` 命令开始操作。克隆会自动设置一个连接（称为“远程”，通常命名为 `origin`）到原始仓库 URL。

这些命令（`init`、`status`、`add`、`commit`、`log`、`clone` 和使用 `.gitignore`）构成使用 Git 的核心。通过对 `config.txt` 文件进行更多变动、添加新文件、暂存并提交它们来练习这些命令。在每个步骤中观察 `git status` 和 `git log` 的输出。掌握这些基本操作对于有效协作和管理数据工程项目的演变很有帮助。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  Git的官方综合指南，涵盖了基本命令和高级概念。
- [GitHub Docs: Getting Started with Git and GitHub](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEBs8WVmUSMZMrU6QLKOkHlyQ_MkMsK-Ixvm_QCwtDuo4EbJ3yN6f5kTvwmEO3EXRLjLbbxf39uJGkjjGMSJukqpkRCq8Ru8dY6306uR8Pv3xfFy8hR1LSAO5WJJTsyxAYv_4C01jeFyd20qirr-xFvfJfLeSiX2L2CuyYe2G2ZBYA31BzjNeAnF-U=) — GitHub Docs (2024)
  Publisher: GitHub
  提供Git基本概念以及将Git与GitHub结合用于项目协作的指南。

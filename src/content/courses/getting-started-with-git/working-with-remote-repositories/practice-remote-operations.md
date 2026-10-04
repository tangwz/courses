---
course: "getting-started-with-git"
chapter: "working-with-remote-repositories"
lesson: "practice-remote-operations"
sourceId: 1032
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories/practice-remote-operations"
title: "实践：克隆、推送和拉取"
description: "通过克隆远程仓库、进行本地更改、推送以及拉取更新，来练习与远程仓库的交互。"
order: 10
plots: []
sourceHash: "9bde2312ada3d56002d1e27301d17714364e089b30c540ab93de7676a3c223b6"
sourceCorrections: []
---

这项动手练习侧重于远程仓库的实际应用。它将指导您完成克隆、推送、抓取和拉取的过程，演示与远程仓库交互的典型工作流程。这将帮助您对这些基本协作命令建立深入的理解。

### 前提条件

开始之前，您需要一个托管远程仓库的地方。GitHub、GitLab 和 Bitbucket 等服务提供免费的公共和私有仓库。

1. **创建账户：** 如果您还没有账户，请在 [GitHub](https://github.com/)、[GitLab](https://gitlab.com/) 或 [Bitbucket](https://bitbucket.org/) 等平台注册一个。
2. **创建新仓库：** 在您选择的平台上，创建一个*新的空*仓库。
   - 给它一个简单的名称，例如 `git-practice-remote`。
   - 确保它是**空的**。暂时不要从平台的选项中初始化 README、许可证或 `.gitignore` 文件。我们希望首先将本地工作推送到一个完全空白的远程仓库。
3. **复制仓库 URL：** 创建后，平台将显示您仓库的 URL。复制 HTTPS URL。它看起来会像 `https://github.com/YourUsername/git-practice-remote.git` 或 `https://gitlab.com/YourUsername/git-practice-remote.git`。克隆时您会用到这个 URL。

### 步骤 1：克隆远程仓库

克隆会在您的计算机上创建远程仓库的本地副本。它还会自动将原始仓库 URL 配置为名为 `origin` 的远程连接。

打开您的终端或命令提示符。切换到您想存储项目的目录（例如您的桌面或 `projects` 文件夹）。然后，运行 `git clone` 命令，将 `<repository_url>` 替换为您复制的 HTTPS URL：

```bash
git clone <repository_url>
```

例如：

```bash
git clone https://github.com/YourUsername/git-practice-remote.git
```

Git 将下载该（目前为空的）仓库。您可能会看到类似“warning: You appear to have cloned an empty repository.”的警告。这是预期的结果。

现在，进入新创建的目录：

```bash
cd git-practice-remote
```

让我们验证远程连接是否已自动设置。使用 `git remote -v` 查看已配置的远程连接：

```bash
git remote -v
```

您应该会看到类似这样的输出，显示 `origin` 远程连接指向您的仓库 URL，用于抓取和推送：

```
origin  https://github.com/YourUsername/git-practice-remote.git (fetch)
origin  https://github.com/YourUsername/git-practice-remote.git (push)
```

### 步骤 2：进行并提交本地更改

现在，让我们向本地仓库添加一些内容。使用文本编辑器或命令行创建一个简单的 `README.md` 文件：

```bash
# 在 Linux/macOS 上
echo "# Git Practice Remote Repository" > README.md

# 在 Windows 上 (命令提示符)
echo # Git Practice Remote Repository > README.md

# 在 Windows 上 (PowerShell)
"# Git Practice Remote Repository" | Out-File -Encoding UTF8 README.md
```

现在，按照标准的 Git 工作流程来暂存和提交这个新文件：

1. **检查状态：** 查看 `README.md` 是否未被跟踪。

   ```bash
   git status
   ```
2. **暂存文件：** 将 `README.md` 添加到暂存区。

   ```bash
   git add README.md
   ```
3. **再次检查状态：** 查看 `README.md` 是否已暂存。

   ```bash
   git status
   ```
4. **提交更改：** 将快照保存到您的本地仓库历史记录中。

   ```bash
   git commit -m "Add initial README file"
   ```

您可以使用 `git log --oneline` 查看您的本地提交：

```bash
git log --oneline
```

您将看到您的提交及其唯一标识符被列出。

### 步骤 3：将本地更改推送到远程

您的提交目前只存在于本地机器上。要共享或备份它，您需要将其推送到 `origin` 远程连接。

命令是 `git push <remote_name> <branch_name>`。在本例中，远程连接是 `origin`，默认分支通常是 `main`（在较旧的设置中可能是 `master`，Git 通常会告诉您）。

```bash
git push origin main
```

- **认证：** 首次通过 HTTPS 推送到远程连接时，Git 可能会提示您输入用户名和密码（或者个人访问令牌，这更安全，并且通常是 GitHub 等平台所要求的）。输入您用于创建仓库的账户凭据。
- **输出：** 如果成功，您将看到输出，表明对象已写入且推送已完成，并提及本地分支 (`main`) 正在跟踪远程分支 (`origin/main`)。

现在，刷新 GitHub/GitLab/Bitbucket 上的仓库页面。您应该会看到您的 `README.md` 文件和提交消息“Add initial README file”。您的本地更改现在已反映在远程服务器上！

### 步骤 4：模拟远程更改

假设协作者（或您，从另一台计算机操作）向远程仓库推送了一个更改。我们可以使用您的托管平台的网页界面轻松模拟这一点。

1. 前往 GitHub/GitLab/Bitbucket 上的仓库页面。
2. 点击 `README.md` 文件。
3. 找到一个“编辑”按钮（通常看起来像铅笔图标）。
4. 向文件中添加新的一行，例如：“This line was added online.”
5. 直接通过网页界面提交更改。使用提交消息，例如“Update README from web”。

远程仓库 (`origin`) 现在包含一个您的本地仓库尚未拥有的提交。

### 步骤 5：抓取和拉取远程更改

让我们将远程上进行的更改拉取到本地仓库。

#### 使用 `git fetch`

首先，我们使用 `git fetch`。此命令会从远程下载新数据，但*不会*将其集成到您的本地工作文件。它会更新您的远程跟踪分支（例如 `origin/main`）。

```bash
git fetch origin
```

您将看到输出表明对象已接收。现在，检查状态：

```bash
git status
```

Git 可能会告诉您：“Your branch is behind 'origin/main' by 1 commit, and can be fast-forwarded.”这意味着您的本地 `main` 分支落后于 `git fetch` 刚刚更新的 `origin/main` 分支。您的工作目录文件 (`README.md`) 仍未更改。

您可以使用 `git log` 查看本地和远程分支的最新提交：

```bash
git log --oneline main origin/main
```

这会显示在线提交已存在于 `origin/main` 上，但尚未存在于您的本地 `main` 上。

#### 使用 `git pull`

要更新您的本地 `main` 分支*以及*您的工作目录文件，您通常使用 `git pull`。此命令实际上是运行 `git fetch`，然后是 `git merge`（或 `git rebase`，取决于配置，但合并是默认设置）。

```bash
git pull origin main
```

由于您的本地 `main` 分支没有与远程历史记录发生分歧（您没有进行任何新的本地提交），这很可能导致“快进”合并。Git 只是将 `main` 分支指针向前移动以匹配 `origin/main`。

现在，检查您的本地 `README.md` 文件的内容：

```bash
# 在 Linux/macOS 上
cat README.md

# 在 Windows 上 (命令提示符)
type README.md

# 在 Windows 上 (PowerShell)
Get-Content README.md
```

您应该会看到“This line was added online.”这一行。

再次检查您的日志：

```bash
git log --oneline
```

您现在将在本地历史记录中看到提交“Update README from web”。您的本地仓库已与远程完全同步。

### 实践总结

您已成功完成了远程工作流的步骤：

1. **克隆**了远程仓库 (`git clone`)。
2. 进行了**本地更改** (`git add`, `git commit`)。
3. 将这些更改**推送**到远程 (`git push origin main`)。
4. 模拟了远程更改。
5. **抓取**了远程更改以更新远程跟踪分支 (`git fetch origin`)。
6. **拉取**了远程更改以更新本地分支和工作目录 (`git pull origin main`)。

这种拉取更改、进行本地提交和推送工作的循环是使用 Git 进行备份和协作的核心。请继续练习这些命令，因为它们是使用共享 Git 仓库的基础。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  一本全面且免费的Git指南，涵盖了Git的所有方面，并有关于远程仓库、克隆、获取、拉取和推送的专门章节。在线版本持续更新。
- [git-remote(1) Manual Page](https://git-scm.com/docs/git-remote) — The Git Community (2024)
  Git远程仓库管理的官方参考手册，提供了git remote、git clone、git fetch、git pull和git push等命令的详细用法和选项。
- [Git remote repositories](https://www.atlassian.com/git/tutorials/git-remote) — Atlassian (2024)
  Publisher: Atlassian
  一个实用的教程，解释了Git中远程仓库的作用，为克隆、获取、拉取和推送等常见操作提供了清晰的解释和工作流程。

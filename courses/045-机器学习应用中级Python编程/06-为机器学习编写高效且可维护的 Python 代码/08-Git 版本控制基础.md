# Git 版本控制基础

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-6-efficient-maintainable-python-ml/git-version-control-ml)

[返回章节目录](README.md) · [返回课程目录](../README.md)

编写可维护、可协作的机器学习 (machine learning)代码是一个重要目标。机器学习项目本身就是迭代的。你会调整算法、尝试各种特征、更改超参数 (parameter) (hyperparameter)，并重构代码。如果没有一个系统来管理这些更改，项目很快就会变得混乱，使得回退到之前可用的状态、理解代码的演变或与他人有效协作变得困难。正是在这种情况下，版本控制系统 (VCS) 变得不可或缺，而 Git 则是软件开发和数据科学社群的事实标准。

### 版本控制对机器学习 (machine learning)为何重要

在机器学习的背景下，版本控制提供多项益处：

1. **可复现性：** Git 允许你为代码的特定版本打上标签，这些版本对应着特定的实验或模型结果。这对于以后或由他人复现研究成果非常必要。
2. **实验跟踪：** 你可以使用 Git 中的分支来隔离不同的实验方法（例如，尝试新的算法、不同的特征集或超参数 (parameter) (hyperparameter)调整策略），而不会扰乱代码的主要工作版本。
3. **协作：** Git 提供了一个框架，允许多个团队成员同时处理同一代码库，系统地合并他们的更改并在出现冲突时解决。
4. **代码恢复：** 它起到了安全网的作用。如果你引入了错误或发现某个方法行不通，Git 可以让你简单地将代码恢复到之前已知良好状态。
5. **理解项目历史：** Git 维护着每次更改的详细日志，包括谁做了更改、何时更改以及为何更改（通过提交信息）。这段历史对于调试和理解项目如何演变非常有价值。

### Git 核心理念与命令

Git 基于**仓库**（通常缩写为“repo”）的理念来运行，它本质上是一个目录，包含你的项目文件和一个隐藏的 `.git` 子目录，Git 在其中存储所有版本历史和元数据。

以下是你将经常用到的一些基本 Git 操作：

- **初始化仓库：** 要开始使用 Git 跟踪项目，请在终端中导航到你的项目目录并运行：

  ```bash
  git init
  ```

  这会创建 `.git` 子目录，将当前目录变为一个 Git 仓库。
- **检查状态：** 要查看仓库的当前状态（哪些文件已修改、已暂存或未被跟踪），请使用：

  ```bash
  git status
  ```
- **暂存更改：** 在保存快照（即一次提交）之前，你需要告诉 Git 你想包含哪些更改。这称为暂存。要暂存特定文件中的更改：

  ```bash
  git add <filename>
  ```

  要暂存当前目录及其子目录中所有已修改和新增的文件：

  ```bash
  git add .
  ```
- **提交更改：** 提交会将你暂存的更改永久保存到仓库的历史记录中。每次提交都有一个唯一标识符，并需要一条描述更改的信息。

  ```bash
  git commit -m "Your descriptive commit message"
  ```

  好的提交信息应简洁但信息丰富（例如，“添加使用 StandardScaler 进行数据缩放”、“重构数据加载函数”、“尝试 Random Forest 分类器”）。
- **查看历史：** 要查看提交序列：

  ```bash
  git log
  ```

  此命令显示提交标识符、作者、日期和信息。

### 用于实验的分支与合并

Git 最强大的功能之一是分支。一个分支代表一个独立的开发线。默认分支通常命名为 `main`（或旧仓库中的 `master`）。

- **创建分支：** 要为实验创建一个新分支，例如测试不同的特征工程方法：

  ```bash
  git branch experiment-feature-scaling
  ```
- **切换分支：** 要开始在新分支上工作：

  ```bash
  git checkout experiment-feature-scaling
  ```

  或者，一步创建并切换到一个新分支：

  ```bash
  git checkout -b experiment-new-model
  ```
- **合并分支：** 一旦你对实验分支上的更改感到满意，就可以将它们合并回你的主要开发线（例如 `main`）。首先，切换回目标分支：

  ```bash
  git checkout main
  ```

  然后，合并实验分支的更改：

  ```bash
  git merge experiment-feature-scaling
  ```

  Git 会尝试自动合并更改。如果两个分支中对同一行进行了冲突的更改，Git 将暂停合并并要求你手动解决冲突。

> 一个简单的 Git 工作流程，说明了如何从 `main` 创建一个 `实验` 分支，在两个分支上进行提交，然后将 `实验` 分支合并回 `main`。

### 使用远程仓库

虽然 Git 在本地工作，但当与托管在 GitHub、GitLab 或 Bitbucket 等平台上的远程仓库一起使用时，其协作能力便会显著发挥。

- **克隆：** 要获取现有远程仓库的本地副本：

  ```bash
  git clone <repository_url>
  ```

  这会下载整个项目历史并设置远程连接（通常命名为 `origin`）。
- **拉取：** 要从远程仓库获取更改并将其合并到你当前的本地分支：

  ```bash
  git pull origin main
  ```

  （将 `main` 替换为相应的分支名称）。在开始新工作或推送自己的更改之前，先拉取更改是一个好习惯。
- **推送：** 要将你的本地提交上传到远程仓库：

  ```bash
  git push origin main
  ```

  （将 `main` 替换为你想推送的分支）。通常，你会将本地分支的提交推送到远程仓库中对应的分支。

### 机器学习 (machine learning)项目中的 Git 实践

- **频繁提交：** 进行小而有逻辑的提交。这使得理解更改和在需要时进行回退变得更容易。
- **使用 `.gitignore`：** 在你的仓库根目录中创建 `.gitignore` 文件，列出 Git 应该忽略的文件和目录（例如，大型数据集、虚拟环境文件夹、临时文件、凭证）。

  ```gitignore
  # 机器学习项目的 .gitignore 示例
  *.csv
  *.pkl
  data/
  models/
  __pycache__/
  *.pyc
  .ipynb_checkpoints/
  venv/
  *.env
  ```
- **跟踪配置：** 将配置文件（例如，包含超参数 (parameter) (hyperparameter)的 YAML 文件）与你的代码一同提交。
- **为功能/实验创建分支：** 广泛使用分支来隔离工作。这能保持你的 `main` 分支整洁且可部署。
- **考虑使用 Git LFS：** 对于像训练好的模型或中等大小的数据集等不易排除的大文件版本管理，可以研究 Git 大型文件存储 (LFS)。它会将 Git 仓库中的大文件替换为文本指针，同时将实际文件内容存储在单独的 LFS 服务器上。

掌握基本的 Git 命令是任何开发者（包括从事机器学习的人员）的基本技能。它提供了管理代码演变、促进协作以及确保机器学习实验可复现性所需的结构，为构建高效和可维护的系统做出重要贡献。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  官方、全面的在线指南，涵盖Git的所有方面，从基础命令到高级工作流。
- [Getting Started with Git](https://docs.github.com/en/get-started/getting-started-with-git) — GitHub, Inc. (2024)
  Publisher: GitHub, Inc.
  提供使用 Git 的实用教程和解释，特别是在与 GitHub 协作方面。
- [Git Large File Storage (Git LFS)](https://git-lfs.github.com/) — Git LFS Contributors (2024)
  Git LFS 的官方网站和文档，解释了其在 Git 仓库中管理大文件的作用，这在机器学习项目中很常见。

---

[上一节](07-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95%E4%BB%8B%E7%BB%8D.md) · [下一节](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E9%87%8D%E6%9E%84%E4%B8%8E%E4%BC%98%E5%8C%96%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%BB%A3%E7%A0%81%E7%89%87%E6%AE%B5.md)

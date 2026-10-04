---
course: "getting-started-with-git"
chapter: "first-git-repository"
lesson: "git-log-command"
sourceId: 1001
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-2-first-git-repository/git-log-command"
title: "查看提交历史 (git log)"
description: "了解 `git log` 命令，查看仓库中提交的历史记录。"
order: 7
plots: []
sourceHash: "7b5bada4ddc8d9abdda6912f8d19d35957a5e908a826670ab2c8549b22b36486"
sourceCorrections: []
---

Git 项目通过将重要更改保存为提交来构建历史。为了有效管理和审查您的工作，了解如何查看此历史记录至关重要。`git log` 命令提供了一种强大的方式来回顾您保存的所有更改。可以将 `git log` 视为您项目的详细日志，记录着每个提交。

在您的终端中，于仓库目录下运行 `git log` 命令，将显示一个按时间顺序排列的提交列表，最新提交在前。

```bash
cd your-repo-directory
git log
```

每个提交的输出通常如下所示：

```
commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0 (HEAD -> main)
Author: Your Name <your.email@example.com>
Date:   Tue May 21 10:30:00 2024 +0000

    Initial commit: Add project structure

commit f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0
Author: Your Name <your.email@example.com>
Date:   Tue May 21 09:45:15 2024 +0000

    Add README file
```

我们来细看每个提交所显示的信息：

- **提交哈希值 (Commit Hash):** 这串长长的字母和数字（例如 `a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0`）是提交的唯一标识符，从技术上讲是一个 SHA-1 哈希值。这个 ID 确保了提交的完整性，并允许 Git（和您）指向历史中的特定时间点。实际上，您通常只需要前 7-10 个字符就能唯一识别一个提交。`(HEAD -> main)` 部分表明 `HEAD`（您在项目历史中的当前位置）正指向 `main` 分支，并且这个提交是该分支上的最新提交。我们稍后会更详细地讨论分支和 `HEAD`。
- **作者 (Author):** 这显示了最初编写这些更改的人的姓名和电子邮件地址。此信息来自您之前设置的 `user.name` 和 `user.email` 配置。
- **日期 (Date):** 时间戳表示提交的创建时间。
- **提交信息 (Commit Message):** 这是您运行 `git commit -m "您的信息"` 时提供的一些描述性文字。如您所见，清晰的信息（如“编写好的提交信息”中讨论的那样）会使日志更有助于理解。

### 自定义日志输出

默认的 `git log` 输出信息量大，但随着项目规模的增长，它可能变得过于冗长。Git 提供了许多选项来自定义输出，以准确显示您所需的信息。以下是一些常用且有帮助的选项：

#### 使用 `--oneline` 获取简洁摘要

要获得一个非常紧凑的视图，请使用 `--oneline` 选项。它只显示每个提交哈希值的前几位和提交信息的主题行：

```bash
git log --oneline
```

输出：

```
a1b2c3d (HEAD -> main) Initial commit: Add project structure
f9e8d7c Add README file
```

这对于快速查看最近的更改很有帮助。

#### 使用 `--stat` 查看更改

如果您想查看每个提交中哪些文件被更改，以及添加或删除了多少行的摘要，请使用 `--stat` 选项：

```bash
git log --stat
```

输出：

```
commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0 (HEAD -> main)
Author: Your Name <your.email@example.com>
Date:   Tue May 21 10:30:00 2024 +0000

    Initial commit: Add project structure

 src/app.js | 1 +
 src/style.css | 1 +
 2 files changed, 2 insertions(+)

commit f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0
Author: Your Name <your.email@example.com>
Date:   Tue May 21 09:45:15 2024 +0000

    Add README file

 README.md | 5 +++++
 1 file changed, 5 insertions(+)
```

这显示了相对于项目根目录的文件路径，以及一个简单的视觉呈现（`+` 表示添加，`-` 表示删除）和一行汇总信息。

#### 使用 `--patch` (`-p`) 查看实际更改

要查看每个提交引入的具体更改（即“diff”或“patch”），请使用 `-p` 或 `--patch` 选项：

```bash
git log -p
```

输出（一个提交的摘录）：

```
commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0 (HEAD -> main)
Author: Your Name <your.email@example.com>
Date:   Tue May 21 10:30:00 2024 +0000

    Initial commit: Add project structure

diff --git a/src/app.js b/src/app.js
new file mode 100644
index 0000000..e69de29
diff --git a/src/style.css b/src/style.css
new file mode 100644
index 0000000..e69de29
```

此输出以类似于 `diff` 命令的格式显示差异。以 `+` 开头的行表示已添加，以 `-` 开头的行表示已删除（尽管在这个简单的添加新文件的示例中没有删除）。这个选项信息内容很全面，有助于理解具体的更改，但可能会产生大量输出。

#### 限制提交数量

如果您只想查看最近的几个提交，可以使用 `-n` 选项，后跟您想查看的提交数量。例如，只查看最近的三个提交：

```bash
git log -n 3
```

您可以组合这些选项。例如，要查看最近两个提交的简洁视图并附带统计信息：

```bash
git log --oneline --stat -n 2
```

输出：

```
a1b2c3d (HEAD -> main) Initial commit: Add project structure
 src/app.js | 1 +
 src/style.css | 1 +
 2 files changed, 2 insertions(+)
f9e8d7c Add README file
 README.md | 5 +++++
 1 file changed, 5 insertions(+)
```

`git log` 命令是您查看使用 `git commit` 所建立历史的窗口。使用这些选项将帮助您有效地查阅和了解项目的演变。随着学习的进行，您会发现 `git log` 是一个不可或缺的辅助手段，用于查看更改、找出特定修改的引入时间以及了解协作伙伴的工作。

## 参考资料

- [git-log](https://git-scm.com/docs/git-log) — Git Community (2024)
  `git log` 命令的官方、最全面的参考资料，详细介绍了所有选项和高级用法。
- [Pro Git](https://git-scm.com/book/en/v2/Git-Basics-Viewing-the-Commit-History) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress; Volume: 2
  一本关于 Git 的权威且免费的书籍，其中有一章专门解释了如何使用 `git log` 及其各种选项查看提交历史。
- [Git log](https://www.atlassian.com/git/tutorials/git-log) — Atlassian (2024)
  Publisher: Atlassian
  一个清晰实用的教程，解释了 `git log` 命令及其常用标志，适用于初学者和快速参考。

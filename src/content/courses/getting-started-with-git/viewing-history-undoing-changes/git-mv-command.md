---
course: "getting-started-with-git"
chapter: "viewing-history-undoing-changes"
lesson: "git-mv-command"
sourceId: 1011
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-3-viewing-history-undoing-changes/git-mv-command"
title: "移动或重命名文件 (git mv)"
description: "学习使用 `git mv` 命令在 Git 仓库中重命名或移动文件的正确方法。"
order: 8
plots: []
sourceHash: "2bb8c004b3301e520c2f7e1050cbb4568771991d0a08bda7aa358a3a093d469b"
sourceCorrections: []
---

随着项目的发展，你将不可避免地需要重新组织文件。你可能会重命名文件，使其更能体现其用途，或者将其移动到其他目录以改进结构。尽管你可以直接使用操作系统的命令（如 Linux/macOS 上的 `mv` 或 Windows 上的 `ren`/`move`），但 Git 提供了一个专门的命令 `git mv`，以便在版本控制环境下更妥善地处理这些操作。

使用 `git mv` 会直接告知 Git 文件已被重命名或移动，而不是将其视为旧文件的删除和新文件的添加（未被追踪）。这有助于 Git 在重命名或移动操作过程中追踪文件的历史。

### 使用 git mv 重命名文件

假设你有一个名为 `temp_script.py` 的文件，你想将其重命名为 `utility_script.py`。你可以简单地使用 `git mv`，而不是先使用操作系统的重命名命令，再执行 `git add` 和 `git rm`：

```bash
# 使用 Git 重命名文件
git mv temp_script.py utility_script.py

# 查看状态
git status
```

`git status` 的输出会类似这样：

```
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        renamed:    temp_script.py -> utility_script.py
```

注意 Git 是如何明确地将其识别为重命名操作的。这个更改已经暂存，为你的下一次提交做好了准备。你不需要单独运行 `git add` 或 `git rm`；`git mv` 会帮你处理。

### 使用 git mv 移动文件

移动文件的方式类似。假设你想把 `utility_script.py` 移动到一个名为 `scripts` 的新目录中。

首先，如果目录不存在，就创建它：

```bash
# 创建新目录（标准操作系统命令）
mkdir scripts
```

然后，使用 `git mv` 移动文件：

```bash
# 将文件移动到新目录
git mv utility_script.py scripts/utility_script.py

# 查看状态
git status
```

同样，`git status` 会显示暂存的更改，表明移动操作：

```
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        renamed:    utility_script.py -> scripts/utility_script.py
```

即使我们把文件移动到了不同的目录，Git 通常也会将其报告为“重命名”，因为从 Git 的角度看，文件的路径（包含目录）已经改变。关键是 Git 知道文件的身份得到了保留。

你也可以在一个步骤中同时移动和重命名文件：

```bash
# 同时移动并重命名文件
git mv old_location/old_name.txt new_directory/new_name.txt
```

### git mv 实际做了什么

`git mv` 命令本质上是一个方便的命令，它执行三个操作：

1. 在你的文件系统上重命名或移动文件（类似于操作系统的 `mv` 或 `ren`/`move` 命令）。
2. 从 Git 索引中移除旧文件路径（类似于 `git rm <old_path>`）。
3. 将新文件路径添加到 Git 索引（类似于 `git add <new_path>`）。

通过使用 `git mv`，你可以确保 Git 正确追踪文件在重命名或移动过程中的历史，这使其成为在仓库中管理文件位置的推荐方式。如果你忘记使用 `git mv`，而是使用了操作系统的工具，Git 会将旧文件视为已删除，新文件视为未追踪。那样你将需要手动 `git rm` 旧路径并 `git add` 新路径才能达到相同的结果。使用 `git mv` 可以简化这个过程。

## 参考资料

- [git-mv Documentation](https://git-scm.com/docs/git-mv) — Git Community (2024)
  `git mv` 命令的官方文档，详细说明了其用法和选项。
- [Pro Git](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository#_moving_files) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  一本关于 Git 的权威且全面的书籍，在基本的仓库操作中涵盖了 `git mv`。

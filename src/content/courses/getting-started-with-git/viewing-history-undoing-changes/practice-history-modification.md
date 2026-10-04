---
course: "getting-started-with-git"
chapter: "viewing-history-undoing-changes"
lesson: "practice-history-modification"
sourceId: 1012
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-3-viewing-history-undoing-changes/practice-history-modification"
title: "练习：检查和修改历史"
description: "通过练习查看差异、取消暂存文件、修订提交和撤销更改来应用你的知识。"
order: 9
plots: []
sourceHash: "62683d91455aa350829aa14ce0b98d13991042691c6a5c81989375b76158de40"
sourceCorrections: []
---

本节将演练常见情况，这些情况需要你查看项目历史、更正暂存或提交中的错误，以及管理 Git 仓库中的文件。

请确保打开终端或命令提示符，并进入一个你一直在使用的 Git 仓库，也许是上一章练习中使用的那个。如果你手头没有，创建一个新目录，进入该目录，然后运行 `git init`。然后，创建几个文件（例如，`README.md`、`script.py`），添加一些初始内容，并使用 `git add .` 和 `git commit -m "Initial commit"` 进行一两次提交。

### 使用 `git diff` 检查更改

首先，让我们看看 `git diff` 如何帮助我们了解更改。

1. **修改文件：** 打开仓库中的一个文件（例如，`README.md`），并添加新行文本或修改现有行。保存文件。
2. **检查状态：** 运行 `git status`。Git 将报告 `README.md` 已被修改但尚未暂存以供提交。

   ```bash
   $ git status
   On branch main # 或者 master，取决于你的默认设置
   Changes not staged for commit:
     (use "git add <file>..." to update what will be committed)
     (use "git restore <file>..." to discard changes in working directory)
           modified:   README.md

   no changes added to commit (use "git add" and/or "git commit -a")
   ```
3. **查看未暂存的更改：** 运行 `git diff`。此命令默认情况下显示工作目录和暂存区（索引）之间的差异。你会看到你添加的行（前缀为 `+`）或删除/更改的行（前缀为 `-`）。

   ```bash
   $ git diff
   diff --git a/README.md b/README.md
   index e69de29..b9ca75a 100644
   --- a/README.md
   +++ b/README.md
   @@ -0,0 +1 @@
   +This is the initial README content.
   +# 加上我添加的新行。 # 这一行将带有 '+' 前缀
   ```

   *(具体输出将取决于你的文件内容和更改)*
4. **暂存更改：** 将修改后的文件添加到暂存区：

   ```bash
   git add README.md
   ```
5. **查看已暂存的更改：** 现在，再次运行 `git diff`。注意它没有显示任何输出。这是因为你工作目录中的更改现在与暂存区匹配。要查看*已*暂存的内容（暂存区和上次提交之间的差异），请使用 `--staged`（或 `--cached`）标志：

   ```bash
   git diff --staged
   ```

   输出将与步骤 3 类似，显示你刚刚暂存的更改。
6. **提交更改：** 让我们提交这个已暂存的更改：

   ```bash
   git commit -m "Update README with additional info"
   ```
7. **比较提交：** 你也可以使用 `git diff` 比较两个提交。让我们比较最新提交 (`HEAD`) 和它之前的一个 (`HEAD~1`)：

   ```bash
   git diff HEAD~1 HEAD
   ```

   这将显示你刚刚进行的上次提交中引入的更改。

### 修正错误：取消暂存和修订

错误在所难免。也许你暂存了错误的文件，或者在上次提交消息中打错了字。

1. **意外暂存：** 修改另一个文件（例如，`script.py`），并*同时*创建一个你*暂时不*想提交的新临时文件（例如，`temp.log`）。现在，不小心将两者都暂存：

   ```bash
   # 修改 script.py
   echo "print('Hello Git!')" > script.py
   # 创建临时文件
   echo "Temporary log data" > temp.log
   # 暂存目录中的所有内容
   git add .
   ```
2. **检查已暂存的文件：** 运行 `git status`。你会看到 `script.py` 和 `temp.log` 都已暂存。
3. **取消暂存临时文件：** 你只打算暂存 `script.py`。使用 `git reset HEAD <file>` 将 `temp.log` 从暂存区移除：

   ```bash
   git reset HEAD temp.log
   ```
4. **验证：** 再次运行 `git status`。`script.py` 应该仍然已暂存，但 `temp.log` 现在应该显示为“未跟踪文件”。

   ```bash
   $ git status
   On branch main
   Changes to be committed:
     (use "git restore --staged <file>..." to unstage)
           modified:   script.py

   Untracked files:
     (use "git add <file>..." to include in what will be committed)
           temp.log
   ```
5. **提交时出现拼写错误：** 提交已暂存的更改，但消息中出现拼写错误：

   ```bash
   git commit -m "Add greating to script" # Oops, typo: "greating"
   ```
6. **修订提交消息：** 你刚刚进行了那个提交。要不创建*新*提交的情况下修正消息，请使用 `git commit --amend`：

   ```bash
   git commit --amend -m "Add greeting to script"
   ```

   这将打开你的默认文本编辑器（除非你像此处一样再次使用 `-m` 标志），让你编辑*最新*提交的消息。如果你检查 `git log -1`，你将看到更正后的消息，并且提交哈希值可能已经改变。**重要提示：** 避免修订你已共享（推送）到远程仓库的提交，因为这会重写历史记录。
7. **修订以添加更改：** 再次稍微修改 `script.py`。暂存更改（`git add script.py`）。现在，意识到这个小改动在逻辑上属于*上一个*提交（“Add greeting to script”）。此时，你可以不用创建新提交，而是再次使用 `--amend` 将这个已暂存的更改添加到上次提交中，并保留现有消息：

   ```bash
   git commit --amend --no-edit
   ```

   `--no-edit` 标志阻止编辑器打开，在将新暂存的更改合并到上次提交中的同时保留以前的消息。检查 `git log -p -1` 以查看修订后的提交中的合并更改。

### 使用 `git revert` 安全地撤销提交

有时，你发现之前的一个提交引入了错误或不想要的更改。你可以撤销 (`revert`) 它，而不是删除它（删除会重写历史记录，对共享仓库来说存在问题）。撤销会创建一个*新*提交，它正好抵消了先前提交引入的更改。

1. **确定目标提交：** 使用 `git log --oneline` 查找要撤销的提交的哈希值。例如，让我们撤销之前进行的“Update README with additional info”提交。复制它的短提交哈希值。
2. **撤销提交：** 运行 `git revert <commit-hash>`：

   ```bash
   # 将 <commit-hash> 替换为你日志中的实际哈希值
   git revert <commit-hash>
   ```
3. **确认撤销提交：** 你的编辑器将打开，其中包含预填充的提交消息，例如“Revert 'Update README with additional info'”。如果需要，你可以编辑此消息，然后保存并关闭编辑器。
4. **验证：** 再次运行 `git log --oneline`。你会看到顶部有一个带有“Revert...”消息的新提交。检查 `README.md` 的内容；被撤销的提交中引入的更改现在应该已消失。原始提交仍然存在于历史记录中，但其影响已被新的撤销提交抵消。

### 使用 `git rm` 和 `git mv` 管理文件

Git 不仅需要跟踪内容更改，还需要跟踪文件删除和重命名。

1. **跟踪并提交文件：** 让我们使用 `temp.log` 文件。暂存并提交它：

   ```bash
   git add temp.log
   git commit -m "Add temporary log file"
   ```
2. **删除文件：** 你决定不再需要 `temp.log`。要从项目中移除它*并*告诉 Git 停止跟踪它，请使用 `git rm`：

   ```bash
   git rm temp.log
   ```

   运行 `ls`（或 Windows 上的 `dir`）将显示文件已从你的工作目录中消失。`git status` 将显示删除已暂存。
3. **提交删除：**

   ```bash
   git commit -m "Remove temporary log file"
   ```
4. **重命名文件：** 让我们将 `script.py` 重命名为 `app.py`。Git 的方法是 `git mv`：

   ```bash
   git mv script.py app.py
   ```

   此命令实际上做两件事：它重命名文件系统上的文件（`mv script.py app.py` 或 `ren script.py app.py`）并暂存此更改（`git add app.py` 和 `git rm script.py`）。
5. **验证并提交重命名：** 检查 `git status`。它应该显示 `renamed: script.py -> app.py`。提交更改：

   ```bash
   git commit -m "Rename script.py to app.py"
   ```

   Git 足以在重命名时跟踪历史记录。

本次练习涵盖了检查更改（`git diff`）、取消暂存（`git reset HEAD`）、修改上次提交（`git commit --amend`）、安全地撤销过去的提交（`git revert`），以及正确删除（`git rm`）和重命名（`git mv`）已跟踪文件。这些是让你的项目历史保持清晰易懂的必备工具。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon, Ben Straub (2014)
  Publisher: Apress
  一本关于Git的指南，涵盖了本节中讨论的所有命令和概念，包括检查历史记录、撤销更改和管理文件。
- [Git Reference Manual: git-diff](https://git-scm.com/docs/git-diff) — The Git Community (2024)
  官方文档，提供Git中检查更改的详细语法和选项，对理解`git diff`命令至关重要。
- [Git Reference Manual: git-commit](https://git-scm.com/docs/git-commit) — The Git Community (2024)
  解释`git commit`命令的官方文档，包括用于修改最新提交的`--amend`选项。
- [Git Reference Manual: git-revert](https://git-scm.com/docs/git-revert) — The Git Community (2024)
  安全撤销过去提交的官方文档，通过创建新提交来恢复更改。

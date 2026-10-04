---
course: "getting-started-with-git"
chapter: "branching-merging-basics"
lesson: "practice-branching-merging"
sourceId: 1022
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-4-branching-merging-basics/practice-branching-merging"
title: "动手实践：分支与合并工作流程"
description: "练习创建功能分支、进行修改并将其合并回主分支的常用流程。"
order: 10
plots: []
sourceHash: "9415bb153394b47f53206007319ec83b7b08f0e554fe9b5b33fc14538149c603"
sourceCorrections: []
---

这个动手练习演示了分支与合并的实际应用。它模拟了一种常见的开发工作流程：为新功能创建单独的分支，进行修改，然后将这些修改整合回主要开发线。你还将模拟并解决合并冲突。

首先，请确保你有一个可用的 Git 仓库。如果你已经完成了第2章的实践部分，可以使用那个仓库。否则，请创建一个新的：

```bash
# 为我们的练习项目创建一个目录
mkdir git-branch-practice
cd git-branch-practice

# 初始化一个 Git 仓库
git init

# 创建一个初始文件并提交
echo "Project Initial Content." > README.md
git add README.md
git commit -m "Initial commit with README"

# 为历史记录添加另一个小修改
echo "Version 1.0" >> README.md
git add README.md
git commit -m "Add version info to README"
```

现在你应该在默认分支（可能名为 `main` 或 `master`）上，有两个提交。你可以使用 `git log --oneline` 来验证这一点。

### 步骤 1：创建并切换到功能分支

假设我们需要在 `README.md` 文件中添加一个新章节，来描述一个新功能。最好是在一个单独的分支上进行此操作。

1. **创建一个新分支**，名为 `add-feature-description`：

   ```bash
   git branch add-feature-description
   ```

   这个命令创建了分支，但尚未切换过去。
2. **列出分支**，查看新分支：

   ```bash
   git branch
   ```

   输出：

   ```
   * main
     add-feature-description
   ```

   星号 (\*) 表示当前活跃的分支 (`main`)。
3. **切换到新分支**：

   ```bash
   git switch add-feature-description
   ```

   （或者，你可以使用 `git checkout add-feature-description`）。
   Git 将确认：`Switched to branch 'add-feature-description'`

   你也可以使用 `git switch -c add-feature-description` 或 `git checkout -b add-feature-description` 一步完成创建和切换。

### 步骤 2：在功能分支上进行修改

现在你已在 `add-feature-description` 分支上，你进行的任何提交都将记录在此分支上，与 `main` 分支分开。

1. **修改文件**：向 `README.md` 添加功能描述。

   ```bash
   echo "" >> README.md # 添加一个空行以作分隔
   echo "## New Feature" >> README.md
   echo "This section describes the amazing new feature." >> README.md
   ```
2. **检查状态**：

   ```bash
   git status
   ```

   输出：

   ```
   On branch add-feature-description
   Changes not staged for commit:
     (use "git add <file>..." to update what will be committed)
     (use "git restore <file>..." to discard changes in working directory)
           modified:   README.md

   no changes added to commit (use "git add" and/or "git commit -a")
   ```
3. **暂存并提交**修改：

   ```bash
   git add README.md
   git commit -m "Add feature description section"
   ```

你的功能分支现在有了一个 `main` 分支没有的提交。

### 步骤 3：切回主分支并进行修改

让我们模拟在功能开发期间主分支上的工作。也许需要修复一个拼写错误。

1. **切回** `main` 分支：

   ```bash
   git switch main
   ```
2. **修改 `main` 上的文件**：让我们修改 `README.md` 中的第一行。用文本编辑器打开 `README.md`，将 `Project Initial Content.` 改为 `Project Initial Content`。（删除末尾的句号）。保存文件。
3. **暂存并提交** `main` 上的此修改：

   ```bash
   git add README.md
   git commit -m "Fix typo in README title"
   ```

此时，`main` 分支和 `add-feature-description` 分支已经分叉。每个分支都有对方没有的一个新提交。

> 合并前的仓库状态。`main` 指向提交 `C3`（拼写修正），`add-feature-description` 指向 `C2`（功能描述）。两个分支都源自提交 `C1`。

### 步骤 4：将功能分支合并到主分支

功能已完成，我们希望将其整合到主项目。

1. **确保你在接收分支**（本例中是 `main`）上。我们在上一步中已经切换了。如果不确定，运行 `git status` 或 `git branch`。
2. **运行合并命令**：

   ```bash
   git merge add-feature-description
   ```

由于两个分支的修改都源自同一个共同的祖先 (`C1`)，并且修改在文件的不同部分（标题与末尾的新章节），Git 很可能可以自动执行合并。因为历史记录分叉了，Git 会创建一个 *合并提交*。

你可能会看到一个编辑器打开，要求输入合并提交信息。默认信息通常就足够了。如果出现编辑器，请保存并关闭它。

输出可能如下所示：

```
Merge made by the 'recursive' strategy.
 README.md | 4 ++++
 1 file changed, 4 insertions(+)
```

3. **检查日志**：

   ```bash
   git log --oneline --graph
   ```

   输出将显示合并提交，将历史记录汇合在一起：

   ```
   *   commit-hash-M (HEAD -> main) Merge branch 'add-feature-description'
   |\
   | * commit-hash-C2 (add-feature-description) Add feature description section
   * | commit-hash-C3 Fix typo in README title
   |/
   * commit-hash-C1 Add version info to README
   * commit-hash-C0 Initial commit with README
   ```

你的 `main` 分支上的 `README.md` 文件现在包含了拼写修正和新功能章节 *两者*。

### 步骤 5：模拟并解决合并冲突

现在，让我们创建一个 Git *无法* 自动合并修改的情况，因为两个分支修改了同一行。

1. 从 `main` 分支（现在已包含第一个功能）**创建另一个分支**：

   ```bash
   git switch -c update-version main
   ```
2. 在此新分支上**修改 `README.md`**。修改版本行：
   修改 `README.md`，使第二行显示 `Version 1.1` 而不是 `Version 1.0`。保存文件。
3. 在 `update-version` 上**暂存并提交**：

   ```bash
   git add README.md
   git commit -m "Update version to 1.1"
   ```
4. **切回 `main`**：

   ```bash
   git switch main
   ```
5. **修改 `main` 上的 `README.md`**：模拟另一个开发者独立更新版本，可能更新为 `Version 1.0.1`。
   修改 `README.md`，使第二行显示 `Version 1.0.1`。保存文件。
6. 在 `main` 上**暂存并提交**：

   ```bash
   git add README.md
   git commit -m "Update version to 1.0.1"
   ```

现在，自它们上次共同的祖先（步骤 4 的合并提交）以来，`main` 和 `update-version` 都修改了同一行（版本行）。

7. **尝试将** `update-version` **合并到 `main`**：

   ```bash
   git merge update-version
   ```

Git 将检测到冲突：

```
Auto-merging README.md
CONFLICT (content): Merge conflict in README.md
Automatic merge failed; fix conflicts and then commit the result.
```

### 步骤 6：解决冲突

Git 无法决定保留哪一行的版本，因此它在文件中标记 (token)了冲突并暂停了合并。

1. **检查状态**：

   ```bash
   git status
   ```

   输出：

   ```
   On branch main
   You have unmerged paths.
     (fix conflicts and run "git commit")
     (use "git merge --abort" to abort the merge)

   Unmerged paths:
     (use "git add <file>..." to mark resolution)
           both modified:   README.md

   no changes added to commit (use "git add" and/or "git commit -a")
   ```

   它清楚地显示 `README.md` 存在未合并的路径（冲突）。
2. 在你的编辑器中**打开冲突文件** (`README.md`)。你将看到 Git 添加的冲突标记：

   ```
   Project Initial Content.
   <<<<<<< HEAD
   Version 1.0.1
   =======
   Version 1.1
   >>>>>>> update-version

   ## New Feature
   This section describes the amazing new feature.
   ```

   - `<<<<<<< HEAD`：表示当前分支 (`main`) 中冲突行的开始。
   - `=======`：分隔了两个分支的冲突行。
   - `>>>>>>> update-version`：表示正在合并的分支 (`update-version`) 中冲突行的结束。
3. **编辑文件以解决冲突**：你需要手动编辑此部分以包含你想要保留的内容。移除冲突标记（`<<<`、`===`、`>>>`）。例如，我们决定 `Version 1.1` 是正确的版本：

   ```
   Project Initial Content.
   Version 1.1

   ## New Feature
   This section describes the amazing new feature.
   ```

   在进行所需更改并移除标记后保存文件。
4. **暂存已解决的文件**：告诉 Git 你已修复冲突：

   ```bash
   git add README.md
   ```
5. **提交合并**：完成合并操作。Git 通常会建议一个提交信息。

   ```bash
   git commit
   ```

   （如果你的编辑器打开，只需保存并关闭默认消息：“Merge branch 'update-version'”）。

冲突已解决，合并已完成。

### 步骤 7：清理分支

一旦功能分支成功合并并且你不再需要它，你就可以删除它。

1. **列出分支**：

   ```bash
   git branch
   ```

   输出：

   ```
   * main
     add-feature-description
     update-version
   ```
2. **删除已合并的分支**：

   ```bash
   git branch -d add-feature-description
   git branch -d update-version
   ```

   输出：

   ```
   Deleted branch add-feature-description (was commit-hash-C2).
   Deleted branch update-version (was commit-hash-...).
   ```

   `-d` 标志只允许在分支已完全合并的情况下删除。如果需要强制删除，请使用 `-D`（大写 D），但要谨慎。

你现在已经成功练习了创建分支、进行独立修改、将它们合并回来（包括非快进合并）、解决合并冲突以及清理废弃分支。这个工作流程构成了许多使用 Git 的协作开发策略的基础。

## 参考资料

- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  Git的综合指南，涵盖分支和合并的基本概念，包括命令和常见工作流程的详细解释。此在线资源持续更新。
- [\`git-merge\` Documentation](https://git-scm.com/docs/git-merge) — The Git Development Community (2024)
  《git merge》命令的官方参考文档，详细说明其用法、策略和选项，有助于理解Git如何合并不同历史。

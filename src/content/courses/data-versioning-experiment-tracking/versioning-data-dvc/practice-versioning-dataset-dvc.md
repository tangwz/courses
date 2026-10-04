---
course: "data-versioning-experiment-tracking"
chapter: "versioning-data-dvc"
lesson: "practice-versioning-dataset-dvc"
sourceId: 4074
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-2-versioning-data-dvc/practice-versioning-dataset-dvc"
title: "动手实践：数据集版本管理"
description: "练习设置 DVC，添加数据集，推送到远程存储，并切换不同版本。"
order: 8
plots: []
sourceHash: "3e28e4954896294fe6b7a88593bd4dbad3286d182f2427d0955a7508ba8fb311"
sourceCorrections: []
---

本次实践操作将指导您完成 DVC 的基本工作流程。它包括在项目中初始化 DVC、追踪数据集、配置远程存储、推送数据、修改数据以及在不同版本间切换。

### 准备工作

开始之前，请确保您已安装以下内容：

1. **Git：** DVC 很大程度上依赖于 Git。如果您没有安装，请从 [git-scm.com](https://git-scm.com/) 下载并安装。
2. **DVC：** 使用 pip 安装 DVC。为了支持常见的云存储选项（如 AWS S3、Google Cloud Storage、Azure Blob），您可以安装可选依赖项。在此示例中，我们将首先使用本地远程存储，因此基本安装就足够了，但安装额外组件是很好的做法：

   ```bash
   pip install dvc[s3,gcs,azure] # 或者只运行 pip install dvc
   ```
3. **Python（可选）：** 我们会使用一个简单的 Python 脚本生成一些示例数据，但您也可以手动创建示例 CSV 文件。

### 1. 项目设置

首先，为我们的练习项目创建一个新目录并进入该目录。然后，初始化 Git 和 DVC。

```bash
mkdir dvc-practice
cd dvc-practice

# 初始化 Git 仓库
git init
Initialized empty Git repository in /path/to/dvc-practice/.git/

# 初始化 DVC
dvc init
Initialized DVC repository.
You can now commit the changes to git.

+---------------------------------------------------------------------+
|                                                                     |
|        DVC 已启用匿名聚合使用分析。                                  |
|     在此处阅读分析文档（以及如何选择退出）：                       |
|              <https://dvc.org/doc/user-guide/analytics>             |
|                                                                     |
+---------------------------------------------------------------------+

接下来？
------------
- 查看文档：<https://dvc.org/doc>
- 获取帮助和分享想法：<https://dvc.org/chat>
- 在 GitHub 上给我们点星：<https://github.com/iterative/dvc>
```

运行 `dvc init` 会执行几项操作：

- 它会创建一个 `.dvc` 目录，用于存放 DVC 的内部信息和配置。
- 它可能会创建或修改 `.dvcignore` 文件，其作用类似于 `.gitignore`，但用于 DVC 特定的模式。
- 它通常会将 `.dvc/cache` 和其他内部路径添加到您的 `.gitignore` 文件中，以防止 Git 追踪数据缓存。

我们将这些初始设置文件提交到 Git：

```bash
git add .dvc .dvcignore .gitignore
git commit -m "初始化 DVC"
```

### 2. 准备示例数据

我们需要一些数据来进行版本管理。让我们创建一个 `data` 目录并在其中添加一个简单的 CSV 文件。您可以手动创建 `data/samples.csv`，或者使用这个小的 Python 脚本：

```python
# create_data.py
import os
import csv

os.makedirs('data', exist_ok=True)

header = ['id', 'value']
data = [
    [1, 10.5],
    [2, 15.2],
    [3, 20.0]
]

with open('data/samples.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print("已创建 data/samples.csv")
```

将其保存为 `create_data.py` 并运行：

```bash
python create_data.py
已创建 data/samples.csv
```

您现在应该有一个包含 `samples.csv` 的 `data` 目录。

### 3. 使用 DVC 追踪数据

现在，让我们告诉 DVC 开始追踪这个数据集。

```bash
dvc add data/samples.csv
```

您会看到输出，表明 DVC 正在处理该文件。此命令主要执行两项操作：

1. 它将 `data/samples.csv` 复制到 DVC 的缓存中（位于 `.dvc/cache` 内部）。缓存文件以其内容哈希命名（通常是 MD5）。
2. 它会创建一个名为 `data/samples.csv.dvc` 的小型文本文件。此文件充当指针或元数据文件，其中包含有关原始数据的信息，包括其哈希值。
3. 它会更新 `.gitignore`，以确保实际数据文件（`data/samples.csv`）不被 Git 追踪。

让我们查看 `.dvc` 文件：

```bash
cat data/samples.csv.dvc
```

输出将大致如下所示（哈希值会因确切文件内容而异）：

```yaml
outs:
- md5: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6 # 示例哈希
  path: samples.csv
```

此文件告诉 DVC，`samples.csv`（相对于 `.dvc` 文件所在位置）与指定的 MD5 哈希值相关联。

### 4. 将数据占位符提交到 Git

现在的主要步骤是将 `.dvc` 占位符文件和已更新的 `.gitignore` 提交到 Git。实际数据（`data/samples.csv`）仍未被 Git 追踪，因为它已在 `.gitignore` 中列出。

```bash
git add data/samples.csv.dvc .gitignore
git commit -m "使用 DVC 追踪初始数据集"
```

您的 Git 历史记录现在记录了数据在此阶段的*状态*（通过 `.dvc` 文件），而不存储大型数据文件本身。

### 5. 配置远程存储

为了共享或备份数据，我们需要配置远程存储。为了简化本练习，我们将在项目目录之外使用本地文件系统上的一个目录来模拟远程位置。在实际项目中，您通常会使用 S3、GCS、Azure Blob 或其他支持的后端。

```bash
# 创建一个目录作为远程存储（如果需要，请调整路径）
mkdir /tmp/dvc-practice-storage

# 将此目录配置为名为 'myremote' 的 DVC 远程
# -d 标志使其成为默认远程
dvc remote add -d myremote /tmp/dvc-practice-storage
```

此命令会更新位于 `.dvc/config` 的 DVC 配置文件。让我们提交此配置更改：

```bash
git add .dvc/config
git commit -m "配置本地 DVC 远程存储"
```

现在，您的项目知道从何处推送和拉取数据。

### 6. 将数据推送到远程存储

配置好远程后，我们可以将追踪的数据（当前位于本地缓存中）推送到远程存储。

```bash
dvc push
```

DVC 检查与当前 Git 提交关联的 `.dvc` 文件，在本地缓存（`.dvc/cache`）中找到对应的数据文件，并将它们上传到 `myremote` 位置（`/tmp/dvc-practice-storage`）。如果您查看 `/tmp/dvc-practice-storage` 目录，您会发现以 MD5 哈希值的前两个字符命名的子目录，其中包含实际数据文件（也以其哈希值命名）。

### 7. 修改数据并追踪更改

让我们模拟更新数据集。修改 `data/samples.csv`，添加新行或更改值。例如，在文件末尾添加一行 `4,25.8`。

```csv
id,value
1,10.5
2,15.2
3,20.0
4,25.8 # 新行
```

现在，询问 DVC 数据的状态：

```bash
dvc status
```

DVC 将当前的 `data/samples.csv` 文件与存储在 `data/samples.csv.dvc` 中的哈希值进行比较，并报告它已被修改。

要追踪这个新版本，请再次使用 `dvc add`：

```bash
dvc add data/samples.csv
```

这会更新 `data/samples.csv.dvc` 内部的哈希值以反映新内容，并将修改后的文件复制到缓存中。

将更新后的 `.dvc` 文件提交到 Git，以记录这个新版本：

```bash
git add data/samples.csv.dvc
git commit -m "更新数据集 (v2)"
```

最后，将新版本的数据推送到远程存储：

```bash
dvc push
```

只有新的数据文件（对应于新的哈希值）将被上传。

### 8. 模拟数据获取

假设您在新机器上克隆了这个仓库，或者您不小心删除了本地数据或缓存。让我们模拟这种情况：

```bash
# 从工作目录中删除数据文件
rm data/samples.csv

# （可选但有说明性）删除本地 DVC 缓存
rm -rf .dvc/cache
```

如果您运行 `git status`，Git 不会注意到 `data/samples.csv` 的缺失，因为它已被忽略。但是，`dvc status` 会显示 `data/samples.csv.dvc` 追踪的数据文件在工作空间中缺失。

要恢复与当前 Git 提交对应的数据（它指向数据的 v2 版本），请使用 `dvc pull`：

```bash
dvc pull
```

DVC 查询 `data/samples.csv.dvc`，找到所需的哈希值，从 `myremote` 存储下载相应文件到本地缓存（如果尚未存在），并在 `data/samples.csv` 放置一个副本（或链接，取决于配置）。验证 `data/samples.csv` 现在已恢复为 v2 内容（包括行 `4,25.8`）。

### 9. 在数据版本之间切换

Git 和 DVC 之间的联系在此处变得明显。您的 Git 历史记录追踪 `.dvc` *指针*文件的不同版本。您可以检出一个较旧的提交，来处理较旧版本的数据。

首先，找到初始数据集版本的提交哈希值（您可以使用 `git log`）。假设提交消息是“使用 DVC 追踪初始数据集”。

```bash
# 检出前一个提交（如果需要，请调整 'HEAD~1'）
git checkout HEAD~1
Note: switching to 'HEAD~1'.
...
HEAD is now at <commit_hash> 使用 DVC 追踪初始数据集
```

现在，您的 `data/samples.csv.dvc` 文件包含*原始*哈希值。然而，您工作空间中的实际 `data/samples.csv` 文件可能仍是 v2 版本（如果您严格遵循了上一步，则可能缺失）。使用 `dvc checkout` 将您的工作空间数据与当前检出的 Git 提交中的 `.dvc` 文件同步：

```bash
dvc checkout
M       data/samples.csv # DVC 指示正在修改文件
```

检查 `data/samples.csv` 的内容。它现在应该包含*原始*数据（不包含行 `4,25.8`）。

要返回最新版本：

```bash
# 切换回 main 分支（或您的工作分支）
git checkout main # 或者您的分支名称，例如 master

# 将数据与最新的 .dvc 文件同步
dvc checkout
M       data/samples.csv
```

验证 `data/samples.csv` 再次包含 v2 数据。

### 总结

在本次实践操作中，您成功使用了 DVC 的核心命令：

- `dvc init`：在 Git 仓库中初始化 DVC。
- `dvc add`：开始追踪数据文件，创建 `.dvc` 元数据文件。
- `git commit`：在 Git 历史记录中保存 `.dvc` 文件（数据指针）的版本。
- `dvc remote add`：配置存储数据的位置（本例中为本地目录）。
- `dvc push`：将数据文件从本地缓存上传到远程存储。
- `dvc pull`：根据 `.dvc` 文件从远程存储下载数据文件。
- `git checkout`：在不同代码/数据指针版本之间切换。
- `dvc checkout`：将工作空间数据与当前 Git 提交中 `.dvc` 文件指定的版本同步。

您已经看到 Git 如何追踪代码以及数据版本的*引用*，而 DVC 则管理实际数据文件及其与远程存储的同步。这种结合使您能够有效地管理大型数据集的版本，使其与代码保持一致，为可复现的机器学习 (machine learning)项目打下基础。

## 参考资料

- [DVC User Guide](https://dvc.org/doc/user-guide) — Iterative.ai (Current)
  数据版本控制（DVC）的官方指南，包含DVC命令和概念的解释及示例。
- [Pro Git](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  Git的官方综合资源，涵盖其功能和版本控制用法。对于理解DVC的基本原理至关重要。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready ML Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书提供了设计和构建生产就绪机器学习系统的综合视角，包括数据管理和可复现性章节。

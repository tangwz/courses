# 克隆现有仓库 (git clone)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories/git-clone-command)

[返回章节目录](README.md) · [返回课程目录](../README.md)

参与一个已经存在于远程平台（例如GitHub或你公司内部服务器）的项目，是Git中的一个常见情景。要开始处理此类项目，你需要获取其远程仓库的本地副本。执行此操作的命令是 `git clone`。

克隆不仅仅是下载文件；它会创建整个仓库的一个完整、独立的副本（一个“克隆”），包括其所有历史记录、分支和提交。这与Git的分布式特性相符——每个克隆都包含完整的项目历史，让你能够离线工作、查看历史记录并独立进行提交。

### `git clone` 如何工作

克隆仓库的基本语法很简单：

```bash
git clone <repository_url>
```

这里，`<repository_url>` 是指向远程仓库的URL。你通常会在GitHub、GitLab或Bitbucket等托管平台上的仓库页面找到这个URL。它通常以 `.git` 结尾，并且可以使用HTTPS或SSH等协议。

例如，使用HTTPS克隆仓库的命令可能如下所示：

```bash
git clone https://github.com/user/repository-name.git
```

当你运行 `git clone` 时，Git会执行以下几项操作：

1. **创建目录：** 默认情况下，Git会在你的本地机器上创建一个以仓库名称命名的目录（例如，上面例子中的 `repository-name`）。
2. **初始化 `.git`：** 在这个新目录内，Git会初始化一个 `.git` 子目录，就像 `git init` 命令一样。这个目录包含仓库的所有元数据、历史记录和配置。
3. **下载对象：** Git从远程仓库下载所有数据（提交、树、Blob对象）。
4. **检出默认分支：** 它会检出项目文件的最新版本的工作副本，通常是从默认分支（常被称为 `main` 或 `master`）中检出。
5. **配置远程连接：** 重要的是，`git clone` 会自动配置一个指向原始仓库URL的远程连接。这个远程连接通常被命名为 `origin`。你可以在新克隆的仓库中运行 `git remote -v` 来验证这一点。

> 一个图示，描述了 `git clone` 过程，其中整个远程仓库（包括其历史记录）被复制以创建一个新的本地仓库目录。

### 克隆到特定目录

如果你想让克隆的仓库位于一个与仓库本身名称不同的目录中，你可以在URL后面指定一个目录名：

```bash
git clone <repository_url> <desired_directory_name>
```

例如：

```bash
git clone https://github.com/user/repository-name.git my-project-folder
```

此命令会将仓库克隆到你当前位置一个名为 `my-project-folder` 的新目录中。

### 克隆之后

一旦你克隆了一个仓库，你就拥有了一个功能完善的本地Git仓库。你可以开始处理文件、暂存更改并进行提交，就像你在前面章节中学到的那样。主要的区别在于，你的仓库现在知道它的“源”——你克隆它所来自的远程仓库。正是这种连接让你能够稍后将本地更改推送到远程仓库，或者拉取他人进行的更新，这些内容我们将在后续章节中介绍。克隆是加入现有项目或创建远程工作备份的标准第一步。

## 参考资料

- [git-clone Documentation](https://git-scm.com/docs/git-clone) — Git contributors (2025)
  这是 `git clone` 命令的官方且最新文档，提供了语法、选项和详细信息。
- [Pro Git (2nd Edition): 2.1 Git Basics - Getting a Git Repository](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  这本广受认可的“Pro Git”书籍的章节详细解释了 `git clone`，涵盖其功能和内部操作。

---

[上一节](04-%E6%9F%A5%E7%9C%8B%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%20%28git%20remote%20-v%29.md) · [下一节](06-%E5%90%91%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%E6%8E%A8%E9%80%81%E6%9B%B4%E6%94%B9%20%28git%20push%29.md)

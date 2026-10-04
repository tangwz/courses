---
course: "getting-started-with-git"
chapter: "working-with-remote-repositories"
lesson: "intro-remote-repositories"
sourceId: 1023
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories/intro-remote-repositories"
title: "远程仓库简介"
description: "理解远程仓库及其在协作和代码共享中的作用。"
order: 1
plots: []
sourceHash: "4ef60d434aa221156236bd6ee52de1c3254d9c947ec4162a46179297f41054c0"
sourceCorrections: []
---

Git 为开发者提供了强大的版本控制功能，支持在本地使用，包括初始化仓库、追踪更改、查看历史记录和管理分支。尽管这些本地操作对于个人项目非常重要，但 Git 分享工作和与他人协作的真正潜力则通过远程仓库得以实现。

远程仓库本质上是你项目的一个版本，它托管在可通过网络（最常见的是互联网）访问的服务器上。可以把它想象成一个中心或你项目历史记录的共享副本。它仍然是一个 Git 仓库，包含与你的本地副本（一旦同步）相同的所有提交、分支和历史记录，但它存在于其他地方。

为什么要使用远程仓库？有几个主要优点：

1. **协作**：这是最常见的原因。远程仓库为多个开发者共同为一个项目做贡献提供了一个共同平台。每个开发者都可以在自己的本地副本上工作，然后将更改推送到远程仓库。其他人可以将这些更改拉取到他们自己的本地副本中，从而整合所有人的工作。
2. **备份与冗余**：将代码托管在远程服务器上提供了备份。如果你的本地机器硬盘故障或本地仓库损坏，远程仓库可以作为安全副本，你可以从中恢复你的工作。
3. **可访问性**：你可以从多个位置访问你的代码。如果你在台式机和笔记本电脑上工作，可以使用远程仓库在它们之间保持项目同步。

需要明白，你的本地仓库和远程仓库是独立的实体。你在本地进行的更改（例如新的提交）不会自动出现在远程仓库上，而其他人推送到远程仓库的更改也不会自动出现在你的本地机器上。你需要使用特定的 Git 命令来明确管理本地仓库及其远程对应仓库之间的信息流动。

> 本地 Git 仓库和远程 Git 仓库之间的关系涉及明确的同步命令。

本章将专注于这些交互。你将学习如何将本地仓库连接到远程仓库、如何从远程源首次下载项目（`git clone`）、如何上传本地提交（`git push`）以及如何从远程下载更新（`git pull` 和 `git fetch`）。我们还将简单看一下专门设计用于托管这些远程仓库的服务。掌握这些操作对于参与协作软件开发或简单有效地分享你的项目非常重要。

## 参考资料

- [Pro Git](https://git-scm.com/book) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  这本Git的官方综合指南涵盖了远程仓库、协作工作流以及所有基本命令，包括克隆、拉取、合并和推送。
- [The Missing Semester of Your CS Education - Version Control (Git) Lecture](https://missing.csail.mit.edu/) — Anish Athalye, Jon Gjengset, Jose Javier Gonzalez Ortiz (2024)
  作为一门知名大学课程的一部分，本讲座介绍了使用Git进行版本控制，包括其分布式特性以及利用远程操作实现共享和协作。

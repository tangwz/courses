---
course: "getting-started-with-git"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories"
sourceId: 373
chapter: "working-with-remote-repositories"
title: "使用远程仓库"
order: 5
description: "学习如何使用 GitHub 或 GitLab 等平台上的远程 Git 仓库与他人合作。克隆、推送、拉取和获取更改。"
hasQuiz: true
---

到目前为止，您一直只在本地机器上使用 Git。然而，当您与他人合作或将您的工作备份到单独的服务器时，Git 的作用会大大提升。本章将介绍远程仓库：即托管在其他地方（通常是互联网或网络上）的项目版本。

您将学习与这些远程仓库进行交互所需的命令。这包括配置连接（`git remote add`, `git remote -v`）、获取远程项目的完整副本（`git clone`）、上传您的本地提交（`git push`），以及将他人所做的更改合并到您的本地仓库中（`git fetch`, `git pull`）。我们还将提及 GitHub、GitLab 和 Bitbucket 等热门托管服务，以及 `$origin$` 等标准的远程命名约定。掌握这些远程操作对于软件协作开发和共享项目来说非常重要。

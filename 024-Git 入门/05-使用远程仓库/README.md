# 第 5 章：使用远程仓库

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories)

[返回课程目录](../README.md)

到目前为止，您一直只在本地机器上使用 Git。然而，当您与他人合作或将您的工作备份到单独的服务器时，Git 的作用会大大提升。本章将介绍远程仓库：即托管在其他地方（通常是互联网或网络上）的项目版本。

您将学习与这些远程仓库进行交互所需的命令。这包括配置连接（`git remote add`, `git remote -v`）、获取远程项目的完整副本（`git clone`）、上传您的本地提交（`git push`），以及将他人所做的更改合并到您的本地仓库中（`git fetch`, `git pull`）。我们还将提及 GitHub、GitLab 和 Bitbucket 等热门托管服务，以及 `$origin$` 等标准的远程命名约定。掌握这些远程操作对于软件协作开发和共享项目来说非常重要。

## 小节

- 1. [远程仓库简介](01-%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%E7%AE%80%E4%BB%8B.md)
- 2. [常用托管平台（GitHub、GitLab、Bitbucket）](02-%E5%B8%B8%E7%94%A8%E6%89%98%E7%AE%A1%E5%B9%B3%E5%8F%B0%EF%BC%88GitHub%E3%80%81GitLab%E3%80%81Bitbucket%EF%BC%89.md)
- 3. [添加远程仓库 (git remote add)](03-%E6%B7%BB%E5%8A%A0%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%20%28git%20remote%20add%29.md)
- 4. [查看远程仓库 (git remote -v)](04-%E6%9F%A5%E7%9C%8B%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%20%28git%20remote%20-v%29.md)
- 5. [克隆现有仓库 (git clone)](05-%E5%85%8B%E9%9A%86%E7%8E%B0%E6%9C%89%E4%BB%93%E5%BA%93%20%28git%20clone%29.md)
- 6. [向远程仓库推送更改 (git push)](06-%E5%90%91%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%E6%8E%A8%E9%80%81%E6%9B%B4%E6%94%B9%20%28git%20push%29.md)
- 7. [从远程仓库获取更改 (git fetch)](07-%E4%BB%8E%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%E8%8E%B7%E5%8F%96%E6%9B%B4%E6%94%B9%20%28git%20fetch%29.md)
- 8. [从远程拉取更改 (git pull)](08-%E4%BB%8E%E8%BF%9C%E7%A8%8B%E6%8B%89%E5%8F%96%E6%9B%B4%E6%94%B9%20%28git%20pull%29.md)
- 9. [理解 \`origin\` 和 \`upstream\`](09-%E7%90%86%E8%A7%A3%20%60origin%60%20%E5%92%8C%20%60upstream%60.md)
- 10. [实践：克隆、推送和拉取](10-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%85%8B%E9%9A%86%E3%80%81%E6%8E%A8%E9%80%81%E5%92%8C%E6%8B%89%E5%8F%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories/quiz)

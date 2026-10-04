---
course: "getting-started-with-git"
chapter: "working-with-remote-repositories"
lesson: "git-hosting-platforms"
sourceId: 1024
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-5-working-with-remote-repositories/git-hosting-platforms"
title: "常用托管平台（GitHub、GitLab、Bitbucket）"
description: "概述流行的Git托管平台，如GitHub、GitLab和Bitbucket及其用途。"
order: 2
plots: []
sourceHash: "2b393e026f63e8f4d50ec18708eab73294e155b89a25c80895ba60ceaed9adbd"
sourceCorrections: []
---

虽然理论上你可以在任何你能访问的服务器上托管Git仓库，但专门的托管平台提供了便捷的网页界面、协作工具、访问控制以及围绕Git仓库构建的其他功能。这些平台是开发者分享代码、追踪问题和管理软件开发周期的中心场所。

我们来看看三种最常用的Git托管平台。了解它们的基本特点将帮助你为自己的项目选择合适的平台，或在参与他人工作时更好地使用它们。

### GitHub

GitHub可以说是最受欢迎的Git仓库托管服务，尤其是在开源社区中。它提供了一个基于网络的Git仓库界面，以及一套旨在促进协作的工具。

- **主要服务：** 托管公共和私有Git仓库。公共仓库对所有人可见且免费托管，这使得GitHub成为开源软件的中心。私有仓库则限制访问。
- **协作功能：** 重要功能包括*拉取请求*（一种提出修改并进行代码审查的机制），*问题追踪*（用于报告错误和请求功能），项目管理工具（如看板），以及用于文档的*维基*。
- **社区：** 其庞大的用户群意味着许多项目、库和开发者都在此平台活跃，提供了充足的学习和贡献机会。
- **自动化与扩展：** GitHub Actions 提供集成式持续集成和持续交付（CI/CD）功能，让你可以直接从仓库自动化测试和部署流程。GitHub Pages 允许直接从仓库托管简单的静态网站。

对于初学者来说，GitHub的广泛使用使其成为一个很好的起点。许多教程和开源项目都使用它，提供了远程仓库工作流的实践经验。

### GitLab

GitLab将自己定位为一个以单一应用形式提供的完整DevOps平台。虽然它提供了类似于GitHub的Git仓库托管服务，但它将更广泛的开发生命周期工具直接集成到平台中。

- **主要服务：** 提供具有细粒度访问控制的Git仓库管理。它提供具有宽松限制的免费公共和私有仓库。
- **集成DevOps：** GitLab 的一个显著特点是其内置的CI/CD、安全扫描、监控和包注册表功能。目标是提供从规划到部署和监控的连贯工具链。
- **部署选项：** GitLab 提供 GitLab.com 上的软件即服务（SaaS）版本和自托管选项（GitLab 社区版/企业版），这对于希望完全控制其基础设施和代码库的组织很有吸引力。
- **功能集：** 通常包括与GitHub类似的问题追踪、维基和代码审查工具等功能。

GitLab 是一个强大的选择，常受寻求一体化解决方案或自托管灵活性的组织青睐。

### Bitbucket

Bitbucket 是由 Atlassian 开发的一项 Git 托管服务，该公司还出品了流行的开发工具，如 Jira（问题追踪和项目管理）以及团队协作和维基软件。

- **主要服务：** 提供Git仓库托管（历史上也支持Mercurial，另一种版本控制系统，尽管Git现在是主要重心）。它为小型团队提供免费计划，包括私有仓库。
- **Atlassian 集成：** Bitbucket 的主要优势在于与 Atlassian 其他产品的紧密集成。如果团队大量使用 Jira 进行项目管理，Bitbucket 能提供非常流畅的工作流，将代码提交、分支和拉取请求直接关联到 Jira 问题。
- **功能：** 包括代码审查的拉取请求、内置CI/CD（Bitbucket Pipelines）和基本问题追踪等标准功能。

Bitbucket 是已使用 Atlassian 生态系统的团队的常见选择，提供顺畅的集成，可以在该特定工具集中提升生产力。

### 选择平台

这三个平台都提供使用 `git clone`、`git push` 和 `git pull` 等命令操作远程Git仓库所需的基本功能。它们都提供适合个人开发者和小型项目的免费层级服务。选择通常归结为以下几点：

1. **社区侧重：** GitHub 在这方面表现出色，尤其对于开源项目而言。
2. **集成DevOps：** GitLab 提供一套全面的内置工具。
3. **Atlassian 生态系统：** Bitbucket 与 Jira 等工具集成效果最佳。
4. **自托管需求：** GitLab 提供流行的自托管选项。

就本课程而言，任何这些平台都非常适合用于练习远程操作。鉴于GitHub的受欢迎程度，我们通常会使用假设是GitHub远程仓库的示例，但核心Git命令（`clone`、`fetch`、`pull`、`push`）无论你选择哪个托管平台，功能都是一样的。主要区别在于网页界面、辅助协作工具和特定集成功能。

## 参考资料

- [About GitHub Docs](https://docs.github.com/en/get-started/on-github/about-github) — GitHub Docs Team (2024)
  Publisher: GitHub
  提供关于 GitHub 的基础信息、其宗旨以及作为 Git 协作和开源项目托管平台的核心功能。
- [About GitLab](https://docs.gitlab.com/) — GitLab Documentation Team (2024)
  Publisher: GitLab
  概述 GitLab 作为完整的 DevOps 平台，详细说明其在仓库管理、CI/CD 和项目生命周期方面的集成功能。
- [Pro Git (2nd Edition)](https://git-scm.com/book/en/v2) — Scott Chacon and Ben Straub (2014)
  Publisher: Apress
  一本权威书籍，涵盖 Git 的所有方面，包括关于远程仓库、分支和协作工作流程的详细讨论。

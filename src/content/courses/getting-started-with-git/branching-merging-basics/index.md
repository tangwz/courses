---
course: "getting-started-with-git"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-git/chapter-4-branching-merging-basics"
sourceId: 372
chapter: "branching-merging-basics"
title: "分支与合并基础操作"
order: 4
description: "学习 Git 分支的基本知识，了解如何进行并行开发以及如何将更改合并回来，包括解决冲突。"
hasQuiz: true
---

之前的章节主要讲解了如何管理项目沿单一路径产生的历史记录。然而，实际开发中，经常需要同时进行多项任务。例如，你可能需要开发一个新功能，同时单独修复已发布版本中的一个错误，而不让两者相互干扰。

Git 通过其分支机制，使这种并行工作成为可能。分支在你的仓库中充当独立的开发路径。本章将介绍使用分支的基本原理和相关命令。你将学习如何创建新分支 (`git branch`)、在它们之间切换 (`git switch` 或 `git checkout`)，以及如何将不同分支上完成的工作合并回来 (`git merge`)。我们还会讲解如何解决合并冲突，这些冲突可能在合并不同修改时发生。最后，还将涵盖分支的管理，包括列出和删除分支。

# 第 4 章：分支与合并基础操作

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-git/chapter-4-branching-merging-basics)

[返回课程目录](../README.md)

之前的章节主要讲解了如何管理项目沿单一路径产生的历史记录。然而，实际开发中，经常需要同时进行多项任务。例如，你可能需要开发一个新功能，同时单独修复已发布版本中的一个错误，而不让两者相互干扰。

Git 通过其分支机制，使这种并行工作成为可能。分支在你的仓库中充当独立的开发路径。本章将介绍使用分支的基本原理和相关命令。你将学习如何创建新分支 (`git branch`)、在它们之间切换 (`git switch` 或 `git checkout`)，以及如何将不同分支上完成的工作合并回来 (`git merge`)。我们还会讲解如何解决合并冲突，这些冲突可能在合并不同修改时发生。最后，还将涵盖分支的管理，包括列出和删除分支。

## 小节

- 1. [Git中的分支是什么？](01-Git%E4%B8%AD%E7%9A%84%E5%88%86%E6%94%AF%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F.md)
- 2. [创建新分支 (git branch)](02-%E5%88%9B%E5%BB%BA%E6%96%B0%E5%88%86%E6%94%AF%20%28git%20branch%29.md)
- 3. [在分支间切换 (git checkout 或 git switch)](03-%E5%9C%A8%E5%88%86%E6%94%AF%E9%97%B4%E5%88%87%E6%8D%A2%20%28git%20checkout%20%E6%88%96%20git%20switch%29.md)
- 4. [列出分支](04-%E5%88%97%E5%87%BA%E5%88%86%E6%94%AF.md)
- 5. [在分支上进行提交](05-%E5%9C%A8%E5%88%86%E6%94%AF%E4%B8%8A%E8%BF%9B%E8%A1%8C%E6%8F%90%E4%BA%A4.md)
- 6. [合并分支（git merge）](06-%E5%90%88%E5%B9%B6%E5%88%86%E6%94%AF%EF%BC%88git%20merge%EF%BC%89.md)
- 7. [理解快进合并](07-%E7%90%86%E8%A7%A3%E5%BF%AB%E8%BF%9B%E5%90%88%E5%B9%B6.md)
- 8. [处理合并冲突](08-%E5%A4%84%E7%90%86%E5%90%88%E5%B9%B6%E5%86%B2%E7%AA%81.md)
- 9. [删除分支 (git branch -d)](09-%E5%88%A0%E9%99%A4%E5%88%86%E6%94%AF%20%28git%20branch%20-d%29.md)
- 10. [动手实践：分支与合并工作流程](10-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%88%86%E6%94%AF%E4%B8%8E%E5%90%88%E5%B9%B6%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-git/chapter-4-branching-merging-basics/quiz)

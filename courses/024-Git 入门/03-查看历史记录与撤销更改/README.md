# 第 3 章：查看历史记录与撤销更改

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-git/chapter-3-viewing-history-undoing-changes)

[返回课程目录](../README.md)

创建提交以保存项目状态后，下一步是学习如何查看您已建立的历史记录并修正错误。本章将侧重于检查更改和修改提交历史记录。

您将练习使用 `git diff` 比较不同版本的工作，使用 `git reset` 取消暂存文件，以及使用 `git commit --amend` 调整最近的提交。此外，您会了解到 `git revert` 如何提供一种安全的方法来撤销较早的提交，以及如何使用 `git rm` 和 `git mv` 管理 Git 跟踪的文件。这些命令提供了必要的工具，帮助您理解项目的演变并有效管理更改。

## 小节

- 1. [比较更改 (git diff)](01-%E6%AF%94%E8%BE%83%E6%9B%B4%E6%94%B9%20%28git%20diff%29.md)
- 2. [查看提交之间的更改](02-%E6%9F%A5%E7%9C%8B%E6%8F%90%E4%BA%A4%E4%B9%8B%E9%97%B4%E7%9A%84%E6%9B%B4%E6%94%B9.md)
- 3. [查看已暂存与未暂存的修改](03-%E6%9F%A5%E7%9C%8B%E5%B7%B2%E6%9A%82%E5%AD%98%E4%B8%8E%E6%9C%AA%E6%9A%82%E5%AD%98%E7%9A%84%E4%BF%AE%E6%94%B9.md)
- 4. [取消暂存文件 (git reset HEAD <file>)](04-%E5%8F%96%E6%B6%88%E6%9A%82%E5%AD%98%E6%96%87%E4%BB%B6%20%28git%20reset%20HEAD%20-file-%29.md)
- 5. [修改最后一次提交 (git commit --amend)](05-%E4%BF%AE%E6%94%B9%E6%9C%80%E5%90%8E%E4%B8%80%E6%AC%A1%E6%8F%90%E4%BA%A4%20%28git%20commit%20--amend%29.md)
- 6. [撤销提交 (git revert)](06-%E6%92%A4%E9%94%80%E6%8F%90%E4%BA%A4%20%28git%20revert%29.md)
- 7. [从 Git 移除文件 (git rm)](07-%E4%BB%8E%20Git%20%E7%A7%BB%E9%99%A4%E6%96%87%E4%BB%B6%20%28git%20rm%29.md)
- 8. [移动或重命名文件 (git mv)](08-%E7%A7%BB%E5%8A%A8%E6%88%96%E9%87%8D%E5%91%BD%E5%90%8D%E6%96%87%E4%BB%B6%20%28git%20mv%29.md)
- 9. [练习：检查和修改历史](09-%E7%BB%83%E4%B9%A0%EF%BC%9A%E6%A3%80%E6%9F%A5%E5%92%8C%E4%BF%AE%E6%94%B9%E5%8E%86%E5%8F%B2.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-git/chapter-3-viewing-history-undoing-changes/quiz)

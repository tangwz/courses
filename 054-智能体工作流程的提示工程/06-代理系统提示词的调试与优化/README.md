# 第 6 章：代理系统提示词的调试与优化

来源：[原章节](https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-6-debugging-optimizing-prompts-agentic-systems)

[返回课程目录](../README.md)

设计人工智能代理的有效提示词是一个迭代的过程。初步设计通常需要调整，以确保代理能按预期运行，并在其工作流程中取得可靠的结果。本章侧重于改进这些提示词的实际操作方面。

您将学会：
*   找出在代理系统中使用提示词时出现的常见问题。
*   运用系统方法迭代和测试您的提示词设计。
*   明白提示词序列，即提示词链，如何影响代理的输出。
*   使用方法分析代理所采取的行动序列。
*   比较不同的提示词变体，以确定哪些对特定任务最有效。
*   使用日志记录和监控来收集数据，以便持续改进提示词。
*   建立组织和版本管理代理提示词的方法。

本章最后是一个实践练习，您将在其中运用这些技术，在一个示例代理工作流程中调试和优化提示词，从而提升其性能和可靠性。

## 小节

- 1. [代理提示实现中的常见问题](01-%E4%BB%A3%E7%90%86%E6%8F%90%E7%A4%BA%E5%AE%9E%E7%8E%B0%E4%B8%AD%E7%9A%84%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98.md)
- 2. [提示词迭代与测试的系统化方法](02-%E6%8F%90%E7%A4%BA%E8%AF%8D%E8%BF%AD%E4%BB%A3%E4%B8%8E%E6%B5%8B%E8%AF%95%E7%9A%84%E7%B3%BB%E7%BB%9F%E5%8C%96%E6%96%B9%E6%B3%95.md)
- 3. [提示链对智能体输出的影响](03-%E6%8F%90%E7%A4%BA%E9%93%BE%E5%AF%B9%E6%99%BA%E8%83%BD%E4%BD%93%E8%BE%93%E5%87%BA%E7%9A%84%E5%BD%B1%E5%93%8D.md)
- 4. [分析智能体行动序列的方法](04-%E5%88%86%E6%9E%90%E6%99%BA%E8%83%BD%E4%BD%93%E8%A1%8C%E5%8A%A8%E5%BA%8F%E5%88%97%E7%9A%84%E6%96%B9%E6%B3%95.md)
- 5. [比较提示变体以提升智能体效果](05-%E6%AF%94%E8%BE%83%E6%8F%90%E7%A4%BA%E5%8F%98%E4%BD%93%E4%BB%A5%E6%8F%90%E5%8D%87%E6%99%BA%E8%83%BD%E4%BD%93%E6%95%88%E6%9E%9C.md)
- 6. [提示词优化的记录与监控](06-%E6%8F%90%E7%A4%BA%E8%AF%8D%E4%BC%98%E5%8C%96%E7%9A%84%E8%AE%B0%E5%BD%95%E4%B8%8E%E7%9B%91%E6%8E%A7.md)
- 7. [智能体提示词的组织与版本管理](07-%E6%99%BA%E8%83%BD%E4%BD%93%E6%8F%90%E7%A4%BA%E8%AF%8D%E7%9A%84%E7%BB%84%E7%BB%87%E4%B8%8E%E7%89%88%E6%9C%AC%E7%AE%A1%E7%90%86.md)
- 8. [实践：优化提示词以解决智能体工作流程问题](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BC%98%E5%8C%96%E6%8F%90%E7%A4%BA%E8%AF%8D%E4%BB%A5%E8%A7%A3%E5%86%B3%E6%99%BA%E8%83%BD%E4%BD%93%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B%E9%97%AE%E9%A2%98.md)

章节测验：[在线测验](https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-6-debugging-optimizing-prompts-agentic-systems/quiz)

# 第 2 章：高级提示以掌控代理

来源：[原章节](https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-2-advanced-prompting-agent-control)

[返回课程目录](../README.md)

基于对上一章代理AI系统组成部分的理解，我们现在侧重于指导这些代理的实用方法。本章提供运用高级提示工程进行精确代理控制的技术。您将学习如何为序列操作构造提示，完善复杂代理动作的指令，以及分配角色或人设以塑造代理的回复。我们将考察如何运用少量示例进行代理引导，应用推理框架如Chain-of-Thought ($CoT$) 和Tree-of-Thought ($ToT$)，并通过提示设计管理代理状态。此外，提示代理进行自我纠正和错误处理的技术也将被涵盖，最终通过实践练习培养连贯的代理行为。

## 小节

- 1. [组织提示词以进行顺序操作](01-%E7%BB%84%E7%BB%87%E6%8F%90%E7%A4%BA%E8%AF%8D%E4%BB%A5%E8%BF%9B%E8%A1%8C%E9%A1%BA%E5%BA%8F%E6%93%8D%E4%BD%9C.md)
- 2. [优化复杂智能体操作的指令](02-%E4%BC%98%E5%8C%96%E5%A4%8D%E6%9D%82%E6%99%BA%E8%83%BD%E4%BD%93%E6%93%8D%E4%BD%9C%E7%9A%84%E6%8C%87%E4%BB%A4.md)
- 3. [给智能体分配角色和人设](03-%E7%BB%99%E6%99%BA%E8%83%BD%E4%BD%93%E5%88%86%E9%85%8D%E8%A7%92%E8%89%B2%E5%92%8C%E4%BA%BA%E8%AE%BE.md)
- 4. [运用少样本示例指导智能体](04-%E8%BF%90%E7%94%A8%E5%B0%91%E6%A0%B7%E6%9C%AC%E7%A4%BA%E4%BE%8B%E6%8C%87%E5%AF%BC%E6%99%BA%E8%83%BD%E4%BD%93.md)
- 5. [智能体提示中的思维链与思维树](05-%E6%99%BA%E8%83%BD%E4%BD%93%E6%8F%90%E7%A4%BA%E4%B8%AD%E7%9A%84%E6%80%9D%E7%BB%B4%E9%93%BE%E4%B8%8E%E6%80%9D%E7%BB%B4%E6%A0%91.md)
- 6. [通过提示词设计管理代理状态](06-%E9%80%9A%E8%BF%87%E6%8F%90%E7%A4%BA%E8%AF%8D%E8%AE%BE%E8%AE%A1%E7%AE%A1%E7%90%86%E4%BB%A3%E7%90%86%E7%8A%B6%E6%80%81.md)
- 7. [提示代理进行自我修正和错误处理](07-%E6%8F%90%E7%A4%BA%E4%BB%A3%E7%90%86%E8%BF%9B%E8%A1%8C%E8%87%AA%E6%88%91%E4%BF%AE%E6%AD%A3%E5%92%8C%E9%94%99%E8%AF%AF%E5%A4%84%E7%90%86.md)
- 8. [实践：设计实现智能体行为一致的提示](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%BE%E8%AE%A1%E5%AE%9E%E7%8E%B0%E6%99%BA%E8%83%BD%E4%BD%93%E8%A1%8C%E4%B8%BA%E4%B8%80%E8%87%B4%E7%9A%84%E6%8F%90%E7%A4%BA.md)

章节测验：[在线测验](https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-2-advanced-prompting-agent-control/quiz)

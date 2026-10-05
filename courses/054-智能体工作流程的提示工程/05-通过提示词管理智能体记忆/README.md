# 第 5 章：通过提示词管理智能体记忆

来源：[原章节](https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-5-managing-agent-memory-prompts)

[返回课程目录](../README.md)

AI智能体回忆过去交流、所学信息和当前目标的能力，对其在复杂多步骤任务中的表现起着决定作用。本章说明提示工程技术如何在智能体系统中帮助进行记忆管理。我们会介绍处理短期记忆的方法，比如智能体上下文窗口中的信息，并讨论从知识库中获取长期信息的做法。您将学会构建提示词，以指导智能体总结数据、查询外部知识库、保持信息一致性，以及在长时间交互中保持任务专注。本章还会提及针对处理不断变化信息流的智能体的实际注意事项。

## 小节

- 1. [记忆在智能体运行中的重要性](01-%E8%AE%B0%E5%BF%86%E5%9C%A8%E6%99%BA%E8%83%BD%E4%BD%93%E8%BF%90%E8%A1%8C%E4%B8%AD%E7%9A%84%E9%87%8D%E8%A6%81%E6%80%A7.md)
- 2. [提示策略：短期记忆与上下文窗口管理](02-%E6%8F%90%E7%A4%BA%E7%AD%96%E7%95%A5%EF%BC%9A%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%E4%B8%8E%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AA%97%E5%8F%A3%E7%AE%A1%E7%90%86.md)
- 3. [提示词的信息精简技巧](03-%E6%8F%90%E7%A4%BA%E8%AF%8D%E7%9A%84%E4%BF%A1%E6%81%AF%E7%B2%BE%E7%AE%80%E6%8A%80%E5%B7%A7.md)
- 4. [为长期信息检索提供提示](04-%E4%B8%BA%E9%95%BF%E6%9C%9F%E4%BF%A1%E6%81%AF%E6%A3%80%E7%B4%A2%E6%8F%90%E4%BE%9B%E6%8F%90%E7%A4%BA.md)
- 5. [构建提示以获取知识库信息](05-%E6%9E%84%E5%BB%BA%E6%8F%90%E7%A4%BA%E4%BB%A5%E8%8E%B7%E5%8F%96%E7%9F%A5%E8%AF%86%E5%BA%93%E4%BF%A1%E6%81%AF.md)
- 6. [管理信息一致性与认知](06-%E7%AE%A1%E7%90%86%E4%BF%A1%E6%81%AF%E4%B8%80%E8%87%B4%E6%80%A7%E4%B8%8E%E8%AE%A4%E7%9F%A5.md)
- 7. [在长时间交互中保持任务重心](07-%E5%9C%A8%E9%95%BF%E6%97%B6%E9%97%B4%E4%BA%A4%E4%BA%92%E4%B8%AD%E4%BF%9D%E6%8C%81%E4%BB%BB%E5%8A%A1%E9%87%8D%E5%BF%83.md)
- 8. [动手实践：用动态信息流提示代理](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%94%A8%E5%8A%A8%E6%80%81%E4%BF%A1%E6%81%AF%E6%B5%81%E6%8F%90%E7%A4%BA%E4%BB%A3%E7%90%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-5-managing-agent-memory-prompts/quiz)

# 第 2 章：进阶提示策略

来源：[原章节](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-2-advanced-prompting-strategies)

[返回课程目录](../README.md)

本章在与大语言模型交互的基本方法之上，介绍了更精细的提示词构建策略。这些技巧旨在提升大语言模型回复的质量、明确性及推理能力，尤其适用于更复杂的任务。

您将学到以下实用方法：

*   **零样本与少样本提示：** 了解何时以及如何在提示词中提供（或不提供）例子以引导模型（$k$-shot learning）。
*   **指令遵循与角色提示：** 编写清晰的指令并分配特定角色，以提高任务执行的贴合度。
*   **结构化输出生成：** 让大语言模型生成可预测格式（例如 JSON 或 Markdown）的技巧。
*   **思维链（CoT）提示：** 鼓励模型逐步阐明其推理过程，通常能提升在需要逻辑或计算的问题上的表现。
*   **自洽性：** 一种通过采样多条推理路径来提升结果可靠性的方法。

掌握这些策略能够更精细地控制大语言模型的行为，有助于开发更复杂且可靠的应用。

## 小节

- 1. [零样本提示](01-%E9%9B%B6%E6%A0%B7%E6%9C%AC%E6%8F%90%E7%A4%BA.md)
- 2. [小样本提示](02-%E5%B0%8F%E6%A0%B7%E6%9C%AC%E6%8F%90%E7%A4%BA.md)
- 3. [指令遵循提示](03-%E6%8C%87%E4%BB%A4%E9%81%B5%E5%BE%AA%E6%8F%90%E7%A4%BA.md)
- 4. [角色设定提示](04-%E8%A7%92%E8%89%B2%E8%AE%BE%E5%AE%9A%E6%8F%90%E7%A4%BA.md)
- 5. [结构化输出格式 (JSON, Markdown)](05-%E7%BB%93%E6%9E%84%E5%8C%96%E8%BE%93%E5%87%BA%E6%A0%BC%E5%BC%8F%20%28JSON%2C%20Markdown%29.md)
- 6. [思维链提示](06-%E6%80%9D%E7%BB%B4%E9%93%BE%E6%8F%90%E7%A4%BA.md)
- 7. [自洽性提示](07-%E8%87%AA%E6%B4%BD%E6%80%A7%E6%8F%90%E7%A4%BA.md)
- 8. [练习：应用高级技巧](08-%E7%BB%83%E4%B9%A0%EF%BC%9A%E5%BA%94%E7%94%A8%E9%AB%98%E7%BA%A7%E6%8A%80%E5%B7%A7.md)

章节测验：[在线测验](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-2-advanced-prompting-strategies/quiz)

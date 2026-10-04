# 第 4 章：使用合成数据提升LLM微调效果

来源：[原章节](https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-4-llm-fine-tuning-synthetic-data-enhancement)

[返回课程目录](../README.md)

在掌握了合成数据在LLM预训练中的应用后，我们现在将注意力转向微调。此阶段旨在使通用LLM适应特定任务、提升指令遵循能力或展现特有的运行方式。合成数据为构建有效微调所需的目标数据集提供了有益来源，尤其是在针对专业需求时真实数据不足或无法获取的情况下。

本章将介绍如何：
*   应用合成数据进行指令微调 (IFT)，使LLM能更好地理解并执行指令。
*   采用Self-Instruct等方法来构建多样化的微调数据集。
*   生成合成示例，以助力LLM在小样本或零样本学习场景中的表现。
*   以合适的格式（如 JSONL）组织合成数据，以适配不同的微调流程。
*   使用合成输入来塑造模型特点，例如写作风格或角色设定。
*   通过一个实践练习来构建针对特定微调目标定制的合成数据集。

## 小节

- 1. [利用生成数据进行指令遵循微调](01-%E5%88%A9%E7%94%A8%E7%94%9F%E6%88%90%E6%95%B0%E6%8D%AE%E8%BF%9B%E8%A1%8C%E6%8C%87%E4%BB%A4%E9%81%B5%E5%BE%AA%E5%BE%AE%E8%B0%83.md)
- 2. [制作有效的合成指令-响应对](02-%E5%88%B6%E4%BD%9C%E6%9C%89%E6%95%88%E7%9A%84%E5%90%88%E6%88%90%E6%8C%87%E4%BB%A4-%E5%93%8D%E5%BA%94%E5%AF%B9.md)
- 3. [构建多样化微调数据集的方法](03-%E6%9E%84%E5%BB%BA%E5%A4%9A%E6%A0%B7%E5%8C%96%E5%BE%AE%E8%B0%83%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84%E6%96%B9%E6%B3%95.md)
- 4. [生成少样本和零样本学习场景的数据](04-%E7%94%9F%E6%88%90%E5%B0%91%E6%A0%B7%E6%9C%AC%E5%92%8C%E9%9B%B6%E6%A0%B7%E6%9C%AC%E5%AD%A6%E4%B9%A0%E5%9C%BA%E6%99%AF%E7%9A%84%E6%95%B0%E6%8D%AE.md)
- 5. [针对不同微调框架的数据组织](05-%E9%92%88%E5%AF%B9%E4%B8%8D%E5%90%8C%E5%BE%AE%E8%B0%83%E6%A1%86%E6%9E%B6%E7%9A%84%E6%95%B0%E6%8D%AE%E7%BB%84%E7%BB%87.md)
- 6. [通过人工生成数据塑造模型行为（风格、角色）](06-%E9%80%9A%E8%BF%87%E4%BA%BA%E5%B7%A5%E7%94%9F%E6%88%90%E6%95%B0%E6%8D%AE%E5%A1%91%E9%80%A0%E6%A8%A1%E5%9E%8B%E8%A1%8C%E4%B8%BA%EF%BC%88%E9%A3%8E%E6%A0%BC%E3%80%81%E8%A7%92%E8%89%B2%EF%BC%89.md)
- 7. [动手实践：创建用于特定任务微调的合成数据集](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%88%9B%E5%BB%BA%E7%94%A8%E4%BA%8E%E7%89%B9%E5%AE%9A%E4%BB%BB%E5%8A%A1%E5%BE%AE%E8%B0%83%E7%9A%84%E5%90%88%E6%88%90%E6%95%B0%E6%8D%AE%E9%9B%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-4-llm-fine-tuning-synthetic-data-enhancement/quiz)

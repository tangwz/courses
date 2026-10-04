# 第 2 章：合成文本生成的核心技术

来源：[原章节](https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-2-core-synthetic-text-generation-techniques)

[返回课程目录](../README.md)

在上一章对合成数据的作用有了基础认识后，我们现在将转向“如何做”：即生成合成文本的具体方法。本章将对这些方法进行实际介绍。

您将学会：
*   实现基于算法和规则的文本生成系统。
*   使用回译作为数据扩充的策略。
*   应用释义模型，为您的文本数据集增加多样性。
*   直接使用大型语言模型（LLM）生成新的数据样本，并侧重于有效的提示设计来控制输出。
*   应用数据遮蔽和扰动技术，生成多样化且尊重隐私的数据。

本章包含一个实践练习，您将使用LLM API生成文本，并将这些技术付诸实践。通过学习这些部分，您将构建一个工具包，用于生成适应不同LLM开发需求的合成文本。

## 小节

- 1. [算法与规则驱动的文本生成](01-%E7%AE%97%E6%B3%95%E4%B8%8E%E8%A7%84%E5%88%99%E9%A9%B1%E5%8A%A8%E7%9A%84%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90.md)
- 2. [借助回译扩充数据](02-%E5%80%9F%E5%8A%A9%E5%9B%9E%E8%AF%91%E6%89%A9%E5%85%85%E6%95%B0%E6%8D%AE.md)
- 3. [使用释义模型丰富文本](03-%E4%BD%BF%E7%94%A8%E9%87%8A%E4%B9%89%E6%A8%A1%E5%9E%8B%E4%B8%B0%E5%AF%8C%E6%96%87%E6%9C%AC.md)
- 4. [使用大型语言模型生成合成样本](04-%E4%BD%BF%E7%94%A8%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90%E5%90%88%E6%88%90%E6%A0%B7%E6%9C%AC.md)
- 5. [通过高效的提示词设计引导生成](05-%E9%80%9A%E8%BF%87%E9%AB%98%E6%95%88%E7%9A%84%E6%8F%90%E7%A4%BA%E8%AF%8D%E8%AE%BE%E8%AE%A1%E5%BC%95%E5%AF%BC%E7%94%9F%E6%88%90.md)
- 6. [数据掩码和数据扰动技术](06-%E6%95%B0%E6%8D%AE%E6%8E%A9%E7%A0%81%E5%92%8C%E6%95%B0%E6%8D%AE%E6%89%B0%E5%8A%A8%E6%8A%80%E6%9C%AF.md)
- 7. [动手实践：使用大型语言模型API生成文本](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8BAPI%E7%94%9F%E6%88%90%E6%96%87%E6%9C%AC.md)

章节测验：[在线测验](https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-2-core-synthetic-text-generation-techniques/quiz)

# 第 3 章：应用合成数据于大型语言模型预训练

来源：[原章节](https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-3-synthetic-data-llm-pretraining-application)

[返回课程目录](../README.md)

大型语言模型的预训练阶段需要大量文本数据。当真实数据稀缺、不足或缺少特定属性时，合成数据为构建或补充预训练数据集提供了一个可行的选择。本章会研究合成数据专门应用于大型语言模型开发的这一重要阶段。

您将学习如何：
*   理解数据量 $V_{data}$ 与预训练效果之间的关联。
*   构建适合预训练阶段使用的大规模合成语料库。
*   实施将合成文本与现有真实数据结合的策略。
*   使用合成生成的内容进行特定方向或目标明确的预训练。
*   生成指令格式的数据以便纳入预训练中。
*   评估合成数据如何影响预训练的结果。
*   通过组建一个小型合成预训练数据集样本来获得实践经验。

## 小节

- 1. [基础模型训练中的数据量与多样性](01-%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E4%B8%AD%E7%9A%84%E6%95%B0%E6%8D%AE%E9%87%8F%E4%B8%8E%E5%A4%9A%E6%A0%B7%E6%80%A7.md)
- 2. [构建大规模合成语料库用于预训练](02-%E6%9E%84%E5%BB%BA%E5%A4%A7%E8%A7%84%E6%A8%A1%E5%90%88%E6%88%90%E8%AF%AD%E6%96%99%E5%BA%93%E7%94%A8%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83.md)
- 3. [合成文本与数据的结合](03-%E5%90%88%E6%88%90%E6%96%87%E6%9C%AC%E4%B8%8E%E6%95%B0%E6%8D%AE%E7%9A%84%E7%BB%93%E5%90%88.md)
- 4. [定向预训练：使用合成生成内容](04-%E5%AE%9A%E5%90%91%E9%A2%84%E8%AE%AD%E7%BB%83%EF%BC%9A%E4%BD%BF%E7%94%A8%E5%90%88%E6%88%90%E7%94%9F%E6%88%90%E5%86%85%E5%AE%B9.md)
- 5. [为预训练阶段生成指令式数据](05-%E4%B8%BA%E9%A2%84%E8%AE%AD%E7%BB%83%E9%98%B6%E6%AE%B5%E7%94%9F%E6%88%90%E6%8C%87%E4%BB%A4%E5%BC%8F%E6%95%B0%E6%8D%AE.md)
- 6. [衡量合成数据对预训练结果的影响](06-%E8%A1%A1%E9%87%8F%E5%90%88%E6%88%90%E6%95%B0%E6%8D%AE%E5%AF%B9%E9%A2%84%E8%AE%AD%E7%BB%83%E7%BB%93%E6%9E%9C%E7%9A%84%E5%BD%B1%E5%93%8D.md)
- 7. [动手实践：构建一个合成预训练数据集片段](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E5%90%88%E6%88%90%E9%A2%84%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE%E9%9B%86%E7%89%87%E6%AE%B5.md)

章节测验：[在线测验](https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-3-synthetic-data-llm-pretraining-application/quiz)

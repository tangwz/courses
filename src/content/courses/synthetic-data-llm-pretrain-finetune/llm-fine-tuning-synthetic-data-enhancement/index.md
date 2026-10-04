---
course: "synthetic-data-llm-pretrain-finetune"
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-4-llm-fine-tuning-synthetic-data-enhancement"
sourceId: 1112
chapter: "llm-fine-tuning-synthetic-data-enhancement"
title: "使用合成数据提升LLM微调效果"
order: 4
description: "学习使用合成数据进行有效LLM微调，涵盖指令微调、数据格式和行为塑造。"
hasQuiz: true
---

在掌握了合成数据在LLM预训练中的应用后，我们现在将注意力转向微调。此阶段旨在使通用LLM适应特定任务、提升指令遵循能力或展现特有的运行方式。合成数据为构建有效微调所需的目标数据集提供了有益来源，尤其是在针对专业需求时真实数据不足或无法获取的情况下。

本章将介绍如何：
*   应用合成数据进行指令微调 (IFT)，使LLM能更好地理解并执行指令。
*   采用Self-Instruct等方法来构建多样化的微调数据集。
*   生成合成示例，以助力LLM在小样本或零样本学习场景中的表现。
*   以合适的格式（如 JSONL）组织合成数据，以适配不同的微调流程。
*   使用合成输入来塑造模型特点，例如写作风格或角色设定。
*   通过一个实践练习来构建针对特定微调目标定制的合成数据集。

---
course: "agentic-llm-memory-architectures"
sourceUrl: "https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-3-designing-memory-systems"
sourceId: 870
chapter: "designing-memory-systems"
title: "设计复杂的记忆系统"
order: 3
description: "了解增强大型语言模型的高级方法，结合多样化和持久的记忆结构。"
hasQuiz: false
---

智能体系统需要有机制来保持状态、从交互中学习，并获取超出即时输入上下文的知识。标准大型语言模型在有限的上下文窗口 $L_{context}$ 下运行，这使得持久记忆对于需要长期连贯性或引用大量信息存储的任务变得不可或缺。

本章将详细说明针对大型语言模型智能体的记忆系统的设计与实现。我们将考察用于管理短期信息的方法，例如对话历史缓冲区和摘要方法。接着，我们会介绍长期记忆解决方案，主要侧重于向量数据库，以便有效进行相关知识的语义检索，使用 $e = f(text)$ 等嵌入。您会学习到高级检索策略，以改善获取的信息质量，以及如何整合结构化数据源，例如知识图谱或关系型数据库。最后，我们将讨论记忆管理的实际方面，包括合并、摘要以及设计智能体的有效读写接口。实践练习涉及整合向量数据库，以提供智能体持久记忆功能。

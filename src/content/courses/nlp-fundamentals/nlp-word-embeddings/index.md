---
course: "nlp-fundamentals"
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-4-nlp-word-embeddings"
sourceId: 601
chapter: "nlp-word-embeddings"
title: "嵌入表示与序列数据理解"
order: 4
description: "了解分布语义以及词嵌入技术，如Word2Vec（CBOW, Skip-gram）和GloVe。"
hasQuiz: true
---

前几章介绍了基于词频的文本预处理和特征表示方法，例如词袋模型（Bag-of-Words）和TF-IDF。这些技术在某些任务中有效，但难以捕捉词语的语义或语境。例如，如果“cat”和“feline”在训练数据中不经常出现在相同的语境下，这些方法会把它们视为完全不相关的词。

本章将重点转向基于词语周围语境来表示词语的方法，从而得到能够编码语义相似性的表示。我们将首先讨论基于词频方法的局限性，并介绍分布语义的理念，即出现在相似语境中的词语往往具有相似的含义。

接下来，我们将研究词嵌入：词语学习到的密集向量表示，例如 $w \in \mathbb{R}^n$。我们将了解流行的算法，如Word2Vec（包括其CBOW和Skip-gram变体）和GloVe（全局词向量表示）。最后，我们将介绍这些嵌入的可视化技术，并学习如何使用现成的预训练嵌入模型，以便将其用于其他自然语言处理任务。

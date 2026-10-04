---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-28-efficient-inference-strategies"
sourceId: 1106
chapter: "efficient-inference-strategies"
title: "高效推理策略"
order: 28
description: "实行加速LLM推理的技术，包括KV缓存、注意力优化和推测解码。"
hasQuiz: false
---

训练大型语言模型计算量很大，但将其应用于实时场景时，仍面临性能方面的难题。逐个生成文本令牌的过程，即自回归解码，本身速度可能较慢，且对内存消耗大。每次生成新令牌通常需要处理整个之前的序列，这会带来较大的计算开销和延迟，尤其是在输出较长时。

本章侧重于优化此生成过程的实用技术。我们将介绍减少冗余计算、高效管理内存以及提高整体吞吐量和降低延迟的方法。我们也会审视一些主要策略，例如键值 (KV) 缓存，以避免为之前的令牌重新计算注意力部分。我们还将研究优化过的注意力实现，例如 FlashAttention，它们可最大程度减少内存读写操作。此外，我们将讨论批处理技术以提高吞吐量，并探讨推测解码以实现更快的生成。实行这些策略对于使大型语言模型在实际部署场景中变得实用且具有成本效益十分必要。

---
course: "nlp-fundamentals"
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-2-nlp-feature-engineering"
sourceId: 599
chapter: "nlp-feature-engineering"
title: "文本特征工程"
order: 2
description: "学习文本特征工程方法，包含 TF-IDF、N-gram 和用于自然语言处理模型的降维技术。"
hasQuiz: true
---

在处理完原始文本数据后，下一步很自然就是将其转变为机器学习算法能理解的数值形式。原始文本字符串不直接适用于大多数模型，因此需要生成有意义的数值特征。

本章将介绍用于文本表示的基本特征工程方法。我们将从词袋模型等简单的基于计数的方法的思路说起，并逐步介绍广泛使用的词频-逆文档频率 (TF-IDF) 加权方式。你将学习 TF-IDF 分数（通常计算为 $TF \times IDF$）如何在文档集合中衡量词语的重要性。

我们还将查看 N-gram（如二元词组、三元词组等）如何用于在特征中包含局部词序和上下文。最后，我们将提及处理文本特征可能存在的较高维度问题的方法，包括特征哈希和像奇异值分解 (SVD) 这样的降维方法。完成本章后，你将能够生成和比较文本数据的不同数值表示形式，为其作为机器学习模型的输入做好准备。

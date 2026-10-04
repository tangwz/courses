---
course: "introduction-quantum-computing"
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-5-basic-quantum-circuits"
sourceId: 1428
chapter: "basic-quantum-circuits"
title: "量子线路入门"
order: 5
description: "设计并模拟完整的量子线路。学习量子不可克隆定理和隐形传态逻辑。"
hasQuiz: false
---

到目前为止，您已经操作了单个量子比特，并使用特定的门对创建了纠缠。本章将重点转向如何将这些元素组合成完整的量子线路。我们将确立线路图的标准规范，让您能够直观地看到量子信息如何随时间在系统中传递。

您还将了解到量子软件设计所遵循的约束。一种主要的限制是量子不可克隆定理，该定理指出，任意量子态 $\lvert \psi \rangle$ 都无法被完美复制。该定理使得量子算法无法使用经典的“复制粘贴”逻辑。相反，您必须依靠其他协议来传递信息。

我们将实现两种此类协议：超密编码和量子隐形传态。这些例子说明了如何将纠缠作为通信资源来使用。学习完本章后，您将能够使用 Python 构建并模拟这些线路，验证信息是否在不违反物理定律的情况下到达目的地。

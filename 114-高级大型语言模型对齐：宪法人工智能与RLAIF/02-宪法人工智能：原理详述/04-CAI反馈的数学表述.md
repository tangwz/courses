# CAI反馈的数学表述

来源：[原文](https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-2-constitutional-ai-theory/cai-mathematical-formulation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

宪法AI（CAI）反馈流程的规范化表述侧重于其监督学习 (supervised learning)（SL）阶段。此阶段旨在通过教导LLM模仿其自身输出的章程化修订版本，从而改进LLM的表现。这些修订版本是通过AI驱动的批改和修订循环生成的。

令$M_{\theta}$表示以$\theta$为参数 (parameter)的基础大型语言模型。给定一个输入提示$x$，模型生成一个初始回应$y_{init}$：
$y_{init} \sim P(y | x; \theta)$
此表示法说明，$y_{init}$是从模型$M_{\theta}$根据输入$x$定义的序列$y$的概率分布中取样得到的。

CAI SL阶段的核心在于根据预定义的章程$C$生成一个更理想的回应$y_{revised}$。这通常分两步完成：

1. **批改生成：** 一个AI系统（通常是同一个LLM $M_{\theta}$被恰当提示，或是一个专门的批改模型$M_{critique}$）在提示$x$和章程$C$的背景下分析初始回应$y_{init}$。它识别$y_{init}$中可能违反$C$中原则的方面，并生成批改$c$。我们可以用函数形式表示此流程：
   $c = \text{批改}(x, y_{init}, C)$
   此批改可能是对缺陷的文字说明，或是一条具体的改进指令。
2. **修订生成：** 另一个AI系统（可能是再次使用$M_{\theta}$，或是一个专门的修订模型$M_{revise}$）将原始提示$x$、初始回应$y_{init}$、批改$c$和章程$C$作为输入。然后它生成一个修订后的回应$y_{revised}$，此回应既处理了批改，又符合$C$。
   $y_{revised} = \text{修订}(x, y_{init}, c, C)$
   类似于初始回应生成，批改和修订步骤可以从概率角度看待，涉及从以各自输入为条件的分布中取样。

下图呈现了针对单个训练实例的数据生成流程：

> 宪法AI中生成单个监督微调 (fine-tuning)示例的数据流程。该流程使用基于章程的AI生成批改和修订，以优化LLM的初始回应。

整个流程对大量多样化提示数据集\ {x\_i\}重复进行，以生成一个符合章程的输入-输出对数据集：
$D_{CAI} = \{ (x_i, y_{revised, i}) \}_{i=1}^N$

CAI SL阶段的主要目的是使用此数据集$D_{CAI}$训练一个新模型$M_{aligned}$（通常从$M_{\theta}$初始化）。训练目标通常是标准监督微调（SFT），即最小化给定提示下修订回应的负对数似然：
$L_{SFT}(\theta_{aligned}) = - \frac{1}{N} \sum_{i=1}^N \log P(y_{revised, i} | x_i; \theta_{aligned})$
这里，$\theta_{aligned}$表示正在微调模型的参数。通过最小化此损失，模型$M_{aligned}$学习直接生成类似于修订后、符合章程的回应，在推理 (inference)时无需显式批改和修订步骤。

需认识到，此表述简化了一个潜在的复杂流程。批改和修订步骤本身可能涉及多次LLM调用、特定提示策略以及采样程序（例如，使用温度缩放），以生成多样且有效的反馈。

此外，虽然此阶段的主要输出是用于SFT的数据集$D_{CAI}$，但生成的数据有时可以被重新利用。例如，对于给定的提示$x_i$，对$(y_{revised, i}, y_{init, i})$隐式定义了一种偏好：根据章程，$y_{revised}$优于$y_{init}$。此偏好数据可能用于训练奖励模型，以供后续强化学习 (reinforcement learning)阶段使用（如在RLAIF中考察的那样，本课程后续讨论），尽管原始CAI框架侧重于使用修订后输出的直接SFT步骤。

这种数学框架强调了CAI如何将高层原则（章程）转化为适合标准机器学习 (machine learning)优化技术（$L_{SFT}$）的具体训练数据（$D_{CAI}$），从而引导LLM的表现，而无需对每个具体交互进行人工标注。其有效性取决于章程的质量以及AI系统根据其生成有意义批改和修订的能力。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2212.08073; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  本文介绍了宪法式人工智能（CAI）框架，详细阐述了通过AI生成的批评和修订来训练大型语言模型（LLM）的自我修正过程，以提高无害性，涵盖了监督学习和强化学习阶段。

---

[上一节](03-%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0%E9%98%B6%E6%AE%B5%EF%BC%88%E6%89%B9%E8%AF%84%E4%B8%8E%E4%BF%AE%E6%94%B9%EF%BC%89.md) · [下一节](05-%E4%B8%8E%E6%8C%87%E4%BB%A4%E9%81%B5%E5%BE%AA%E7%9A%84%E5%85%B3%E8%81%94.md)

---
course: "fine-tuning-small-language-model"
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-7-model-merging-and-deployment"
sourceId: 1507
chapter: "model-merging-and-deployment"
title: "模型合并与部署"
order: 7
description: "将 LoRA 权重与基础模型合并，并使用现代推理框架部署最终模型。"
hasQuiz: false
---

到目前为止，您已经完成对参数高效适配器的训练与评估。微调过程产出了一组 LoRA 权重，这些权重充当原始模型的修改项。为了在生产环境中高效运行模型，需要将这些更新后的权重与基础架构结合起来。

本章介绍将模型从训练产物转变为部署应用所需的具体步骤。首先，您将把训练好的 LoRA 适配器融合到基础模型层中。对于基础权重矩阵 $W$ 和低秩适配器矩阵 $A$ 与 $B$，合并后的权重矩阵计算如下：

$$W' = W + AB$$

此操作使模型能够独立运行，无需在推理时动态应用适配器层，从而消除了额外的计算开销。

合并权重后，您会将最终模型导出为 Safetensors 格式。这种格式提供了一种安全且快速的机制，可将张量直接加载到内存中。随后，您将配置 vLLM 来运行模型，以实现高吞吐量的本地推理。最后，您将编写一个 RESTful API 来封装该模型服务，以便外部应用程序可以通过编程方式发送文本生成请求。在本节结束时，您将拥有一套部署在本地服务器上、能够处理即时查询且功能完备的特定任务语言模型。

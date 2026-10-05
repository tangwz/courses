# 训练后合并LoRA权重

来源：[原文](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-4-advanced-lora-variants/merging-lora-weights)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当大语言模型 (LLM) 通过低秩适配 (LoRA) 进行微调 (fine-tuning)时，会得到两组不同的参数 (parameter)：基础模型的原始权重 (weight) ($W_0$) 和训练过的低秩适配器矩阵 ($A$ 和 $B$)。推理 (inference)时，适配层的输出通过结合原始层的输出与从LoRA路径获得的输出计算，并乘以缩放因子$\frac{\alpha}{r}$。数学上，对于给定输入 $x$，经LoRA适配的权重矩阵 $W$ 的修改后前向传播为：

$h = W_0 x + \frac{\alpha}{r} B A x$

虽然保持LoRA适配器独立提供了灵活性，使你能够动态加载、卸载，甚至将不同的适配器与同一基础模型结合使用，但在某些情况下，将适配器权重直接并入基础模型权重是有利的。这个过程称为LoRA权重合并。

### 合并过程说明

合并LoRA权重 (weight)涉及计算有效权重更新 $\Delta W = \frac{\alpha}{r} B A$，并将其直接添加到原始权重矩阵 $W_0$。结果是一个新的权重矩阵 $W_{merged}$，它包含了学到的适配：

$W_{merged} = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$

计算完成后，$W_{merged}$ 在模型层中替代原始权重矩阵 $W_0$。使用这个特定合并模型进行推理 (inference)时，不再需要独立的LoRA矩阵 $A$ 和 $B$。该层随后像使用 $W_{merged}$ 的标准层一样运行：

$h = W_{merged} x$

此计算在训练完成后离线进行。你为模型中每个经LoRA适配的层执行此计算。

### 合并LoRA权重 (weight)的优点

合并适配器提供了实际益处，主要与部署和推理 (inference)性能有关：

1. **部署简化：** 合并后的模型在结构上与原始基础模型相同。它只包含组合权重矩阵 ($W_{merged}$)，推理时不需要对LoRA路径进行任何特殊处理。这大幅简化了部署流程，因为你通常可以使用为常规LLM设计的标准工具和基础设施，无需修改。你分发的是一组权重，就像原始预训练 (pre-training)模型一样。
2. **潜在推理加速：** 在标准LoRA设置中，每次通过适配层的正向传播需要两条矩阵乘法路径：一条用于原始权重 ($W_0 x$)，另一条用于LoRA适配器 ($\frac{\alpha}{r} B A x$)，之后是加法。合并预先计算了组合权重矩阵 $W_{merged}$。因此，正向传播只需一次矩阵乘法 ($W_{merged} x$)。这可以减少计算开销，并可能降低推理延迟，特别是在每层计算最小化很重要的环境中。

   > 合并LoRA权重前后正向传播计算的比较。合并通过预计算组合权重简化了计算图。
3. **框架兼容性：** 合并后的模型是标准模型检查点。它们可由各种推理框架或库轻松加载和使用，这些框架或库可能不原生支持处理独立的PEFT适配器。

### 权衡与重要考量

虽然合并提供了优点，但也伴随着权衡：

1. **灵活性丧失：** 合并通常是不可逆的操作，除非你保留原始基础模型权重 (weight)和独立的适配器权重 ($A$ 和 $B$)。一旦合并，你就失去了在同一基础模型上动态切换不同适配器的能力，除非创建多个完整大小的模型副本。合并后调整LoRA缩放因子$\alpha$也是不可能的。如果你需要使用同一基础模型但不同适配来服务多个任务，保持适配器独立通常更有效。
2. **存储影响：** LoRA适配器与基础模型相比非常小（通常是兆字节对吉字节）。如果你为许多不同任务微调 (fine-tuning)单个基础模型，存储大量小型适配器在存储上效率显著更高，比存储大量完整大小的合并模型。合并为每个你合并的适配器创建一个与原始基础模型大致相同大小的模型检查点。

   
   
   [交互图表：存储对比 (示意)](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-4-advanced-lora-variants/merging-lora-weights#plot-5i3w4r)
   
   <details>
   <summary>图表数据（JSON）</summary>
   
   ```json
   {
     "data": [
       {
         "type": "bar",
         "name": "基础模型",
         "x": [
           "存储需求"
         ],
         "y": [
           14000
         ],
         "marker": {
           "color": "#495057"
         }
       },
       {
         "type": "bar",
         "name": "10个LoRA适配器 (独立)",
         "x": [
           "存储需求"
         ],
         "y": [
           100
         ],
         "marker": {
           "color": "#228be6"
         }
       },
       {
         "type": "bar",
         "name": "10个合并模型",
         "x": [
           "存储需求"
         ],
         "y": [
           140000
         ],
         "marker": {
           "color": "#fa5252"
         }
       }
     ],
     "layout": {
       "title": {
         "text": "存储对比 (示意)",
         "font": {
           "size": 14
         }
       },
       "barmode": "group",
       "xaxis": {
         "title": {
           "text": "场景",
           "font": {
             "size": 12
           }
         }
       },
       "yaxis": {
         "title": {
           "text": "大致存储空间 (MB)",
           "font": {
             "size": 12
           }
         },
         "type": "log"
       },
       "legend": {
         "font": {
           "size": 10
         }
       },
       "margin": {
         "l": 60,
         "r": 20,
         "t": 40,
         "b": 40
       }
     }
   }
   ```
   
   </details>
   
   

   > 存储需求示意对比。存储多组小型LoRA适配器在空间上效率显著更高，比存储多个完整大小的合并模型。注意Y轴的对数刻度。数值假设一个约7B参数 (parameter)的模型 (FP16约14GB) 和约10MB的适配器。
3. **精度考量：** 合并操作 ($W_0 + \frac{\alpha}{r} B A$) 通常使用模型的原生精度（例如 float32 或 float16）执行。如果你在训练期间使用了量化 (quantization)技术，例如使用QLoRA时，基础模型 ($W_0$) 可能以4位或8位格式存储，合并需要谨慎处理。通常，量化后的基础模型权重需要先反量化回更高精度 (如 float16)，然后才能准确执行加法。结果的 $W_{merged}$ 将处于这种更高精度格式。

### 实际操作

像Hugging Face的`PEFT` (参数 (parameter)高效微调 (fine-tuning)) 这样的库提供了方便的方法来执行此合并。通常，在加载基础模型并附加训练过的LoRA适配器后，你可以调用 `model.merge_and_unload()` 这样的函数。此函数会遍历各层，为每个经LoRA适配的模块计算 $W_{merged}$，用使用合并权重 (weight)的标准层替换专用LoRA层实现，并从内存中移除现在多余的适配器参数 ($A$, $B$)。

```python
# 使用Hugging Face PEFT库的示例
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 加载基础模型
base_model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# 加载训练过的LoRA适配器
adapter_path = "./my-lora-adapter"
model = PeftModel.from_pretrained(model, adapter_path)

# 将适配器权重合并到基础模型中
# 这会原地修改模型
model = model.merge_and_unload()

# 模型现在是一个标准的AutoModelForCausalLM实例
# 其中的LoRA适配已并入其权重。
# 它可以像任何常规Hugging Face模型一样保存、加载和使用。
# model.save_pretrained("./merged_model")
```

总而言之，合并LoRA权重是一种有价值的训练后技术，用于简化部署并可能优化特定任务适配的推理 (inference)性能。它以动态适配器加载的灵活性，换取了标准模型检查点的操作简便性。是否合并的决定取决于你的具体部署限制、对多任务灵活性的需求以及存储考量。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍低秩适应（LoRA）的原始研究论文，详细阐述了其数学公式和用于适应大型语言模型的核心机制。
- [PeftModel.merge_and_unload](https://huggingface.co/docs/peft/main/en/package_reference/peft.peft_model.PeftModel#peft.PeftModel.merge_and_unload) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，提供了`merge_and_unload`函数用于将LoRA权重集成到基础模型中的实际细节和用法。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了QLoRA，一种用于微调量化模型的方法，强调了将LoRA权重与量化基础模型合并时相关的精度考虑。

---

[上一节](01-LoRA%20%E5%88%9D%E5%A7%8B%E5%8C%96%E7%AD%96%E7%95%A5.md) · [下一节](03-%E9%87%8F%E5%8C%96LoRA%20%28QLoRA%29%EF%BC%9A%E5%8E%9F%E7%90%86.md)

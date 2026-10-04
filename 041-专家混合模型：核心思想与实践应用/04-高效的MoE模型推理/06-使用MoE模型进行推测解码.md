# 使用MoE模型进行推测解码

来源：[原文](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference/speculative-decoding-moe)

[返回章节目录](README.md) · [返回课程目录](../README.md)

自回归 (autoregressive)生成面临一个持续的延迟问题，即使在专家混合（MoE）模型中应用了像卸载和量化 (quantization)这样的内存节省方法来处理其静态占用。每个令牌都是按顺序生成的，需要大型MoE模型进行完整的正向传递，这个过程受限于内存带宽和计算开销。推测解码提供了一种有效的方法，通过改变令牌生成方式来减少这种延迟。

这项策略基于一个简单的观察：大型、高性能的模型速度慢，而小型、性能较低的模型速度快。推测解码同时使用两种模型：

1. **目标模型：** 我们的大型、高性能MoE模型。它准确但每个令牌的延迟高。
2. **草稿模型：** 一个小得多、速度更快的模型。这可以是MoE的精简版，或者只是一个参数 (parameter)少得多的标准Transformer模型。它的用处在于速度，而非完善。

草稿模型不再是MoE模型一次生成一个令牌，而是快速生成多个未来令牌的“草稿”。然后，大型MoE模型在一个单独的并行正向传递中检查整个草稿，这比多次顺序传递快得多。

### 推测解码循环

该过程在一个循环中运行，生成令牌直到满足停止条件。每次迭代都包含一个草稿与验证周期。

1. **起草：** 小型草稿模型接收当前序列，并自回归 (autoregressive)地生成一个包含 $k$ 个令牌的短续写。例如，如果当前文本是“The quick brown fox”，草稿模型可能会建议“jumps over the”。这一步速度很快。
2. **验证：** 大型MoE模型接收原始序列加上 $k$ 个草稿令牌（“The quick brown fox jumps over the”）。它执行一次正向传递，计算每个位置上*下一个*令牌的概率分布。
3. **接受：** 我们将草稿模型的预测与目标MoE的输出进行比较。从第一个草稿令牌（“jumps”）开始，我们检查它是否是MoE模型很可能选择的令牌。
   - 如果令牌匹配（或通过采样检查），我们接受该令牌并移动到草稿中的下一个令牌（“over”）。
   - 这个过程持续进行，直到我们发现不匹配，或者草稿被完全接受。
4. **修正：** 第一次有令牌被拒绝时，我们丢弃它以及草稿中所有后续令牌。然后，我们使用MoE模型在该位置计算出的更准确的概率分布，采样一个新的、正确的令牌。
5. **继续：** 该过程从新扩展序列的末尾重复进行。

下图说明了这一工作流程。其要点是，开销大的目标MoE模型在每个周期中只被调用一次，以并行验证多个令牌，从而分摊其高延迟。

> 推测解码过程。一个快速的草稿模型生成多个候选令牌，然后由较慢、更准确的MoE模型单次传递进行验证。

### 性能提升与实现细节

推测解码带来的速度提升与每个周期平均接受的令牌数量直接相关。如果草稿模型能很好地近似目标MoE，它就可以持续生成多个正确的令牌，从而使系统能够跳过多次顺序且耗时的正向传递。例如，如果每个周期平均接受三个令牌，则生成过程的实际运行时间可以减少近三分之二。

一个高层次的实现将遵循以下逻辑：

```python
# 推测解码的伪代码
def speculative_decode(prompt, target_moe_model, draft_model, max_len, k):
    tokens = tokenize(prompt)

    while len(tokens) < max_len:
        # 1. 使用快速模型起草 k 个令牌
        draft_tokens = []
        context = tokens
        for _ in range(k):
            next_token = draft_model.generate(context, 1)
            draft_tokens.append(next_token)
            context.append(next_token)

        # 2. 调用目标模型一次，验证所有 k+1 个令牌
        # 输入包含原始序列加上草稿
        target_logits = target_moe_model.forward(tokens + draft_tokens)

        # 3. 比较并接受/拒绝
        accepted_count = 0
        for i in range(k):
            # 检查草稿令牌是否与目标模型最有可能的令牌匹配
            draft_token_i = draft_tokens[i]
            target_distribution_i = softmax(target_logits[len(tokens) + i - 1])

            if is_accepted(draft_token_i, target_distribution_i):
                tokens.append(draft_token_i)
                accepted_count += 1
            else:
                # 4. 从目标模型的分布中采样进行修正
                corrected_token = sample(target_distribution_i)
                tokens.append(corrected_token)
                break # 退出验证循环

        # 如果所有草稿令牌都被接受，则采样最后一个令牌
        if accepted_count == k:
            final_distribution = softmax(target_logits[-1])
            final_token = sample(final_distribution)
            tokens.append(final_token)

    return detokenize(tokens)
```

> **注意：** 接受函数 `is_accepted` 可以是一个简单的贪婪检查（即 `argmax(target_distribution_i) == draft_token_i`），也可以是一种更复杂的采样方法，如拒绝采样，以保留原始模型的输出分布。

### 权衡与考虑

推测解码虽然有效，但也需要仔细调整并引入自身的权衡。

- **草稿模型选择：** 理想的草稿模型需要权衡。它必须比MoE模型快很多，但也要足够准确，以实现高接受率。过小的模型会生成低质量的草稿，导致频繁拒绝和很小的加速效果。草稿模型的一个好选择通常是目标MoE的精简版，因为它经过专门训练来模仿大型模型的行为。
- **内存开销：** 此技术需要将目标MoE和草稿模型都加载到GPU内存中。尽管草稿模型很小，但这增加了服务大型MoE已经存在的庞大内存压力。这种用内存换取延迟的权衡必须根据可用的硬件进行评估。
- **草稿长度 ($k$)：** 要起草的令牌数量 $k$ 是一个重要的超参数 (parameter) (hyperparameter)。较大的 $k$ 能提供更高的潜在加速，但也会增加早期拒绝的概率，从而浪费了用于生成草稿后期部分的计算。最佳的 $k$ 取决于任务和草稿模型的质量。对于代码生成等高度可预测的任务，较大的 $k$ 可能有效。对于更具创造性或复杂的任务，较小的 $k$ 通常更安全。

通过巧妙地将快速但不完善的草稿与缓慢但准确的验证结合起来，推测解码有效地并行化了自回归 (autoregressive)过程。它是减少MoE推理 (inference)端到端延迟的有力工具，使得这些大型稀疏模型更适用于交互式应用。

## 参考资料

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed (2024)
  Journal: arXiv preprint arXiv:2401.04088; DOI: [10.48550/arXiv.2401.04088](https://doi.org/10.48550/arXiv.2401.04088)
  介绍了Mixtral 8x7B模型，这是一个稀疏激活的专家混合模型，为推测解码中提到的“目标MoE模型”提供了关键背景，并展示了这种大规模MoE架构的性能。
- [Speculative Decoding: Faster LLM Inference with Small Models](https://blog.google/technology/ai/speculative-decoding-faster-llm-inference-small-models/) — Alex M. Dai, Shibo Wang (2023)
  Publisher: Google AI Blog
  一篇Google AI博客文章，提供了推测解码的易懂解释，包括其原理和优势，是理解该技术的重要概览。

---

[上一节](05-MoE%E5%B1%82%E9%87%8F%E5%8C%96%E6%8A%80%E6%9C%AF.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%BC%98%E5%8C%96%E6%8E%A8%E7%90%86%E7%AE%A1%E7%BA%BF.md)

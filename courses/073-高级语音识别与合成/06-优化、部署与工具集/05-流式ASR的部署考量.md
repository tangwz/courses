# 流式ASR的部署考量

来源：[原文](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-6-optimization-deployment-toolkits/streaming-asr-deployment)

[返回章节目录](README.md) · [返回课程目录](../README.md)

部署增量处理到达音频的自动语音识别 (ASR) 系统（通常称之为流式ASR），与预先获取整个音频文件的批处理相比，带来独特的挑战。其主要目的是以最小延迟提供准确的转录，以支持实时应用，如实时字幕、语音助手和命令控制。实现此目标需要仔细评估模型架构、处理策略和性能权衡。

### 延迟：主要制约

延迟是流式ASR中最重要的因素。用户希望获得近乎即时的反馈。我们通常通过两种方式衡量延迟：

1. **实时因子 (RTF)：** 处理时间与音频时长的比率。为了真正的实时性能，RTF必须小于1。
   $RTF = \frac{\text{处理时间}}{\text{音频时长}}$
   RTF为0.5表示系统在0.5秒内处理1秒的音频。
2. **首字延迟 (TTFT) / 感知延迟：** 用户说出一个词与第一个相应转录词出现之间的延迟。这直接影响用户体验。
3. **句末延迟：** 用户说完话与最终、稳定转录可用之间的延迟。

延迟源于多种因素：网络传输（如适用）、音频缓冲、模型内的计算时间（算法延迟），以及可能的解码算法。

### 基于分块的处理

流式模型对分割成小块的传入音频进行操作，通常范围是几十到几百毫秒。ASR模型在每个分块到达时进行处理。

> 基于分块处理的简化视图。音频分块到达，被缓冲，由模型处理，并生成中间结果。后续的分块更新上下文 (context)并完善假设。

分块大小的选择涉及权衡：

- **较小分块：** 延迟更低（结果更快出现），但为模型的每个处理步骤提供的上下文较少，可能影响准确性，特别是对于需要未来上下文（前瞻）的模型。
- **较大分块：** 延迟更高，但每一步提供更多上下文，可能提高准确性。

某些模型架构，特别是那些涉及双向处理或分块内特定注意力机制 (attention mechanism)的架构，可能需要少量未来的音频上下文，这被称为算法前瞻。这种前瞻会增加固有的延迟，因为系统必须等待该未来音频才能处理当前分块。

### 适合流式处理的架构

并非所有ASR架构都同样适合流式处理。严重依赖对整个输入序列进行注意的模型（如标准双向Transformer或未经修改的注意力编码器-解码器）本质上难以有效流式处理，否则会带来显著的延迟代价或复杂的近似。为流式处理设计或调整的架构包括：

- **RNN换能器 (RNN-T)：** 明确为流式处理而设计。它联合处理音频编码器输出和预测的文本序列，单调地发出与输入音频流对齐 (alignment)的输出标签。
- **连接时序分类 (CTC)：** 可以通过处理分块和发出输出来适应流式处理。通常与重叠分块或有限未来上下文 (context)等技术结合，以提高边界预测。
- **因果或时间受限自注意力 (self-attention) (Transformers)：** Transformer模型可以通过将自注意力机制 (attention mechanism)限制为只关注过去和当前上下文（因果注意力），或者在分块处理方案中关注有限的未来上下文窗口来使其可流式处理。

这些架构允许模型仅基于已处理的音频（加上任何定义的前瞻）进行预测，从而实现增量输出生成。

### 缓冲策略

输入音频和中间模型状态或输出假设都需要有效的缓冲区管理。

- **输入音频缓冲区：** 在传入音频分块送入模型之前，用于保存它们。其大小与处理分块大小以及任何所需的前瞻有关。
- **输出假设缓冲区/解码器状态：** 存储转录的当前最佳猜测以及任何必要的状信息（如集束搜索假设或语言模型状态），以确保分块间的连续性。

低效的缓冲可能引入额外延迟或导致音频数据丢失。

### 端点检测和语音活动检测 (VAD)

在连续流式处理中，一个重要的挑战是确定用户何时说完（一个话语边界）。这个过程被称为端点检测或语音活动检测 (VAD)。如果没有有效的端点检测，ASR系统可能：

1. 无限期地等待更多音频，在最终确定转录前引入长时间延迟。
2. 继续处理静音或背景噪声，浪费计算资源。
3. 过早地切断用户，导致转录不完整。

VAD算法范围从简单的基于能量的方法到训练用于区分语音和非语音片段的复杂神经网络 (neural network)分类器。VAD通常与ASR系统紧密结合。它可能分析原始音频、声学特征，甚至ASR模型内部状态（如CTC空白概率）来做出判断。在激进的端点检测（低延迟但切断语音的风险较高）和保守的端点检测（更安全但延迟更高）之间存在权衡。

### 部分假设与最终假设

流式系统通常提供：

1. **部分结果：** 随着音频分块的处理而生成的中间转录。这些结果延迟低但不稳定，并会随着更多上下文 (context)的到来而变化。它们在用户界面中对于即时反馈很有用。
2. **最终结果：** 在端点检测机制检测到话语结束之后生成的稳定转录。这种假设通常更准确，因为它包含了话语的完整上下文（或大部分片段）。

管理部分结果到最终结果的转换和显示对于流畅的用户体验很重要。

### 计算资源与优化

流式ASR系统通常需要持续运行或处理大量并发用户，对计算资源提出较高的要求。本章前面讨论的优化技术，如量化 (quantization)（降低数值精度，例如从FP32到INT8）和模型剪枝（移除冗余权重 (weight)），经常应用于流式模型。这些技术减少模型大小和计算成本（$FLOPs$），有助于达到所需的低RTF并在服务器甚至边缘设备上高效运行。ONNX Runtime或TensorRT等优化推理 (inference)引擎常用于在目标硬件（CPU、GPU、专用加速器）上高效执行这些优化模型。

有效部署流式ASR涉及平衡延迟、准确性和计算成本。它需要选择合适的模型架构，仔细调整分块和缓冲策略，实施有效的端点检测，并使用模型优化技术来满足实时交互的需求。

## 参考资料

- [Efficient Streaming Transformer-based ASR for Production](https://doi.org/10.21437/Interspeech.2020-1681) — Yu-An He, Tian-An Li, Liang-Yan Gui, Lei Xie, Zheng-Wen Zhang (2020)
  Journal: Proceedings of Interspeech 2020; Publisher: ISCA (International Speech Communication Association); Pages: 4525-4529; DOI: [10.21437/Interspeech.2020-1681](https://doi.org/10.21437/Interspeech.2020-1681)
  讨论了在实时流媒体环境中部署基于Transformer的ASR系统的设计选择和优化策略。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2023)
  Publisher: Pearson
  一本全面的教材，涵盖了ASR基础知识，包括各种架构（CTC、RNN-T、Transformer）以及与实时处理和部署相关的内容。（第四版，草稿）
- [Live Speech Transcription with Sequence-to-Sequence Models](https://doi.org/10.21437/Interspeech.2019-1506) — Nitesh Jaitly, Yonghui Wu, Ruoming Pang, Yong Li, Chung-Cheng Chiu, Anjuli Kannan, Xiang Li, Rachel Batson, David Rybach, Patrick Nguyen, Navdeep Jaitly (2019)
  Journal: Proceedings of Interspeech 2019; Publisher: ISCA; Pages: 3673-3677; DOI: [10.21437/Interspeech.2019-1506](https://doi.org/10.21437/Interspeech.2019-1506)
  讨论了构建低延迟、高准确度实时语音转录系统的架构选择和挑战。

---

[上一节](04-%E4%BC%98%E5%8C%96%E6%8E%A8%E7%90%86%E5%BC%95%E6%93%8E%EF%BC%88ONNX%20Runtime%2C%20TensorRT%EF%BC%89.md) · [下一节](06-%E5%AE%9E%E6%97%B6%E6%96%87%E6%9C%AC%E8%BD%AC%E8%AF%AD%E9%9F%B3%EF%BC%88TTS%EF%BC%89%E7%9A%84%E9%83%A8%E7%BD%B2%E8%80%83%E8%99%91.md)

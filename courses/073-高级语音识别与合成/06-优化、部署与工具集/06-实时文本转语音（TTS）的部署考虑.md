# 实时文本转语音（TTS）的部署考虑

来源：[原文](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-6-optimization-deployment-toolkits/real-time-tts-deployment)

[返回章节目录](README.md) · [返回课程目录](../README.md)

部署能够参与实时对话的文本转语音（TTS）系统，与离线生成音频相比，面临特殊难题。在语音助手、交互式代理或辅助工具等应用中，用户期望即时反馈。用户提示与合成语音之间长时间的停顿会严重影响用户体验。这里对部署低延迟、高响应TTS系统所需的具体性能目标和技术方案进行讨论。

### 延迟：主要障碍

实时TTS中最重要的因素是延迟。我们通常通过两种方式衡量：

1. **首字节时间（TTFB）：** 这是从系统收到文本输入（或其最后一部分）到第一块音频数据可供播放的时长。对于交互式系统，最小化TTFB很必要，以营造响应迅速的感觉。TTFB超过几百毫秒会使交互显得迟缓。
2. **实时因子（RTF）：** 定义为合成音频所用时间与合成音频本身的持续时间之比：
   
   $$
   RTF = \frac{\text{合成时间}}{\text{音频时长}}
   $$
   
   对于一个能与语音同步生成音频的系统，RTF必须小于1.0。理想情况下，对于实时交互（合成应在播放结束前完成，以考虑缓冲和网络抖动），目标RTF通常要低得多，可能低于0.1甚至0.05，这取决于硬件和模型复杂度。

延迟产生于TTS管线中的多个阶段：

- **文本前端：** 文本标准化、语音转换和语言特征提取通常贡献相对较少的延迟，但复杂的文本分析会增加等待时间。
- **声学模型：** 从文本特征生成声学特征（如梅尔谱图）。自回归 (autoregressive)模型（例如Tacotron 2）按顺序生成特征，使得推理 (inference)时间与输出长度成比例，本质上较慢。非自回归模型（例如FastSpeech 2、Glow-TTS）并行生成特征，大幅减少这一瓶颈。
- **声码器：** 从声学特征合成最终音频波形。这通常是计算最密集的步骤。自回归声码器（例如WaveNet、WaveRNN）逐个样本生成音频，质量很高但推理非常慢。现代非自回归声码器（例如Parallel WaveGAN、HiFi-GAN、WaveGlow）通过并行波形生成提供显著加速，使其更适合实时应用，尽管有时会有细微的质量权衡。

### 管理计算负载

实现低TTFB和RTF需要仔细管理计算资源。

- **模型优化：** 前面讨论过的技术，例如量化 (quantization)（降低数值精度）、剪枝（去除冗余模型权重 (weight)）和知识蒸馏 (knowledge distillation)（训练一个较小模型模仿较大模型），是基础。这些技术减少模型大小和浮点运算，直接提升目标硬件上的推理 (inference)速度。
- **硬件加速：** 通常需要使用GPU或专用加速器（如移动设备上的TPU或NPU）。ONNX Runtime或TensorRT等优化推理运行时可以通过融合操作和利用硬件特定指令来进一步加快执行速度。
- **模型架构选择：** 选择非自回归 (autoregressive)声学模型和声码器通常是满足严格实时要求的先决条件。在序列长度上并行计算的能力是一个主要优势。

### 流式合成

流式TTS不是等到整个话语合成完毕再开始播放，而是生成并以小块形式传输音频。这显著降低了感知延迟，尤其是TTFB。

> 流式TTS的工作流程。文本被分块，通过管线增量处理，并将音频块发送到播放缓冲，从而减少感知到的启动延迟。

有效实现流式传输需要考虑以下方面：

- **分块逻辑：** 确定文本分割的合适边界（例如，句尾、逗号，甚至固定长度），避免引入不自然的停顿或在思考中打断语义单元。
- **状态管理：** 如果使用适应流式传输的自回归 (autoregressive)组件，管理块之间的隐藏状态对保持连贯性很重要。
- **边界伪影：** 确保音频块之间平滑过渡。在声码阶段可能需要重叠相加/保存等技术，特别是对于某些声码器类型，以避免可听见的咔嗒声或不连续性。非自回归模型通常更自然地处理这个问题，因为它们在很大程度上独立处理块。

### 缓存

对于某些常用短语或回复的应用（例如，“好的”、“正在呼叫联系人...”、标准问候语），缓存可以显著减少延迟。

- **波形缓存：** 最直接的方法是缓存常用、固定文本输入的最终生成音频。这为已知短语提供最低延迟，但需要存储和识别可缓存输入的机制。
- **中间特征缓存：** 缓存文本前端的输出（音素、持续时间）甚至声学特征是可行的，但通常效果不佳。文本前端缓存节省的时间很少，而如果韵律或说话人身份改变，声学特征缓存可能无法重用。

缓存在受限场景中特别有用，例如简单的IVR系统或设备控制命令。

### 系统架构选择

整体系统设计影响实时性能：

- **设备端 vs. 服务器端：**
  - **服务器端：** 允许使用托管在高性能硬件（GPU）上的更大、更强的模型。但它会引入请求和返回音频流的网络延迟，这种延迟可能不稳定，并抵消快速服务器端合成的优势。
  - **设备端：** 消除了网络延迟，提供最即时的响应可能。这需要高度优化的模型（使用本章中的技术），能够高效运行在资源受限的硬件（智能手机、智能音箱）上。混合方法也很常见，即简单/常见请求在设备端处理，复杂请求发送到服务器。
- **缓冲：** 客户端需要一个音频输出缓冲来平滑播放。合成过程可能以略微可变的速度生成音频块，而音频硬件需要连续流。缓冲适应这种差异，但需要仔细确定大小——太小，播放可能会卡顿；太大，则会增加整体感知延迟。

部署实时TTS需要整体方法，结合模型优化、架构选择（非自回归 (autoregressive)模型）、高效实现（流式传输、缓存），以及利用合适的硬件和推理 (inference)引擎。平衡合成质量、计算成本和响应能力是将TTS应用于交互式应用的核心难题。

## 参考资料

- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) — Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2006.04558](https://doi.org/10.48550/arXiv.2006.04558)
  介绍了一种对低延迟文本转语音很重要的非自回归声学模型。
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://neurips.cc/paper/2020/file/92c3b88b172a6a683e783457199c43d7-Paper.pdf) — Jungil Kong, Jaehyeon Kim, Jaekyoung Bae (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: NeurIPS; Volume: 33; DOI: [10.48550/arXiv.2010.05646](https://doi.org/10.48550/arXiv.2010.05646)
  提出了一种适用于实时语音生成的高效声码器架构。
- [Streaming End-to-End Neural Speech Synthesis](https://doi.org/10.1609/aaai.v35i12.17498) — Yi Ren, Jinglin Liu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu (2021)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence; Publisher: AAAI (Association for the Advancement of Artificial Intelligence); Volume: 35; Pages: 11090-11098; DOI: [10.1609/aaai.v35i12.17498](https://doi.org/10.1609/aaai.v35i12.17498)
  探讨了用于流式神经文本转语音以降低感知延迟的技术。

---

[上一节](05-%E6%B5%81%E5%BC%8FASR%E7%9A%84%E9%83%A8%E7%BD%B2%E8%80%83%E9%87%8F.md) · [下一节](07-%E8%AF%AD%E9%9F%B3%E5%A4%84%E7%90%86%E5%B7%A5%E5%85%B7%E5%8C%85%EF%BC%88ESPnet%2C%20NeMo%2C%20Coqui%EF%BC%89%E6%A6%82%E8%BF%B0.md)

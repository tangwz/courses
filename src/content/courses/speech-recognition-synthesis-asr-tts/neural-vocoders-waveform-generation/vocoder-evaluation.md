---
course: "speech-recognition-synthesis-asr-tts"
chapter: "neural-vocoders-waveform-generation"
lesson: "vocoder-evaluation"
sourceId: 3210
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-5-neural-vocoders-waveform-generation/vocoder-evaluation"
title: "合成音频质量评估"
description: "评估合成音频真实度和自然度的客观与主观方法。"
order: 7
plots: ["plots/3210-0.json"]
sourceHash: "5c62770bc31584abdacc4630e413361fdeb16c8c7a864ccb72df6253c99f8dba"
sourceCorrections: []
---

评估神经声码器的输出对于知晓其性能和比较不同模型非常重要。与传统声码器中常见的嗡嗡声或模糊不清等明显局限不同，神经声码器旨在实现与真实人声在听觉上难以区分的效果。这一更高的标准要求更精细的评估方法，包括自动化信号分析和人耳判断。仅仅生成一个波形是不够的；我们需要衡量这个波形听起来*有多好*。

### 客观评估指标

客观指标通过数学方式分析合成波形，并与真实（原始）录音进行比较。它们提供可重复的自动化评估，但可能不总与人耳感知完全一致。

- **对数频谱距离（LSD）：** 此指标衡量真实音频信号与合成音频信号之间对数功率谱的平均差异，通常逐帧计算。它衡量频谱内容的相似程度。LSD值越低，表示相似度越高。公式如下：

  
  $$
  LSD = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \frac{1}{K} \sum_{k=1}^{K} (10 \log_{10} |S_t(k)|^2 - 10 \log_{10} |\hat{S}_t(k)|^2)^2}
  $$
  

  此处，$S_t(k)$ 和 $\hat{S}_t(k)$ 分别表示原始信号和合成信号在时间帧 $t$ 和频率 bin $k$ 处的频谱幅度。$T$ 是总帧数，$K$ 是频率 bin 的数量。
- **梅尔倒谱失真（MCD）：** 这是语音合成中一个常用指标，用于衡量真实音频与合成音频之间梅尔倒谱系数（MCCs）的欧几里得距离。由于MCCs是根据人耳听觉系统对频率的感知（使用梅尔刻度）得出的，MCD被认为比原始频谱距离在听觉上更相关。它通常以分贝（dB）表示，值越低越好。计算通常涉及动态时间规整（DTW）来对齐 (alignment)MCC序列，然后再计算距离：

  
  $$
  MCD [dB] = \frac{10}{\ln 10} \sqrt{2 \sum_{d=1}^{D} (mcc_d - \hat{mcc}_d)^2}
  $$
  

  该公式计算对齐后特定维度 $d$（通常为13到40维）的失真。最终的 MCD 是所有帧的平均值。
- **信噪比（SNR）/ 分段信噪比（SegSNR）：** 信噪比（SNR）衡量原始信号功率与误差（原始信号与合成信号之差）功率的比值。通常SNR越高越好。分段信噪比（SegSNR）在短段（例如20-30毫秒）上计算SNR并取平均值，这通常比全局SNR与感知质量有更好的关联，因为它避免了安静片段在总计算中被响亮片段主导。然而，SNR指标可能对相位差异过于敏感，并且可能无法完全反映感知的自然度。
- **PESQ（语音质量感知评估）：** PESQ 在 ITU-T 建议 P.862 中定义，是一种用于预测主观听觉质量的算法。它通过复杂的听觉变换模型将原始参考信号与受损（合成）信号进行比较。输出分数通常在-0.5到4.5之间，接近平均主观意见得分（MOS），分数越高表示听觉质量越好。PESQ应用广泛，但需要真实参考信号。
- **STOI（短时客观清晰度）：** 该指标旨在通过衡量干净参考语音与处理后语音在不同频段短时帧内的时域包络相关性来预测语音清晰度。它产生0到1之间的分数，值越高表示清晰度越好。虽然主要用于噪声抑制评估，但它也能为合成语音的清晰度提供见解。

客观指标提供有价值的定量数据，但不能完全反映所有情况。一个模型可能在LSD或MCD上取得优异成绩，但仍然产生人类听者觉得不悦的不易察觉的瑕疵。

### 主观评估指标

主观测试涉及人类听众对合成音频质量进行评级。它们被认为是听觉质量的最终衡量标准，但妥善进行会更耗时且成本更高。

- **平均主观意见得分（MOS）：** 这是最常见的主观测试。一组听众根据绝对质量或自然度量表对音频样本进行评分，通常从1到5分。

  - 5: 优秀（与自然语音无法区分）
  - 4: 良好（存在一些小瑕疵）
  - 3: 一般（有明显瑕疵，但不令人厌烦）
  - 2: 差（有令人厌烦的瑕疵）
  - 1: 糟糕（完全不自然或无法理解）

  MOS测试需要仔细设置：受控的听音环境（例如，安静的房间、耳机），足够大且多样化的听众群体，清晰的指示，以及对结果进行统计分析（包括置信区间）以确保可靠性。
- **比较测试（A/B 或 A/B/X）：** 与绝对评级不同，听众直接比较两个（A/B）或更多样本。在A/B测试中，听众表达对样本A和样本B的偏好，或者表示没有偏好。在A/B/X测试中，听众听到A、B，然后是X（X是A或B），必须辨别X是匹配A还是B。这有助于衡量系统间的偏好和区分度。CMOS（比较MOS）分数通常从A/B测试中得出，表示在某个量表（例如-3到+3）上的平均偏好强度。
- **MUSHRA（多刺激隐藏参考与锚点）：** MUSHRA 由 ITU-R 建议 BS.1534 定义，在比较质量相近的多个系统时非常有用。听众同时收到多个刺激：原始参考（隐藏），被测各系统的输出，以及一个或多个低质量的“锚点”。听众根据参考，在0到100的连续量表上对每个刺激（如果提供的话，显式参考除外）进行评分。这种设置有助于听众校准他们的评分，并为高质量音频（其中差异可能很小）提供灵敏的衡量方法。

主观测试提供了对人类如何感知合成语音的最直接评估，反映了自然度、悦耳度以及客观指标可能遗漏的瑕疵等方面。



![不同声码器的示范性MOS评分](plots/3210-0.json)



> 示例平均主观意见得分（MOS），比较了传统声码器（Griffin-Lim）与几种神经声码器。分数越高表示感知的自然度越好。请注意，实际分数很大程度上取决于具体的模型、训练数据和测试条件。

### 实际考量

评估声码器时，请考虑以下几点：

1. **参考质量：** 真实音频的质量是基本。客观指标需要干净的参考信号，主观测试则依赖它进行比较（无论是显式的还是隐式的）。
2. **声码器与端到端：** 您是在独立评估声码器（使用真实的声学特征），还是作为完整TTS系统的一部分进行评估（使用预测的声学特征）？后者反映了系统整体性能，但会将声码器质量与声学模型的质量混合在一起。两种评估方式通常都有参考价值。
3. **数据多样性：** 确保评估涵盖不同说话者、说话风格和内容，以评估声码器的泛化能力。
4. **权衡：** 高质量声码器（如自回归 (autoregressive)模型）计算成本可能很高。评估可能还需要在考虑感知质量的同时，兼顾推理 (inference)速度、模型大小和实时部署的适用性。

最终，一个全面的评估策略将自动化客观指标用于快速迭代和诊断，并结合严格的主观测试来确认感知质量和自然度的真实改进。选择合适的组合取决于项目的目标和可用资源。

## 参考资料

- [Perceptual evaluation of speech quality (PESQ): An objective method for end-to-end speech quality assessment of narrowband telephone networks and speech codecs](https://www.itu.int/rec/T-REC-P.862-200102-I/en) — International Telecommunication Union (ITU-T) (2001)
  Journal: ITU-T Recommendation P.862; Publisher: International Telecommunication Union (ITU-T)
  定义了PESQ算法用于客观语音质量评估，是内容中提及的广泛使用的度量标准。
- [Method for the subjective assessment of intermediate quality levels of coding systems](https://www.itu.int/rec/R-REC-BS.1534-3-201509-I/en) — International Telecommunication Union (ITU-R) (2015)
  Journal: ITU-R Recommendation BS.1534-3; Publisher: International Telecommunication Union
  规定了MUSHRA方法，一种用于主观评估音频质量（特别是细微差异）的标准。
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) — Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu (2016)
  Journal: arXiv preprint arXiv:1609.03499; Publisher: arXiv; DOI: [10.48550/arXiv.1609.03499](https://doi.org/10.48550/arXiv.1609.03499)
  介绍了一种基础的神经声码器，并讨论了高质量合成语音感知评估的必要性。

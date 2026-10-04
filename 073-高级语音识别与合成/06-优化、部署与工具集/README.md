# 第 6 章：优化、部署与工具集

来源：[原章节](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-6-optimization-deployment-toolkits)

[返回课程目录](../README.md)

构建高性能的ASR和TTS模型仅仅是整个过程的一部分。让这些模型在实际使用中高效运行，会带来一系列技术难题。本章侧重于解决模型开发与真实环境部署之间的衔接问题。

我们将考察模型优化方法，包括量化、剪枝和知识蒸馏，以减少它们的计算开销($FLOPs$)和内存需求。接下来我们转向部署策略，讨论像ONNX Runtime和TensorRT这样的优化运行时，并处理流式ASR和低延迟TTS的特定需求。此外，还将提供流行语音处理框架的概览，以指导您的实现工作。

## 小节

- 1. [语音模型量化](01-%E8%AF%AD%E9%9F%B3%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96.md)
- 2. [模型剪枝与稀疏化](02-%E6%A8%A1%E5%9E%8B%E5%89%AA%E6%9E%9D%E4%B8%8E%E7%A8%80%E7%96%8F%E5%8C%96.md)
- 3. [ASR/TTS 的知识蒸馏](03-ASR-TTS%20%E7%9A%84%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F.md)
- 4. [优化推理引擎（ONNX Runtime, TensorRT）](04-%E4%BC%98%E5%8C%96%E6%8E%A8%E7%90%86%E5%BC%95%E6%93%8E%EF%BC%88ONNX%20Runtime%2C%20TensorRT%EF%BC%89.md)
- 5. [流式ASR的部署考量](05-%E6%B5%81%E5%BC%8FASR%E7%9A%84%E9%83%A8%E7%BD%B2%E8%80%83%E9%87%8F.md)
- 6. [实时文本转语音（TTS）的部署考虑](06-%E5%AE%9E%E6%97%B6%E6%96%87%E6%9C%AC%E8%BD%AC%E8%AF%AD%E9%9F%B3%EF%BC%88TTS%EF%BC%89%E7%9A%84%E9%83%A8%E7%BD%B2%E8%80%83%E8%99%91.md)
- 7. [语音处理工具包（ESPnet, NeMo, Coqui）概述](07-%E8%AF%AD%E9%9F%B3%E5%A4%84%E7%90%86%E5%B7%A5%E5%85%B7%E5%8C%85%EF%BC%88ESPnet%2C%20NeMo%2C%20Coqui%EF%BC%89%E6%A6%82%E8%BF%B0.md)
- 8. [实践：优化语音模型](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BC%98%E5%8C%96%E8%AF%AD%E9%9F%B3%E6%A8%A1%E5%9E%8B.md)

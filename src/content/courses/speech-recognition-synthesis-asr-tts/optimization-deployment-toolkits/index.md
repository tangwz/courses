---
course: "speech-recognition-synthesis-asr-tts"
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-6-optimization-deployment-toolkits"
sourceId: 695
chapter: "optimization-deployment-toolkits"
title: "优化、部署与工具集"
order: 6
description: "语音模型优化技术，用于真实环境部署以及常用工具集的介绍。"
hasQuiz: false
---

构建高性能的ASR和TTS模型仅仅是整个过程的一部分。让这些模型在实际使用中高效运行，会带来一系列技术难题。本章侧重于解决模型开发与真实环境部署之间的衔接问题。

我们将考察模型优化方法，包括量化、剪枝和知识蒸馏，以减少它们的计算开销($FLOPs$)和内存需求。接下来我们转向部署策略，讨论像ONNX Runtime和TensorRT这样的优化运行时，并处理流式ASR和低延迟TTS的特定需求。此外，还将提供流行语音处理框架的概览，以指导您的实现工作。

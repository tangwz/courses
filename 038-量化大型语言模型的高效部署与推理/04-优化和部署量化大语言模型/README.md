# 第 4 章：优化和部署量化大语言模型

来源：[原章节](https://apxml.com/zh/courses/quantized-llm-deployment/chapter-4-optimizing-deploying-quantized-llms)

[返回课程目录](../README.md)

在应用量化技术并评估其影响后，下一步是让这些模型投入实际使用。本章讨论有效部署量化大语言模型 (LLM) 的实际方面。

您将了解与量化技术相辅相成的优化方法，例如专用内核的使用和高效的注意力机制。我们将指导您选择和使用为量化模型定制的合适部署框架，包括 Text Generation Inference (TGI)、vLLM、NVIDIA TensorRT-LLM 和 ONNX Runtime。本章还涉及硬件特定调优，尤其是针对 GPU，以及在生产环境中对这些优化模型进行容器化、扩展和监控的重要策略。结束时，您将能够选择正确的工具并为您的量化 LLM 建立有效的部署流程。

## 小节

- 1. [量化后的推理优化技术](01-%E9%87%8F%E5%8C%96%E5%90%8E%E7%9A%84%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF.md)
- 2. [选择合适的部署框架](02-%E9%80%89%E6%8B%A9%E5%90%88%E9%80%82%E7%9A%84%E9%83%A8%E7%BD%B2%E6%A1%86%E6%9E%B6.md)
- 3. [使用文本生成推理 (TGI) 进行部署](03-%E4%BD%BF%E7%94%A8%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90%E6%8E%A8%E7%90%86%20%28TGI%29%20%E8%BF%9B%E8%A1%8C%E9%83%A8%E7%BD%B2.md)
- 4. [借助 vLLM 实现高吞吐量推理](04-%E5%80%9F%E5%8A%A9%20vLLM%20%E5%AE%9E%E7%8E%B0%E9%AB%98%E5%90%9E%E5%90%90%E9%87%8F%E6%8E%A8%E7%90%86.md)
- 5. [使用 NVIDIA TensorRT-LLM 进行 GPU 优化](05-%E4%BD%BF%E7%94%A8%20NVIDIA%20TensorRT-LLM%20%E8%BF%9B%E8%A1%8C%20GPU%20%E4%BC%98%E5%8C%96.md)
- 6. [使用ONNX Runtime进行部署](06-%E4%BD%BF%E7%94%A8ONNX%20Runtime%E8%BF%9B%E8%A1%8C%E9%83%A8%E7%BD%B2.md)
- 7. [容器化与扩展策略](07-%E5%AE%B9%E5%99%A8%E5%8C%96%E4%B8%8E%E6%89%A9%E5%B1%95%E7%AD%96%E7%95%A5.md)
- 8. [监控已部署的量化模型](08-%E7%9B%91%E6%8E%A7%E5%B7%B2%E9%83%A8%E7%BD%B2%E7%9A%84%E9%87%8F%E5%8C%96%E6%A8%A1%E5%9E%8B.md)
- 9. [动手实操：通过推理服务器部署](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E6%93%8D%EF%BC%9A%E9%80%9A%E8%BF%87%E6%8E%A8%E7%90%86%E6%9C%8D%E5%8A%A1%E5%99%A8%E9%83%A8%E7%BD%B2.md)

# 第 5 章：保存、加载和部署模型

来源：[原章节](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-5-pytorch-model-persistence-deployment)

[返回课程目录](../README.md)

成功训练模型后，接下来的步骤是保存您的成果并准备将其用于应用程序。本章侧重于PyTorch框架中这些必要的训练后操作，同时与您可能在TensorFlow中了解到的做法进行比较。

您将学习：
*   管理模型持久化，使用PyTorch的`state_dict`，并与TensorFlow的SavedModel和HDF5格式进行对比。
*   区分保存整个模型与仅保存其参数（$权重$和$偏差$），并了解检查点策略。
*   使用TorchScript为模型部署做准备，它有助于PyTorch模型的序列化和优化。
*   了解ONNX（开放神经网络交换），以实现不同框架间的模型互操作性。
*   简要介绍TorchServe，用于PyTorch模型服务。

完成本章后，您将能够有效地保存、加载、检查和准备您的PyTorch模型以应对各种部署场景，并将您现有的TensorFlow知识调整到PyTorch生态系统中。

## 小节

- 1. [模型保存：TensorFlow 格式与 PyTorch \`state_dict\`](01-%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%EF%BC%9ATensorFlow%20%E6%A0%BC%E5%BC%8F%E4%B8%8E%20PyTorch%20%60state_dict%60.md)
- 2. [保存和加载完整模型与仅保存参数](02-%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E5%AE%8C%E6%95%B4%E6%A8%A1%E5%9E%8B%E4%B8%8E%E4%BB%85%E4%BF%9D%E5%AD%98%E5%8F%82%E6%95%B0.md)
- 3. [PyTorch 训练中的检查点策略](03-PyTorch%20%E8%AE%AD%E7%BB%83%E4%B8%AD%E7%9A%84%E6%A3%80%E6%9F%A5%E7%82%B9%E7%AD%96%E7%95%A5.md)
- 4. [在PyTorch中查看模型架构和权重](04-%E5%9C%A8PyTorch%E4%B8%AD%E6%9F%A5%E7%9C%8B%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E5%92%8C%E6%9D%83%E9%87%8D.md)
- 5. [TorchScript 在模型序列化方面的简介](05-TorchScript%20%E5%9C%A8%E6%A8%A1%E5%9E%8B%E5%BA%8F%E5%88%97%E5%8C%96%E6%96%B9%E9%9D%A2%E7%9A%84%E7%AE%80%E4%BB%8B.md)
- 6. [使用 ONNX 实现框架互操作性](06-%E4%BD%BF%E7%94%A8%20ONNX%20%E5%AE%9E%E7%8E%B0%E6%A1%86%E6%9E%B6%E4%BA%92%E6%93%8D%E4%BD%9C%E6%80%A7.md)
- 7. [TorchServe PyTorch 模型服务概览](07-TorchServe%20PyTorch%20%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E6%A6%82%E8%A7%88.md)
- 8. [动手实践：模型持久化与TorchScript基础](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%A8%A1%E5%9E%8B%E6%8C%81%E4%B9%85%E5%8C%96%E4%B8%8ETorchScript%E5%9F%BA%E7%A1%80.md)

章节测验：[在线测验](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-5-pytorch-model-persistence-deployment/quiz)

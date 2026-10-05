# PyTorch入门

来源：[PyTorch入门](https://apxml.com/zh/courses/getting-started-with-pytorch)

学习PyTorch的基本原理，用于构建和训练深度学习模型。本课程涵盖张量、自动微分、神经网络模块、数据加载及训练流程。适合具备Python和机器学习基本知识的开发者和工程师。

预计学时：18 小时

先修要求：Python和机器学习基本知识

## 课程目录

### 1. [PyTorch 入门与环境搭建](01-PyTorch%20%E5%85%A5%E9%97%A8%E4%B8%8E%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/README.md)

- 1. [PyTorch 是什么？](01-PyTorch%20%E5%85%A5%E9%97%A8%E4%B8%8E%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/01-PyTorch%20%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F.md)
- 2. [安装与环境配置](01-PyTorch%20%E5%85%A5%E9%97%A8%E4%B8%8E%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/02-%E5%AE%89%E8%A3%85%E4%B8%8E%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE.md)
- 3. [张量介绍](01-PyTorch%20%E5%85%A5%E9%97%A8%E4%B8%8E%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/03-%E5%BC%A0%E9%87%8F%E4%BB%8B%E7%BB%8D.md)
- 4. [创建张量](01-PyTorch%20%E5%85%A5%E9%97%A8%E4%B8%8E%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/04-%E5%88%9B%E5%BB%BA%E5%BC%A0%E9%87%8F.md)
- 5. [基本张量操作](01-PyTorch%20%E5%85%A5%E9%97%A8%E4%B8%8E%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/05-%E5%9F%BA%E6%9C%AC%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C.md)
- 6. [与 NumPy 的关联](01-PyTorch%20%E5%85%A5%E9%97%A8%E4%B8%8E%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/06-%E4%B8%8E%20NumPy%20%E7%9A%84%E5%85%B3%E8%81%94.md)
- 7. [动手实践：环境配置与张量基本操作](01-PyTorch%20%E5%85%A5%E9%97%A8%E4%B8%8E%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%E4%B8%8E%E5%BC%A0%E9%87%8F%E5%9F%BA%E6%9C%AC%E6%93%8D%E4%BD%9C.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-1-pytorch-fundamentals-setup/quiz)

### 2. [高级张量操作](02-%E9%AB%98%E7%BA%A7%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C/README.md)

- 1. [张量索引与切片](02-%E9%AB%98%E7%BA%A7%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C/01-%E5%BC%A0%E9%87%8F%E7%B4%A2%E5%BC%95%E4%B8%8E%E5%88%87%E7%89%87.md)
- 2. [张量的重塑与维度调整](02-%E9%AB%98%E7%BA%A7%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C/02-%E5%BC%A0%E9%87%8F%E7%9A%84%E9%87%8D%E5%A1%91%E4%B8%8E%E7%BB%B4%E5%BA%A6%E8%B0%83%E6%95%B4.md)
- 3. [张量的合并与分割](02-%E9%AB%98%E7%BA%A7%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C/03-%E5%BC%A0%E9%87%8F%E7%9A%84%E5%90%88%E5%B9%B6%E4%B8%8E%E5%88%86%E5%89%B2.md)
- 4. [理解广播机制](02-%E9%AB%98%E7%BA%A7%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C/04-%E7%90%86%E8%A7%A3%E5%B9%BF%E6%92%AD%E6%9C%BA%E5%88%B6.md)
- 5. [张量数据类型](02-%E9%AB%98%E7%BA%A7%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C/05-%E5%BC%A0%E9%87%8F%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B.md)
- 6. [CPU 与 GPU 张量](02-%E9%AB%98%E7%BA%A7%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C/06-CPU%20%E4%B8%8E%20GPU%20%E5%BC%A0%E9%87%8F.md)
- 7. [练习：张量操作技巧](02-%E9%AB%98%E7%BA%A7%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C/07-%E7%BB%83%E4%B9%A0%EF%BC%9A%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C%E6%8A%80%E5%B7%A7.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-2-advanced-tensor-manipulations/quiz)

### 3. [Autograd 自动微分](03-Autograd%20%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/README.md)

- 1. [自动微分的原理](03-Autograd%20%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/01-%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E7%9A%84%E5%8E%9F%E7%90%86.md)
- 2. [PyTorch 计算图](03-Autograd%20%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/02-PyTorch%20%E8%AE%A1%E7%AE%97%E5%9B%BE.md)
- 3. [张量与梯度计算 (\`requires_grad\`)](03-Autograd%20%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/03-%E5%BC%A0%E9%87%8F%E4%B8%8E%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97%20%28%60requires_grad%60%29.md)
- 4. [执行反向传播 (\`backward()\`)](03-Autograd%20%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/04-%E6%89%A7%E8%A1%8C%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%20%28%60backward%28%29%60%29.md)
- 5. [访问梯度（\`.grad\`）](03-Autograd%20%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/05-%E8%AE%BF%E9%97%AE%E6%A2%AF%E5%BA%A6%EF%BC%88%60.grad%60%EF%BC%89.md)
- 6. [禁用梯度追踪](03-Autograd%20%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/06-%E7%A6%81%E7%94%A8%E6%A2%AF%E5%BA%A6%E8%BF%BD%E8%B8%AA.md)
- 7. [梯度累积](03-Autograd%20%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/07-%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF.md)
- 8. [动手实践：Autograd 运用](03-Autograd%20%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9AAutograd%20%E8%BF%90%E7%94%A8.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-3-automatic-differentiation-autograd/quiz)

### 4. [使用 \`torch.nn\` 搭建模型](04-%E4%BD%BF%E7%94%A8%20%60torch.nn%60%20%E6%90%AD%E5%BB%BA%E6%A8%A1%E5%9E%8B/README.md)

- 1. [\`torch.nn.Module\` 基类](04-%E4%BD%BF%E7%94%A8%20%60torch.nn%60%20%E6%90%AD%E5%BB%BA%E6%A8%A1%E5%9E%8B/01-%60torch.nn.Module%60%20%E5%9F%BA%E7%B1%BB.md)
- 2. [定义自定义网络架构](04-%E4%BD%BF%E7%94%A8%20%60torch.nn%60%20%E6%90%AD%E5%BB%BA%E6%A8%A1%E5%9E%8B/02-%E5%AE%9A%E4%B9%89%E8%87%AA%E5%AE%9A%E4%B9%89%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84.md)
- 3. [常见层：线性、卷积、循环](04-%E4%BD%BF%E7%94%A8%20%60torch.nn%60%20%E6%90%AD%E5%BB%BA%E6%A8%A1%E5%9E%8B/03-%E5%B8%B8%E8%A7%81%E5%B1%82%EF%BC%9A%E7%BA%BF%E6%80%A7%E3%80%81%E5%8D%B7%E7%A7%AF%E3%80%81%E5%BE%AA%E7%8E%AF.md)
- 4. [激活函数 (ReLU, Sigmoid, Tanh)](04-%E4%BD%BF%E7%94%A8%20%60torch.nn%60%20%E6%90%AD%E5%BB%BA%E6%A8%A1%E5%9E%8B/04-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%20%28ReLU%2C%20Sigmoid%2C%20Tanh%29.md)
- 5. [简单模型的顺序容器](04-%E4%BD%BF%E7%94%A8%20%60torch.nn%60%20%E6%90%AD%E5%BB%BA%E6%A8%A1%E5%9E%8B/05-%E7%AE%80%E5%8D%95%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%A1%BA%E5%BA%8F%E5%AE%B9%E5%99%A8.md)
- 6. [损失函数 (\`torch.nn\` 损失)](04-%E4%BD%BF%E7%94%A8%20%60torch.nn%60%20%E6%90%AD%E5%BB%BA%E6%A8%A1%E5%9E%8B/06-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%20%28%60torch.nn%60%20%E6%8D%9F%E5%A4%B1%29.md)
- 7. [优化器 (\`torch.optim\`)](04-%E4%BD%BF%E7%94%A8%20%60torch.nn%60%20%E6%90%AD%E5%BB%BA%E6%A8%A1%E5%9E%8B/07-%E4%BC%98%E5%8C%96%E5%99%A8%20%28%60torch.optim%60%29.md)
- 8. [练习：构建一个简单网络](04-%E4%BD%BF%E7%94%A8%20%60torch.nn%60%20%E6%90%AD%E5%BB%BA%E6%A8%A1%E5%9E%8B/08-%E7%BB%83%E4%B9%A0%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%BD%91%E7%BB%9C.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-4-building-models-torch-nn/quiz)

### 5. [高效数据处理](05-%E9%AB%98%E6%95%88%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/README.md)

- 1. [对专用数据加载器的需求](05-%E9%AB%98%E6%95%88%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/01-%E5%AF%B9%E4%B8%93%E7%94%A8%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E5%99%A8%E7%9A%84%E9%9C%80%E6%B1%82.md)
- 2. [使用 \`torch.utils.data.Dataset\`](05-%E9%AB%98%E6%95%88%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/02-%E4%BD%BF%E7%94%A8%20%60torch.utils.data.Dataset%60.md)
- 3. [内置数据集（例如：TorchVision）](05-%E9%AB%98%E6%95%88%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/03-%E5%86%85%E7%BD%AE%E6%95%B0%E6%8D%AE%E9%9B%86%EF%BC%88%E4%BE%8B%E5%A6%82%EF%BC%9ATorchVision%EF%BC%89.md)
- 4. [数据变换 (\`torchvision.transforms\`)](05-%E9%AB%98%E6%95%88%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/04-%E6%95%B0%E6%8D%AE%E5%8F%98%E6%8D%A2%20%28%60torchvision.transforms%60%29.md)
- 5. [使用 \`torch.utils.data.DataLoader\`](05-%E9%AB%98%E6%95%88%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/05-%E4%BD%BF%E7%94%A8%20%60torch.utils.data.DataLoader%60.md)
- 6. [自定义DataLoader用法](05-%E9%AB%98%E6%95%88%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/06-%E8%87%AA%E5%AE%9A%E4%B9%89DataLoader%E7%94%A8%E6%B3%95.md)
- 7. [动手实践：构建数据处理流程](05-%E9%AB%98%E6%95%88%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E6%B5%81%E7%A8%8B.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/quiz)

### 6. [实现训练循环](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/README.md)

- 1. [训练循环的构成](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/01-%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF%E7%9A%84%E6%9E%84%E6%88%90.md)
- 2. [设置模型、损失函数和优化器](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/02-%E8%AE%BE%E7%BD%AE%E6%A8%A1%E5%9E%8B%E3%80%81%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%92%8C%E4%BC%98%E5%8C%96%E5%99%A8.md)
- 3. [使用 DataLoader 遍历数据](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/03-%E4%BD%BF%E7%94%A8%20DataLoader%20%E9%81%8D%E5%8E%86%E6%95%B0%E6%8D%AE.md)
- 4. [前向传播：获取预测结果](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/04-%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%EF%BC%9A%E8%8E%B7%E5%8F%96%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.md)
- 5. [计算损失](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/05-%E8%AE%A1%E7%AE%97%E6%8D%9F%E5%A4%B1.md)
- 6. [反向传播：计算梯度](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/06-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%EF%BC%9A%E8%AE%A1%E7%AE%97%E6%A2%AF%E5%BA%A6.md)
- 7. [使用优化器更新权重](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/07-%E4%BD%BF%E7%94%A8%E4%BC%98%E5%8C%96%E5%99%A8%E6%9B%B4%E6%96%B0%E6%9D%83%E9%87%8D.md)
- 8. [梯度清零](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/08-%E6%A2%AF%E5%BA%A6%E6%B8%85%E9%9B%B6.md)
- 9. [实现评估循环](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/09-%E5%AE%9E%E7%8E%B0%E8%AF%84%E4%BC%B0%E5%BE%AA%E7%8E%AF.md)
- 10. [保存和加载模型检查点](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/10-%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B%E6%A3%80%E6%9F%A5%E7%82%B9.md)
- 11. [动手实践：完整训练流程](06-%E5%AE%9E%E7%8E%B0%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF/11-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%8C%E6%95%B4%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/quiz)

### 7. [常用模型结构介绍](07-%E5%B8%B8%E7%94%A8%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E4%BB%8B%E7%BB%8D/README.md)

- 1. [卷积神经网络 (CNN) 概述](07-%E5%B8%B8%E7%94%A8%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E4%BB%8B%E7%BB%8D/01-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28CNN%29%20%E6%A6%82%E8%BF%B0.md)
- 2. [在PyTorch中构建一个简单的CNN](07-%E5%B8%B8%E7%94%A8%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E4%BB%8B%E7%BB%8D/02-%E5%9C%A8PyTorch%E4%B8%AD%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84CNN.md)
- 3. [理解CNN层的输入/输出形状](07-%E5%B8%B8%E7%94%A8%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E4%BB%8B%E7%BB%8D/03-%E7%90%86%E8%A7%A3CNN%E5%B1%82%E7%9A%84%E8%BE%93%E5%85%A5-%E8%BE%93%E5%87%BA%E5%BD%A2%E7%8A%B6.md)
- 4. [循环神经网络 (RNN) 概述](07-%E5%B8%B8%E7%94%A8%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E4%BB%8B%E7%BB%8D/04-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28RNN%29%20%E6%A6%82%E8%BF%B0.md)
- 5. [在PyTorch中构建一个简单的RNN](07-%E5%B8%B8%E7%94%A8%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E4%BB%8B%E7%BB%8D/05-%E5%9C%A8PyTorch%E4%B8%AD%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84RNN.md)
- 6. [循环神经网络（RNN）的序列数据输入处理](07-%E5%B8%B8%E7%94%A8%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E4%BB%8B%E7%BB%8D/06-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88RNN%EF%BC%89%E7%9A%84%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E8%BE%93%E5%85%A5%E5%A4%84%E7%90%86.md)
- 7. [LSTM 和 GRU 简要介绍](07-%E5%B8%B8%E7%94%A8%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E4%BB%8B%E7%BB%8D/07-LSTM%20%E5%92%8C%20GRU%20%E7%AE%80%E8%A6%81%E4%BB%8B%E7%BB%8D.md)
- 8. [实践：实现基本CNN和RNN](07-%E5%B8%B8%E7%94%A8%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E4%BB%8B%E7%BB%8D/08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%9F%BA%E6%9C%ACCNN%E5%92%8CRNN.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures/quiz)

### 8. [监控与调试模型](08-%E7%9B%91%E6%8E%A7%E4%B8%8E%E8%B0%83%E8%AF%95%E6%A8%A1%E5%9E%8B/README.md)

- 1. [PyTorch开发中常见错误](08-%E7%9B%91%E6%8E%A7%E4%B8%8E%E8%B0%83%E8%AF%95%E6%A8%A1%E5%9E%8B/01-PyTorch%E5%BC%80%E5%8F%91%E4%B8%AD%E5%B8%B8%E8%A7%81%E9%94%99%E8%AF%AF.md)
- 2. [调试张量形状不匹配](08-%E7%9B%91%E6%8E%A7%E4%B8%8E%E8%B0%83%E8%AF%95%E6%A8%A1%E5%9E%8B/02-%E8%B0%83%E8%AF%95%E5%BC%A0%E9%87%8F%E5%BD%A2%E7%8A%B6%E4%B8%8D%E5%8C%B9%E9%85%8D.md)
- 3. [检查设备放置 (CPU/GPU)](08-%E7%9B%91%E6%8E%A7%E4%B8%8E%E8%B0%83%E8%AF%95%E6%A8%A1%E5%9E%8B/03-%E6%A3%80%E6%9F%A5%E8%AE%BE%E5%A4%87%E6%94%BE%E7%BD%AE%20%28CPU-GPU%29.md)
- 4. [检查梯度问题（消失/爆炸）](08-%E7%9B%91%E6%8E%A7%E4%B8%8E%E8%B0%83%E8%AF%95%E6%A8%A1%E5%9E%8B/04-%E6%A3%80%E6%9F%A5%E6%A2%AF%E5%BA%A6%E9%97%AE%E9%A2%98%EF%BC%88%E6%B6%88%E5%A4%B1-%E7%88%86%E7%82%B8%EF%BC%89.md)
- 5. [使用 TensorBoard 可视化训练进度](08-%E7%9B%91%E6%8E%A7%E4%B8%8E%E8%B0%83%E8%AF%95%E6%A8%A1%E5%9E%8B/05-%E4%BD%BF%E7%94%A8%20TensorBoard%20%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E8%BF%9B%E5%BA%A6.md)
- 6. [训练/评估期间的指标记录](08-%E7%9B%91%E6%8E%A7%E4%B8%8E%E8%B0%83%E8%AF%95%E6%A8%A1%E5%9E%8B/06-%E8%AE%AD%E7%BB%83-%E8%AF%84%E4%BC%B0%E6%9C%9F%E9%97%B4%E7%9A%84%E6%8C%87%E6%A0%87%E8%AE%B0%E5%BD%95.md)
- 7. [在 PyTorch 中使用 Python 调试器 (pdb)](08-%E7%9B%91%E6%8E%A7%E4%B8%8E%E8%B0%83%E8%AF%95%E6%A8%A1%E5%9E%8B/07-%E5%9C%A8%20PyTorch%20%E4%B8%AD%E4%BD%BF%E7%94%A8%20Python%20%E8%B0%83%E8%AF%95%E5%99%A8%20%28pdb%29.md)
- 8. [实践：调试与可视化](08-%E7%9B%91%E6%8E%A7%E4%B8%8E%E8%B0%83%E8%AF%95%E6%A8%A1%E5%9E%8B/08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%B0%83%E8%AF%95%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- [章节测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-8-monitoring-debugging-models/quiz)

## 学习目标

- **张量操作**：创建、处理PyTorch张量，并对其执行操作。
- **自动微分**：理解并应用PyTorch的自动求导系统进行梯度计算。
- **神经网络构建**：使用`torch.nn`模块、层和激活函数构建神经网络。
- **数据加载**：使用PyTorch数据集和数据加载器高效准备和加载数据。
- **模型训练**：为深度学习模型实现完整的训练和评估循环。
- **模型保存与加载**：保存和加载已训练的PyTorch模型及检查点。
- **基本结构**：构建简单的卷积神经网络（CNN）和循环神经网络（RNN）。

## 课程项目

[课程项目](PROJECT.md)

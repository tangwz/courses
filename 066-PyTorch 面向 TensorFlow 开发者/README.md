# PyTorch 面向 TensorFlow 开发者

来源：[PyTorch 面向 TensorFlow 开发者](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers)

本课程旨在帮助已熟悉TensorFlow的开发者顺利转用PyTorch。它会将TensorFlow的思路与方法与PyTorch的对应部分进行对照，内容包括张量操作、使用`torch.nn`构建模型、通过`torch.utils.data`处理数据，以及编写自定义训练循环。您将获得将TensorFlow技能转化为PyTorch应用，从而有效地开发和训练机器学习 (machine learning)模型的实际经验。

预计学时：28 小时

先修要求：需具备TensorFlow使用经验。

## 课程目录

### 1. [衔接 TensorFlow 与 PyTorch：核心要点](01-%E8%A1%94%E6%8E%A5%20TensorFlow%20%E4%B8%8E%20PyTorch%EF%BC%9A%E6%A0%B8%E5%BF%83%E8%A6%81%E7%82%B9/README.md)

- 1. [从 TensorFlow 到 PyTorch：开发者指南](01-%E8%A1%94%E6%8E%A5%20TensorFlow%20%E4%B8%8E%20PyTorch%EF%BC%9A%E6%A0%B8%E5%BF%83%E8%A6%81%E7%82%B9/01-%E4%BB%8E%20TensorFlow%20%E5%88%B0%20PyTorch%EF%BC%9A%E5%BC%80%E5%8F%91%E8%80%85%E6%8C%87%E5%8D%97.md)
- 2. [TensorFlow 计算图与 PyTorch 动态计算的对比](01-%E8%A1%94%E6%8E%A5%20TensorFlow%20%E4%B8%8E%20PyTorch%EF%BC%9A%E6%A0%B8%E5%BF%83%E8%A6%81%E7%82%B9/02-TensorFlow%20%E8%AE%A1%E7%AE%97%E5%9B%BE%E4%B8%8E%20PyTorch%20%E5%8A%A8%E6%80%81%E8%AE%A1%E7%AE%97%E7%9A%84%E5%AF%B9%E6%AF%94.md)
- 3. [张量比较：tf.Tensor 与 torch.Tensor](01-%E8%A1%94%E6%8E%A5%20TensorFlow%20%E4%B8%8E%20PyTorch%EF%BC%9A%E6%A0%B8%E5%BF%83%E8%A6%81%E7%82%B9/03-%E5%BC%A0%E9%87%8F%E6%AF%94%E8%BE%83%EF%BC%9Atf.Tensor%20%E4%B8%8E%20torch.Tensor.md)
- 4. [基础张量运算：对比视角](01-%E8%A1%94%E6%8E%A5%20TensorFlow%20%E4%B8%8E%20PyTorch%EF%BC%9A%E6%A0%B8%E5%BF%83%E8%A6%81%E7%82%B9/04-%E5%9F%BA%E7%A1%80%E5%BC%A0%E9%87%8F%E8%BF%90%E7%AE%97%EF%BC%9A%E5%AF%B9%E6%AF%94%E8%A7%86%E8%A7%92.md)
- 5. [自动微分：GradientTape 与 Autograd 对比](01-%E8%A1%94%E6%8E%A5%20TensorFlow%20%E4%B8%8E%20PyTorch%EF%BC%9A%E6%A0%B8%E5%BF%83%E8%A6%81%E7%82%B9/05-%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%EF%BC%9AGradientTape%20%E4%B8%8E%20Autograd%20%E5%AF%B9%E6%AF%94.md)
- 6. [NumPy 在 PyTorch 和 TensorFlow 中的结合](01-%E8%A1%94%E6%8E%A5%20TensorFlow%20%E4%B8%8E%20PyTorch%EF%BC%9A%E6%A0%B8%E5%BF%83%E8%A6%81%E7%82%B9/06-NumPy%20%E5%9C%A8%20PyTorch%20%E5%92%8C%20TensorFlow%20%E4%B8%AD%E7%9A%84%E7%BB%93%E5%90%88.md)
- 7. [设备管理：CPU和GPU控制](01-%E8%A1%94%E6%8E%A5%20TensorFlow%20%E4%B8%8E%20PyTorch%EF%BC%9A%E6%A0%B8%E5%BF%83%E8%A6%81%E7%82%B9/07-%E8%AE%BE%E5%A4%87%E7%AE%A1%E7%90%86%EF%BC%9ACPU%E5%92%8CGPU%E6%8E%A7%E5%88%B6.md)
- 8. [动手实践：张量操作与自动求导](01-%E8%A1%94%E6%8E%A5%20TensorFlow%20%E4%B8%8E%20PyTorch%EF%BC%9A%E6%A0%B8%E5%BF%83%E8%A6%81%E7%82%B9/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C%E4%B8%8E%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC.md)
- [章节测验](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-1-pytorch-tensorflow-core-concepts/quiz)

### 2. [搭建神经网络：从 Keras 到 torch.nn](02-%E6%90%AD%E5%BB%BA%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%9A%E4%BB%8E%20Keras%20%E5%88%B0%20torch.nn/README.md)

- 1. [定义网络组件：Keras 层与 torch.nn.Module](02-%E6%90%AD%E5%BB%BA%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%9A%E4%BB%8E%20Keras%20%E5%88%B0%20torch.nn/01-%E5%AE%9A%E4%B9%89%E7%BD%91%E7%BB%9C%E7%BB%84%E4%BB%B6%EF%BC%9AKeras%20%E5%B1%82%E4%B8%8E%20torch.nn.Module.md)
- 2. [模型架构：Keras API 与 PyTorch 的 nn.Module](02-%E6%90%AD%E5%BB%BA%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%9A%E4%BB%8E%20Keras%20%E5%88%B0%20torch.nn/02-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%EF%BC%9AKeras%20API%20%E4%B8%8E%20PyTorch%20%E7%9A%84%20nn.Module.md)
- 3. [常见层类型：对比实现](02-%E6%90%AD%E5%BB%BA%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%9A%E4%BB%8E%20Keras%20%E5%88%B0%20torch.nn/03-%E5%B8%B8%E8%A7%81%E5%B1%82%E7%B1%BB%E5%9E%8B%EF%BC%9A%E5%AF%B9%E6%AF%94%E5%AE%9E%E7%8E%B0.md)
- 4. [激活函数：比较与分析](02-%E6%90%AD%E5%BB%BA%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%9A%E4%BB%8E%20Keras%20%E5%88%B0%20torch.nn/04-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%EF%BC%9A%E6%AF%94%E8%BE%83%E4%B8%8E%E5%88%86%E6%9E%90.md)
- 5. [PyTorch中的权重初始化方法](02-%E6%90%AD%E5%BB%BA%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%9A%E4%BB%8E%20Keras%20%E5%88%B0%20torch.nn/05-PyTorch%E4%B8%AD%E7%9A%84%E6%9D%83%E9%87%8D%E5%88%9D%E5%A7%8B%E5%8C%96%E6%96%B9%E6%B3%95.md)
- 6. [访问和修改模型参数及层](02-%E6%90%AD%E5%BB%BA%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%9A%E4%BB%8E%20Keras%20%E5%88%B0%20torch.nn/06-%E8%AE%BF%E9%97%AE%E5%92%8C%E4%BF%AE%E6%94%B9%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0%E5%8F%8A%E5%B1%82.md)
- 7. [动手实践：构建等效模型](02-%E6%90%AD%E5%BB%BA%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%9A%E4%BB%8E%20Keras%20%E5%88%B0%20torch.nn/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E7%AD%89%E6%95%88%E6%A8%A1%E5%9E%8B.md)
- [章节测验](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users/quiz)

### 3. [数据加载与预处理：从 tf.data 到 torch.utils.data](03-%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86%EF%BC%9A%E4%BB%8E%20tf.data%20%E5%88%B0%20torch.utils.data/README.md)

- 1. [数据结构：tf.data.Dataset 与 torch.utils.data.Dataset](03-%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86%EF%BC%9A%E4%BB%8E%20tf.data%20%E5%88%B0%20torch.utils.data/01-%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%EF%BC%9Atf.data.Dataset%20%E4%B8%8E%20torch.utils.data.Dataset.md)
- 2. [批处理与迭代：TensorFlow DataLoaders 与 PyTorch DataLoaders](03-%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86%EF%BC%9A%E4%BB%8E%20tf.data%20%E5%88%B0%20torch.utils.data/02-%E6%89%B9%E5%A4%84%E7%90%86%E4%B8%8E%E8%BF%AD%E4%BB%A3%EF%BC%9ATensorFlow%20DataLoaders%20%E4%B8%8E%20PyTorch%20DataLoaders.md)
- 3. [数据增强：TensorFlow 方法与 torchvision.transforms](03-%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86%EF%BC%9A%E4%BB%8E%20tf.data%20%E5%88%B0%20torch.utils.data/03-%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA%EF%BC%9ATensorFlow%20%E6%96%B9%E6%B3%95%E4%B8%8E%20torchvision.transforms.md)
- 4. [在 PyTorch 中实现自定义数据集](03-%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86%EF%BC%9A%E4%BB%8E%20tf.data%20%E5%88%B0%20torch.utils.data/04-%E5%9C%A8%20PyTorch%20%E4%B8%AD%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86.md)
- 5. [使用 PyTorch 变换进行数据预处理](03-%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86%EF%BC%9A%E4%BB%8E%20tf.data%20%E5%88%B0%20torch.utils.data/05-%E4%BD%BF%E7%94%A8%20PyTorch%20%E5%8F%98%E6%8D%A2%E8%BF%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.md)
- 6. [在 PyTorch 中构建高效数据管道](03-%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86%EF%BC%9A%E4%BB%8E%20tf.data%20%E5%88%B0%20torch.utils.data/06-%E5%9C%A8%20PyTorch%20%E4%B8%AD%E6%9E%84%E5%BB%BA%E9%AB%98%E6%95%88%E6%95%B0%E6%8D%AE%E7%AE%A1%E9%81%93.md)
- 7. [动手实践：创建自定义数据集和数据加载器](03-%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86%EF%BC%9A%E4%BB%8E%20tf.data%20%E5%88%B0%20torch.utils.data/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86%E5%92%8C%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E5%99%A8.md)
- [章节测验](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/quiz)

### 4. [训练与评估：Keras 方法到 PyTorch 循环的对应](04-%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%EF%BC%9AKeras%20%E6%96%B9%E6%B3%95%E5%88%B0%20PyTorch%20%E5%BE%AA%E7%8E%AF%E7%9A%84%E5%AF%B9%E5%BA%94/README.md)

- 1. [训练模式：TensorFlow 的 fit 方法与 PyTorch 训练循环](04-%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%EF%BC%9AKeras%20%E6%96%B9%E6%B3%95%E5%88%B0%20PyTorch%20%E5%BE%AA%E7%8E%AF%E7%9A%84%E5%AF%B9%E5%BA%94/01-%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%BC%8F%EF%BC%9ATensorFlow%20%E7%9A%84%20fit%20%E6%96%B9%E6%B3%95%E4%B8%8E%20PyTorch%20%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)
- 2. [TensorFlow 和 PyTorch 中的损失函数](04-%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%EF%BC%9AKeras%20%E6%96%B9%E6%B3%95%E5%88%B0%20PyTorch%20%E5%BE%AA%E7%8E%AF%E7%9A%84%E5%AF%B9%E5%BA%94/02-TensorFlow%20%E5%92%8C%20PyTorch%20%E4%B8%AD%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 3. [优化算法：TensorFlow 和 PyTorch 优化器](04-%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%EF%BC%9AKeras%20%E6%96%B9%E6%B3%95%E5%88%B0%20PyTorch%20%E5%BE%AA%E7%8E%AF%E7%9A%84%E5%AF%B9%E5%BA%94/03-%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%EF%BC%9ATensorFlow%20%E5%92%8C%20PyTorch%20%E4%BC%98%E5%8C%96%E5%99%A8.md)
- 4. [PyTorch 中的梯度计算与权重更新](04-%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%EF%BC%9AKeras%20%E6%96%B9%E6%B3%95%E5%88%B0%20PyTorch%20%E5%BE%AA%E7%8E%AF%E7%9A%84%E5%AF%B9%E5%BA%94/04-PyTorch%20%E4%B8%AD%E7%9A%84%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97%E4%B8%8E%E6%9D%83%E9%87%8D%E6%9B%B4%E6%96%B0.md)
- 5. [性能指标：Keras 指标与 PyTorch 对应的实现](04-%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%EF%BC%9AKeras%20%E6%96%B9%E6%B3%95%E5%88%B0%20PyTorch%20%E5%BE%AA%E7%8E%AF%E7%9A%84%E5%AF%B9%E5%BA%94/05-%E6%80%A7%E8%83%BD%E6%8C%87%E6%A0%87%EF%BC%9AKeras%20%E6%8C%87%E6%A0%87%E4%B8%8E%20PyTorch%20%E5%AF%B9%E5%BA%94%E7%9A%84%E5%AE%9E%E7%8E%B0.md)
- 6. [PyTorch 中的模型评估循环](04-%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%EF%BC%9AKeras%20%E6%96%B9%E6%B3%95%E5%88%B0%20PyTorch%20%E5%BE%AA%E7%8E%AF%E7%9A%84%E5%AF%B9%E5%BA%94/06-PyTorch%20%E4%B8%AD%E7%9A%84%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E5%BE%AA%E7%8E%AF.md)
- 7. [训练控制：Keras 回调与 PyTorch 自定义逻辑](04-%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%EF%BC%9AKeras%20%E6%96%B9%E6%B3%95%E5%88%B0%20PyTorch%20%E5%BE%AA%E7%8E%AF%E7%9A%84%E5%AF%B9%E5%BA%94/07-%E8%AE%AD%E7%BB%83%E6%8E%A7%E5%88%B6%EF%BC%9AKeras%20%E5%9B%9E%E8%B0%83%E4%B8%8E%20PyTorch%20%E8%87%AA%E5%AE%9A%E4%B9%89%E9%80%BB%E8%BE%91.md)
- 8. [动手实践：实现一个完整的训练和评估循环](04-%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%EF%BC%9AKeras%20%E6%96%B9%E6%B3%95%E5%88%B0%20PyTorch%20%E5%BE%AA%E7%8E%AF%E7%9A%84%E5%AF%B9%E5%BA%94/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AA%E5%AE%8C%E6%95%B4%E7%9A%84%E8%AE%AD%E7%BB%83%E5%92%8C%E8%AF%84%E4%BC%B0%E5%BE%AA%E7%8E%AF.md)
- [章节测验](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs/quiz)

### 5. [保存、加载和部署模型](05-%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E5%92%8C%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/README.md)

- 1. [模型保存：TensorFlow 格式与 PyTorch \`state_dict\`](05-%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E5%92%8C%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/01-%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%EF%BC%9ATensorFlow%20%E6%A0%BC%E5%BC%8F%E4%B8%8E%20PyTorch%20%60state_dict%60.md)
- 2. [保存和加载完整模型与仅保存参数](05-%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E5%92%8C%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/02-%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E5%AE%8C%E6%95%B4%E6%A8%A1%E5%9E%8B%E4%B8%8E%E4%BB%85%E4%BF%9D%E5%AD%98%E5%8F%82%E6%95%B0.md)
- 3. [PyTorch 训练中的检查点策略](05-%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E5%92%8C%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/03-PyTorch%20%E8%AE%AD%E7%BB%83%E4%B8%AD%E7%9A%84%E6%A3%80%E6%9F%A5%E7%82%B9%E7%AD%96%E7%95%A5.md)
- 4. [在PyTorch中查看模型架构和权重](05-%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E5%92%8C%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/04-%E5%9C%A8PyTorch%E4%B8%AD%E6%9F%A5%E7%9C%8B%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E5%92%8C%E6%9D%83%E9%87%8D.md)
- 5. [TorchScript 在模型序列化方面的简介](05-%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E5%92%8C%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/05-TorchScript%20%E5%9C%A8%E6%A8%A1%E5%9E%8B%E5%BA%8F%E5%88%97%E5%8C%96%E6%96%B9%E9%9D%A2%E7%9A%84%E7%AE%80%E4%BB%8B.md)
- 6. [使用 ONNX 实现框架互操作性](05-%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E5%92%8C%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/06-%E4%BD%BF%E7%94%A8%20ONNX%20%E5%AE%9E%E7%8E%B0%E6%A1%86%E6%9E%B6%E4%BA%92%E6%93%8D%E4%BD%9C%E6%80%A7.md)
- 7. [TorchServe PyTorch 模型服务概览](05-%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E5%92%8C%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/07-TorchServe%20PyTorch%20%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E6%A6%82%E8%A7%88.md)
- 8. [动手实践：模型持久化与TorchScript基础](05-%E4%BF%9D%E5%AD%98%E3%80%81%E5%8A%A0%E8%BD%BD%E5%92%8C%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%A8%A1%E5%9E%8B%E6%8C%81%E4%B9%85%E5%8C%96%E4%B8%8ETorchScript%E5%9F%BA%E7%A1%80.md)
- [章节测验](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-5-pytorch-model-persistence-deployment/quiz)

### 6. [面向 TensorFlow 用户的高级 PyTorch 功能](06-%E9%9D%A2%E5%90%91%20TensorFlow%20%E7%94%A8%E6%88%B7%E7%9A%84%E9%AB%98%E7%BA%A7%20PyTorch%20%E5%8A%9F%E8%83%BD/README.md)

- 1. [理解和使用 PyTorch 钩子](06-%E9%9D%A2%E5%90%91%20TensorFlow%20%E7%94%A8%E6%88%B7%E7%9A%84%E9%AB%98%E7%BA%A7%20PyTorch%20%E5%8A%9F%E8%83%BD/01-%E7%90%86%E8%A7%A3%E5%92%8C%E4%BD%BF%E7%94%A8%20PyTorch%20%E9%92%A9%E5%AD%90.md)
- 2. [分布式训练方法](06-%E9%9D%A2%E5%90%91%20TensorFlow%20%E7%94%A8%E6%88%B7%E7%9A%84%E9%AB%98%E7%BA%A7%20PyTorch%20%E5%8A%9F%E8%83%BD/02-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E6%96%B9%E6%B3%95.md)
- 3. [使用 PyTorch AMP 进行混合精度训练](06-%E9%9D%A2%E5%90%91%20TensorFlow%20%E7%94%A8%E6%88%B7%E7%9A%84%E9%AB%98%E7%BA%A7%20PyTorch%20%E5%8A%9F%E8%83%BD/03-%E4%BD%BF%E7%94%A8%20PyTorch%20AMP%20%E8%BF%9B%E8%A1%8C%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83.md)
- 4. [剖析 PyTorch 代码以查找性能瓶颈](06-%E9%9D%A2%E5%90%91%20TensorFlow%20%E7%94%A8%E6%88%B7%E7%9A%84%E9%AB%98%E7%BA%A7%20PyTorch%20%E5%8A%9F%E8%83%BD/04-%E5%89%96%E6%9E%90%20PyTorch%20%E4%BB%A3%E7%A0%81%E4%BB%A5%E6%9F%A5%E6%89%BE%E6%80%A7%E8%83%BD%E7%93%B6%E9%A2%88.md)
- 5. [PyTorch 生态系统概述：torchvision、torchaudio、torchtext](06-%E9%9D%A2%E5%90%91%20TensorFlow%20%E7%94%A8%E6%88%B7%E7%9A%84%E9%AB%98%E7%BA%A7%20PyTorch%20%E5%8A%9F%E8%83%BD/05-PyTorch%20%E7%94%9F%E6%80%81%E7%B3%BB%E7%BB%9F%E6%A6%82%E8%BF%B0%EF%BC%9Atorchvision%E3%80%81torchaudio%E3%80%81torchtext.md)
- 6. [PyTorch 模型调试策略](06-%E9%9D%A2%E5%90%91%20TensorFlow%20%E7%94%A8%E6%88%B7%E7%9A%84%E9%AB%98%E7%BA%A7%20PyTorch%20%E5%8A%9F%E8%83%BD/06-PyTorch%20%E6%A8%A1%E5%9E%8B%E8%B0%83%E8%AF%95%E7%AD%96%E7%95%A5.md)
- 7. [动手实践：实现 Hook 与模型性能分析](06-%E9%9D%A2%E5%90%91%20TensorFlow%20%E7%94%A8%E6%88%B7%E7%9A%84%E9%AB%98%E7%BA%A7%20PyTorch%20%E5%8A%9F%E8%83%BD/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%20Hook%20%E4%B8%8E%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90.md)
- [章节测验](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-6-advanced-pytorch-features-tf-users/quiz)

## 学习目标

- **转换TensorFlow思路**：将TensorFlow的核心组成部分（张量、图、层、优化器）与PyTorch的对应内容进行映射。
- **掌握PyTorch的动态计算模式**：理解并运用PyTorch的动态计算图，以实现灵活的模型构建。
- **开发PyTorch模型**：使用`torch.nn`、`torch.optim`和自定义训练循环来构建、训练和评估神经网络。
- **管理数据处理流程**：使用`torch.utils.data`和`torchvision.transforms`实现高效的数据加载和预处理流程。
- **处理模型持久化**：保存、加载PyTorch模型并进行部署准备，包括对TorchScript的介绍。
- **适配TensorFlow工作流程**：将现有的TensorFlow工作流程和思考方式迁移到PyTorch生态系统。

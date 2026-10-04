---
course: "basics-ml-deployment"
chapter: "preparing-model-deployment"
lesson: "saving-trained-models"
sourceId: 3955
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-2-preparing-model-deployment/saving-trained-models"
title: "保存训练好的模型"
description: "了解训练后保存模型状态的必要性。"
order: 1
plots: []
sourceHash: "685890f8cc3614fbc472aad0c09f462af523ba7ae837679692ccdd84c5f253c2"
sourceCorrections: []
---

一个训练好的机器学习 (machine learning)模型，例如一个能区分垃圾邮件的分类器，或是一个能预测房价的回归模型，能够从数据中学习模式。在模型训练期间，它会调整内部参数 (parameter)以进行准确的预测。这种学习状态，它代表着模型的智能，目前仅存在于您电脑的活动内存（RAM）中。

当您的训练脚本运行结束，或者您关闭开发环境时会发生什么？就像一个未保存的文档，所有这些学习到的信息都会消失。模型对象及其有价值的参数都不复存在了。如果您想在明天、或在另一个程序中再次使用该模型，或与同事分享，您将不得不从头开始重新训练它。重新训练可能非常耗时且计算成本高昂，特别是对于大型数据集或复杂模型。

这就是保存您训练好的模型变得非常重要的地方。我们需要一种方式来捕捉模型的状态——其结构、学习到的参数以及进行预测所需的一切，并将其持久化存储，通常是在文件中。可以将其想象为在特定时间点拍摄训练模型的快照。

通过保存模型，您可以达成几个重要的目标：

1. **持久性：** 模型可以在训练脚本或会话结束后依然存在。您可以在需要时随时重新加载它，而无需重新训练。
2. **可复用性：** 保存的模型可以加载到不同的脚本或应用程序中，用于对新的、未见过的数据进行预测。
3. **分享：** 您可以与他人分享保存的模型文件，让他们使用您训练好的模型。
4. **部署：** 这是本课程的主要议题。保存的模型是将您的模型在生产环境中可用的起点，例如将其整合到网络应用或更大的软件系统中。

将内存中的模型对象转换为适合保存到文件的格式的过程，通常称为**序列化**。反之，将文件重新加载到内存中以重建模型对象则称为**反序列化**。在接下来的部分中，我们将查看特定的Python工具，它们能让您高效地执行这种保存和加载操作。请记住，在训练之后，保存模型通常是您为部署做准备时迈出的第一个具体步骤。

## 参考资料

- [Save and load Keras models](https://www.tensorflow.org/guide/keras/save_and_serialize) — Neel Kovelamudi, Francois Chollet (2023)
  TensorFlow 官方指南，详细介绍使用 TensorFlow 原生 SavedModel 格式和 H5 格式保存 Keras 模型的方法，这对于深度学习模型部署很重要。
- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) — Matthew Inkawhich (2018)
  Publisher: PyTorch Foundation
  PyTorch 官方教程，解释了保存和加载 PyTorch 模型的方法，这对于深度学习模型的持久化至关重要。

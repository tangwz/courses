# 使用 DataLoader 遍历数据

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/iterating-through-data-dataloader)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`DataLoader` 对象设计用于管理训练数据，处理批处理、数据混洗以及可能的并行加载。它有助于在训练循环中高效地迭代数据。作为 Python 可迭代对象，`DataLoader` 能够简单地在每个训练周期中系统地为模型提供数据批次。

标准做法是使用 Python 的 `for` 循环。在每次迭代中，`DataLoader` 会生成一个数据批次，通常包含输入特征及其对应的目标标签。

```python
# 假设这些已定义并配置好：
# train_dataloader = DataLoader(your_dataset, batch_size=64, shuffle=True)
# model = YourNeuralNetwork()
# loss_fn = torch.nn.CrossEntropyLoss() # 损失函数示例
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 优化器示例
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device) # 确保模型在正确的设备上

num_epochs = 10 # 遍历数据集的次数示例

# 外部循环，用于处理训练周期
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    # 将模型设置为训练模式。
    # 这会启用诸如 dropout 和批归一化更新之类的功能。
    model.train()

    # 内部循环，用于处理一个训练周期内的批次
    # 遍历 DataLoader 提供的批次
    for batch_idx, data_batch in enumerate(train_dataloader):
        # 1. 解包批次
        # 结构取决于您的 Dataset 的 __getitem__ 方法。
        # 对于监督学习，通常是 (inputs, labels)。
        inputs, labels = data_batch

        # 2. 将数据移动到目标设备（GPU 或 CPU）
        # 这必须与模型所在的设备匹配。
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ---> 接下来的步骤（前向传播、损失计算、反向传播、优化） <---
        # ---> 使用“inputs”和“labels”在此处进行。             <---
        # （这些将在后续部分详细说明）

        # 后续逻辑放置的示例占位符：
        # predictions = model(inputs)
        # loss = loss_fn(predictions, labels)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # 可选：定期打印进度
        if batch_idx % 100 == 0:
            current_batch_size = len(inputs) # 获取当前批次的大小
            # 将 0.0 替换为实际计算出的损失值，用于记录
            current_loss = 0.0
            print(f"  Batch {batch_idx}: [{current_batch_size} samples] Current Loss: {current_loss:.4f}") # 示例日志

    # ---> 通常在此之后进行验证数据的评估循环 <---
    # （我们将在本章后面介绍评估循环）

print("训练完成！")
```

让我们来分析一下这个内部循环的主要部分：

1. **训练周期循环**：外部的 `for epoch in range(num_epochs):` 循环控制着我们遍历整个数据集的次数。
2. **设置训练模式**：`model.train()` 在每个训练周期开始时被调用。这很要紧，因为像 `torch.nn.Dropout` 或 `torch.nn.BatchNorm2d` 这样的层在训练期间（例如，应用 dropout、更新运行统计数据）与评估时有不同的行为。这个调用能确保它们处于正确的模式。
3. **遍历 DataLoader**：`for batch_idx, data_batch in enumerate(train_dataloader):` 是核心的迭代操作。`enumerate` 提供了一个批次计数器 (`batch_idx`)，而 `train_dataloader` 一次生成一个 `data_batch`。
4. **解包数据**：我们将 `data_batch` 解包成 `inputs` 和 `labels`。`DataLoader` 返回的结构与它所包装的 `Dataset` 的 `__getitem__` 方法返回的结构直接对应。对于典型的监督任务，这通常是包含特征和目标的元组或列表。
5. **将数据移动到设备**：`inputs.to(device)` 和 `labels.to(device)` 是不可或缺的步骤。神经网络 (neural network)计算，特别是模型的前向传播，要求模型的参数 (parameter)和输入数据位于*相同的*计算设备上（例如，都位于 CPU，或都位于特定的 GPU）。这一步将 `DataLoader` 获取的批次数据（通常在 CPU 内存中）移动到放置模型的设备上。未能进行此同步是运行时错误的常见原因。这还能确保，如果 `device` 设置为 CUDA 设备，计算可以从 GPU 加速中受益。

值得一提的是，如果你的 `DataLoader` 在初始化时设置了 `drop_last=False`（这是默认设置），一个训练周期中生成的最后一个批次可能包含比指定 `batch_size` 更少的样本。如果数据集中的样本总数不能被批次大小完全整除，就会发生这种情况。PyTorch 操作通常能很好地处理可变批次大小，但如果你进行任何假定固定批次大小的计算（例如对固定数量的损失求平均），请注意这一点。

当数据批次 (`inputs`, `labels`) 成功加载到目标设备后，你现在已为训练迭代循环中的主要计算步骤做好了充分准备：

- 将 `inputs` 送入模型以获得预测（前向传播）。
- 使用损失函数 (loss function)将预测与 `labels` 进行比较（计算损失）。
- 根据损失计算梯度（反向传播 (backpropagation)）。
- 使用优化器更新模型的权重 (weight)（更新权重）。

这些步骤针对每个批次重复执行，构成了模型训练过程的核心，也是后续章节的重点。

## 参考资料

- [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) — PyTorch Core Team (2025)
  Publisher: PyTorch Foundation
  详细说明 DataLoader 类、其构造函数参数以及它如何进行批处理、洗牌和数据加载的官方文档。
- [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  Dataset 基类的官方文档，说明其在数据访问抽象和与 DataLoader 集成方面的作用。
- [PyTorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) — PyTorch Core Team (2025)
  Publisher: PyTorch Foundation
  一个介绍性教程，展示了完整的训练循环，包括使用 DataLoader 进行数据批次迭代。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习理论基础的综合教科书，包括优化、小批量梯度下降和训练周期。

---

[上一节](02-%E8%AE%BE%E7%BD%AE%E6%A8%A1%E5%9E%8B%E3%80%81%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%92%8C%E4%BC%98%E5%8C%96%E5%99%A8.md) · [下一节](04-%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%EF%BC%9A%E8%8E%B7%E5%8F%96%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.md)

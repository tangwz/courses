# PyTorch 模型调试策略

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-6-advanced-pytorch-features-tf-users/debugging-pytorch-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

即使经过最周密的规划，缺陷仍是软件开发中无法避免的一部分，机器学习 (machine learning)代码也不例外。PyTorch 的动态特性常因其灵活性而备受推崇，但有时与 TensorFlow 基于图的执行相比，会带来独特的调试挑战，特别是对于习惯 TensorFlow 1.x 的用户。然而，这种动态性也意味着您可以更直接地使用标准 Python 调试工具。这里概述了您在开发 PyTorch 模型时可能遇到的常见问题，并提供了识别和解决这些问题的实用方法和工具。

### PyTorch 常见问题及其解决方案

理解常见的错误模式可以显著加快调试过程。以下是一些经常遇到的问题：

#### 1. 张量形状不匹配

这或许是任何张量库中最常见的运行时错误。

- **症状：** 您会看到 `RuntimeError` 消息，如“mat1 and mat2 shapes cannot be multiplied”、“size mismatch, m1: [A x B], m2: [C x D]”，或与广播相关的错误。
- **常见原因：**
  - `nn.Linear` 层中 `in_features` 或 `out_features` 不正确。
  - 卷积层中的通道数不匹配。
  - 在将卷积层输出传递到线性层之前，忘记将其展平（例如，未使用 `tensor.view(batch_size, -1)`）。
  - 数据预处理中的错误影响了张量维度。
- **调试策略：**
  - **随意打印形状：** 在您怀疑的操作之前和之后插入 `print(f"Tensor X shape: {x.shape}")`。
  - **交互式调试：** 使用 `import pdb; pdb.set_trace()` 或您的 IDE 调试器暂停执行，并在不同位置检查 `tensor.shape` 属性。
  - **小样本例子：** 如果您的输入数据复杂，尝试通过有问题的层或模型段落一个预期形状的虚拟张量：

    ```python
    # 假设 model 是您的 nn.Module 实例
    # 并且您怀疑在某个特定输入大小附近存在问题
    dummy_input = torch.randn(1, 3, 224, 224) # 图像模型的例子
    try:
        output = model(dummy_input)
        print(f"Dummy output shape: {output.shape}")
    except Exception as e:
        print(f"虚拟输入出现错误: {e}")
    ```

#### 2. 损失或梯度中出现 NaN 或 Inf 值

数值不稳定性会迅速使训练偏离轨道。

- **症状：** 损失变为 `NaN`（非数字）或 `inf`（无穷大），或模型权重 (weight)变为 `NaN`/`inf`。梯度可能爆炸到非常大的值或消失到零。
- **常见原因：**
  - **学习率过高：** 这可能导致权重剧烈波动和发散。
  - **不稳定的运算：** 对零或负数取对数（例如，`torch.log(x)`，其中 `x <= 0`），除以非常小的数或零。
  - **梯度爆炸：** 在 RNN 或深度网络中常见。梯度在反向传播 (backpropagation)过程中呈指数级增长。
  - **梯度消失：** 梯度变得非常小，实际停止了早期层的学习。在深度网络中使用某些激活函数 (activation function)（如 Sigmoid）时常见。
  - **输入数据问题：** 未归一化 (normalization)的数据或极端异常值。
- **调试策略：**
  - **`torch.autograd.set_detect_anomaly(True)`：** 这是一个有效的工具。将其上下文 (context)管理器包裹您的训练步骤（前向和反向传播）：

    ```python
    # 在训练脚本的开头
    # torch.autograd.set_detect_anomaly(True) # 适用于旧版 PyTorch

    # 在训练循环中
    # for data, target in train_loader:
    #     optimizer.zero_grad()
    #     with torch.autograd.detect_anomaly(): # 首选方式
    #         output = model(data)
    #         loss = criterion(output, target)
    #         loss.backward()
    #     optimizer.step()
    ```

    这会打印一个堆栈跟踪，指出在反向传播中首次产生 `NaN` 或 `inf` 的操作。它会增加开销，因此仅在调试时使用。
  - **检查问题函数的输入：** 如果 `log(x)` 导致问题，请在 `log` 操作之前打印 `x`。
  - **降低学习率：** 如果您怀疑存在不稳定性，这是简单的第一步。
  - **梯度裁剪：** 限制梯度的幅度。

    ```python
    # 在 loss.backward() 之后和 optimizer.step() 之前
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    ```
  - **权重初始化：** 确保合理的权重初始化。
  - **归一化：** 应用批归一化或其他归一化层。确保输入数据正确归一化。
  - **检查梯度：** 在 `loss.backward()` 之后检查所有参数 (parameter)的 `param.grad`。

    ```python
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"参数: {name}, 梯度均值: {param.grad.mean()}, 梯度标准差: {param.grad.std()}")
        else:
            print(f"参数: {name}, 梯度为 None")
    ```

#### 3. 模型未学习

损失停滞不前，或准确率保持在随机水平。

- **症状：** 训练和验证损失不降低，或验证指标未显示改善。
- **常见原因：**
  - **优化器逻辑错误：** 忘记 `optimizer.step()` 或 `optimizer.zero_grad()`。错误放置 `optimizer.zero_grad()`（例如，在某些特殊情况下放置在 `loss.backward()` 之前，尽管通常它位于循环开始处）。
  - **梯度未流动：**
    - 参数的 `requires_grad=False`，而它们本应可训练。
    - 在应该作为计算图一部分用于梯度计算的张量上使用 `.detach()`。
    - 不可微分或未被 autograd 跟踪的操作。
  - **学习率过低/过高：** 过低则学习速度慢到难以察觉。过高则可能超调或发散（导致 NaN 或不稳定）。
  - **损失函数 (loss function)不正确：** 对回归任务使用分类损失，反之亦然。输出激活不匹配（例如，`nn.BCELoss` 没有 Sigmoid）。
  - **数据问题：**
    - 标签不正确或混洗不当。
    - 数据归一化缺失或不正确。
    - 数据不足或数据不能代表问题。
  - **模型结构中的错误：** 层连接方式或自定义模块实现中的逻辑缺陷。
  - **`forward` 传播中的隐性错误：** `forward` 传播可能在没有 Python 错误的情况下执行，但产生数学上不正确的结果。
- **调试策略：**
  - **单批次过拟合 (overfitting)：** 这是一项基本的健全性检查。如果您的模型无法在极小批次数据（例如，2-4 个样本）上达到接近零的损失，那么存在一个重要问题。

    ```python
    # 获取单个批次
    data_iter = iter(train_loader)
    sample_data, sample_targets = next(data_iter)

    # 在此单个批次上训练多个 epoch
    for epoch in range(100): # 或更多
        optimizer.zero_grad()
        output = model(sample_data)
        loss = criterion(output, sample_targets)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"周期 {epoch}, 损失: {loss.item()}")
    ```
  - **验证 `optimizer.step()` 和 `optimizer.zero_grad()` 的位置。**
  - **检查 `param.requires_grad`：**

    ```python
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    ```
  - **检查梯度：** 确保您期望训练的层的梯度不为 `None`。微小但非零的梯度是可以的；所有权重都为 `None` 或零的梯度则有问题。
  - **从简单开始：** 在尝试复杂的自定义模型之前，使用标准的、经过充分测试的结构（例如，用于图像的 ResNet18）。
  - **检查数据加载器和预处理：** 手动检查 DataLoader 中的几个批次。图像是否正确归一化？标签是否与输入对齐 (alignment)？
  - **验证损失函数和输出激活：** 对于 `nn.CrossEntropyLoss`，模型输出应为原始 logits。对于 `nn.BCELoss`，输出应通过 `torch.sigmoid`，并且目标应为 0 或 1。

#### 4. CUDA 错误

这些错误表示 GPU 使用方面的问题。

- **症状：** `RuntimeError: CUDA out of memory`、`RuntimeError: CUDA error: an illegal memory access was encountered`，或设备断言错误。
- **常见原因：**
  - **批次大小过大：** 这是“内存不足”错误最常见的原因。
  - **模型过大：** 模型参数和中间激活不适合在 GPU 上。
  - **张量累积：** 在多次迭代中将张量存储在 Python 列表或字典中而未分离它们或将其移至 CPU。例如，累积损失值：

    ```python
    # 不良：累积计算图
    # all_losses_gpu = []
    # for ...:
    #   loss = criterion(output, target)
    #   all_losses_gpu.append(loss) # loss 仍在 GPU 上并带有图

    # 良好：仅存储 Python 浮点数
    all_losses_scalar = []
    for i in range(num_iterations): # 伪代码
        # ... 前向传播 ...
        loss = criterion(output, target)
        # ... 反向传播, 优化器步骤 ...
        all_losses_scalar.append(loss.item()) # .item() 获取 Python 数字，并分离
    ```
  - **GPU 内存碎片化不足：** 有时，即使理论上有足够的可用内存，它也可能过于碎片化而无法分配大型张量。
- **调试策略：**
  - **减少批次大小：** 解决 OOM 错误最简单的第一步。
  - **`torch.cuda.empty_cache()`：** 这可以释放 GPU 上未使用的缓存内存。然而，它不释放活跃使用的内存。请谨慎使用，因为它会减慢执行速度。
  - **将张量移至 CPU：** 如果张量不再需要在 GPU 上进行计算，请移动它们：`tensor_cpu = tensor_gpu.cpu()`。
  - **`del tensor_gpu`：** 如果张量很大且不再需要，请显式删除它们。结合 `torch.cuda.empty_cache()` 有时会有帮助。
  - **梯度累积：** 顺序处理小批次并在调用 `optimizer.step()` 之前累积梯度。这允许在不增加每步内存使用的情况下获得更大的有效批次大小。
  - **混合精度训练 (AMP)：** 可以显著减少内存使用。（本章前面已介绍）。
  - **模型并行或检查点：** 对于非常大的模型，这些是更高级的方法。`torch.utils.checkpoint` 权衡计算量和内存。

### PyTorch 调试工具包

PyTorch 的 Pythonic 特性提供了对一系列有用调试工具的访问。

#### Print 语句

朴素的 `print()` 语句通常是检查张量形状、数据类型、设备或中间值最快的方法。
`print(f"层 X 输出: {output.shape}, {output.dtype}, {output.device}, 均值: {output.mean().item()}")`

#### `torch.autograd.set_detect_anomaly(True)`

如前所述，这个上下文 (context)管理器对于追踪反向传播 (backpropagation)中 `NaN` 或 `Inf` 错误的来源非常有价值。

```python
with torch.autograd.detect_anomaly():
    loss.backward()
```

#### Python 调试器 (`pdb` 或 IDE 集成调试器)

PyTorch 代码就是 Python 代码。您可以在任何地方插入 `import pdb; pdb.set_trace()` 进入 Python 调试器。在那里，您可以检查变量、单步执行代码并运行命令。大多数 IDE（如 VS Code、PyCharm）都提供与 PyTorch 配合使用的复杂图形调试器。

#### 检查钩子

钩子允许您将函数附加到 `nn.Module` 实例或张量上，以检查（或修改）激活和梯度，而无需更改模块的 `forward` 方法。

- **前向钩子 (`register_forward_hook`)：** 在模块的 `forward` 传播后运行。用于检查激活或特征图很有用。

  ```python
  def print_activation_shape(module, input_tensor, output_tensor):
      print(f"模块: {module.__class__.__name__}")
      print(f"  输入形状: {input_tensor[0].shape if isinstance(input_tensor, tuple) else input_tensor.shape}")
      print(f"  输出形状: {output_tensor.shape}")

  # 在特定层上注册钩子
  model.conv1.register_forward_hook(print_activation_shape)
  ```
- **张量钩子 (`register_hook`)：** 当计算张量的梯度时运行。用于检查特定张量的梯度很有用。

  ```python
  def print_grad(grad):
      print(f"梯度形状: {grad.shape}, 均值: {grad.mean()}")

  # 假设 'x' 是一个需要梯度的输入张量
  # x = torch.randn(10, 20, requires_grad=True)
  # y = model(x) 
  # y.register_hook(print_grad) # 在 y 上注册钩子，将打印 d(loss)/dy 的梯度
  # loss.backward()
  ```

  注意：`module.register_full_backward_hook` 和 `module.register_backward_hook` 提供了访问模块输入和输出梯度的途径。

#### 验证梯度流动

在 `loss.backward()` 后，检查具有 `requires_grad=True` 的参数 (parameter)和中间张量的 `.grad` 属性。

```python
# 在 loss.backward() 之后
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"警告: {name} 没有梯度")
    elif torch.all(param.grad == 0):
        print(f"警告: {name} 的梯度全部为零")
```

如果 `.grad` 为 `None`，表示该参数不属于导致损失的计算图，或者其 `requires_grad` 为 `False`。如果它全是零，则可能表明存在诸如 ReLU 死亡或激活饱和等问题。

#### 可视化计算图

对于复杂的模型，理解计算图会有帮助。像 `torchviz` 这样的库可以生成图的图示。

```python
# pip install torchviz
from torchviz import make_dot

# 假设 'loss' 是您图的最终输出
# 并且 model 是您的 nn.Module
graph_viz = make_dot(loss, params=dict(model.named_parameters()))
graph_viz.render("computation_graph", format="png") # 保存为 PNG 文件
```

这有助于识别图中分离的部分或不正确的连接。下面是这样一种图的简化表示：

> 数据流经一个简单模型，从输入到损失，以及反向传播过程中随后的梯度传播。

### 调试的系统方法

有条理的方法通常比随机试错更有效：

1. **稳定复现：** 隔离出能够可靠触发缺陷的最小、最简单的代码和数据片段。这可能意味着固定随机种子（`torch.manual_seed(0)`）。
2. **简化：**
   - **模型：** 从一个非常简单的模型开始（例如，一个线性层）。如果可行，逐渐增加复杂度。
   - **数据：** 使用一小部分固定的数据子集。单个样本或一小批次。
   - **训练循环：** 删除增强、回调或任何非必要组件。
3. **假设并测试：** 对原因形成假设。设计一个小型实验或添加特定的 print 语句/调试器断点来验证它。
4. **隔离问题组件：** 如果您的模型有多个部分，尽可能单独测试它们。
5. **仔细阅读错误信息：** PyTorch 的错误信息和堆栈跟踪通常包含有关错误发生位置和原因的精确信息。不要只看最后一行。
6. **版本控制 (Git)：** 当代码处于可用状态时提交。如果您引入了一个缺陷，可以使用 `git diff` 查看更改或恢复到已知良好状态。
7. **检查环境：** 确保您的 PyTorch 版本、CUDA 版本和其他依赖项兼容并符合您的预期。

### 连接您的 TensorFlow 调试经验

如果您来自 TensorFlow，这里有一些需要注意的地方：

- **动态图与静态图：** 在 TensorFlow 1.x 中，许多错误发生在图构建期间。在 PyTorch（以及使用 Eager Execution 的 TensorFlow 2.x）中，错误通常发生在运行时，使其感觉更像标准 Python 缺陷。这意味着 Python 调试器非常有效。
- **没有 `tf.Session`：** 您不需要会话来运行操作。张量会立即计算。这使得用于张量值的 `print()` 语句按预期工作，而无需 `tf.print` 或在会话中计算。
- **调试自定义训练循环：** TensorFlow Keras 的 `model.fit()` 抽象了许多细节。当您在 PyTorch 中编写自定义训练循环时，您拥有更多控制权，但也承担更多责任。常见错误包括忘记 `optimizer.zero_grad()`、`loss.backward()` 或 `optimizer.step()`。然而，详细的控制意味着您可以在循环的任何位置插入调试逻辑。
- **设备放置：** 尽管 TensorFlow 通常更隐式地处理设备放置，但在 PyTorch 中您使用 `.to(device)` 更为明确。与张量位于不同设备相关的错误（“Expected all tensors to be on the same device”）很常见，但通常通过确保操作的所有输入都在目标设备上即可轻松修复。

调试是一项习得的技能，它将系统性排查与经验直觉相结合。通过理解常见的 PyTorch 问题并使用其调试工具，您可以高效地识别和解决模型中的问题。请记住，当您遇到特别棘手的缺陷时，PyTorch 论坛和文档是很好的资源。

## 参考资料

- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本关于 PyTorch 的实用指南，涵盖模型开发和常见问题，包括调试策略。
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供构建和调试稳健机器学习系统的广阔视角，为 PyTorch 模型开发提供相关见解。
- [On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.pdf) — Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, and Yoshua Bengio (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning (ICML); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 28; Pages: 1310-1318; DOI: [10.5555/3042817.3042823](https://doi.org/10.5555/3042817.3042823)
  一篇基础性论文，讨论训练深度网络的挑战，特别是梯度消失和梯度爆炸问题，以及梯度裁剪等解决方法。
- [pytorchviz GitHub Repository](https://github.com/szagoruyko/pytorchviz) — Sergey Zagoruyko (2024)
  提供用于可视化 PyTorch 计算图的工具，有助于理解张量流并在调试时识别分离的组件。

---

[上一节](05-PyTorch%20%E7%94%9F%E6%80%81%E7%B3%BB%E7%BB%9F%E6%A6%82%E8%BF%B0%EF%BC%9Atorchvision%E3%80%81torchaudio%E3%80%81torchtext.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%20Hook%20%E4%B8%8E%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90.md)

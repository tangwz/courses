# 实践：将错误缓解应用于VQC

来源：[原文](https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-7-hardware-error-mitigation-qml/practice-applying-error-mitigation-vqc)

[返回章节目录](README.md) · [返回课程目录](../README.md)

我们将理论付诸实践。我们讨论了噪声如何困扰近期量子设备，并介绍了错误缓解技术。现在，你将运用此类技术之一，即零噪声外推（ZNE），以提升在模拟带噪声后端上运行的变分量子分类器（VQC）的性能。

本次动手练习假定你熟悉以下内容：

- 使用Qiskit或Pennylane等库构建和训练基础VQC。
- 定义量子线路，包括特征映射和参数 (parameter)化变分量子线路（ansätze）。
- 模拟量子线路，理想情况下具备添加噪声模型的能力。

### 任务：使用带噪声的VQC分类Moons数据集

我们将使用scikit-learn的`make_moons`数据集处理一个标准二分类问题。我们的目标是训练一个VQC来区分这两个类别。

1. **数据集：** 生成一个小型`make_moons`数据集。
2. **VQC架构：** 我们将使用一个简单的VQC结构：
   - **特征映射：** 将二维输入数据编码到，例如，2个量子位的状态中。一种常见选择是角度编码，可能与一些纠缠相结合。例如，Qiskit中的`ZZFeatureMap`或Pennylane中的角度嵌入 (embedding)层。
   - **变分量子线路（Ansatz）：** 一个短深度参数 (parameter)化线路，包含旋转门和CNOT门。例如，Qiskit的`RealAmplitudes`或Pennylane的`StronglyEntanglingLayers`，重复几次。
   - **测量：** 测量可观测量的期望值，如 $\langle Z_0 \rangle$（第一个量子位上的泡利Z），以产生分类输出。
   - **成本函数：** 一个标准二元交叉熵或均方误差损失，用于比较VQC输出与真实标签。
   - **优化器：** 经典优化器，例如Adam或SPSA。

### 步骤1：建立无噪声基线

首先，在*理想*（无噪声）量子模拟器上实现并训练你的VQC。这为你所选架构和超参数 (parameter) (hyperparameter)下的最佳性能提供了一个参考点。

```python
# 伪代码示例（使用Pennylane风格的语法）
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# --- 数据 ---
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
y_one_hot = np.array([[1, 0] if label == 0 else [0, 1] for label in y]) # 针对某些损失函数的格式
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

# --- VQC设置 ---
n_qubits = 2
dev_ideal = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    # 例子：简单角度嵌入
    qml.AngleEmbedding(x, wires=range(n_qubits))

def ansatz(params):
    # 例子：基础变分层
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))

@qml.qnode(dev_ideal)
def circuit(params, x):
    feature_map(x)
    ansatz(params)
    # 测量量子位0上泡利Z的期望值
    return qml.expval(qml.PauliZ(0))

def cost_fn(params, X_batch, y_batch):
    predictions = [circuit(params, x) for x in X_batch]
    # 将期望值（-1到1）映射到概率（0到1）
    probs = (np.stack(predictions) + 1) / 2
    # 简单的MSE损失用于演示（其他损失如交叉熵也很常见）
    # 注意：确保y_batch的格式与你选择的损失函数正确匹配 [样本数] 或 [样本数, 类别数]
    # 假设y_batch代表类别1的概率（需要根据实际标签格式进行调整）
    loss = np.mean((probs - y_batch[:, 1])**2) 
    return loss

# --- 训练（理想情况）---
# 初始化参数（例如，使用StronglyEntanglingLayers.shape）
init_params = np.random.uniform(0, 2 * np.pi, (1, n_qubits, 3)) # 示例形状

opt = qml.AdamOptimizer(stepsize=0.1)
params = init_params
print("Training VQC (Ideal)...")
for iteration in range(50):
    # 批处理示例（实践中使用合适的批处理）
    params, cost = opt.step_and_cost(lambda p: cost_fn(p, X_train, y_train), params)
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Cost: {cost:.4f}")

# --- 评估（理想情况）---
test_predictions_ideal = [(circuit(params, x) + 1) / 2 for x in X_test]
# 将概率转换为类别标签（0或1）
test_labels_ideal = (np.array(test_predictions_ideal) > 0.5).astype(int) 
y_test_labels = np.argmax(y_test, axis=1) # 将one-hot编码转换回标签
accuracy_ideal = np.mean(test_labels_ideal == y_test_labels)
print(f"\nIdeal VQC Test Accuracy: {accuracy_ideal:.4f}")
```

你应观察到无噪声模拟器上训练收敛性和测试准确率都相当不错。记录此准确率作为你的基线。

### 步骤2：引入硬件噪声

现在，我们来模拟噪声的影响。我们将使用一个基本噪声模型，例如去极化噪声，在每个CNOT门之后应用。许多量子库提供了构建和应用噪声模型到模拟器的工具。

```python
# 示例（Qiskit Aer风格的噪声模型定义）
from qiskit_aer.noise import NoiseModel, depolarizing_error

# 定义去极化错误概率
error_prob = 0.01 # 1%错误概率

# 为双量子位门创建去极化错误通道
depol_error = depolarizing_error(error_prob, 2)

# 构建噪声模型：将错误应用于CNOT门
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depol_error, ["cx"]) # 应用于'cx'门

print(f"\nNoise Model:\n{noise_model}")

# --- 使用噪声模拟 ---
# 重新运行VQC训练和评估，但配置模拟器
# 以使用'noise_model'。具体机制取决于库。
# 示例（Pennylane与Qiskit插件）：
# dev_noisy = qml.device("qiskit.aer", wires=n_qubits, noise_model=noise_model)
# 重新定义qnode以使用dev_noisy：
# @qml.qnode(dev_noisy)
# def circuit_noisy(params, x): ... (相同的线路定义)
# 使用circuit_noisy和相同的成本函数/优化器重新训练。

# --- 训练（有噪声情况）---
# 重置参数或使用理想参数作为起点
params_noisy = init_params 
opt_noisy = qml.AdamOptimizer(stepsize=0.1) # 如有需要，重置优化器状态
print("\nTraining VQC (Noisy)...")
# 假设circuit_noisy和dev_noisy已如上定义
# 你需要重新定义cost_fn以使用circuit_noisy
# def cost_fn_noisy(params, X_batch, y_batch): ... 使用circuit_noisy ...
# for iteration in range(50):
#    params_noisy, cost_noisy = opt_noisy.step_and_cost(lambda p: cost_fn_noisy(p, X_train, y_train), params_noisy)
#    if iteration % 10 == 0:
#        print(f"Iteration {iteration}, Cost: {cost_noisy:.4f}")

# --- 评估（有噪声情况）---
# 使用训练好的params_noisy和有噪声的线路
# test_predictions_noisy = [(circuit_noisy(params_noisy, x) + 1) / 2 for x in X_test]
# test_labels_noisy = (np.array(test_predictions_noisy) > 0.5).astype(int)
# accuracy_noisy = np.mean(test_labels_noisy == y_test_labels)
# print(f"\nNoisy VQC Test Accuracy: {accuracy_noisy:.4f}") 

# 预期有噪声结果的占位符 - 请替换为实际模拟结果
accuracy_noisy = accuracy_ideal * 0.7 # 模拟显著的性能下降
print(f"\n(Simulated) Noisy VQC Test Accuracy: {accuracy_noisy:.4f}")
```

正如预期的那样，噪声显著降低了性能。优化器可能难以收敛，最终测试准确率可能会远低于理想基线。

### 步骤3：应用零噪声外推（ZNE）

ZNE通过有意*增加*线路执行中的噪声，在多个噪声水平下测量得到的期望值，然后外推回零噪声限制来发挥作用。

一种放大与特定门（如CNOT）相关的噪声的常见方法是使用*酉折叠*。对于一个门 $U$，我们可以将其替换为 $U (U^\dagger U)^k$。如果 $U$ 有噪声，每次应用都会引入更多噪声。噪声比例因子大致为 $c = 2k + 1$。

- **比例因子 $c=1$：** 原始有噪声线路（$k=0$）。
- **比例因子 $c=3$：** 每个 $U$ 被替换为 $U U^\dagger U$（$k=1$）。
- **比例因子 $c=5$：** 每个 $U$ 被替换为 $U U^\dagger U U^\dagger U$（$k=2$）。

我们对多个比例因子（例如，$c=1, 3, 5$）运行线路，以得到有噪声的期望值 $E_1, E_3, E_5$。然后，我们对这些点 $(c, E_c)$ 拟合一个模型（例如，线性、二次、指数），并外推以找到 $c=0$ 时的估计值。

```python
# ZNE的实现
# Mitiq (https://mitiq.readthedocs.io/) 等库可以自动化此过程。
# 这是手动实现的思路：

def get_noisy_expectation(params, x, noise_model, scale_factor):
    """
在缩放噪声下运行线路的函数。
    实践中，这包括修改线路（门折叠）
    或指示模拟器/硬件后端缩放噪声。
    """
    # 这高度依赖于框架和后端。
    # 占位符：假设我们可以直接模拟缩放后的噪声
    # 实际上，你会使用门折叠或特定的后端选项。
    
    # 示例：通过调整error_prob模拟缩放（简单近似）
    effective_prob = noise_model.get_error("cx").probabilities[1] * scale_factor 
    scaled_error = depolarizing_error(min(effective_prob, 1.0), 2) # 将概率限制在1
    scaled_noise_model = NoiseModel()
    scaled_noise_model.add_all_qubit_quantum_error(scaled_error, ["cx"])
    
    # 假设存在一个接受噪声模型的函数`run_noisy_circuit`
    noisy_expval = run_noisy_circuit(params, x, scaled_noise_model) 
    return noisy_expval

def zne_expectation(params, x, noise_model, scale_factors=[1, 3, 5]):
    """计算期望值的ZNE估计。"""
    noisy_values = []
    for c in scale_factors:
        # 此函数需要根据实际库的功能进行修改
        expval = get_noisy_expectation(params, x, noise_model, scale_factor=c) 
        noisy_values.append(expval)
    
    # 执行外推（例如，使用numpy进行线性拟合）
    coeffs = np.polyfit(scale_factors, noisy_values, 1) # 线性拟合：ax + b
    zero_noise_estimate = coeffs[1] # c=0时的值（截距）
    
    # 更高级：Richardson外推法、指数拟合等。
    # zero_noise_estimate = extrapolate(scale_factors, noisy_values, method='richardson') 
    
    return zero_noise_estimate

# --- 修改VQC以使用ZNE ---
# 核心改变是将期望值计算封装在
# 成本函数和最终预测步骤中。

# 示例：重新定义成本函数以使用zne_expectation
# def cost_fn_zne(params, X_batch, y_batch):
#    predictions = [zne_expectation(params, x, noise_model) for x in X_batch]
#    probs = (np.stack(predictions) + 1) / 2
#    loss = np.mean((probs - y_batch[:, 1])**2) 
#    return loss

# --- 训练（有噪声+ZNE）---
# 重置参数和优化器
params_zne = init_params
opt_zne = qml.AdamOptimizer(stepsize=0.1) # 使用相同的优化器设置
print("\nTraining VQC (Noisy + ZNE)...")
# for iteration in range(50): # 注意：ZNE会显著增加运行时间！
#    params_zne, cost_zne = opt_zne.step_and_cost(lambda p: cost_fn_zne(p, X_train, y_train), params_zne)
#    if iteration % 10 == 0:
#        print(f"Iteration {iteration}, Cost: {cost_zne:.4f}")

# --- 评估（有噪声+ZNE）---
# 使用训练好的params_zne和zne_expectation函数
# test_predictions_zne = [(zne_expectation(params_zne, x, noise_model) + 1) / 2 for x in X_test]
# test_labels_zne = (np.array(test_predictions_zne) > 0.5).astype(int)
# accuracy_zne = np.mean(test_labels_zne == y_test_labels)
# print(f"\nNoisy VQC + ZNE Test Accuracy: {accuracy_zne:.4f}")

# 预期ZNE结果的占位符 - 请替换为实际模拟结果
accuracy_zne = accuracy_ideal * 0.9 # 模拟显著的恢复
print(f"\n(Simulated) Noisy VQC + ZNE Test Accuracy: {accuracy_zne:.4f}")
```

> **注意：** Mitiq等库提供了简化量子执行函数封装并自动处理折叠和外推的功能（例如，`mitiq.zne.execute_with_zne`）。在实践中强烈推荐使用此类库。

### 步骤4：比较结果

现在，比较测试准确率：

1. **理想（无噪声）：** 理论上的最佳性能。
2. **有噪声：** 因噪声模型导致的性能下降。
3. **有噪声+ZNE：** 应用错误缓解后的性能。

你应该观察到ZNE缓解后的准确率显著优于纯粹有噪声的准确率，理想情况下可恢复大部分因噪声损失的性能。可视化此比较。



[交互图表：VQC测试准确率比较](https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-7-hardware-error-mitigation-qml/practice-applying-error-mitigation-vqc#plot-ms9q8t)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "理想（无噪声）",
        "有噪声",
        "有噪声 + ZNE"
      ],
      "y": [
        0.92,
        0.64,
        0.85
      ],
      "type": "bar",
      "marker": {
        "color": [
          "#74c0fc",
          "#ff8787",
          "#69db7c"
        ]
      }
    }
  ],
  "layout": {
    "title": "VQC测试准确率比较",
    "yaxis": {
      "title": "准确率",
      "range": [
        0,
        1
      ]
    },
    "xaxis": {
      "title": "模拟条件"
    },
    "width": 600,
    "height": 400
  }
}
```

</details>



> VQC测试准确率在理想条件、模拟噪声和使用ZNE进行噪声缓解下的比较。使用了占位符值；请替换为你的实际结果。

### 讨论与考量

- **开销：** ZNE对于*每个*期望值计算都需要多次线路执行（每个比例因子一次）。这会显著增加计算成本，高于标准有噪声模拟。
- **外推选择：** 外推的准确率取决于所选模型（线性、指数等）与噪声尺度和期望值之间关系的拟合程度。这可能与噪声模型相关。
- **噪声缩放方法：** 门折叠是一种常见技术，但存在其他方法。其有效性取决于能否准确放大主要噪声源。
- **局限性：** ZNE在噪声水平不过高且噪声放大方法与实际硬件噪声良好对应时效果最佳。它不能完美地纠正所有错误。

此练习展示了应用错误缓解技术的实际流程。尽管ZNE会增加开销，但它可以是一种有效工具，改善从带噪声量子硬件获得的结果，使我们更接近发挥QML算法的潜力。请记住，ZNE只是一种工具；概率错误消除（PEC）、动态解耦和纠错（长期来看）等技术提供了替代或补充方法。

## 参考资料

- [Error mitigation on near-term quantum devices with a variational quantum eigensolver](https://www.nature.com/articles/s41586-019-1040-7) — Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Chow, J. M., and Gambetta, J. M. (2019)
  Journal: Nature; Volume: 567; Pages: 491-495; DOI: [10.1038/s41586-019-1040-7](https://doi.org/10.1038/s41586-019-1040-7)
  一篇开创性论文，介绍并实验验证了零噪声外推（ZNE）作为噪声量子设备的纠错技术。
- [Mitiq Documentation: Zero-Noise Extrapolation](https://mitiq.readthedocs.io/en/stable/guide/zne.html) — Unitary Fund (2024)
  Mitiq 库实现零噪声外推的官方指南，提供实用细节和不同外推方法的代码示例。
- [Variational Quantum Algorithms](https://www.nature.com/articles/s42254-021-00348-9) — Cerezo, M., Arrasmith, A., Gover, R., Kolawole, S., Mills, F., Rudolph, L., and Coles, P. J. (2021)
  Journal: Nature Reviews Physics; Volume: 3; Pages: 625-644; DOI: [10.1038/s42254-021-00348-9](https://doi.org/10.1038/s42254-021-00348-9)
  对变分量子算法（VQA）的全面评论，包括变分量子分类器（VQC），涵盖其理论基础、架构和应用。
- [Quantum machine learning: a review and perspective](https://doi.org/10.1063/5.0091990) — K. Bharti, A. Cervera-Lierta, T. Kyriienko, R. Albarran-Arriagada, and A. Pérez-Salinas (2022)
  Journal: Applied Physics Reviews; Publisher: AIP Publishing; Volume: 9; Pages: 041315; DOI: [10.1063/5.0091990](https://doi.org/10.1063/5.0091990)
  一篇关于量子机器学习的广泛综述，讨论了算法、硬件考量、噪声效应以及量子机器学习中纠错技术的现状。

---

[上一节](06-%E5%9C%A8%E5%AE%9E%E9%99%85%E9%87%8F%E5%AD%90%E8%AE%BE%E5%A4%87%E4%B8%8A%E8%BF%9B%E8%A1%8CQML%E7%AE%97%E6%B3%95%E5%9F%BA%E5%87%86%E6%B5%8B%E8%AF%95.md)

---
course: "advanced-tensorflow"
chapter: "model-deployment-optimization"
lesson: "practice-deploying-tf-serving"
sourceId: 2940
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-6-model-deployment-optimization/practice-deploying-tf-serving"
title: "动手实践：使用 TF Serving 部署模型"
description: "保存训练好的模型，并使用 TensorFlow Serving 在本地部署，然后发送推理请求。"
order: 8
plots: []
sourceHash: "8dd46d4d91efdf51a50c64b4dbf75d5446d2d42e50c715ef8ef13bfd6635f88c"
sourceCorrections: []
---

机器学习 (machine learning)模型训练完成后，将其部署以提供预测服务是首要要求。TensorFlow Serving 是一个专为此目的设计的高性能系统。本次实践将逐步演示如何保存训练好的 TensorFlow 模型，并使用在 Docker 容器内运行的 TensorFlow Serving 在本地对其进行部署。随后，我们将使用其 REST API 与已部署模型进行交互。

本实践假设您的系统上已安装并运行 Docker，并且拥有一个已安装 TensorFlow 的可用 Python 环境。

### 1. 准备并保存模型

首先，我们创建一个非常简单的 Keras 模型。对于本例，我们不需要复杂的架构；主要看部署操作。

```python
import tensorflow as tf
import numpy as np
import os
import shutil # 用于目录清理

# 定义一个简单模型
def create_simple_model():
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Dense(3, activation='softmax') # 示例输出形状
    ])
    # 编译是为了保存签名，但我们此处不进行训练
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

model = create_simple_model()

# 生成一些模拟数据，仅用于展示输入/输出形状
print("模型概览：")
model.summary()
dummy_input = np.random.rand(1, 4)
print(f"\n模拟输入形状：{dummy_input.shape}")
dummy_output = model.predict(dummy_input)
print(f"模拟输出形状：{dummy_output.shape}")
print(f"模拟输出：{dummy_output}")

# --- 保存模型 ---
# 定义模型保存路径。
# TF Serving 要求模型位于带版本号的目录中。
model_dir = 'simple_model'
version = 1
export_path = os.path.join(model_dir, str(version))

# 如果目录已存在则进行清理
if os.path.isdir(export_path):
    print(f"删除现有目录：{export_path}")
    shutil.rmtree(model_dir)

print(f"\n保存模型到：{export_path}")

# 以 TensorFlow 的 SavedModel 格式保存模型
# 此格式包含模型架构、权重和服务签名。
tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True, # 可选，但推荐
    signatures=None, # Keras 会自动生成一个默认的“serving_default”签名
    options=None
)

print(f"\n模型保存成功！")
print(f"目录结构在 {model_dir} 下：")
for root, dirs, files in os.walk(model_dir):
    indent = ' ' * 4 * (root.count(os.sep) - model_dir.count(os.sep))
    print(f"{indent}{os.path.basename(root)}/")
    file_indent = ' ' * 4 * (root.count(os.sep) - model_dir.count(os.sep) + 1)
    for f in files:
        print(f"{file_indent}{f}")
```

执行此代码会创建一个名为 `simple_model` 的目录，其中包含一个子目录 `1`（即版本号）。在 `1` 中，您会找到定义计算图的 `saved_model.pb` 文件，以及诸如 `variables`（包含模型权重 (weight)）和可能的 `assets` 等子目录。这种 `SavedModel` 格式正是 TensorFlow Serving 所需的。

### 2. 通过 Docker 启动 TensorFlow Serving

接下来，我们将使用 Docker 运行官方 TensorFlow Serving 镜像，并将其指向我们已保存的模型。请打开您的终端或命令提示符。

首先，请确保您拥有最新的服务镜像：

```bash
docker pull tensorflow/serving
```

接下来，运行容器。您需要将 `/path/to/your/simple_model` 替换为您刚刚在主机上创建的 `simple_model` 目录的**绝对路径**。

```bash
# 请确保您位于包含 'simple_model' 文件夹的目录中
# 或者提供 'simple_model' 的完整绝对路径

# 使用绝对路径的示例（请替换为您的实际路径）：
# 在 Linux/macOS 上：
# docker run -p 8501:8501 --mount type=bind,source=/home/user/my_projects/advanced_tf/simple_model,target=/models/my_simple_classifier -e MODEL_NAME=my_simple_classifier -t tensorflow/serving

# 在 Windows（使用 PowerShell）上：
# docker run -p 8501:8501 --mount type=bind,source=C:\Users\YourUser\MyProjects\advanced_tf\simple_model,target=/models/my_simple_classifier -e MODEL_NAME=my_simple_classifier -t tensorflow/serving

# 使用当前目录的示例（从 'simple_model' 的父目录运行）：
# 在 Linux/macOS 上：
docker run -p 8501:8501 --mount type=bind,source=$(pwd)/simple_model,target=/models/my_simple_classifier -e MODEL_NAME=my_simple_classifier -t tensorflow/serving &

# 在 Windows（使用 PowerShell）上：
# docker run -p 8501:8501 --mount type=bind,source=${PWD}/simple_model,target=/models/my_simple_classifier -e MODEL_NAME=my_simple_classifier -t tensorflow/serving
```

让我们解释一下此命令：

- `docker run`：启动一个新容器。
- `-p 8501:8501`：将您主机上的 8501 端口映射到容器内的 8501 端口。这是 TF Serving REST API 的默认端口。
- `--mount type=bind,source=<host_path>,target=/models/my_simple_classifier`：这是重要部分。它使得您主机上的 `simple_model` 目录（`source`）在容器内的路径 `/models/my_simple_classifier`（`target`）下可用。TensorFlow Serving 默认配置为在容器内的 `/models` 目录中查找模型。我们在服务环境中将模型命名为 `my_simple_classifier`。
- `-e MODEL_NAME=my_simple_classifier`：此环境变量明确告知 TensorFlow Serving 从 `/models` 目录加载哪个模型。该名称必须与 `--mount` 选项中 `target` 部分使用的子目录名称匹配。
- `-t tensorflow/serving`：指定要使用的 Docker 镜像。
- `&` (Linux/macOS)：在后台运行容器（可选）。

运行此命令后，Docker 将拉取镜像（如果您本地没有）并启动容器。您应该会看到 TensorFlow Serving 的日志输出，表明它正在查找模型，并希望成功加载 `my_simple_classifier`。请查找类似以下内容的行：

```
... Successfully loaded servable version {name: my_simple_classifier version: 1} ...
... Running gRPC ModelServer at 0.0.0.0:8500 ...
... Exporting HTTP/REST API at 0.0.0.0:8501 ...
```

如果您看到错误，请仔细检查 `--mount` 命令中的 `source` 路径；它必须是包含版本子目录（`1`）的正确绝对路径。

### 3. 通过 REST API 发送推理 (inference)请求

TF Serving 运行且模型加载后，我们现在可以发送预测请求了。我们将使用 Python 的 `requests` 库与 REST 端点进行交互。

创建一个新 Python 脚本或使用 Jupyter notebook：

```python
import requests
import json
import numpy as np

# 准备与模型输入形状 (1, 4) 兼容的示例输入数据
# 注意：JSON 序列化需要是列表的列表形式
input_data = np.random.rand(2, 4).tolist() # 创建 2 个样本

# REST API 端点格式为：
# http://<host>:<port>/v1/models/<model_name>[:predict]
# 或 http://<host>:<port>/v1/models/<model_name>/versions/<version>[:predict]
url = 'http://localhost:8501/v1/models/my_simple_classifier:predict'
# url = 'http://localhost:8501/v1/models/my_simple_classifier/versions/1:predict' # 同样适用

# 请求载荷必须是 JSON 对象。
# 对于默认的“serving_default”签名（以及许多常见情况），
# 键是“instances”，值是输入示例的列表。
data = json.dumps({"instances": input_data})

# 设置内容类型头
headers = {"content-type": "application/json"}

# 发送 POST 请求
try:
    response = requests.post(url, data=data, headers=headers)
    response.raise_for_status() # 对不良状态码（4xx 或 5xx）抛出异常

    # 解析 JSON 响应
    predictions = response.json()['predictions']

    print("请求 URL：", url)
    print("发送的输入数据（第一个样本）：", input_data[0])
    print("\n响应状态码：", response.status_code)
    print("收到的预测（第一个样本）：", predictions[0])
    print(f"\n收到 {len(predictions)} 个预测。")

except requests.exceptions.RequestException as e:
    print(f"请求出错：{e}")
    # 如果在 Docker 中运行，请检查容器日志：docker logs <container_id>
except KeyError:
    print("错误：响应中未找到 'predictions'。")
    print("响应内容：", response.text) # 打印原始响应以便调试
```

当您运行此脚本时，它会构建一个 JSON 载荷，将您的输入数据放在 `"instances"` 下。它通过 HTTP POST 请求将此载荷发送到 TensorFlow Serving 端点。如果成功，TF Serving 会使用加载的模型处理输入并返回预测结果，然后将其打印出来。

输出应类似于以下内容（具体的预测值会有所不同）：

```
请求 URL： http://localhost:8501/v1/models/my_simple_classifier:predict
发送的输入数据（第一个样本）： [0.123, 0.456, 0.789, 0.987]

响应状态码： 200
收到的预测（第一个样本）： [0.25, 0.45, 0.3] # softmax 的示例概率

收到 2 个预测。
```

### 4. 清理

完成实验后，您可以停止 TensorFlow Serving 容器。使用 `docker ps` 查找其 ID，然后停止它：

```bash
# 查找容器 ID
docker ps

# 停止容器（将 <container_id> 替换为实际 ID）
docker stop <container_id>

# 可选：删除容器
docker rm <container_id>
```

如果不再需要，您还可以删除之前创建的 `simple_model` 目录。

本实践展示了使用 TF Serving 部署 TensorFlow 模型的基本流程。您以 `SavedModel` 格式保存了模型，使用 Docker 启动服务容器并挂载模型目录，并成功向模型的 REST 端点查询预测结果。这为将更复杂的模型部署到生产环境奠定了基础。您可以通过查看 TF Serving 的配置选项、批量请求以获得更高吞吐量 (throughput)，或使用 gRPC 接口以实现潜在的更低延迟通信来扩展此功能。

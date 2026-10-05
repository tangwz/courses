# 使用 REST 和 gRPC 部署 TensorFlow Serving 模型

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-6-model-deployment-optimization/deploying-tf-serving-rest-grpc)

[返回章节目录](README.md) · [返回课程目录](../README.md)

对于已打包成 SavedModel 的模型，使其可用于推理 (inference)是主要考虑事项。TensorFlow Serving 是一个专用的高性能系统，专门为在生产环境中部署机器学习 (machine learning)模型而设计。它接收您的 SavedModels 并通过网络端点公开它们，允许客户端应用程序轻松发送推理请求并接收预测。

TensorFlow Serving 主要提供两种通信协议：REST（表征性状态传输）和 gRPC（gRPC 远程过程调用）。了解如何与这些协议交互对于将部署的模型集成到大型应用程序中非常重要。

### 设置 TensorFlow Serving (简要说明)

在与 API 交互之前，您需要让 TensorFlow Serving 与您的模型一起运行。虽然详细设置超出本节的讨论范围，但一种常见的方法是使用 Docker。假设您已安装 Docker 并且您的 SavedModel 位于 `/path/to/your/model/1`（其中 `1` 表示版本号），您可以像这样启动服务器：

```bash
docker run -p 8501:8501 --mount type=bind,source=/path/to/your/model/,target=/models/my_model \
-e MODEL_NAME=my_model -t tensorflow/serving
```

这个命令：

- 运行官方的 TensorFlow Serving Docker 镜像。
- 将主机上的 8501 端口映射到容器内的 8501 端口（REST 的默认端口）。8500 端口通常用于 gRPC。
- 将本地模型目录挂载到容器内的 `/models/my_model`。
- 将环境变量 `MODEL_NAME` 设置为 `my_model`，这是您在 API 调用中引用模型的方式。

> **注意：** 对于 gRPC 访问，您通常也需要映射 8500 端口（`-p 8500:8500`）。确保您的模型目录结构遵循 TF Serving 约定（`model_name/version_number/saved_model.pb` 和 assets/variables）。

### 通过 REST 进行交互

RESTful API 通常是与 TF Serving 开始交互的更简单方式，它利用标准 HTTP 方法和 JSON 进行数据交换。它与各种编程语言和工具广泛兼容。

#### REST 端点结构

TensorFlow Serving 通过可预测的 URL 公开模型。最常见的预测端点是：

`POST http://<host>:<port>/v1/models/<model_name>[:predict]`

或者，如果您想针对特定版本：

`POST http://<host>:<port>/v1/models/<model_name>/versions/<version_number>[:predict]`

- `<host>`: TF Serving 运行的主机名或 IP 地址（例如 `localhost`）。
- `<port>`: 为 REST API 映射的端口（默认为 8501）。
- `<model_name>`: 分配给您模型的名称（例如 Docker 命令中的 `my_model`）。
- `<version_number>`: 可选的查询特定版本。如果省略，TF Serving 通常使用可用的最新版本。

#### 请求格式

POST 请求的主体必须是一个 JSON 对象。结构取决于您模型的签名，但一种常见格式使用 `instances` 键。与 `instances` 关联的值是一个列表，其中每个元素表示一个输入实例（或一批实例，具体取决于您的模型如何期望输入）。

假设您的模型期望一个名为 `input_features` 的输入张量，其形状为 `(batch_size, 784)`（例如扁平化的 MNIST 图像）：

```json
{
  "instances": [
    [0.0, 0.1, ..., 0.9],
    [0.5, 0.2, ..., 0.0],
    ...
  ]
}
```

如果您的模型在其签名中定义了多个命名输入（例如 `image_input` 和 `metadata_input`），您可以使用不同的格式，为每个实例提供一个字典：

```json
{
  "instances": [
    {
      "image_input": [[0.0, ...], [0.1, ...]],
      "metadata_input": [1.0, 2.5]
    },
    {
      "image_input": [[0.5, ...], [0.3, ...]],
      "metadata_input": [0.5, 1.2]
    }
  ]
}
```

您可以使用 `saved_model_cli` 工具检查您的 SavedModel 的签名，以确定预期的输入名称和格式：

```bash
saved_model_cli show --dir /path/to/your/model/1 --tag_set serve --signature_def serving_default
```

#### Python REST 示例

下面是您如何使用 Python 的 `requests` 库发送请求：

```python
import requests
import json
import numpy as np

# 假设 TF Serving 在 localhost:8501 上运行，模型名为 'my_model'
url = "http://localhost:8501/v1/models/my_model:predict"

# 示例：创建两个模拟输入实例（例如，扁平化的 28x28 图像）
# 替换为您实际的数据预处理
input_data = np.random.rand(2, 784).tolist()

# 构建请求载荷
request_payload = json.dumps({"instances": input_data})

# 设置请求头
headers = {"content-type": "application/json"}

try:
    # 发送 POST 请求
    response = requests.post(url, data=request_payload, headers=headers)
    response.raise_for_status() # 对于错误的HTTP状态码（4xx或5xx）抛出异常

    # 解析 JSON 响应
    predictions = response.json()['predictions']

    print("收到预测结果：")
    print(predictions)

except requests.exceptions.RequestException as e:
    print(f"请求出错：{e}")
except json.JSONDecodeError:
    print(f"解码 JSON 响应出错：{response.text}")
except KeyError:
    print(f"响应中未找到 'predictions'：{response.json()}")
```

#### REST 优点和缺点

- **优点：** 简单，易于调试（使用 `curl` 或 Web 浏览器等工具），广泛兼容。
- **缺点：** 由于 HTTP 开销和 JSON 序列化/反序列化导致延迟较高，对于大型载荷或高吞吐量 (throughput)场景效率较低。

### 通过 gRPC 进行交互

gRPC 是一个由 Google 开发的现代高性能 RPC 框架。它使用 HTTP/2 作为传输协议，并使用 Protocol Buffers (Protobufs) 作为接口定义语言和消息交换格式。gRPC 通常比 REST 提供更低的延迟和更高的吞吐量 (throughput)，使其适用于对性能要求高的应用程序。

#### gRPC 设置

要使用 gRPC，您通常需要：

1. TensorFlow Serving API 定义（`.proto` 文件）。
2. 适用于您客户端语言的 gRPC 库。
3. 从 `.proto` 文件生成的客户端代码（存根）。

幸运的是，`tensorflow-serving-api` Python 包提供了必要的库和预生成的存根。

```bash
pip install grpcio tensorflow-serving-api
```

#### Python gRPC 示例

通过 gRPC 交互包括创建 Protobuf 请求对象，建立到服务器的通道，创建客户端存根，以及进行远程过程调用。

```python
import grpc
import numpy as np
import tensorflow as tf

# 导入生成的 gRPC 类
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# 假设 TF Serving 在 localhost:8500 上运行，模型名为 'my_model'
server_address = 'localhost:8500'
model_name = 'my_model'
# 如果不是 'serving_default'，请指定签名名称
# signature_name = 'serving_default'

try:
    # 创建一个 gRPC 通道
    channel = grpc.insecure_channel(server_address)

    # 创建一个存根（客户端）
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # 示例：创建两个模拟输入实例
    # 替换为您实际的数据预处理
    input_data = np.random.rand(2, 784).astype(np.float32)

    # 创建一个 PredictRequest Protobuf 对象
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    # request.model_spec.signature_name = signature_name # 如果需要，取消注释

    # 将输入数据映射到签名中正确的输入张量名称
    # 使用 'tf.make_tensor_proto' 将 NumPy 数组转换为 TensorProto
    # 确保数据类型与模型预期的输入类型匹配
    request.inputs['input_features'].CopyFrom(
        tf.make_tensor_proto(input_data, shape=input_data.shape, dtype=tf.float32)
    )

    # 为请求设置超时（例如 10 秒）
    timeout_seconds = 10.0
    result_future = stub.Predict.future(request, timeout_seconds)
    result = result_future.result() # 等待响应

    # 解析响应（这是一个 PredictResponse Protobuf 对象）
    # 通过其名称访问输出张量（例如 'output_scores'）
    # 使用 'tf.make_ndarray' 将 TensorProto 转换回 NumPy 数组
    # 将 'output_scores' 替换为您实际的输出张量名称
    predictions = tf.make_ndarray(result.outputs['output_scores'])

    print("收到预测结果：")
    print(predictions)

except grpc.RpcError as e:
    print(f"gRPC 错误：{e.status()}")
    print(f"详情：{e.details()}")
except Exception as e:
    print(f"发生意外错误：{e}")

finally:
    # 如果通道已打开，请确保关闭
    if 'channel' in locals() and channel:
        channel.close()
```

> **重要提示：** 示例中的输入和输出张量名称（`'input_features'`、`'output_scores'`）必须与 SavedModel 签名定义中定义的名称完全一致。使用 `saved_model_cli` 查找这些名称。

#### gRPC 优点和缺点

- **优点：** 高性能（低延迟、高吞吐量），高效的二进制序列化（Protobufs），支持双向流，非常适合微服务通信。
- **缺点：** 与 REST 相比设置更复杂，载荷可读性较低，需要特定的客户端库和代码生成（尽管通常由 `tensorflow-serving-api` 等包提供）。

### 在 REST 和 gRPC 之间选择

选择取决于您的具体需求：

- **简单性和兼容性：** 如果集成便捷性和广泛兼容性是主要考虑因素，REST 通常就足够了，特别是对于毫秒级延迟差异不重要的应用程序。
- **性能：** 对于高吞吐量 (throughput)系统、实时推理 (inference)或延迟敏感型应用程序，gRPC 由于 HTTP/2 和 Protobuf 的效率，通常能提供明显更好的性能。
- **内部微服务：** gRPC 通常更受内部服务间通信的青睐，因为性能要求高，且开发人员可以控制客户端和服务器实现。

> 客户端应用程序可以通过 HTTP/1.1 上的 REST 与 JSON 载荷（通常在 8501 端口）连接到 TensorFlow Serving，或通过 HTTP/2 上的性能更好的 gRPC 与 Protobuf 载荷（通常在 8500 端口）连接。

掌握与 TensorFlow Serving 的 REST 和 gRPC 交互方式，您将获得灵活地有效部署模型的能力，根据特定使用场景，选择在性能需求和开发简易性之间取得良好平衡的协议。这为构建可扩展的机器学习 (machine learning)推理服务奠定了基础。

## 参考资料

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) — TensorFlow Authors (2021)
  Publisher: Google
  提供使用TensorFlow Serving部署TensorFlow模型的官方指南和详细说明，涵盖设置、REST和gRPC API。
- [gRPC - A high performance, open source universal RPC framework](https://grpc.io/) — gRPC Authors (2024)
  gRPC官方网站，解释了其架构、优势和使用方法，这对于理解TF Serving中的gRPC协议至关重要。
- [Protocol Buffers Developer Documentation](https://developers.google.com/protocol-buffers/) — Google (2023)
  Publisher: Google
  Protocol Buffers的官方文档，它是gRPC在TensorFlow Serving中用于高效数据交换的序列化格式。
- [SavedModel Guide](https://www.tensorflow.org/guide/saved_model) — TensorFlow Authors (2024)
  详细介绍了SavedModel格式的官方指南，SavedModel是TensorFlow模型的通用格式，也是TensorFlow Serving的输入。

---

[上一节](02-TensorFlow%20Serving%20%E7%AE%80%E4%BB%8B.md) · [下一节](04-%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95.md)

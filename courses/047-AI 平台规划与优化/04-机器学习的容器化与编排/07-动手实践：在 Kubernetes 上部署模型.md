# 动手实践：在 Kubernetes 上部署模型

来源：[原文](https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-4-containerization-orchestration-for-ml/practical-deploying-model-on-kubernetes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

本实践练习通过引导您完成一个完整的工作流程：将一个简单的模型服务应用打包成 Docker 容器，然后将其部署到本地 Kubernetes 集群。您将编写 Docker 和 Kubernetes 所需的配置档案，使用 `kubectl` 来管理应用，最后，测试实际运行的端点。此过程反映了将机器学习 (machine learning)服务迁移到生产环境的常见方式。

### 前提条件

开始之前，请确保您的机器上已安装并配置以下工具：

- **Docker Desktop 或 Docker Engine：** 用于构建容器镜像。
- **本地 Kubernetes 集群：** 最简单的方式是启用 Docker Desktop 附带的集群。或者，您也可以安装 [Minikube](https://minikube.sigs.k8s.io/docs/start/)。
- **kubectl：** Kubernetes 命令行工具。通常与 Docker Desktop 或 Minikube 一同安装。您可以通过运行 `kubectl version --client` 来验证其安装情况。

### 模型服务应用

我们将使用一个简单的 Flask Web 应用，它提供来自预训练 (pre-training) Scikit-learn 模型的预测。该模型是一个在 Iris 数据集上训练的经典分类器。目标是保持应用简单，以便我们将注意力放在部署架构上。

首先，为此项目创建一个名为 `iris-k8s-app` 的新目录。在该目录中，创建以下三个档案。

**1. `model.joblib`**

您无需自行创建此档案。本次练习中，我们假设您有一个已保存的预训练模型。其具体内容不重要，只需我们的 Python 脚本能够加载它即可。

**2. `requirements.txt`**

此档案列出了我们的应用所需的 Python 库。

```text
flask==2.2.2
scikit-learn==1.1.2
joblib==1.2.0
numpy==1.23.4
```

**3. `app.py`**

这是我们的 Python 脚本。它加载模型并创建一个 Flask Web 服务器，带有一个 `/predict` 端点，该端点接受包含花卉测量数据的 POST 请求并返回预测结果。

```python
import joblib
import numpy as np
from flask import Flask, request, jsonify

# 初始化 Flask 应用
app = Flask(__name__)

# 加载预训练模型
# 在实际场景中，您会有这个模型档案。
# 本实验中，我们将模拟它的存在和加载。
# model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json(force=True)

        # 提取特征并转换为模型所需的 numpy 数组
        # 期望包含 4 个浮点数的列表：[萼片长度, 萼片宽度, 花瓣长度, 花瓣宽度]
        features = np.array(data['features']).reshape(1, -1)

        # 模拟预测，因为我们没有实际的模型档案。
        # 在实际运行中，这将是：prediction = model.predict(features)
        # 我们将基于花瓣长度（特征索引 2）模拟一个预测
        if features[0, 2] < 2.5:
            prediction_result = 'setosa'
        else:
            prediction_result = 'versicolor_or_virginica'

        # 以 JSON 格式返回预测结果
        return jsonify({'prediction': prediction_result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # 在主机 0.0.0.0 上运行应用，使其可从容器内访问
    app.run(host='0.0.0.0', port=5000)
```

> **模型说明：** Python 代码中包含一行 `joblib.load`，该行被注释掉并替换为模拟逻辑。这简化了实验，消除了下载或训练模型的需要，使我们能够完全专注于容器化和编排步骤。

### 步骤 1：使用 Dockerfile 对应用进行容器化

第一步是创建一个 `Dockerfile`，它定义了如何构建包含我们的应用及其依赖项的镜像。

在您的项目目录中创建一个名为 `Dockerfile` 的档案，内容如下：

```dockerfile
# 从精简的 Python 基础镜像开始
FROM python:3.9-slim

# 设置容器内的工作目录
WORKDIR /app

# 将 requirements 档案复制到容器中
COPY requirements.txt .

# 安装 Python 依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 将应用的其余代码复制到容器中
COPY . .

# 暴露应用运行的端口
EXPOSE 5000

# 定义运行应用的命令
CMD ["python", "app.py"]
```

#### 构建 Docker 镜像

现在，在 `iris-k8s-app` 目录中打开您的终端，运行以下命令来构建镜像。我们将它标记 (token)为 `iris-app:v1`。

```bash
docker build -t iris-app:v1 .
```

构建完成后，您可以通过在本地运行容器来验证它。

```bash
docker run -p 5000:5000 iris-app:v1
```

您应该会看到 Flask 的输出，表示服务器正在运行。您可以使用 `Ctrl+C` 停止它。

### 步骤 2：定义 Kubernetes 部署

Kubernetes 的 **Deployment** 是一种资源对象，用于管理一组相同的 Pod。它确保指定数量的应用副本正在运行，并处理更新或回滚。

创建一个名为 `deployment.yaml` 的档案，内容如下：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: iris-server
  template:
    metadata:
      labels:
        app: iris-server
    spec:
      containers:
      - name: iris-app-container
        image: iris-app:v1
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 5000
```

我们来拆解此档案：

- **`replicas: 2`**：指示 Kubernetes 维护我们应用的两个运行实例（Pod），以确保可用性。
- **`selector`**：这告诉 Deployment 要管理哪些 Pod。它会查找带有标签 `app: iris-server` 的 Pod。
- **`template`**：此部分定义将要创建的 Pod。它包含：
  - **`metadata.labels`**：一个标签 `app: iris-server` 被附加到每个 Pod。这是选择器找到它们的方式。
  - **`spec.containers`**：定义要在 Pod 内运行的容器。
  - **`image: iris-app:v1`**：指定要使用的 Docker 镜像。
  - **`imagePullPolicy: IfNotPresent`**：告诉 Kubernetes 如果本地存在镜像则使用它，而不是尝试从远程注册表拉取。这对于本地开发很有用。
  - **`containerPort: 5000`**：通知 Kubernetes 容器在端口 5000 监听。

### 步骤 3：使用 Service 暴露 Deployment

Kubernetes 中的 Pod 是短暂的，并且其内部 IP 地址可能发生变化。为了提供一个稳定的网络端点来访问我们的应用，我们使用 **Service**。

我们将使用 `NodePort` 服务类型，它在集群中每个节点的一个静态端口上暴露应用。这是一种在开发过程中访问应用的直接方式。

创建一个名为 `service.yaml` 的档案：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: iris-service
spec:
  type: NodePort
  selector:
    app: iris-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
```

以下是此档案定义的内容：

- **`type: NodePort`**：在集群节点的 IP 地址上的一个特定端口暴露此服务。
- **`selector`**：这是关键的连接。该服务将把流量路由到任何带有 `app: iris-server` 标签的 Pod，这与我们的 Deployment 创建的 Pod 相匹配。
- **`ports`**：此部分映射网络端口。它表明服务将在端口 `80` 上接受流量，并将其转发到所选 Pod 上的 `targetPort: 5000`。

> 我们创建的 Kubernetes 对象之间的关系。用户向稳定的服务发送请求，然后该服务将流量转发到由 Deployment 管理的任意 Pod。

### 步骤 4：将清单应用到集群

镜像构建完成且清单档案已编写好，您现在可以部署应用了。对每个档案运行 `kubectl apply` 命令。

```bash
# 应用部署配置
kubectl apply -f deployment.yaml

# 应用服务配置
kubectl apply -f service.yaml
```

您可以检查资源的状态：

```bash
# 检查部署是否正在进行
kubectl get deployment

# 预期输出：
# NAME              READY   UP-TO-DATE   AVAILABLE   AGE
# iris-deployment   2/2     2            2           15s

# 检查两个 Pod 是否正在运行
kubectl get pods

# 预期输出：
# NAME                               READY   STATUS    RESTARTS   AGE
# iris-deployment-5c68f6d787-abcde   1/1     Running   0          25s
# iris-deployment-5c68f6d787-fghij   1/1     Running   0          25s

# 检查服务是否已创建
kubectl get service iris-service

# 预期输出：
# NAME           TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
# iris-service   NodePort   10.101.102.103  <none>        80:31234/TCP   35s
```

请注意服务的 `PORT(S)` 列。集群已分配一个高位端口（例如，`31234`），它映射到服务的内部端口 `80`。这就是您将用于访问应用的端口。

### 步骤 5：测试已部署的模型

要测试服务，您需要集群节点的 IP 地址和 `NodePort`。对于本地集群，IP 通常是 `localhost`，或者可以通过 `minikube ip` 找到。

如果使用 Minikube，您可以通过一个命令获取完整的 URL：

```bash
minikube service iris-service --url
```

这将输出一个类似 `http://192.168.49.2:31234` 的 URL。

现在，使用 `curl` 向您服务的 `/predict` 端点发送一个带有示例数据的 POST 请求。使用您在上一步中获得的 URL。

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}' \
  http://<YOUR_NODE_IP>:<YOUR_NODE_PORT>/predict
```

您应该会从在您的某个 Pod 内运行的模型收到一个 JSON 响应：

```json
{"prediction":"setosa"}
```

恭喜您，您已成功地将一个机器学习 (machine learning)应用容器化并部署到 Kubernetes 上。

### 清理

要从集群中移除资源，请使用 `kubectl delete` 命令，并带上相同的清单档案。

```bash
kubectl delete -f service.yaml
kubectl delete -f deployment.yaml
```

本实验说明了将机器学习 (machine learning)模型部署为可伸缩、可管理服务的常见模式。通过在 YAML 档案中声明式地定义您的应用，您可以轻松地在任何 Kubernetes 环境中复制、伸缩和管理您的部署。

## 参考资料

- [Kubernetes Documentation](https://kubernetes.io/docs/) — Kubernetes Authors (2024)
  Publisher: Cloud Native Computing Foundation (CNCF)
  理解Kubernetes概念、架构和使用的综合资源，与本节描述的部署步骤直接相关。
- [Docker Documentation](https://docs.docker.com/) — Docker, Inc. (2024)
  Publisher: Docker, Inc.
  Docker容器化、使用Dockerfile构建镜像以及容器管理的重要指南，直接支持应用程序的初始打包。
- [Kubernetes in Action, Second Edition](https://www.manning.com/books/kubernetes-in-action-second-edition) — Marko Lukša, Kevin Conner (2026)
  Publisher: Manning Publications
  本书对Kubernetes进行全面介绍，详细涵盖Pod、部署和服务等核心概念，对编排原理提供更深入的理解。
- [Practical MLOps: How to Take Your Machine Learning Models to Production](https://oreil.ly/practical-mlops) — Noah Gift, Alfredo Deza (2021)
  Publisher: O'Reilly Media
  涵盖MLOps的完整生命周期，包括在Kubernetes等平台上部署模型的策略，帮助读者将本实践练习置于更广泛的生产化背景中。

---

[上一节](06-%E4%BD%BF%E7%94%A8%20Kubeflow%20%E6%9E%84%E5%BB%BA%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%A1%E9%81%93.md) · [下一节](../05-%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5/01-%E8%AF%86%E5%88%AB%E6%80%A7%E8%83%BD%E7%93%B6%E9%A2%88.md)

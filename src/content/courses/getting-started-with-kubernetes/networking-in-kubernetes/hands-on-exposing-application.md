---
course: "getting-started-with-kubernetes"
chapter: "networking-in-kubernetes"
lesson: "hands-on-exposing-application"
sourceId: 7322
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-4-networking-in-kubernetes/hands-on-exposing-application"
title: "动手实践：暴露应用程序"
description: "通过动手实践为 Deployment 创建 Service，并使用 NodePort 或 Ingress 资源将其暴露到集群外部。"
order: 7
plots: []
sourceHash: "cc825040c2f22479aa9db97606fbd5cb910f0583b549d641d27b9dcdd5e410df"
sourceCorrections: []
---

在 Kubernetes 网络配置中，将运行中的应用程序暴露到外部是一项基础任务。这通常涉及两种方法：使用 `NodePort` 类型的 Service 进行直接访问，以及配置 `Ingress` 资源以实现更灵活的基于主机的路由。

### 前提条件

在开始之前，请确保你有一个正在运行的本地 Kubernetes 集群（例如 Minikube），并且已配置好 `kubectl` 与之交互。

首先，让我们部署一个简单的 Web 应用程序用于实验。该程序是一个响应 HTTP 请求的 Nginx 服务器。将以下配置保存为 `webapp-deployment.yaml`：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
    spec:
      containers:
      - name: nginx-container
        image: nginx:1.21
        ports:
        - containerPort: 80
```

将此配置应用到集群以创建 Deployment 及其关联的 Pod：

```sh
kubectl apply -f webapp-deployment.yaml
```

验证 Pod 是否正在运行：

```sh
kubectl get pods -l app=webapp
```

你应该能看到两个状态为 `Running` 的 Pod。这些 Pod 拥有 IP 地址，但只能在集群内部访问。我们的目标是让它们可以从外部访问。

### 使用 NodePort Service 暴露应用程序

第一步是创建一个 Service，为 `webapp` Pod 提供稳定的端点。我们将使用 `NodePort` 类型，它会在集群的每个节点上暴露一个特定端口。这是一种将外部流量引入应用程序的直接方式，在开发环境中非常常用。

将以下配置保存为 `webapp-nodeport-service.yaml`：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: webapp-service
spec:
  type: NodePort
  selector:
    app: webapp
  ports:
    - protocol: TCP
      port: 80         # Service 在集群内部可用的端口
      targetPort: 80   # Service 转发流量到的 Pod 端口
      # nodePort: 30080 # 可选，可以指定一个端口。如果省略，Kubernetes 会自动分配一个。
```

配置详情如下：

- `spec.type: NodePort`：指定 Service 的类型。
- `spec.selector.app: webapp`：这是关联的关键。Service 会将流量路由到任何带有 `app: webapp` 标签的 Pod。
- `spec.ports`：此部分定义端口映射。到达 Service `port: 80` 的流量将被转发到所选 Pod 的 `targetPort: 80`。Kubernetes 还会从节点上分配一个高位端口（`nodePort`），该端口会转发到 Service 的 `80` 端口。

应用该 Service 配置：

```sh
kubectl apply -f webapp-nodeport-service.yaml
```

现在，查看创建的 Service：

```sh
kubectl get service webapp-service
```

输出结果类似于：

```
NAME             TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
webapp-service   NodePort   10.108.146.13   <none>        80:31501/TCP   25s
```

注意 `PORT(S)` 列。它显示 Service 内部 `CLUSTER-IP` 的 `80` 端口已映射到了一个 `nodePort`（在本例中是 `31501`）。你获取的端口号可能会有所不同。发送到集群中任何节点的 `31501` 端口的流量都会被路由到 `webapp` Pod 的 `80` 端口。

> NodePort Service 的流量流向。外部请求命中工作节点 IP 地址上的指定 NodePort，Kubernetes 随后通过 Service 将请求路由到其中一个目标 Pod。

要访问它，可以使用 `minikube service` 命令，这是一个便捷的捷径：

```sh
minikube service webapp-service --url
```

此命令会在浏览器中打开应用程序或打印出一个 URL，例如 `http://192.168.49.2:31501`。访问该 URL 将看到由其中一个 Pod 提供的 Nginx 默认欢迎页面。

### 使用 Ingress 暴露应用程序

`NodePort` 虽然有效但存在局限性。通常情况下，你不会希望向用户暴露随意的非标准高位端口。`Ingress` 提供了一种更高级的外部访问管理方式，支持 HTTP/HTTPS 路由、基于主机的路由以及基于路径的路由。

要使用 `Ingress`，集群中需要运行 Ingress 控制器。对于 Minikube，可以通过一条简单的命令启用内置的 Nginx Ingress 控制器：

```sh
minikube addons enable ingress
```

启用插件后，等待一分钟让控制器 Pod 启动。你可以在 `ingress-nginx` 命名空间中检查其状态：`kubectl get pods -n ingress-nginx`。

现在，让我们创建一个 `Ingress` 资源。该资源将定义一条规则：任何发往主机 `webapp.example.com` 的流量都应发送到我们的 `webapp-service`。请注意，为了让 `Ingress` 正常工作，它指向的 Service 不一定非要是 `NodePort` 类型。它可以是默认的 `ClusterIP` 类型，因为 Ingress 控制器运行在集群内部，可以直接访问内部 Service IP。我们可以继续使用当前的 `webapp-service`。

将以下配置保存为 `webapp-ingress.yaml`：

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: webapp-ingress
spec:
  rules:
  - host: "webapp.example.com"
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: webapp-service
            port:
              number: 80
```

应用 Ingress 配置：

```sh
kubectl apply -f webapp-ingress.yaml
```

检查 Ingress 的状态：

```sh
kubectl get ingress webapp-ingress
```

分配地址可能需要一点时间。准备就绪后，输出将显示 Ingress 控制器的地址。

```
NAME             CLASS    HOSTS                ADDRESS          PORTS   AGE
webapp-ingress   nginx    webapp.example.com   192.168.49.2     80      50s
```

最后一步是让你的本地机器将 `webapp.example.com` 解析到集群的 IP 地址。我们可以通过编辑本地的 `hosts` 文件来实现。

1. **查找 Minikube IP：** 运行 `minikube ip`。
2. **编辑 hosts 文件：**

   - 在 macOS 或 Linux 上：`sudo nano /etc/hosts`
   - 在 Windows 上：以管理员身份编辑 `C:\Windows\System32\drivers\etc\hosts`。
3. **添加条目：** 添加新的一行，将 Minikube IP 映射到你选择的主机名。例如，如果你的 Minikube IP 是 `192.168.49.2`：

   ```
   192.168.49.2  webapp.example.com
   ```

保存文件。现在，你的电脑就知道发往 `webapp.example.com` 的请求该送往何处了。

> Ingress 的流量流向。外部请求到达 Ingress 控制器。控制器检查请求的主机名，根据 Ingress 资源中的路由规则，将请求转发给相应的后端 Service。

你现在可以使用主机名通过 `curl` 或网页浏览器访问应用程序：

```sh
curl http://webapp.example.com
```

你应该能看到相同的 Nginx 欢迎页面。你已成功使用基于主机的路由暴露了应用程序，这是一种更灵活且符合生产实践的模式。

### 清理

为了保持集群整洁，建议删除在此实践环节中创建的资源。你可以通过引用创建它们时使用的配置文件来删除它们。

```sh
kubectl delete -f webapp-ingress.yaml
kubectl delete -f webapp-nodeport-service.yaml
kubectl delete -f webapp-deployment.yaml
```

另外，请记得删除你添加到 `/etc/hosts` 文件中的条目。

## 参考资料

- [Services, Load Balancing, and Networking](https://kubernetes.io/docs/concepts/services-networking/service/) — Kubernetes Authors (2024)
  Journal: Kubernetes Documentation
  Kubernetes 服务的官方文档，包含 `NodePort`、`ClusterIP` 和 `LoadBalancer` 类型。此文档说明服务如何实现对一组 Pod 的网络访问。
- [Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/) — Kubernetes Authors (2024)
  Journal: Kubernetes Documentation
  Kubernetes Ingress 的官方指南，详细说明其外部访问目的、路由规则以及 Ingress 控制器的作用。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-and-running/9781098110154/) — Brendan Burns, Joe Beda, Kelsey Hightower, Lachlan Evenson (2022)
  Publisher: O'Reilly Media
  一本内容全面的书籍，提供在 Kubernetes 中部署和管理应用程序的实用指导，其中包含有关服务和 Ingress 等网络概念的部分。
- [minikube documentation](https://minikube.sigs.k8s.io/docs/) — Kubernetes Authors (2025)
  Journal: Minikube Documentation
  Minikube 的官方文档，提供本地 Kubernetes 集群设置和管理说明，包括本实践环节中使用的特定命令。

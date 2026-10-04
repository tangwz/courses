---
course: "getting-started-with-kubernetes"
chapter: "networking-in-kubernetes"
lesson: "exposing-services-with-ingress"
sourceId: 7321
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-4-networking-in-kubernetes/exposing-services-with-ingress"
title: "通过 Ingress 暴露服务"
description: "了解如何使用 Ingress 控制器和 Ingress 资源来管理服务的外部访问，通常用于 HTTP/S 路由。"
order: 6
plots: []
sourceHash: "d8bba4ac8c4847cf900fbe3ec53325aa0f7752a01c847589832ee77875bf97c0"
sourceCorrections: []
---

虽然 `NodePort` 和 `LoadBalancer` 等 `Service` 对象提供了暴露应用程序的实用方法，但它们通常伴随着运维上的权衡。`NodePort` 在每个节点上通过一个随机的高端口暴露应用，这对于标准的 Web 流量并不理想。而 `LoadBalancer` 类型的服务通常会为每个服务配置一个专用的、特定于云平台的 L4 负载均衡器。随着应用规模扩大，包含许多微服务时，这种方式可能会变得成本高昂且难以管理。对于通过 HTTP 和 HTTPS 通信的 Web 应用，需要一种更智能的应用层路由机制。

这就是 `Ingress` 资源提供更完善解决方案的地方。Ingress 并不是一种 Service 类型。相反，它是一个 API 对象，充当集群路由规则的集合。它允许你定义外部流量（主要是 HTTP 和 HTTPS）应如何根据主机名或 URL 路径指向内部 Service。这为你的整个应用技术栈提供了一个单一且稳定的入口点，整合了路由逻辑并简化了外部访问管理。

### Ingress 控制器 (Ingress Controller)

有一点非常重要：Ingress 资源本身不执行任何操作。它只是一组被动的规则。要使这些规则生效，你需要在集群中运行 **Ingress 控制器**。Ingress 控制器是实际的软件，是以 Pod 形式运行的特定工作负载，它监听集群中定义的 Ingress 资源，并配置反向代理（如 NGINX、HAProxy 或 Traefik）来实现指定的路由。

大多数 Kubernetes 集群默认不安装 Ingress 控制器。你或集群管理员必须选择并部署一个。一旦 Ingress 控制器运行起来，它就会监视 Kubernetes API 服务器中任何新增或更新的 Ingress 资源，并相应地重新配置自身。

流量流向通常是这样的：外部客户端向单个负载均衡器发送请求，该负载均衡器将流量引导至 Ingress 控制器。然后，控制器检查请求的主机名和路径，参考从 Ingress 资源中获取的路由规则，并将流量转发到相应的内部 Service 及其后端的 Pod。

> Ingress 控制器接收所有传入流量，并根据你在 Ingress 资源中定义的规则将其路由到不同的内部服务。

### 定义 Ingress 清单

与其它 Kubernetes 对象一样，Ingress 资源也是使用 YAML 清单定义的。其规格（spec）包含路由流量的规则。这些规则可以基于请求的主机名（基于主机的路由）或其 URL 路径（基于路径的路由）。

让我们看一个定义了基于路径路由的清单。在这里，指向 `example.com/api` 的流量被发送到一个服务，而指向 `example.com/` 的流量被发送到另一个服务。

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-application-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ui-service
            port:
              number: 80
```

让我们拆解一下 `spec` 部分：

- **`rules`**: 路由规则列表。每条规则应用于特定的主机，虽然这里没有指定主机，这意味着它适用于所有入站流量。
- **`http.paths`**: 路径及其对应后端的列表。
- **`path`**: 要匹配的 URL 路径。在本例中为 `/api` 和 `/`。
- **`pathType`**: 指定 `path` 的匹配方式。`Prefix` 匹配以指定路径开头的任何 URL。`Exact` 则要求完全匹配。
- **`backend.service`**: 定义匹配该路径的流量的目标 Service。它指定了 Service 的名称 (`name`) 和要连接的端口号 (`port`)。

注解（Annotations），如 `nginx.ingress.kubernetes.io/rewrite-target`，通常用于提供针对所使用的特定 Ingress 控制器的额外配置。

你还可以配置基于主机的路由，通常称为虚拟托管。这允许你通过 Ingress 控制器管理的同一个外部 IP 地址，将不同主机名的流量引导至不同的服务。

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: virtual-host-ingress
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ui-service
            port:
              number: 80
```

在这个例子中，对 `api.example.com` 的请求被路由到 `api-service`，而对 `app.example.com` 的请求被路由到 `ui-service`。

### TLS 终止 (TLS Termination)

Ingress 的另一个显著能力是管理 TLS 终止。你无需在每个单独的应用 Pod 中配置 TLS 证书，而是可以在 Ingress 控制器处集中处理这一职责。这简化了证书管理，意味着集群内部从控制器到 Pod 的流量可以使用标准的、未加密的 HTTP。

要启用 TLS，首先需要将证书和私钥存储在 Kubernetes 的 `Secret` 中。

```bash
kubectl create secret tls my-tls-secret --cert=path/to/tls.crt --key=path/to/tls.key
```

然后，在 Ingress 清单中引用此 Secret：

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: secure-ingress
spec:
  tls:
  - hosts:
    - app.example.com
    secretName: my-tls-secret
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ui-service
            port:
              number: 80
```

`spec` 中的 `tls` 部分指示 Ingress 控制器对所有发往 `app.example.com` 的流量使用来自 `my-tls-secret` 的证书。控制器将处理与客户端的 TLS 握手，然后将解密后的流量转发给 `ui-service`。

通过使用 Ingress，你获得了一种强大且灵活的方法来管理应用的外部访问。它整合了入口点，降低了与多个负载均衡器相关的成本，并在集群边缘集中处理了路由和安全逻辑。

## 参考资料

- [Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/) — Kubernetes Authors (2024)
  提供了 Kubernetes Ingress API 对象的定义和操作原理，及其在路由外部 HTTP/HTTPS 流量到集群服务中的作用。
- [Kubernetes in Action](https://www.manning.com/books/kubernetes-in-action) — Marko Lukša (2017)
  Publisher: Manning Publications
  对 Kubernetes 概念进行了广泛的解释，包括关于网络、服务和 Ingress 的详细章节，为其操作提供了全面的背景信息。

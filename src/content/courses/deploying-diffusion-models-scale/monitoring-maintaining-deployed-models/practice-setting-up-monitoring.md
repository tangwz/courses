---
course: "deploying-diffusion-models-scale"
chapter: "monitoring-maintaining-deployed-models"
lesson: "practice-setting-up-monitoring"
sourceId: 5471
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-5-monitoring-maintaining-deployed-models/practice-setting-up-monitoring"
title: "动手实践：建立监控"
description: "为已部署的服务配置基本监控（例如，指标抓取、简单仪表盘）。"
order: 8
plots: ["plots/5471-0.json"]
sourceHash: "7aabd03e426f19659686096526596d5cae0250c33a620f055d3e3b02284446d7"
sourceCorrections: []
---

为已部署的扩散模型推理 (inference)服务建立一个基本的监控系统，通常涉及使用 Prometheus 收集指标和使用 Grafana 进行可视化，这是 MLOps 环境中常用的技术栈。您需要有一个正在运行的容器化扩散模型 API 服务，并且可以在您的部署环境（如 Kubernetes 或虚拟机）中访问。

### 为您的应用程序添加指标上报功能

第一步是让您的应用程序以 Prometheus 可理解的格式暴露相关指标。我们将使用 Prometheus 暴露格式。大多数 Web 框架都有相应的 Prometheus 客户端库。对于 Python FastAPI 应用程序，`prometheus-fastapi-instrumentator` 库是一个方便的选择，您也可以直接使用标准的 `prometheus_client`。

让我们为 FastAPI 应用程序添加功能以暴露核心指标：

1. **安装必要的库：**

   ```bash
   pip install prometheus-fastapi-instrumentator prometheus-client
   ```
2. **修改您的 FastAPI 应用程序：**

   ```python
   from fastapi import FastAPI
   from prometheus_fastapi_instrumentator import Instrumentator
   import time
   import random
   import uvicorn

   # 假设 'generate_image' 是您的核心扩散模型推理函数
   async def generate_image(prompt: str):
       # 模拟生成时间（请替换为实际的模型调用）
       start_time = time.time()
       processing_time = random.uniform(5.0, 15.0) # 模拟 5-15 秒
       time.sleep(processing_time)
       # 模拟偶尔发生的错误
       if random.random() < 0.05: # 5% 错误率
           raise ValueError("Simulated generation error")
       end_time = time.time()
       latency_ms = (end_time - start_time) * 1000
       # 在实际场景中，您还需要返回图像数据
       return {"prompt": prompt, "latency_ms": latency_ms, "status": "success"}

   app = FastAPI()

   # 为应用程序添加指标上报功能
   Instrumentator().instrument(app).expose(app)

   @app.post("/generate")
   async def handle_generate(payload: dict):
       prompt = payload.get("prompt", "a default prompt")
       try:
           result = await generate_image(prompt)
           return result
       except Exception as e:
           # 您可能需要更复杂的错误处理/状态码
           return {"error": str(e), "status": "failure"}, 500

   # 如果需要，添加自定义指标（示例：GPU 利用率）
   # from prometheus_client import Gauge
   # GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'Current GPU Utilization (%)')
   # 您需要一个单独的进程/线程来监控 nvidia-smi 或类似工具
   # 并定期更新此仪表。
   # Example: GPU_UTILIZATION.set(get_current_gpu_utilization())

   if __name__ == "__main__":
       uvicorn.run(app, host="0.0.0.0", port=8000)
   ```

运行此应用程序并访问 `/metrics` 端点（例如，`http://localhost:8000/metrics`）将显示 Prometheus 指标，包括自动上报的指标，如 `http_requests_total`、`http_request_duration_seconds`，以及您可能添加的任何自定义指标。`prometheus-fastapi-instrumentator` 自动提供延迟直方图，这对于计算百分位数很有用。

### 设置 Prometheus

Prometheus 需要配置为定期“抓取”（获取）应用程序 `/metrics` 端点的指标。假设您已运行 Prometheus（例如，作为 Docker 容器或 Kubernetes 服务），您将更新其配置文件（`prometheus.yml`）。

```yaml
# prometheus.yml（示例片段）
global:
  scrape_interval: 15s # 抓取目标的频率

scrape_configs:
  - job_name: 'diffusion-api'
    static_configs:
      - targets: ['<your-diffusion-api-ip-or-hostname>:8000'] # 替换为实际的目标地址
    # 如果在 Kubernetes 中运行，请使用服务发现而不是静态配置
    # kubernetes_sd_configs:
    # - role: endpoints
    # reabel_configs: ... （用于选择正确的服务/Pod）
```

更新配置后，重新启动或重新加载 Prometheus。它将开始从您的服务收集指标。

### 使用 Grafana 可视化指标

Grafana 允许您创建仪表盘来可视化存储在 Prometheus 中的数据。

1. **添加 Prometheus 作为数据源：** 在 Grafana 中，导航到 Configuration -> Data Sources -> Add data source。选择 Prometheus 并输入您的 Prometheus 服务器可访问的 URL（例如，`http://prometheus:9090`）。
2. **创建仪表盘：** 创建一个新的仪表盘并添加面板。

   - **请求速率面板：**

     - **查询：** `rate(http_requests_total{job="diffusion-api", handler="/generate"}[5m])`
     - **可视化：** 时间序列图。这显示了在过去 5 分钟内，`/generate` 端点每秒的平均请求速率。
   - **P95 延迟面板：**

     - **查询：** `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="diffusion-api", handler="/generate"}[5m])) by (le))`
     - **可视化：** 时间序列图或统计面板。这使用由指标上报功能自动创建的直方图指标来计算 `/generate` 请求的第 95 百分位延迟。如果您的端点路径不同，请调整处理程序标签。
     - **单位：** 将 Y 轴单位设置为“秒”。
   - **错误率面板：**

     - **查询：** `sum(rate(http_requests_total{job="diffusion-api", handler="/generate", status_code=~"5.."}[5m])) / sum(rate(http_requests_total{job="diffusion-api", handler="/generate"}[5m]))`
     - **可视化：** 时间序列图或统计面板。这计算了 `/generate` 端点的 5xx 错误与总请求的比例。如果需要，乘以 100 并将单位设置为“%”。
   - **GPU 利用率面板（如果存在自定义指标）：**

     - **查询：** `gpu_utilization_percent{job="diffusion-api"}`
     - **可视化：** 仪表盘或时间序列图。
     - **单位：** 将单位设置为“%”。

以下是 P95 延迟的可视化示例：



![P95 生成延迟（秒）](plots/5471-0.json)



> `/generate` 端点随时间变化的 P95 请求延迟，以秒为单位测量。监控此项有助于发现性能下降。

### 基本告警

监控与告警结合使用时效果最佳。您可以直接在 Grafana 中配置告警，或使用 Prometheus 的 Alertmanager。

- **Grafana 告警：** 在面板设置中（例如，错误率面板），您可以定义告警规则。例如，如果 5 分钟平均错误率超过 2%，则触发告警。
- **Alertmanager：** 对于更复杂的路由和去重，请配置 Prometheus 根据 Prometheus 配置文件中定义的规则将告警发送到 Alertmanager。

Prometheus 告警规则示例（放置在单独的规则文件中）：

```yaml
# alert.rules.yml
groups:
- name: DiffusionAPIRules
  rules:
  - alert: HighAPIErrorRate
    expr: sum(rate(http_requests_total{job="diffusion-api", handler="/generate", status_code=~"5.."}[5m])) / sum(rate(http_requests_total{job="diffusion-api", handler="/generate"}[5m])) > 0.02
    for: 5m # 如果条件持续 5 分钟为真，则触发告警
    labels:
      severity: warning
    annotations:
      summary: 扩散 API 检测到高错误率
      description: '作业 {{ $labels.job }} 的处理程序 {{ $labels.handler }} 在过去 5 分钟内的错误率超过 2%。'

  - alert: HighP95Latency
    expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="diffusion-api", handler="/generate"}[5m])) by (le)) > 15
    for: 10m # 如果 P95 延迟持续 10 分钟高于 15 秒，则触发告警
    labels:
      severity: critical
    annotations:
      summary: 扩散 API 检测到高 P95 延迟
      description: '作业 {{ $labels.job }} 的处理程序 {{ $labels.handler }} 在过去 10 分钟内的 P95 延迟高于 15 秒（当前值：{{ $value }}秒）。'
```

### 后续步骤

本实践提供了一个基础的监控设置。对于生产系统，可以考虑通过以下方式进行扩展：

- 整合结构化日志（例如，将日志发送到 Elasticsearch 或 Loki），并使用跟踪 ID 将日志与指标关联起来。
- 实现分布式跟踪（例如，使用 OpenTelemetry），以了解请求在多个服务间的流向。
- 添加与您的扩散模型相关的更具体的自定义指标（例如，生成过程中的特定步骤时间、缓存命中率）。
- 在 Alertmanager 中设置更复杂的告警规则和通知渠道。
- 使用云提供商工具或成本管理平台监控资源成本。

即使实施了这种基本监控，您也能大大了解已部署扩散模型的运行状况和性能，从而帮助您诊断问题、优化性能并确保大规模部署的可靠性。

## 参考资料

- [Prometheus Documentation](https://prometheus.io/docs/) — The Prometheus Authors (2024)
  Publisher: Cloud Native Computing Foundation
  Prometheus官方指南，包含安装、配置、数据模型、PromQL和Alertmanager。
- [Grafana Documentation](https://grafana.com/docs/grafana/latest/) — The Grafana Team (2024)
  Publisher: Grafana Labs
  Grafana官方指南，包含数据源配置、仪表盘创建和告警设置。
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于MLOps的综合书籍，讨论了大规模构建、部署和监控机器学习系统的实践。
- [Python Client for Prometheus Documentation](https://prometheus.github.io/client_python/) — The Prometheus Authors (2024)
  Prometheus官方Python客户端文档，详细说明了如何对Python应用程序进行插桩以暴露指标。

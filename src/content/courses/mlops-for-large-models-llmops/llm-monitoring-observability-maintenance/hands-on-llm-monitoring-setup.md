---
course: "mlops-for-large-models-llmops"
chapter: "llm-monitoring-observability-maintenance"
lesson: "hands-on-llm-monitoring-setup"
sourceId: 3217
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-5-llm-monitoring-observability-maintenance/hands-on-llm-monitoring-setup"
title: "动手实践：设置基本的LLM监控"
description: "为LLM端点设置基本监控，跟踪延迟和吞吐量。"
order: 9
plots: ["plots/3217-0.json"]
sourceHash: "b49b9533c84a35d991c318f72abc5c2649f33a2d142772f936ffbd4712b9152d"
sourceCorrections: []
---

此处提供了动手指南，教您如何为模拟的LLM推理 (inference)端点设置基本监控，侧重于两个核心性能指标：延迟和吞吐量 (throughput)。监控LLM特定指标是运营稳定性和性能的基础。尽管在此不部署完整的LLM，但所展示的原理和技术可直接应用于各种情况。

我们将使用Python和FastAPI框架来创建一个模拟LLM端点的简单Web服务器，并使用`prometheus_client`库进行插桩，以暴露可被Prometheus等监控系统抓取的指标。

### 1. 设置模拟LLM端点

首先，请确保您已安装FastAPI、Uvicorn（一个ASGI服务器）以及Prometheus客户端库：

```bash
pip install fastapi uvicorn prometheus-client requests
```

现在，让我们创建一个简单的FastAPI应用（`main.py`），它模拟一个LLM端点。它将包含一个人工延迟来模拟处理时间，并为Prometheus暴露一个`/metrics`端点。

```python
# main.py
import time
import random
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Summary, generate_latest, REGISTRY
from prometheus_client import start_http_server # 如果不使用框架集成，则用于独立的指标服务器

# --- Prometheus 指标 ---
# 使用Histogram记录请求延迟（可计算分位数）
REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'LLM端点请求的延迟',
    ['endpoint']
)

# 使用Counter记录总请求数
REQUEST_COUNT = Counter(
    'llm_request_total',
    'LLM端点总请求数',
    ['endpoint', 'method', 'http_status']
)

# 使用Summary记录请求延迟（替代方案，在客户端计算分位数）
# REQUEST_LATENCY_SUMMARY = Summary(
#     'llm_request_latency_summary_seconds',
#     'LLM端点请求的延迟摘要',
#     ['endpoint']
# )

app = FastAPI()

# 捕获所有请求指标的中间件
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start_time = time.time()
    endpoint = request.url.path

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500 # 内部服务器错误
        raise e from None # 重新抛出异常
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        # REQUEST_LATENCY_SUMMARY.labels(endpoint=endpoint).observe(latency) # 如果使用Summary
        REQUEST_COUNT.labels(
            endpoint=endpoint,
            method=request.method,
            http_status=status_code
        ).inc()

        # 重要：如果在设置响应前发生异常，
        # 确保我们仍然记录带有错误状态的请求计数。
        # try/finally 块有助于管理此情况。请注意，如果 call_next
        # 抛出异常，响应对象可能不可用，
        # 因此在 except 块中设置 status_code。

    return response

@app.post("/predict")
async def predict(payload: dict):
    """模拟一个LLM预测端点。"""
    # 模拟LLM处理时间（例如，50毫秒到500毫秒）
    processing_time = random.uniform(0.05, 0.5)
    time.sleep(processing_time)

    # 模拟一个简单响应
    input_text = payload.get("text", "")
    response_text = f"Simulated response for: {input_text[:20]}..."
    return {"prediction": response_text}

@app.get("/metrics")
async def metrics():
    """暴露Prometheus指标。"""
    from starlette.responses import Response
    return Response(generate_latest(REGISTRY), media_type="text/plain")

# 可选：如果未通过Uvicorn的多工作进程运行或在FastAPI上下文之外运行，
# 您可以启动一个单独的指标服务器。通常集成式更好。
# if __name__ == "__main__":
#     start_http_server(8001) # 在端口8001启动Prometheus客户端服务器
#     # Run FastAPI app separately using uvicorn main:app --reload --port 8000
```

**解释：**

- 我们定义了两个主要指标：`REQUEST_LATENCY`（一个直方图）和`REQUEST_COUNT`（一个计数器）。直方图适合用于延迟，因为它允许在服务器端（Prometheus）计算分位数（例如p95、p99），这通常是性能监控的首选。计数器跟踪累积总数。
- 我们使用FastAPI的中间件机制（`@app.middleware("http")`）来拦截每个请求。在处理请求（`call_next`）之前，我们记录开始时间。处理后，我们计算持续时间（`latency`）并使用`REQUEST_LATENCY`直方图对其进行观察。我们还增加`REQUEST_COUNT`，并用端点、HTTP方法和状态码对其进行标记 (token)。
- `/predict` 端点使用`time.sleep()`模拟工作。
- `/metrics` 端点使用 `prometheus_client` 库中的 `generate_latest(REGISTRY)` 返回Prometheus期望的文本格式的所有已注册指标。

### 2. 运行模拟端点

将上述代码保存为`main.py`并使用Uvicorn运行：

```bash
uvicorn main:app --reload --port 8000
```

您的模拟LLM API现在正在`http://localhost:8000`上运行。

### 3. 生成负载并观察指标

现在，让我们向`/predict`端点发送一些请求。您可以使用一个简单的Python脚本（`load_test.py`）或`curl`或`hey`等工具。

```python
# load_test.py
import requests
import time
import random
import concurrent.futures

API_URL = "http://localhost:8000/predict"
METRICS_URL = "http://localhost:8000/metrics"

def send_request(i):
    payload = {"text": f"This is test input number {i}."}
    try:
        start = time.time()
        response = requests.post(API_URL, json=payload, timeout=2) # 添加超时
        latency = time.time() - start
        if response.status_code == 200:
            print(f"Request {i}: Status {response.status_code}, Latency: {latency:.4f}s")
        else:
            print(f"Request {i}: Status {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Request {i}: Failed - {e}")
        return False
    time.sleep(random.uniform(0.01, 0.1)) # 请求之间稍作延迟

if __name__ == "__main__":
    num_requests = 100
    max_workers = 10 # 模拟并发用户

    print(f"正在使用 {max_workers} 个工作进程发送 {num_requests} 个请求...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_requests)]
        concurrent.futures.wait(futures)

    print("\n负载生成完成。")
    print(f"请在以下地址查看指标：{METRICS_URL}")
```

运行负载生成器：

```bash
python load_test.py
```

当负载生成器运行时（或运行结束后），在浏览器中或使用`curl`访问指标端点：`http://localhost:8000/metrics` 或 `curl http://localhost:8000/metrics`。

您将看到类似以下内容的输出（值会有所不同）：

```text
# HELP llm_request_latency_seconds LLM端点请求的延迟
# TYPE llm_request_latency_seconds histogram
llm_request_latency_seconds_bucket{endpoint="/predict",le="0.005"} 0.0
llm_request_latency_seconds_bucket{endpoint="/predict",le="0.01"} 0.0
llm_request_latency_seconds_bucket{endpoint="/predict",le="0.025"} 0.0
llm_request_latency_seconds_bucket{endpoint="/predict",le="0.05"} 2.0
llm_request_latency_seconds_bucket{endpoint="/predict",le="0.075"} 10.0
llm_request_latency_seconds_bucket{endpoint="/predict",le="0.1"} 25.0
llm_request_latency_seconds_bucket{endpoint="/predict",le="0.25"} 68.0
llm_request_latency_seconds_bucket{endpoint="/predict",le="0.5"} 100.0
llm_request_latency_seconds_bucket{endpoint="/predict",le="0.75"} 100.0
# ... 更多分桶 ...
llm_request_latency_seconds_bucket{endpoint="/predict",le="+Inf"} 100.0
llm_request_latency_seconds_sum{endpoint="/predict"} 28.7345...
llm_request_latency_seconds_count{endpoint="/predict"} 100.0
# HELP llm_request_total LLM端点总请求数
# TYPE llm_request_total counter
llm_request_total{endpoint="/predict",http_status="200",method="POST"} 100.0
llm_request_total{endpoint="/metrics",http_status="200",method="GET"} 5.0
# ... 其他指标 ...
```

这种基于文本的格式旨在供Prometheus抓取。

### 4. 使用Prometheus和Grafana进行监控

在完整设置中，您会配置Prometheus以定期从应用程序实例的`/metrics`端点“抓取”（获取）指标。然后，Grafana将被配置为使用Prometheus作为数据源来可视化这些指标。

**Prometheus 配置（片段）：**

```yaml
# prometheus.yml（示例片段）
scrape_configs:
  - job_name: 'llm-service'
    static_configs:
      - targets: ['localhost:8000'] # 替换为您的实际端点
```

**PromQL查询示例（用于Grafana或Prometheus UI）：**

- **`/predict`端点的请求速率（每秒）：**
  `rate(llm_request_total{endpoint="/predict", method="POST"}[5m])`
  *（计算计数器在过去5分钟内的每秒平均增长率）*
- **`/predict`端点的95百分位延迟（p95）：**
  `histogram_quantile(0.95, sum(rate(llm_request_latency_seconds_bucket{endpoint="/predict"}[5m])) by (le))`
  *（根据过去5分钟的直方图分桶计算95百分位延迟）*
- **平均延迟：**
  `sum(rate(llm_request_latency_seconds_sum{endpoint="/predict"}[5m])) / sum(rate(llm_request_latency_seconds_count{endpoint="/predict"}[5m]))`

**可视化：**

您可以在Grafana中创建仪表板，显示这些指标随时间的变化。例如，一个仪表板可能会显示：

> 基本监控流程：客户端与已插桩API交互，API通过一个端点暴露指标，Prometheus抓取这些指标。Grafana查询Prometheus，在仪表板上显示可视化内容。



![模拟LLM端点性能](plots/3217-0.json)



> 示例可视化图表显示了模拟的p95延迟、平均延迟和请求吞吐量 (throughput)随时间的变化。请注意10:15左右的延迟峰值，它与吞吐量的下降相对应。

### 5. 解释与后续步骤

通过观察这些基本指标，您可以开始回答重要问题：

- **延迟是否增加？** 这可能表明基础设施饱和、请求模式变化或模型服务器本身存在问题。
- **吞吐量 (throughput)是否下降？** 这通常与延迟增加或错误相关联。
- **错误率（例如，来自`llm_request_total`的5xx状态码）是否增加？** 这表明后端存在问题。

本实践展示了如何为基本性能监控对应用程序进行插桩。LLMOps监控的范围要远不止于此，还包括：

- **基础设施指标：** GPU利用率、GPU内存使用、网络带宽（通常通过节点导出器或云提供商代理收集）。
- **成本指标：** 将使用情况与云计费API或内部成本分配系统关联。
- **输出质量指标：** 实施采样、自动化检查（毒性、PII）以及可能的人工反馈循环，以评估LLM响应的质量和安全性。
- **漂移检测：** 监控输入提示分布和输出特性随时间的变化。

建立基础的延迟和吞吐量监控提供了可靠运行LLM所需的初步可见性，并为构建更完善的可观察性提供了前提。

## 参考资料

- [Prometheus Documentation](https://prometheus.io/docs/) — The Prometheus Authors (2024)
  Publisher: Cloud Native Computing Foundation (CNCF)
  Prometheus 官方资源，解释了其监控系统、指标类型、用于仪表化的客户端库以及 PromQL 数据查询语言。
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  FastAPI 框架的官方指南，该框架用于构建模拟的 LLM 推理端点。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready AI](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本涵盖 MLOps 实践的书籍，其中包含关于在生产环境中监控机器学习模型和基础设施的专门章节。
- [Grafana Documentation](https://grafana.com/docs/) — Grafana Labs (2024)
  Grafana 的官方文档，展示了如何可视化 Prometheus 抓取的时间序列数据。

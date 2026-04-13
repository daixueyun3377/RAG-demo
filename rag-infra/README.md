# RAG 基础设施一键部署

## 包含服务
- **Milvus** (向量数据库) - 端口 19530
- **Redis** (缓存/对话记忆) - 端口 6379
- **Langfuse** (可观测性) - 端口 3000

## 前置条件
- Docker + Docker Compose 已安装

## 启动

```bash
cd rag-infra
docker compose up -d
```

## 验证

```bash
# 查看所有容器状态
docker compose ps

# 测试 Milvus
curl http://localhost:9091/healthz

# 测试 Redis
docker exec redis redis-cli ping

# 测试 Langfuse
# 浏览器打开 http://localhost:3000 注册账号
```

## 停止

```bash
docker compose down        # 停止（保留数据）
docker compose down -v     # 停止并删除数据（慎用）
```

## Python 连接示例

```python
# Milvus
from pymilvus import connections
connections.connect(host="localhost", port="19530")

# Redis
import redis
r = redis.Redis(host="localhost", port=6379)

# Langfuse
from langfuse import Langfuse
langfuse = Langfuse(
    public_key="pk-xxx",   # 从 Langfuse Web UI 获取
    secret_key="sk-xxx",
    host="http://localhost:3000"
)
```

## 注意事项
- Langfuse 首次访问需要注册管理员账号
- 生产环境请修改 docker-compose.yml 中的密码和 secret
- 数据存储在 Docker volumes 中，不会因容器重启丢失

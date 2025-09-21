# unstructured-ingest-clickzetta

ClickZetta连接器，专为Unstructured数据处理管道打造的企业级数据湖仓解决方案。

本项目基于 [Unstructured-IO/unstructured-ingest](https://github.com/Unstructured-IO/unstructured-ingest) 扩展开发，提供了与ClickZetta数据湖仓平台的深度集成。

## 🚀 核心特性

### 相对于上游项目新增的功能

#### 1. ClickZetta SQL连接器 (`clickzetta`)
- **完整的数据湖仓集成**：支持从ClickZetta数据库表读取和写入非结构化数据
- **智能批量处理**：自动优化的批量上传，支持大规模数据处理
- **向量化支持**：原生支持向量嵌入存储，兼容多种向量维度（512/768/1024/1536）
- **中文优化**：针对中文环境优化的错误处理和日志提示
- **连接池管理**：智能的数据库连接和会话管理

#### 2. ClickZetta Volume连接器 (`clickzetta_volume`)
- **云原生存储**：支持ClickZetta Volume存储系统的文件操作
- **灵活的卷管理**：支持用户卷、表卷等多种卷类型
- **高级文件操作**：包括上传、下载、删除、正则过滤等
- **路径智能解析**：自动处理复杂的文件路径和卷名解析
- **环境变量集成**：支持多种环境变量命名约定

#### 3. 企业级功能增强
- **增强的错误处理**：提供详细的中文错误信息和故障排除指南
- **性能优化**：
  - 批量处理机制减少数据库连接开销
  - 智能缓冲区管理防止内存溢出
  - 优化的数据传输和序列化
- **兼容性增强**：
  - 支持通义千问等第三方API的特殊处理
  - OpenAI客户端SSL优化
  - 向后兼容原有配置格式

#### 4. DashScope嵌入支持 (`dashscope`)
- **阿里云通义千问集成**：完整支持DashScope TextEmbedding API
- **多模型支持**：支持text-embedding-v1/v2/v4等多个版本
- **智能重试机制**：带指数退避的重试策略，处理API限流
- **批量处理优化**：支持批量嵌入和单文本嵌入
- **统计监控**：详细的API调用统计和成功率监控

#### 5. 多湖仓知识库构建系统 (`multi_lakehouse_kb_builder`)
- **批量部署**：支持一键部署到多个ClickZetta Lakehouse实例
- **智能表管理**：自动创建schema、管理Raw表和Silver表结构
- **并行/串行执行**：支持两种部署模式，适应不同性能需求
- **数据验证**：自动验证部署结果，检测向量质量问题
- **健康检查**：连接状态诊断和知识库健康评估
- **知识管理**：支持添加、删除、搜索自定义知识条目

#### 6. Jupyter Notebook示例 (`examples/notebooks/`)
- **`Unstructured_data_ETL_from_local_to_Lakehouse_tongyi.ipynb`**：
  - 完整的本地文档到ClickZetta Lakehouse的ETL流程
  - DashScope text-embedding-v4集成和向量化处理
  - Raw表和Silver表的创建和管理
  - 倒排索引和向量索引的自动创建
  - RAG检索和相似度搜索演示
  - 支持知识库内容动态添加和管理
- **`databricks_delta_tables.ipynb`**：
  - Databricks Delta Tables集成示例（继承自上游项目）

#### 7. 开发和测试工具
- **完整的测试套件**：包含SQL和Volume连接器的集成测试
- **Docker化部署**：支持容器化部署和测试环境
- **CI/CD优化**：定制的GitHub Actions工作流

## 📦 安装

### 基础安装
```bash
git clone https://github.com/yunqiqiliang/unstructured-ingest-clickzetta.git
cd unstructured-ingest-clickzetta
pip install -e .
```

### ClickZetta依赖
```bash
# 基础ClickZetta连接器
pip install -r requirements/connectors/clickzetta.txt

# DashScope嵌入支持
pip install -r requirements/embed/dashscope.txt
```

## 🔧 使用方式

### ClickZetta SQL连接器示例

```python
from unstructured_ingest.processes.connectors.sql.clickzetta import (
    ClickzettaConnectionConfig,
    ClickzettaAccessConfig,
    ClickzettaUploader,
    ClickzettaUploaderConfig
)

# 配置连接（需要7个关键参数）
connection_config = ClickzettaConnectionConfig(
    service="your-service-url",        # 服务URL
    username="your-username",          # 用户名
    instance="your-instance",          # 实例ID
    workspace="your-workspace",        # 工作空间/数据库名
    vcluster="your-vcluster",         # 虚拟集群名
    schema="your-schema",             # Schema名称
    access_config=ClickzettaAccessConfig(password="your-password")  # 访问配置
)

# 配置上传
upload_config = ClickzettaUploaderConfig(
    table_name="your_table",
    batch_size=1000
)

# 执行数据上传
uploader = ClickzettaUploader(
    connection_config=connection_config,
    upload_config=upload_config
)
```

### ClickZetta Volume连接器示例

```python
from unstructured_ingest.processes.connectors.fsspec.clickzetta_volume import (
    ClickZettaVolumeConnectionConfig,
    ClickZettaVolumeIndexer,
    ClickZettaVolumeIndexerConfig
)

# 环境变量配置
export CLICKZETTA_USERNAME="your-username"
export CLICKZETTA_PASSWORD="your-password"
export CLICKZETTA_SERVICE="your-service-url"
# ... 其他环境变量

# 列举卷中文件
connection_config = ClickZettaVolumeConnectionConfig()
index_config = ClickZettaVolumeIndexerConfig(
    volume="your-volume",
    remote_path="path/to/files/",
    regexp=".*\\.pdf$"  # 只匹配PDF文件
)

indexer = ClickZettaVolumeIndexer(
    connection_config=connection_config,
    index_config=index_config
)
files = indexer.list_files()
```

### DashScope嵌入器示例

```python
from unstructured_ingest.embed.dashscope import (
    DashScopeEmbeddingConfig,
    DashScopeEmbeddingEncoder
)

# 配置DashScope嵌入器
config = DashScopeEmbeddingConfig(
    api_key="your-dashscope-api-key",
    model_name="text-embedding-v4",  # 支持v1/v2/v4
    max_retries=3,
    retry_delay=1.0,
    enable_debug_logging=True
)

# 创建嵌入器
encoder = DashScopeEmbeddingEncoder(config)

# 嵌入文档
elements = [{"text": "ClickZetta是云原生数据湖仓平台"}]
embedded_elements = encoder.embed_documents(elements)

# 查看统计
stats = encoder.get_stats()
print(f"成功率: {stats['success_rate_percent']}%")
```

### 多湖仓知识库构建示例

```bash
# 进入知识库构建目录
cd multi_lakehouse_kb_builder

# 快速部署到所有Lakehouse
./run_with_current_env.sh deploy

# 验证部署结果
python validate_kb_simple.py

# 管理知识库内容
python manage_knowledge_simple.py
```

## 📊 Jupyter Notebook使用示例

### 运行完整的ETL流程

```bash
# 启动Jupyter Notebook
jupyter notebook

# 打开示例notebook
# examples/notebooks/Unstructured_data_ETL_from_local_to_Lakehouse_tongyi.ipynb
```

### Notebook功能亮点

1. **环境准备**：
   ```python
   # 自动切换到本地开发版本
   !pip install -e /path/to/unstructured-ingest-clickzetta/

   # 验证DashScope支持
   from unstructured_ingest.processes.embedder import EmbedderConfig
   ```

2. **DashScope配置**：
   ```python
   # 配置DashScope text-embedding-v4
   embedding_provider = "dashscope"
   embedding_model_name = "text-embedding-v4"
   embeddings_dimensions = 1024
   api_key = os.getenv("DASHSCOPE_API_KEY")
   ```

3. **表结构创建**：
   ```python
   # 自动创建Raw表和Silver表
   # 包含向量索引和倒排索引
   INDEX embeddings_vec_index USING vector properties (
       "scalar.type" = "f32",
       "distance.function" = "cosine_distance"
   )
   ```

4. **Pipeline执行**：
   ```python
   # 使用DashScope嵌入器的完整Pipeline
   pipeline = Pipeline.from_configs(
       embedder_config=EmbedderConfig(
           embedding_provider="dashscope",
           embedding_model_name="text-embedding-v4",
           embedding_api_key=api_key,
       ),
       # ... 其他配置
   )
   ```

5. **RAG检索演示**：
   ```python
   # 向量相似度搜索
   query_text = "创建索引的语法是什么？"
   results = retrieve_documents(conn, query_text)

   # 动态添加知识库内容
   kb = "ClickZetta是云器、Singdata的技术品牌..."
   embedded_kb = get_embedding(kb)
   ```

## 📋 环境变量配置

支持多种命名约定的环境变量：

```bash
# ClickZetta连接配置（支持CLICKZETTA_*、CZ_*、cz_*前缀）
CLICKZETTA_USERNAME=your-username
CLICKZETTA_PASSWORD=your-password
CLICKZETTA_SERVICE=your-service-url
CLICKZETTA_INSTANCE=your-instance
CLICKZETTA_WORKSPACE=your-workspace
CLICKZETTA_SCHEMA=your-schema
CLICKZETTA_VCLUSTER=your-vcluster

# OpenAI API配置（支持自定义base_url）
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=your-custom-endpoint  # 可选，支持通义千问等

# DashScope API配置
DASHSCOPE_API_KEY=your-dashscope-api-key  # 阿里云通义千问API密钥
```

## 🧪 测试

```bash
# 运行ClickZetta连接器测试
pytest test/integration/connectors/sql/test_clickzetta.py

# 运行所有集成测试
pytest test/integration/

# 测试DashScope嵌入功能
python -c "
from unstructured_ingest.embed.dashscope import DashScopeEmbeddingConfig, DashScopeEmbeddingEncoder
config = DashScopeEmbeddingConfig(api_key='your-key', model_name='text-embedding-v4')
encoder = DashScopeEmbeddingEncoder(config)
result = encoder.embed_query('测试文本')
print(f'嵌入维度: {len(result)}')
"

# 运行多湖仓知识库构建测试
cd multi_lakehouse_kb_builder && python test_kb_deployment.py
```

## 📚 与上游项目的关系

本项目基于官方 [Unstructured-IO/unstructured-ingest](https://github.com/Unstructured-IO/unstructured-ingest) 项目：

- **上游兼容**：定期同步上游更新，保持与最新版本的兼容性
- **功能扩展**：在保持原有功能的基础上，专门针对ClickZetta平台进行深度集成
- **企业优化**：针对企业级使用场景进行性能和稳定性优化

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目遵循与上游项目相同的开源许可证。
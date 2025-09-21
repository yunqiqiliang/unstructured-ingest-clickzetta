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

### 步骤1：环境准备和验证

```python
# 1. 安装本地开发版本
!pip uninstall unstructured-ingest -y -q
!pip install -e /path/to/unstructured-ingest-clickzetta/ -q

# 2. 验证DashScope支持
from unstructured_ingest.processes.embedder import EmbedderConfig
test_config = EmbedderConfig(
    embedding_provider="dashscope",
    embedding_model_name="text-embedding-v4",
    embedding_api_key="test"
)
print("✅ DashScope 支持已成功添加")
```

### 步骤2：配置环境变量和参数

```python
import os
import dotenv

# 加载环境变量
dotenv.load_dotenv('.env')

# DashScope配置
api_key = os.getenv("DASHSCOPE_API_KEY")
embedding_provider = "dashscope"
embedding_model_name = "text-embedding-v4"
embeddings_dimensions = 1024

# ClickZetta连接参数
_username = os.getenv("cz_username")
_password = os.getenv("cz_password")
_service = os.getenv("cz_service")
_instance = os.getenv("cz_instance")
_workspace = os.getenv("cz_workspace")
_schema = os.getenv("cz_schema")
_vcluster = os.getenv("cz_vcluster")

# 表名配置
index_and_table_prefix = "dashscope_v4_1024_2048_20250611_"
raw_table_name = f"{index_and_table_prefix}yunqi_raw_elements"
silver_table_name = f"{index_and_table_prefix}yunqi_elements"
```

### 步骤3：创建数据库连接和表结构

```python
from clickzetta.connector import connect

# 创建连接函数
def get_connection(password, username, service, instance, workspace, schema, vcluster):
    return connect(
        password=password, username=username, service=service,
        instance=instance, workspace=workspace, schema=schema, vcluster=vcluster
    )

# 建立连接
conn = get_connection(_password, _username, _service, _instance, _workspace, _schema, _vcluster)

# 执行SQL的工具函数
def execute_sql(conn, sql_statement: str):
    with conn.cursor() as cur:
        cur.execute(sql_statement)
        return cur.fetchall()

# 创建Raw表和Silver表（包含向量索引）
execute_sql(conn, raw_table_ddl)  # 详见notebook中的完整DDL
execute_sql(conn, silver_table_ddl)  # 包含倒排索引和向量索引
```

### 步骤4：配置并运行ETL Pipeline

```python
from unstructured_ingest.interfaces import ProcessorConfig
from unstructured_ingest.pipeline.pipeline import Pipeline
from unstructured_ingest.processes.chunker import ChunkerConfig
from unstructured_ingest.processes.connectors.local import (
    LocalIndexerConfig, LocalDownloaderConfig, LocalConnectionConfig
)
from unstructured_ingest.processes.embedder import EmbedderConfig
from unstructured_ingest.processes.partitioner import PartitionerConfig
from unstructured_ingest.processes.connectors.sql.clickzetta import (
    ClickzettaConnectionConfig, ClickzettaAccessConfig,
    ClickzettaUploadStagerConfig, ClickzettaUploaderConfig
)

# 创建Pipeline
pipeline = Pipeline.from_configs(
    context=ProcessorConfig(verbose=False, tqdm=False, num_processes=2),

    # 本地文件输入
    indexer_config=LocalIndexerConfig(
        input_path=os.getenv("LOCAL_FILE_INPUT_DIR"),
        file_glob="**/*",
        recursive=True
    ),
    downloader_config=LocalDownloaderConfig(),
    source_connection_config=LocalConnectionConfig(),

    # 文档解析配置
    partitioner_config=PartitionerConfig(
        partition_by_api=False,
        strategy="hi_res",
        additional_partition_args={
            "split_pdf_page": True,
            "split_pdf_allow_failed": True,
            "split_pdf_concurrency_level": 1
        }
    ),

    # 文档分块配置
    chunker_config=ChunkerConfig(
        chunking_strategy="by_title",
        chunk_max_characters=2048,
        chunk_overlap=512,
        chunk_combine_text_under_n_chars=200,
    ),

    # DashScope嵌入配置
    embedder_config=EmbedderConfig(
        embedding_provider="dashscope",
        embedding_model_name="text-embedding-v4",
        embedding_api_key=api_key,
    ),

    # ClickZetta目标配置
    destination_connection_config=ClickzettaConnectionConfig(
        access_config=ClickzettaAccessConfig(password=_password),
        username=_username, service=_service, instance=_instance,
        workspace=_workspace, schema=_schema, vcluster=_vcluster,
    ),
    stager_config=ClickzettaUploadStagerConfig(),
    uploader_config=ClickzettaUploaderConfig(
        table_name=raw_table_name,
        documents_original_source="https://yunqi.tech/documents"
    ),
)

# 运行Pipeline
print("🚀 运行 Pipeline...")
pipeline.run()
```

### 步骤5：数据转换和清洗

```python
# 从Raw表转换数据到Silver表
clean_transformation_sql = f"""
INSERT overwrite {_schema}.{silver_table_name}
SELECT
    id, record_locator, type, record_id, element_id, filetype,
    file_directory, filename, last_modified, languages, page_number, text,
    CAST(embeddings AS VECTOR({embeddings_dimensions})) AS embeddings,
    parent_id, is_continuation, orig_elements, element_type, coordinates,
    link_texts, link_urls, email_message_id, sent_from, sent_to, subject,
    url, version, date_created, date_modified, date_processed, text_as_html,
    emphasized_text_contents, emphasized_text_tags,
    "https://yunqi.tech/documents" as documents_source
FROM {_schema}.{raw_table_name};
"""

execute_sql(conn, clean_transformation_sql)
print("✅ 数据转换完成")
```

### 步骤6：RAG检索和知识库管理

```python
import dashscope
from dashscope import TextEmbedding
import pandas as pd

# 设置DashScope API
dashscope.api_key = api_key

def get_embedding(query):
    """使用DashScope获取嵌入"""
    response = TextEmbedding.call(model="text-embedding-v4", input=query)
    if response.status_code == 200:
        return response.output['embeddings'][0]['embedding']
    else:
        raise Exception(f"DashScope API error: {response.message}")

def retrieve_documents(conn, query: str, num_results: int = 10):
    """向量相似度搜索"""
    embedding = get_embedding(query)

    with conn.cursor() as cur:
        stmt = f"""
            SELECT "vector_embedding" as retrieve_method, record_locator, type,
                   filename, text, orig_elements,
                   cosine_distance(embeddings, cast({embedding} as vector({embeddings_dimensions}))) AS score
            FROM {silver_table_name}
            ORDER BY score ASC LIMIT {num_results}
        """
        cur.execute(stmt)
        results = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return pd.DataFrame(results, columns=columns)

# 示例：搜索相关文档
query_text = "创建索引的语法是什么？"
results_df = retrieve_documents(conn, query_text)
print(f"找到 {len(results_df)} 个相关文档")

# 添加自定义知识
kb_text = "ClickZetta是云器、Singdata的技术品牌..."
embedded_kb = get_embedding(kb_text)
add_kb_sql = f"""
INSERT INTO {_schema}.{silver_table_name} (
  id, type, record_id, element_id, filetype, last_modified, languages,
  text, embeddings, date_created, date_modified, date_processed
) VALUES (
  uuid(), 'UserInput', uuid(), uuid(), 'text', CURRENT_TIMESTAMP, '["zh-cn"]',
  '{kb_text}', CAST('{embedded_kb}' AS vector(float,{embeddings_dimensions})),
  CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
);
"""
execute_sql(conn, add_kb_sql)
print("✅ 知识库内容添加完成")
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
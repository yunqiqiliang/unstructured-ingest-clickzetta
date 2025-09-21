# unstructured-ingest-clickzetta

[![PyPI version](https://badge.fury.io/py/unstructured-ingest-clickzetta.svg)](https://badge.fury.io/py/unstructured-ingest-clickzetta)
[![Python](https://img.shields.io/pypi/pyversions/unstructured-ingest-clickzetta.svg)](https://pypi.org/project/unstructured-ingest-clickzetta/)

**基于 [Unstructured Ingest](https://github.com/Unstructured-IO/unstructured-ingest) 的 [ClickZetta](https://www.yunqi.tech/) 连接器扩展包**

这是一个为 [ClickZetta 数据湖仓平台](https://www.yunqi.tech/documents/) 定制的 Unstructured 数据ETL处理工具包，提供完整的文档解析、向量化和存储以及检索解决方案。

## 🎯 项目定位

### 📦 这是什么？

`unstructured-ingest-clickzetta` 是一个 **Python 包**，它：

1. **扩展了 Unstructured 生态系统**：在原有 70+ 连接器基础上，新增 [ClickZetta](https://www.yunqi.tech/) 专用连接器
2. **提供完整的 CLI 工具**：`unstructured-ingest-clickzetta` 命令，支持 [ClickZetta](https://www.yunqi.tech/) SQL 和 Volume 操作
3. **包含向量嵌入能力**：集成阿里云 DashScope API，支持更优的中文文档向量化
4. **附带企业级工具集**：多湖仓知识库部署和管理系统

### 🔧 核心能力

#### 1️⃣ **[ClickZetta](https://www.yunqi.tech/) SQL 连接器** (`clickzetta`)
- 将文档解析后存储到 [ClickZetta](https://www.yunqi.tech/) 数据库表
- 支持向量嵌入和 RAG 检索系统构建
- 批量处理优化，适合大规模数据迁移

#### 2️⃣ **[ClickZetta](https://www.yunqi.tech/) Volume 连接器** (`clickzetta-volume`)
- 处理 [ClickZetta](https://www.yunqi.tech/) Volume 存储系统中的文件
- 支持用户卷、表卷、命名卷等多种类型
- 兼容 S3 协议，标准化文件操作

#### 3️⃣ **DashScope 嵌入器** (`dashscope`)
- 阿里云通义千问 API 集成
- 支持 text-embedding-v1/v2/v4 多种模型
- 智能重试和批量处理优化

#### 4️⃣ **企业级工具集** (`multi_lakehouse_kb_builder`)
- 交互式多湖仓知识库部署系统
- 批量操作、健康监控、内容管理
- 自动化部署脚本和验证工具

## 🚀 使用场景

### 💻 场景1：独立CLI工具

**适用于**：DevOps、数据工程师、自动化脚本

特点：开箱即用，无需额外文件，适合生产环境和CI/CD集成

```bash
# 安装后直接使用
pip install unstructured-ingest-clickzetta

# 处理文档到 ClickZetta 表
unstructured-ingest-clickzetta clickzetta \
  --table-name "documents" \
  --local-input-path "/docs"

# 处理 Volume 中的文件
unstructured-ingest-clickzetta clickzetta-volume \
  --volume "data-lake" \
  --remote-path "raw-docs/"
```

### 🔧 场景2：Python开发集成

**适用于**：开发者、应用集成、功能扩展

特点：提供完整的Python API，支持二次开发和定制功能

```python
# 在应用中集成 ClickZetta 连接器
from unstructured_ingest.processes.connectors.sql.clickzetta import ClickzettaConnectionConfig
from unstructured_ingest.embed.dashscope import DashScopeEmbeddingEncoder
from unstructured_ingest.pipeline.pipeline import Pipeline

# 构建自定义数据处理流程
pipeline = Pipeline.from_configs(
    # ... 配置参数
)
```

扩展开发场景：
- 🔌 扩展连接器：开发新的数据源连接器
- 🧠 自定义嵌入器：集成其他向量嵌入服务
- 🔄 定制Pipeline：修改数据处理流程
- 📊 增强功能：添加数据质量检查、格式转换等

### 🏢 场景3：企业级知识库部署

**适用于**：企业用户、批量部署、运维管理

特点：交互式管理系统，支持多实例批量操作和健康监控

📖 **详细文档** → [multi_lakehouse_kb_builder/README.md](./multi_lakehouse_kb_builder/README.md)

```bash
# 获取企业级工具集
git clone https://github.com/yunqiqiliang/unstructured-ingest-clickzetta.git
cd unstructured-ingest-clickzetta

# 启动交互式管理系统
./multi_lakehouse_kb_builder/run.sh
```

---

## ⚡ 快速开始

### 📦 安装

```bash
# 安装主包（自动包含 ClickZetta 依赖）
pip install unstructured-ingest-clickzetta

# 安装可选依赖（用于向量嵌入）
pip install dashscope pandas
```

### 🔧 基础使用

#### 环境变量配置
```bash
# ClickZetta连接配置
export CLICKZETTA_USERNAME="your-username"
export CLICKZETTA_PASSWORD="your-password"
export CLICKZETTA_SERVICE="your-service-url"
export CLICKZETTA_INSTANCE="your-instance"
export CLICKZETTA_WORKSPACE="your-workspace"
export CLICKZETTA_SCHEMA="your-schema"
export CLICKZETTA_VCLUSTER="your-vcluster"

# API密钥配置
export DASHSCOPE_API_KEY="your-dashscope-key"
```

#### CLI方式示例
```bash
# 基础文档处理
unstructured-ingest-clickzetta clickzetta \
  --table-name "my_documents" \
  --local-input-path "/path/to/documents"

# 包含向量嵌入的处理
unstructured-ingest-clickzetta clickzetta \
  --table-name "knowledge_base" \
  --local-input-path "/docs" \
  --embedding-provider "dashscope" \
  --embedding-model-name "text-embedding-v4"

# ClickZetta Volume文件处理
unstructured-ingest-clickzetta clickzetta-volume \
  --volume "data-lake" \
  --remote-path "documents/" \
  --regexp ".*\\.pdf$"
```

#### Python API示例
```python
from unstructured_ingest.pipeline.pipeline import Pipeline
from unstructured_ingest.processes.connectors.sql.clickzetta import (
    ClickzettaConnectionConfig, ClickzettaAccessConfig, ClickzettaUploaderConfig
)
from unstructured_ingest.processes.embedder import EmbedderConfig

# 创建处理流水线
pipeline = Pipeline.from_configs(
    # ... 完整配置见详细文档
    destination_connection_config=ClickzettaConnectionConfig(...),
    uploader_config=ClickzettaUploaderConfig(table_name="documents"),
    embedder_config=EmbedderConfig(
        embedding_provider="dashscope",
        embedding_model_name="text-embedding-v4"
    )
)

# 运行流水线
pipeline.run()
```

## 🚀 详细功能特性

### 相对于上游项目新增的功能

#### 1. ClickZetta SQL连接器 (`clickzetta`)
- **完整的数据湖仓集成**：支持从ClickZetta数据库表读取和写入非结构化数据
- **智能批量处理**：自动优化的批量上传，支持大规模数据处理
- **向量化支持**：原生支持向量嵌入存储，兼容多种向量维度（512/768/1024/1536）
- **中文优化**：针对中文环境优化的错误处理和日志提示
- **连接池管理**：智能的数据库连接和会话管理

#### 2. ClickZetta Volume连接器 (`clickzetta_volume`)
- **云原生存储**：支持ClickZetta Volume存储系统的文件操作
- **多卷类型支持**：
  - **用户卷**：`volume:user://~/path` - 个人存储空间
  - **表卷**：`volume:table://table_name/path` - 表关联存储
  - **命名卷**：`volume://volume_name/path` - 自定义命名卷
- **高级文件操作**：上传、下载、删除、列举、正则过滤
- **智能路径解析**：自动处理复杂的Volume URL格式
- **S3兼容协议**：使用标准S3/S3A协议，确保兼容性
- **环境变量集成**：支持CLICKZETTA_*、CZ_*、cz_*多种前缀

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

#### 4. DashScope嵌入器 (`dashscope`)
- **阿里云通义千问集成**：完整支持DashScope TextEmbedding API
- **多模型支持**：
  - **text-embedding-v1**：512维向量，基础模型
  - **text-embedding-v2**：1536维向量，增强模型
  - **text-embedding-v4**：1024维向量，最新优化模型（推荐）
- **智能重试机制**：指数退避重试策略，自动处理API限流
- **批量处理优化**：
  - 单文档嵌入：`embed_query(text)`
  - 批量文档嵌入：`embed_documents(elements)`
  - 自动批量分组，提升API调用效率
- **统计监控**：实时API调用统计、成功率监控、错误追踪
- **配置灵活**：支持自定义重试次数、超时时间、调试日志

#### 5. 多湖仓知识库构建系统 (`multi_lakehouse_kb_builder`) ⭐
- **🚀 智能启动脚本**：`./multi_lakehouse_kb_builder/run.sh` 一键启动，自动环境检测
- **🎛️ 交互式操作界面**：友好的菜单系统，支持所有功能操作
- **📦 批量部署**：支持一键部署到多个ClickZetta Lakehouse实例
- **🧠 智能表管理**：自动创建schema、管理Raw表和Silver表结构
- **⚡ 并行/串行执行**：支持两种部署模式，适应不同性能需求
- **🔍 数据验证**：自动验证部署结果，检测向量质量问题
- **🏥 健康检查**：连接状态诊断和知识库健康评估
- **📚 知识管理**：支持添加、删除、搜索自定义知识条目

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

## 📦 安装方式

### 方式1：PyPI安装（推荐生产使用）

```bash
# 安装主包
pip install unstructured-ingest-clickzetta

# 安装其他依赖包（ClickZetta包已自动安装）
pip install dashscope pandas

# 验证安装
unstructured-ingest-clickzetta --help
unstructured-ingest-clickzetta clickzetta --help
unstructured-ingest-clickzetta clickzetta-volume --help
```

### 方式2：源码安装（开发使用）

```bash
git clone https://github.com/yunqiqiliang/unstructured-ingest-clickzetta.git
cd unstructured-ingest-clickzetta
pip install -e .

# 安装开发依赖
pip install -r requirements/connectors/clickzetta.txt
pip install -r requirements/embed/dashscope.txt
```

### PyPI包信息

- **包名**：`unstructured-ingest-clickzetta`
- **当前版本**：`1.2.18.dev1`
- **PyPI页面**：https://pypi.org/project/unstructured-ingest-clickzetta/
- **CLI命令**：
  - `unstructured-ingest-clickzetta` (主命令)
  - `unstructured-ingest` (兼容原版)

## 📋 使用指南

### CLI高级参数

```bash
# PDF文档处理
unstructured-ingest-clickzetta clickzetta \
  --table-name "pdfs" \
  --local-input-path "/pdfs" \
  --strategy "hi_res" \
  --split-pdf-page

# 向量化知识库构建
unstructured-ingest-clickzetta clickzetta \
  --table-name "kb_vectors" \
  --local-input-path "/knowledge" \
  --embedding-provider "dashscope" \
  --embedding-model-name "text-embedding-v4" \
  --chunking-strategy "by_title" \
  --chunk-max-characters 2048
```

### Python开发环境设置

```bash
# 获取源码进行二次开发
git clone https://github.com/yunqiqiliang/unstructured-ingest-clickzetta.git
cd unstructured-ingest-clickzetta
pip install -e .

# 验证开发环境
python -c "from unstructured_ingest.processes.connectors.sql.clickzetta import ClickzettaConnectionConfig; print('开发环境就绪')"
```

---

## 📊 Jupyter Notebook使用指南

### 主要步骤概览

1. **环境准备和配置**：安装依赖包，设置环境变量
2. **数据库连接**：建立ClickZetta连接，创建Raw表和Silver表
3. **Pipeline配置**：配置完整的ETL流水线（文档解析+向量化+存储）
4. **数据转换**：从Raw表清洗数据到Silver表
5. **RAG检索**：实现向量相似度搜索和知识库管理

详细代码请参考Notebook文件：`examples/notebooks/Unstructured_data_ETL_from_local_to_Lakehouse_tongyi.ipynb`

### 使用示例代码

#### ClickZetta Volume连接器
```python
from unstructured_ingest.processes.connectors.fsspec.clickzetta_volume import (
    ClickZettaVolumeConnectionConfig, ClickZettaVolumeIndexer
)

# 列举卷中的PDF文件
indexer = ClickZettaVolumeIndexer(
    connection_config=ClickZettaVolumeConnectionConfig(),
    index_config=ClickZettaVolumeIndexerConfig(
        volume="your-volume",
        remote_path="path/to/files/",
        regexp=".*\\.pdf$"
    )
)
files = indexer.list_files()
```

#### DashScope嵌入器
```python
from unstructured_ingest.embed.dashscope import DashScopeEmbeddingEncoder

# 文档向量化
encoder = DashScopeEmbeddingEncoder(config)
elements = [{"text": "ClickZetta是云原生数据湖仓平台"}]
embedded_elements = encoder.embed_documents(elements)
```

#### 企业级工具集
```bash
# 批量部署和管理
cd multi_lakehouse_kb_builder
./run.sh deploy
python validate_kb_simple.py
```

### 运行示例Notebook

```bash
# 启动Jupyter
jupyter notebook

# 打开示例文件
# examples/notebooks/Unstructured_data_ETL_from_local_to_Lakehouse_tongyi.ipynb
```

### 核心功能演示

1. **环境准备和DashScope配置**
2. **ClickZetta表结构创建（包含向量索引）**
3. **完整ETL Pipeline执行**
4. **RAG检索和知识库管理**

## 📋 环境变量配置

```bash
# ClickZetta连接配置（支持CLICKZETTA_*、CZ_*、cz_*前缀）
CLICKZETTA_USERNAME=your-username
CLICKZETTA_PASSWORD=your-password
CLICKZETTA_SERVICE=your-service-url
CLICKZETTA_INSTANCE=your-instance
CLICKZETTA_WORKSPACE=your-workspace
CLICKZETTA_SCHEMA=your-schema
CLICKZETTA_VCLUSTER=your-vcluster

# API密钥配置
DASHSCOPE_API_KEY=your-dashscope-api-key
OPENAI_API_KEY=your-api-key  # 可选
OPENAI_BASE_URL=your-custom-endpoint  # 可选，支持通义千问等
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

## 📚 参考文档

### [ClickZetta](https://www.yunqi.tech/) 官方资源
- **[ClickZetta 官网](https://www.yunqi.tech/)** - 产品介绍和解决方案
- **[ClickZetta 文档中心](https://www.yunqi.tech/documents/)** - 完整的技术文档

### 技术文档参考
- **[Unstructured 官方文档](https://docs.unstructured.io/)** - 上游项目文档
- **[DashScope API 文档](https://help.aliyun.com/zh/dashscope/)** - 阿里云通义千问 API
- **[PyPI 项目页面](https://pypi.org/project/unstructured-ingest-clickzetta/)** - 包发布信息

### 开源代码仓库
- **[GitHub 仓库](https://github.com/yunqiqiliang/unstructured-ingest-clickzetta)** - 本项目源码
- **[上游项目仓库](https://github.com/Unstructured-IO/unstructured-ingest)** - Unstructured Ingest 官方仓库
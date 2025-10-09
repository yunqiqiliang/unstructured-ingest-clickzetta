# unstructured-ingest-clickzetta

[![PyPI version](https://badge.fury.io/py/unstructured-ingest-clickzetta.svg)](https://badge.fury.io/py/unstructured-ingest-clickzetta)
[![Python](https://img.shields.io/pypi/pyversions/unstructured-ingest-clickzetta.svg)](https://pypi.org/project/unstructured-ingest-clickzetta/)

**A [ClickZetta](https://www.yunqi.tech/) connector extension built on top of [Unstructured Ingest](https://github.com/Unstructured-IO/unstructured-ingest).**

This package tailors the Unstructured ETL toolchain for the [ClickZetta Lakehouse Platform](https://www.yunqi.tech/documents/), delivering an end-to-end solution for document parsing, vectorization, storage and retrieval.

---

## 🎯 Project Scope

### 📦 What Is It?

`unstructured-ingest-clickzetta` is a **Python package** that:

1. **Extends the Unstructured ecosystem** – adds ClickZetta-specific connectors on top of the 70+ upstream connectors.
2. **Ships with a complete CLI** – the `unstructured-ingest-clickzetta` command supports both ClickZetta SQL and Volume operations.
3. **Includes embedding capabilities** – integrates Alibaba Cloud DashScope (Tongyi Qianwen) for high-quality Chinese embeddings.
4. **Provides enterprise tooling** – bundles a multi-lakehouse knowledge base deployment and management toolkit.

### 🔧 Core Capabilities

#### 1️⃣ ClickZetta SQL Connector (`clickzetta`)
- Stores parsed documents into ClickZetta database tables.
- Offers native vector embedding support for RAG-style retrieval.
- Optimized for high-throughput batch migration.

#### 2️⃣ ClickZetta Volume Connector (`clickzetta-volume`)
- Operates on files stored in ClickZetta Volume (user, table and named volumes).
- Supports upload, download, delete, listing and regex filtering.
- Uses an S3-compatible protocol for consistent file operations.

#### 3️⃣ DashScope Embedder (`dashscope`)
- Integrates Tongyi Qianwen text embedding APIs.
- Supports `text-embedding-v1/v2/v3/v4`.
- Built-in retries, batching and usage statistics.

#### 4️⃣ Enterprise Toolkit (`multi_lakehouse_kb_builder`)
- Interactive multi-lakehouse knowledge base deployment console.
- Bulk operations, health monitoring and content management.
- Automation scripts for deployment, validation and synchronization.

#### 5️⃣ Jupyter Notebooks (`examples/notebooks/`)
- End-to-end ETL walkthroughs from local documents to ClickZetta.
- Demonstrates DashScope integration, table creation and RAG retrieval.
- Includes upstream Databricks Delta table examples for reference.

#### 6️⃣ Development & Testing Tools
- Comprehensive integration tests covering SQL and Volume connectors.
- Dockerized deployment flows and CI/CD friendly scripts.

---

## 🚀 Usage Scenarios

### 🔧 Scenario 1 · Python Integration

**Target users:** developers, solution engineers, teams extending existing apps.

Provides complete Python APIs for customization:

```python
# Use the ClickZetta SQL connector inside an application
from unstructured_ingest.processes.connectors.sql.clickzetta import ClickzettaConnectionConfig
from unstructured_ingest.embed.dashscope import DashScopeEmbeddingEncoder
from unstructured_ingest.pipeline.pipeline import Pipeline

pipeline = Pipeline.from_configs(
    # ... full configuration in documentation
)

# Work with ClickZetta Volume
from unstructured_ingest.processes.connectors.fsspec.clickzetta_volume import (
    ClickzettaVolumeConnectionConfig, ClickzettaVolumeIndexer, ClickzettaVolumeIndexerConfig
)

connection_config = ClickzettaVolumeConnectionConfig()
indexer = ClickzettaVolumeIndexer(
    connection_config=connection_config,
    index_config=ClickzettaVolumeIndexerConfig(
        volume="your-volume",
        remote_path="documents/",
        regexp=".*\\.pdf$"
    )
)
files = indexer.list_files()

# Use DashScope embeddings
from unstructured_ingest.embed.dashscope import (
    DashScopeEmbeddingConfig, DashScopeEmbeddingEncoder,
)

config = DashScopeEmbeddingConfig(
    api_key="your-dashscope-api-key",
    model_name="text-embedding-v4",
    max_retries=3,
    retry_delay=1.0,
)
encoder = DashScopeEmbeddingEncoder(config)

result = encoder.embed_query("ClickZetta is a cloud-native lakehouse platform.")
elements = [{"text": "Document 1"}, {"text": "Document 2"}]
embedded_elements = encoder.embed_documents(elements)
```

### 💻 Scenario 2 · Stand-alone CLI

**Target users:** DevOps, data engineers, automation workloads.

Ready-to-use command line interface:

```bash
# Install and run
pip install unstructured-ingest-clickzetta

# Ingest local documents into a ClickZetta table
unstructured-ingest-clickzetta clickzetta \
  --table-name "documents" \
  --local-input-path "/docs"

# Process files in a ClickZetta volume
unstructured-ingest-clickzetta clickzetta-volume \
  --volume-type "named" \
  --volume-name "data-lake" \
  --remote-path "raw-docs/"
```

### 🏢 Scenario 3 · Enterprise Knowledge Base Deployment

**Target users:** enterprise platform teams, large-scale operations, managed services.

Interactive management console bundled in `multi_lakehouse_kb_builder`:

```bash
git clone https://github.com/yunqiqiliang/unstructured-ingest-clickzetta.git
cd unstructured-ingest-clickzetta

# Launch the interactive management workflow
./multi_lakehouse_kb_builder/run.sh
```

Full documentation: [multi_lakehouse_kb_builder/README.md](./multi_lakehouse_kb_builder/README.md)

---

## ⚡ Quick Start

### 📦 Installation

```bash
# Install the core package (ClickZetta dependencies included)
pip install unstructured-ingest-clickzetta

# Optional: install embedding helpers
pip install dashscope pandas
```

### 🔧 Configure Environment Variables

```bash
# ClickZetta connection (supports CLICKZETTA_*, CZ_* and cz_* prefixes)
export CLICKZETTA_USERNAME="your-username"
export CLICKZETTA_PASSWORD="your-password"
export CLICKZETTA_SERVICE="your-service-url"
export CLICKZETTA_INSTANCE="your-instance"
export CLICKZETTA_WORKSPACE="your-workspace"
export CLICKZETTA_SCHEMA="your-schema"
export CLICKZETTA_VCLUSTER="your-vcluster"

# API keys
export DASHSCOPE_API_KEY="your-dashscope-key"
export OPENAI_API_KEY="your-openai-key"          # optional
export OPENAI_BASE_URL="https://your-endpoint"    # optional, compatible with Tongyi Qianwen
```

### 🚀 Run with the CLI

```bash
# Basic ingestion
unstructured-ingest-clickzetta clickzetta \
  --table-name "my_documents" \
  --local-input-path "/path/to/documents"

# With embeddings enabled
unstructured-ingest-clickzetta clickzetta \
  --table-name "knowledge_base" \
  --local-input-path "/docs" \
  --embedding-provider "dashscope" \
  --embedding-model-name "text-embedding-v4"

# ClickZetta Volume operations
unstructured-ingest-clickzetta clickzetta-volume \
  --volume-type "named" \
  --volume-name "data-lake" \
  --remote-path "documents/" \
  --regexp ".*\\.pdf$"
```

### 🧑‍💻 Python API

```python
from unstructured_ingest.pipeline.pipeline import Pipeline
from unstructured_ingest.processes.connectors.sql.clickzetta import (
    ClickzettaConnectionConfig,
    ClickzettaAccessConfig,
    ClickzettaUploaderConfig,
)
from unstructured_ingest.processes.embedder import EmbedderConfig

pipeline = Pipeline.from_configs(
    source_config=...,  # see documentation for full options
    destination_connection_config=ClickzettaConnectionConfig(...),
    access_config=ClickzettaAccessConfig(...),
    uploader_config=ClickzettaUploaderConfig(table_name="documents"),
    embedder_config=EmbedderConfig(
        embedding_provider="dashscope",
        embedding_model_name="text-embedding-v4",
    ),
)

pipeline.run()
```

---

## 🚀 Feature Highlights vs. Upstream

### 1. ClickZetta SQL Connector (`clickzetta`)
- End-to-end lakehouse integration for reading/writing unstructured data.
- Intelligent batching to minimize connection overhead.
- Native embedding storage supporting vector sizes 512/768/1024/1536.
- Enhanced logging and error messages for Chinese environments.
- Connection pooling and session management tuned for ClickZetta.

### 2. ClickZetta Volume Connector (`clickzetta_volume`)
- Works with user volumes (`volume:user://`), table volumes (`volume:table://`) and named volumes (`volume://`).
- Smart path parsing for complex Volume URLs.
- Advanced file operations: upload, download, delete, list and regex filtering.
- S3/S3A compatible, so existing tooling continues to work.
- Reads environment variables with `CLICKZETTA_*`, `CZ_*` and `cz_*` prefixes.

### 3. Enterprise Enhancements
- Detailed Chinese-language diagnostics and troubleshooting hints.
- Performance improvements: batching, intelligent buffering and optimized serialization.
- Compatibility tweaks for third-party APIs (Tongyi Qianwen, OpenAI SSL tweaks, legacy config formats).

### 4. DashScope Embedder (`dashscope`)
- Full integration with DashScope TextEmbedding APIs.
- Model support:
  - `text-embedding-v1`: 512 dimensions, general-purpose.
  - `text-embedding-v2`: 1536 dimensions, advanced model.
  - `text-embedding-v3`: 1024 dimensions, performance optimized.
  - `text-embedding-v4`: 1024 dimensions, latest generation (recommended).
- Exponential backoff retries and automatic rate-limit handling.
- Single-document (`embed_query`) and batch (`embed_documents`) helpers.
- Real-time statistics tracking success rate and errors.
- Flexible configuration for retry counts, timeouts and debug logging.

### 5. Multi-Lakehouse Knowledge Base Builder (`multi_lakehouse_kb_builder`) ⭐
- 🚀 One-click bootstrap script: `./multi_lakehouse_kb_builder/run.sh`.
- 🎛️ Interactive menu-driven interface.
- 📦 Batch deployments across multiple ClickZetta lakehouse instances.
- 🧠 Automatic schema creation for raw and silver tables.
- ⚡ Parallel/serial execution modes for different workload profiles.
- 🔍 Deployment validation and vector quality checks.
- 🏥 Health diagnostics for connections and knowledge base status.
- 📚 Knowledge management: add, delete and search custom entries.

### 6. Notebook Walkthroughs (`examples/notebooks/`)
- `Unstructured_data_ETL_from_local_to_Lakehouse_tongyi.ipynb`
  - Complete pipeline from local docs to ClickZetta lakehouse.
  - DashScope text-embedding-v4 integration and vectorization.
  - Raw/Silver table creation and management.
  - Automatic inverted and vector index creation.
  - RAG retrieval and similarity search demos.
  - Dynamic knowledge base content management.
- `databricks_delta_tables.ipynb`
  - Upstream example for Databricks Delta tables.

---

## 📦 Installation Options

### ✅ PyPI (recommended for production)

```bash
pip install unstructured-ingest-clickzetta
pip install dashscope pandas          # optional helpers

# Verify installation
unstructured-ingest-clickzetta --help
unstructured-ingest-clickzetta clickzetta --help
unstructured-ingest-clickzetta clickzetta-volume --help
```

### 🛠️ From Source (for development)

```bash
git clone https://github.com/yunqiqiliang/unstructured-ingest-clickzetta.git
cd unstructured-ingest-clickzetta
pip install -e .

# Install connector-specific dependencies
pip install -r requirements/connectors/clickzetta.txt
pip install -r requirements/embed/dashscope.txt
```

### 🧰 Troubleshooting

If you encounter errors such as:

```
'dashscope' is not a valid choice for embedding_provider
No module named 'unstructured_ingest.processes.connectors.fsspec.clickzetta_volume'
```

**Likely cause:** a conflicting installation of the upstream `unstructured-ingest` package.

**Fix (automatic):**

```bash
python fix_dependencies.py
```

**Fix (manual):**

```bash
pip uninstall unstructured-ingest -y
pip install -e .                       # development version
# or
pip install unstructured-ingest-clickzetta
```

Verify the fix:

```python
from unstructured_ingest.processes.embedder import EmbedderConfig
config = EmbedderConfig(embedding_provider="dashscope")  # should succeed
```

### 📦 PyPI Metadata

- **Package name:** `unstructured-ingest-clickzetta`
- **Current version:** `1.3.1`
- **PyPI page:** https://pypi.org/project/unstructured-ingest-clickzetta/
- **CLI commands:** `unstructured-ingest-clickzetta` (primary) and `unstructured-ingest` (upstream compatible alias)

---

## 📋 Usage Guide

### CLI Advanced Parameters

```bash
# High-resolution PDF processing
unstructured-ingest-clickzetta clickzetta \
  --table-name "pdfs" \
  --local-input-path "/pdfs" \
  --strategy "hi_res" \
  --additional-partition-args '{"split_pdf_page": true}'

# Build a vectorized knowledge base
unstructured-ingest-clickzetta clickzetta \
  --table-name "kb_vectors" \
  --local-input-path "/knowledge" \
  --embedding-provider "dashscope" \
  --embedding-model-name "text-embedding-v4" \
  --chunking-strategy "by_title" \
  --chunk-max-characters 2048
```

### Developer Environment

```bash
git clone https://github.com/yunqiqiliang/unstructured-ingest-clickzetta.git
cd unstructured-ingest-clickzetta
pip install -e .

python -c "from unstructured_ingest.processes.connectors.sql.clickzetta import ClickzettaConnectionConfig; print('Development environment ready')"
```

---

## 📊 Notebook Guide

### Workflow Overview

1. Environment setup and DashScope configuration.
2. Connect to ClickZetta; create raw and silver tables.
3. Configure end-to-end pipelines (parsing + embeddings + storage).
4. Cleanse raw data into silver tables.
5. Implement RAG search and knowledge base management.

### Quick Commands

```bash
# Start Jupyter
jupyter notebook

# Open:
# examples/notebooks/Unstructured_data_ETL_from_local_to_Lakehouse_tongyi.ipynb
```

### Code Samples

```python
from unstructured_ingest.processes.connectors.fsspec.clickzetta_volume import (
    ClickzettaVolumeConnectionConfig, ClickzettaVolumeIndexer, ClickzettaVolumeIndexerConfig
)

indexer = ClickzettaVolumeIndexer(
    connection_config=ClickzettaVolumeConnectionConfig(),
    index_config=ClickzettaVolumeIndexerConfig(
        volume="your-volume",
        remote_path="path/to/files/",
        regexp=".*\\.pdf$",
    ),
)
files = indexer.list_files()
```

```python
from unstructured_ingest.embed.dashscope import DashScopeEmbeddingEncoder

encoder = DashScopeEmbeddingEncoder(config)
elements = [{"text": "ClickZetta is a cloud-native lakehouse platform."}]
embedded_elements = encoder.embed_documents(elements)
```

```bash
cd multi_lakehouse_kb_builder
./run.sh deploy
python validate_kb_simple.py
```

---

## 📦 Docker

Build images that include the CLI, enterprise tooling and optional Streamlit WebUI:

```bash
make build-docker-webui-local
make run-docker-webui-local
```

---

## 🧪 Testing

```bash
# SQL connector tests
pytest test/integration/connectors/sql/test_clickzetta.py

# Full integration suite
pytest test/integration/

# Validate DashScope embeddings
python - <<'PY'
from unstructured_ingest.embed.dashscope import DashScopeEmbeddingConfig, DashScopeEmbeddingEncoder
config = DashScopeEmbeddingConfig(api_key='your-key', model_name='text-embedding-v4')
encoder = DashScopeEmbeddingEncoder(config)
result = encoder.embed_query('Sample text')
print(f'Embedding dimensions: {len(result)}')
PY

# Enterprise toolkit checks
cd multi_lakehouse_kb_builder && python test_kb_deployment.py
```

---

## 🔗 Relationship with Upstream

This project builds on [Unstructured-IO/unstructured-ingest](https://github.com/Unstructured-IO/unstructured-ingest):

- **Upstream compatible** – regularly synced with upstream releases.
- **Feature extensions** – deep integration tailored for ClickZetta.
- **Enterprise enhancements** – performance and stability improvements for production workloads.

---

## 🤝 Contributing

Issues and pull requests are welcome! Share your ideas, bug reports and improvements—community contributions are highly appreciated.

---

## 📄 License

Distributed under the same open-source license as the upstream Unstructured Ingest project.

---

## 📚 References

### ClickZetta Resources
- **[Official Website](https://www.yunqi.tech/)** – product overview and solutions.
- **[Documentation Center](https://www.yunqi.tech/documents/)** – complete technical docs.

### Technical References
- **[Unstructured Docs](https://docs.unstructured.io/)** – upstream documentation.
- **[DashScope API Docs](https://help.aliyun.com/zh/dashscope/)** – Alibaba Cloud API reference.
- **[PyPI Project Page](https://pypi.org/project/unstructured-ingest-clickzetta/)** – package distribution details.

### Source Repositories
- **[GitHub Repository](https://github.com/yunqiqiliang/unstructured-ingest-clickzetta)** – this project.
- **[Upstream Repository](https://github.com/Unstructured-IO/unstructured-ingest)** – Unstructured Ingest.

# 安装指南

## 当前推荐安装方式

### 从GitHub安装（推荐）
```bash
pip install git+https://github.com/yunqiqiliang/unstructured-ingest-clickzetta.git@v1.3.1
```

### 从本地源码安装
```bash
git clone https://github.com/yunqiqiliang/unstructured-ingest-clickzetta.git
cd unstructured-ingest-clickzetta
pip install -e .
```

## 安装验证

安装后可以验证：
```python
import unstructured_ingest
print(f"版本: {unstructured_ingest.__version__}")

# 验证ClickZetta连接器
from unstructured_ingest.processes.connectors.sql.clickzetta import ClickzettaConnectionConfig

# 验证DashScope嵌入器
from unstructured_ingest.processes.embedder import EmbedderConfig
```

## PyPI发布状态

- ✅ V1.3.1版本已发布到GitHub
- ⏳ PyPI发布准备中（metadata兼容性问题需要解决）
- 🔧 构建配置需要调整以支持PyPI上传

## Notebook使用

examples/notebooks/中的Jupyter notebook已更新为使用GitHub安装方式，确保用户可以直接使用最新的V1.3.1版本。
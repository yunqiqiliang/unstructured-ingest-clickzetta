# 安装指南

## 推荐安装方式

### 从PyPI安装（推荐）
```bash
pip install unstructured-ingest-clickzetta
```

### 从GitHub安装最新版本
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

- ✅ 包已发布到PyPI: https://pypi.org/project/unstructured-ingest-clickzetta/
- ✅ 当前可用版本: 1.2.18.dev2
- ✅ V1.3.1版本已发布到GitHub
- ⏳ V1.3.1版本PyPI上传遇到metadata兼容性问题，正在解决

## 版本选择

- **稳定使用**: `pip install unstructured-ingest-clickzetta` (PyPI版本)
- **最新功能**: `pip install git+https://github.com/yunqiqiliang/unstructured-ingest-clickzetta.git@v1.3.1` (GitHub版本)

## Notebook使用

examples/notebooks/中的Jupyter notebook已更新为优先使用PyPI安装，提供GitHub安装作为获取最新功能的选项。
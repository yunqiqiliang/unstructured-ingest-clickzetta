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
- ✅ 当前可用版本: 1.3.1
- ✅ V1.3.1版本已发布到GitHub和PyPI
- ✅ 所有metadata兼容性问题已解决

## 版本选择

- **推荐使用**: `pip install unstructured-ingest-clickzetta` (PyPI最新版本 1.3.1)
- **开发版本**: `pip install git+https://github.com/yunqiqiliang/unstructured-ingest-clickzetta.git` (GitHub最新代码)

## Notebook使用

examples/notebooks/中的Jupyter notebook已更新为优先使用PyPI安装，提供GitHub安装作为获取最新功能的选项。
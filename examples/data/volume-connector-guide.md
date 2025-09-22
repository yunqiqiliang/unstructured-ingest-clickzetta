# ClickZetta Volume 连接器使用指南

## 概述

ClickZetta Volume 连接器是用于访问 ClickZetta Lakehouse 存储卷的专用连接器，支持文件的上传、下载和管理操作。

## 连接配置

### 基本参数
```python
from unstructured_ingest.processes.connectors.fsspec.clickzetta_volume import (
    ClickzettaVolumeConnectionConfig,
    ClickzettaVolumeAccessConfig
)

config = ClickzettaVolumeConnectionConfig(
    access_config=ClickzettaVolumeAccessConfig(
        password="your-password"
    ),
    username="your-username",
    service="your-service-url",
    instance="your-instance-id",
    workspace="your-workspace",
    schema="your-schema",
    vcluster="your-vcluster"
)
```

### 环境变量配置
```bash
# ClickZetta 连接参数
CLICKZETTA_USERNAME=your-username
CLICKZETTA_PASSWORD=your-password
CLICKZETTA_SERVICE=your-service-url
CLICKZETTA_INSTANCE=your-instance-id
CLICKZETTA_WORKSPACE=your-workspace
CLICKZETTA_SCHEMA=your-schema
CLICKZETTA_VCLUSTER=your-vcluster

# Volume 特定参数
CLICKZETTA_VOLUME_PATH=/path/to/volume
CLICKZETTA_VOLUME_BUCKET=your-bucket-name
```

## 支持的操作

### 文件上传
将本地文件上传到 ClickZetta Volume：

```python
from unstructured_ingest.processes.connectors.fsspec.clickzetta_volume import (
    ClickzettaVolumeUploader,
    ClickzettaVolumeUploaderConfig
)

uploader = ClickzettaVolumeUploader(
    connection_config=config,
    upload_config=ClickzettaVolumeUploaderConfig(
        remote_path="/data/documents/"
    )
)
```

### 文件下载
从 ClickZetta Volume 下载文件到本地：

```python
from unstructured_ingest.processes.connectors.fsspec.clickzetta_volume import (
    ClickzettaVolumeDownloader,
    ClickzettaVolumeDownloaderConfig
)

downloader = ClickzettaVolumeDownloader(
    connection_config=config,
    download_config=ClickzettaVolumeDownloaderConfig(
        download_dir="/local/download/path"
    )
)
```

### 文件索引
扫描和索引 Volume 中的文件：

```python
from unstructured_ingest.processes.connectors.fsspec.clickzetta_volume import (
    ClickzettaVolumeIndexer,
    ClickzettaVolumeIndexerConfig
)

indexer = ClickzettaVolumeIndexer(
    connection_config=config,
    index_config=ClickzettaVolumeIndexerConfig(
        remote_path="/data/",
        recursive=True,
        file_glob="*.md"
    )
)
```

## 最佳实践

### 1. 路径管理
- 使用有意义的目录结构组织文件
- 避免在根目录存放大量文件
- 定期清理临时文件

### 2. 性能优化
- 对于大文件，考虑分批处理
- 使用适当的并发参数
- 监控传输速度和错误率

### 3. 安全考虑
- 定期轮换访问凭证
- 使用环境变量存储敏感信息
- 限制Volume访问权限范围

## 故障排除

### 常见错误

**连接超时**
```
解决方案：检查网络连接和服务地址配置
```

**权限不足**
```
解决方案：验证用户权限和Volume访问策略
```

**文件路径错误**
```
解决方案：确认Volume路径格式和文件存在性
```

### 调试模式
启用详细日志以便排查问题：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
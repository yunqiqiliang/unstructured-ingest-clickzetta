# 云器 Lakehouse & DashScope ETL 管道文档

本目录包含了完整的用户文档和技术参考资料，帮助您快速上手并深入使用云器 Lakehouse 和 DashScope ETL 管道。

## 文档结构

### 📚 [用户指南](user_guide.md)
完整的用户指南，包含：
- 快速开始和环境配置
- 三大核心使用场景详解
- CLI 命令参考
- 最佳实践和性能优化
- 故障排查和企业级部署

### 🔧 [API 参考](api_reference.md)
详细的 API 技术文档，包含：
- 所有类和方法的完整签名
- 参数说明和返回值定义
- 配置选项和工具函数
- 性能调优参数
- 测试和验证工具

## 快速导航

### 🚀 新用户开始
1. 阅读 [用户指南 - 概述](user_guide.md#概述)
2. 完成 [环境配置](user_guide.md#环境配置)
3. 运行 [快速开始](user_guide.md#快速开始) 示例

### 💼 使用场景选择
- **SQL 数据处理** → [场景 1: 云器 Lakehouse SQL 数据处理](user_guide.md#场景-1-云器-lakehouse-sql-数据处理)
- **文件系统操作** → [场景 2: 云器 Lakehouse Volume 文件处理](user_guide.md#场景-2-云器-lakehouse-volume-文件处理)
- **端到端管道** → [场景 3: 完整 ETL 管道](user_guide.md#场景-3-完整-etl-管道)

### 🔍 问题解决
- **连接问题** → [故障排查 - 连接失败](user_guide.md#故障排查)
- **性能优化** → [最佳实践 - 性能优化](user_guide.md#最佳实践)
- **API 详情** → [API 参考文档](api_reference.md)

### 🏢 企业部署
- **Docker 部署** → [企业级部署 - Docker 化](user_guide.md#docker-化部署)
- **Kubernetes** → [企业级部署 - K8s 配置](user_guide.md#kubernetes-配置)
- **生产配置** → [企业级部署 - 生产环境](user_guide.md#生产环境配置)

## 核心概念

### 云器 Lakehouse 连接器
- **SQL 连接器**: 用于关系数据库表的批量处理和向量化
- **Volume 连接器**: 用于文件系统操作，支持用户卷、表卷、命名卷

### DashScope 嵌入服务
- **4 个模型版本**: v1/v2/v3/v4，支持不同维度和长度
- **批量处理**: 优化的批处理机制提升性能
- **自动重试**: 内置重试机制保证稳定性

### ETL 管道流程
```
原始数据 → 索引 → 下载 → 处理 → 嵌入 → 上传 → 存储
```

## 支持的数据格式

### 输入格式
- **文档**: PDF, DOCX, TXT, HTML, MD
- **数据**: JSON, NDJSON, CSV, SQL 结果集
- **图像**: PNG, JPG (OCR 处理)

### 输出格式
- **向量数据**: 云器 Lakehouse 表格式
- **文件存储**: 云器 Lakehouse Volume 格式
- **中间结果**: JSON, NDJSON

## 版本信息

- **当前版本**: 1.2.18-dev2
- **Python 要求**: 3.8+
- **云器 Lakehouse**: 所有当前版本
- **DashScope**: API v1

## 获取帮助

### 常见问题
1. **安装问题**: 检查 Python 版本和依赖
2. **连接问题**: 验证环境变量配置
3. **性能问题**: 调整批处理大小
4. **数据问题**: 查看日志和错误信息

### 支持渠道
- 📖 查阅文档: [用户指南](user_guide.md) 和 [API 参考](api_reference.md)
- 🐛 报告问题: 提交 GitHub Issue
- 💡 功能建议: 创建 Feature Request

### 相关链接
- [云器 Lakehouse 官方文档](https://www.yunqi.tech/)
- [DashScope API 文档](https://help.aliyun.com/document_detail/611472.html)

## 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](../LICENSE) 文件。

# Developers' Guide

## Local testing

When testing from a local checkout rather than a pip-installed version of `unstructured`,
just execute `unstructured_ingest/main.py`, e.g.:

    PYTHONPATH=. ./unstructured_ingest/main.py \
       s3 \
       --remote-url s3://utic-dev-tech-fixtures/small-pdf-set/ \
       --anonymous \
       --output-dir s3-small-batch-output \
       --num-processes 2

## Adding Source Data Connectors

To add a source connector, refer to [local.py](unstructured_ingest/processes/connectors/local.py) as an example that implements the two relevant abstract base classes with their associated configs.

If the connector has an available `fsspec` implementation, then refer to [s3.py](unstructured_ingest/processes/connectors/fsspec/s3.py).

Make sure to update the source registry via `add_source_entry` using a unique key for the source type. This will expose it as an available connector.

Create at least one folder [examples/ingest](examples/ingest) with an easily reproducible
script that shows the new connector in action.

Finally, to ensure the connector remains stable, add a new script test_unstructured_ingest/test-ingest-\<the-new-data-source\>.sh similar to [test_unstructured_ingest/test-ingest-s3.sh](test_unstructured_ingest/test-ingest-s3.sh), and append a line invoking the new script in [test_unstructured_ingest/test-ingest.sh](test_unstructured_ingest/test-ingest.sh).

You'll notice that the unstructured outputs for the new documents are expected
to be checked into CI under test_unstructured_ingest/expected-structured-output/\<folder-name-relevant-to-your-dataset\>. So, you'll need to `git add` those json outputs so that `test-ingest.sh` passes in CI.

Double check that the connector is optimized for the best fan out, check [here](#parallel-execution) for more details.

## Adding Destination Data Connectors

To add a source connector, refer to [local.py](unstructured_ingest/processes/connectors/local.py) as an example that implements the uploader abstract base classes with the associated configs.

If the connector has an available `fsspec` implementation, then refer to [s3.py](unstructured_ingest/processes/connectors/fsspec/s3.py).

Make sure to update the destination registry via `add_source_entry` using a unique key for the source type. This will expose it as an available connector.

Similar tests and examples should be added to demonstrate/validate the use of the destination connector similar to the steps laid out for a source connector.

Double check that the connector is optimized for the best fan out, check [here](#parallel-execution) for more details.

### The checklist:

In checklist form, the above steps are summarized as:

- [ ] Create a new file under [connectors/](unstructured_ingest/processes/connectors/) implementing the base classes required depending on if it's a new source or destination connector.
  - [ ] If the IngestDoc relies on a connection or session that could be reused, the subclass of `BaseConnectorConfig` implements a session handle to manage connections. The ConnectorConfig subclass should also inherit from `ConfigSessionHandleMixin` and the IngestDoc subclass should also inherit from `IngestDocSessionHandleMixin`. Check [here](https://github.com/Unstructured-IO/unstructured/pull/1058/files#diff-dae96d30f58cffe1b348c036d006b48bdc7e2e47fbd7c8ec1c45d63face1542d) for a detailed example.
  - [ ] Indexer should fetch appropriate metadata from the source that can be used to reference the doc in the pipeline and detect if there are any changes from what might already exist locally.
  - [ ] Add the relevant decorators from `unstructured.ingest.error` on top of relevant methods to handle errors such as a source connection error, destination connection error, or a partition error.
  - [ ] Register the required information via `add_source_entry` or `add_source_entry` to expose the new connectors.
- [ ] Update the CLI to expose the new connectors via CLI params
  - [ ] Add a new file under [cmds](unstructured_ingest/cli/cmds)
  - [ ] Add the command base classes from the file above in the [__init__.py](unstructured_ingest/cli/cmds/__init__.py). This will expose it in the CLI.
- [ ] Update [unstructured_ingest/cli](unstructured_ingest/cli) with support for the new connector.
- [ ] Create a folder under [examples/ingest](examples/ingest) that includes at least one well documented script.
- [ ] Add a script test_unstructured_ingest/[src|dest\/test-ingest-\<the-new-data-source\>.sh. It's json output files should have a total of no more than 100K.
- [ ] Git add the expected outputs under test_unstructured_ingest/expected-structured-output/\<folder-name-relevant-to-your-dataset\> so the above test passes in CI.
- [ ] Add a line to [test_unstructured_ingest/test-ingest.sh](test_unstructured_ingest/test-ingest.sh) invoking the new test script.
- [ ] Make sure the tests for the connector are running and not skipped by reviewing the logs in CI.
- [ ] If additional python dependencies are needed for the new connector:
  - [ ] Add them as an extra to [setup.py](unstructured/setup.py).
  - [ ] Update the Makefile, adding a target for `install-ingest-<name>` and adding another `pip-compile` line to the `pip-compile` make target. See [this commit](https://github.com/Unstructured-IO/unstructured/commit/ab542ca3c6274f96b431142262d47d727f309e37) for a reference.
  - [ ] The added dependencies should be imported at runtime when the new connector is invoked, rather than as top-level imports.
  - [ ] Add the decorator `unstructured.utils.requires_dependencies` on top of each class instance or function that uses those connector-specific dependencies e.g. for `GitHubConnector` should look like `@requires_dependencies(dependencies=["github"], extras="github")`
  - [ ] Run `make tidy` and `make check` to ensure linting checks pass.
- [ ] Update ingest documentation [here](https://github.com/Unstructured-IO/docs)
- [ ] For team members that are developing in the original repository:
  - [ ] If there are secret variables created for the connector tests, make sure to:
    - [ ] add the secrets into GitHub (contact someone with access)
    - [ ] include the secret variables in [`ci.yml`](https://github.com/Unstructured-IO/unstructured/blob/main/.github/workflows/ci.yml) and [`ingest-test-fixtures-update-pr.yml`](https://github.com/Unstructured-IO/unstructured/blob/main/.github/workflows/ingest-test-fixtures-update-pr.yml)
    - [ ] add a make install line in the workflow configurations to be able to provide the workflow machine with the required dependencies on the connector while testing
    - [ ] Whenever necessary, use the [ingest update test fixtures](https://github.com/Unstructured-IO/unstructured/actions/workflows/ingest-test-fixtures-update-pr.yml) workflow to update the test fixtures.

## Design References

The ingest flow is similar to an ETL pipeline that gets defined at runtime based on user input:

![unstructured ingest diagram](assets/pipeline.png)



### Steps
* `Indexer`: This is responsible for reaching out to the source location and pulling in metadata for each document that will need to be downloaded and processed
* `Downloader`: Using the information generated from the indexer, download the content as files on the local file system for processing. This may require manipulation of the data to prep it for partitioning.
* `Uncompressor`: If enabled, this will look for any supported compressed files (tar and zip are currently supported) and expands those.
* `Partitioner`: Generated the structured enriched content from the local files that have been pulled down. Both local and api-based partitioning is supported, with api-based partitioning set to run async while local set to run via multiprocessing.
* `Chunker`: Optionally chunk the partitioned content. Can also be run locally or via the api, with async/multiprocessing set in the same approach as the partitioner.
* `Embedder`: Create vector embeddings for each element in the structured output. Most of these are via an API call (i.e. AWS Bedrock) and therefor run async but can also use a local huggingface model which will run via multiprocessing.
* `Stager`: This is an optional step that won't apply for most pipelines. If the data needs to be modified from the existing structure to better support the upload, such as converting it to a csv for tabular-based destinations (sql dbs).
* `Uploader`: Write the local content to the destination. If none if provided, the local one will be used which writes the final result to a location on the local filesystem. If batch uploads are needed, this will run in a single process with access to all docs. If batch is not supported, all docs can be uploaded concurrently using the async approach.


### Sequence Diagram
![unstructured ingest sequence diagram](assets/sequence.png)


### Parallel Execution
For each step in the pipeline, a choice can be made when to run it async to support IO heavy tasks vs multiprocessing for CPU heavy loads. This choice should be make with care
because if enabling async, that code will be run in a single process with the assumption that the async support will provide better fan out and optimization that running the code
with a multiprocessing pool fan out. If the underlying code is completely blocking but the async flag is enabled, this will run as if it's a normal for loop and will get worse performance
than is simply run via multiprocessing. One option to help support IO heavy tasks that might not support async yet is wrapping it in a `run_in_executor()` call. Otherwise, it is common that
the underlying SDKs have an async version to run the same network calls without blocking the event loop.

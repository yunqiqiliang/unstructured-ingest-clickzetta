# ClickZetta 多Lakehouse知识库批量部署系统

## 概述

该系统用于将文档知识库批量部署到多个ClickZetta Lakehouse实例。支持自动创建schema、管理表结构、处理文档向量化，并提供串行和并行两种部署模式。

## 快速开始

```bash
# 进入项目目录
cd /Users/liangmo/Documents/GitHub/unstructured-ingest-clickzetta

# 如果你已经激活了虚拟环境（如 unstructured311）
./multi_lakehouse_kb_builder/run_with_current_env.sh deploy

# 或者使用Makefile（会自动检测当前环境）
make -C multi_lakehouse_kb_builder deploy

# 或者直接运行Python脚本
python multi_lakehouse_kb_builder/deploy_kb_simple.py
```

## 功能特性

- ✅ 从`~/.clickzetta/connections.json`自动读取所有Lakehouse连接
- ✅ 自动检查并创建`clickzetta_doc_kb` schema
- ✅ 智能管理Raw表和Silver表（清空数据但不删除表结构）
- ✅ 使用DashScope（通义千问）进行文档向量化
- ✅ 支持串行和并行批量部署
- ✅ 详细的部署日志和错误处理
- ✅ 部署结果统计和报告
- ✅ **自动数据验证**：
  - 验证Raw表和Silver表的行数是否匹配
  - 检测全零或问题向量（超过50%为0的向量）
  - 验证所有向量维度的一致性（1024维）
- ✅ **连接和健康检查**：
  - 检查所有Lakehouse连接的可用性
  - 诊断知识库的部署状态
  - 分析向量数据质量和分布
  - 生成健康评分和问题报告
- ✅ **知识库管理**：
  - 添加自定义知识条目
  - 支持交互式输入和文件导入
  - 批量更新多个Lakehouse
  - 支持JSON、CSV、TXT等格式

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│              知识库批量部署系统                       │
├─────────────────────────────────────────────────────┤
│  1. 连接配置读取器                                   │
│     └─> 读取 ~/.clickzetta/connections.json         │
│                                                     │
│  2. Lakehouse环境准备器                             │
│     ├─> 检查/创建 clickzetta_doc_kb schema         │
│     └─> 检查/清空 Raw表 和 Silver表                 │
│                                                     │
│  3. 知识库构建器                                    │
│     ├─> 文档处理 (分片、向量化)                     │
│     ├─> 数据写入 Raw表                              │
│     └─> 数据转换写入 Silver表                       │
│                                                     │
│  4. 批量执行控制器                                  │
│     ├─> 并行/串行执行策略                           │
│     └─> 错误处理和重试机制                          │
└─────────────────────────────────────────────────────┘
```

## 前置要求

### 1. 安装 uv

本项目使用 `uv` 作为Python包管理器：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 brew (macOS)
brew install uv
```

### 2. 环境依赖

环境依赖会通过 `uv sync` 自动安装，主要包括：
- Python 3.11（必须）
- clickzetta
- dashscope  
- pandas
- unstructured-ingest（从本地wheel安装）

### 3. 配置文件

确保 `~/.clickzetta/connections.json` 存在且格式正确：

```json
{
  "system_config": {
    "embedding": {
      "dashscope": {
        "api_key": "sk-your-dashscope-api-key"
      }
    }
  },
  "connections": [
    {
      "connection_name": "production",
      "service": "api.clickzetta.com",
      "username": "your_username",
      "password": "your_password",
      "instance": "your_instance",
      "workspace": "default",
      "schema": "default",
      "vcluster": "default"
    }
  ]
}
```

API密钥优先级：
1. 配置文件中的 `system_config.embedding.dashscope.api_key`
2. 环境变量 `DASHSCOPE_API_KEY`
3. 默认值（仅用于测试）

### 4. 环境变量

```bash
# DashScope API密钥（可选，建议在配置文件中设置）
export DASHSCOPE_API_KEY="sk-your-api-key"
```

### 5. 文档目录

默认文档路径：`/Users/liangmo/yunqidoc/cn_markdown_20250526`

## 使用方法

### 方法一：在已激活的虚拟环境中使用（如 unstructured311）

```bash
# 进入项目目录
cd /Users/liangmo/Documents/GitHub/unstructured-ingest-clickzetta

# 使用当前环境运行脚本
./multi_lakehouse_kb_builder/run_with_current_env.sh deploy

# 或直接运行Python脚本
python multi_lakehouse_kb_builder/deploy_kb_simple.py
python multi_lakehouse_kb_builder/validate_kb_simple.py

# 或使用Makefile（会自动检测当前环境）
make -C multi_lakehouse_kb_builder deploy
make -C multi_lakehouse_kb_builder validate
make -C multi_lakehouse_kb_builder help
```

### 方法二：使用启动脚本

```bash
# 智能启动脚本（自动检测环境）
./multi_lakehouse_kb_builder/run.sh

# 快速启动脚本（直接使用.venv）
./multi_lakehouse_kb_builder/run_quick.sh deploy
```

启动脚本会自动：
1. 检测是否已有虚拟环境
2. 仅在需要时同步依赖
3. 显示交互式菜单

### 方法三：使用 uv 直接运行

#### 1. 测试环境

```bash
# 设置Python版本
uv python pin 3.11

# 同步环境
uv sync

# 运行测试
uv run python multi_lakehouse_kb_builder/test_kb_deployment.py
```

#### 2. 简单部署（交互式）

```bash
uv run python multi_lakehouse_kb_builder/deploy_kb_simple.py
```

选项说明：
- **选项1**：串行部署到所有Lakehouse（推荐，更稳定）
- **选项2**：并行部署到所有Lakehouse（更快，但需要注意资源）
- **选项3**：部署到特定Lakehouse（通过名称匹配）
- **选项4**：测试模式（只部署到第一个连接）

#### 3. 高级部署

使用命令行参数进行更精细的控制：

```bash
# 串行部署到所有Lakehouse
uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --mode serial

# 并行部署（5个工作线程）
uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --mode parallel --workers 5

# 只部署到包含"prod"的连接
uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --filter prod

# 排除测试环境
uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --exclude test

# 使用自定义文档目录
uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --docs /path/to/your/docs
```

### 方法四：Python代码使用

```python
from multi_lakehouse_kb_builder import BatchKnowledgeBaseDeployer

# 创建部署器
deployer = BatchKnowledgeBaseDeployer(
    config_path="~/.clickzetta/connections.json",
    doc_path="/Users/liangmo/yunqidoc/cn_markdown_20250526",
    execution_mode="serial"  # 或 "parallel"
)

# 部署到所有Lakehouse
results = deployer.deploy_to_all_lakehouse()

# 打印结果
deployer.print_summary(results)
```

### 数据验证

部署完成后，系统会自动进行数据验证。您也可以单独运行验证：

#### 交互式验证

```bash
uv run python multi_lakehouse_kb_builder/validate_kb_simple.py
```

#### 命令行验证

```bash
# 验证所有Lakehouse
uv run python multi_lakehouse_kb_builder/kb_data_validator.py

# 验证特定Lakehouse
uv run python multi_lakehouse_kb_builder/kb_data_validator.py --filter production

# 保存验证报告
uv run python multi_lakehouse_kb_builder/kb_data_validator.py --output validation_report.json
```

#### Python代码验证

```python
from multi_lakehouse_kb_builder import BatchKnowledgeBaseValidator

# 创建验证器
validator = BatchKnowledgeBaseValidator()

# 执行验证
results = validator.validate_all_deployments()

# 打印总结
validator.print_validation_summary(results)

# 保存详细报告
validator.save_validation_results(results, "validation_report.json")
```

### 连接和健康检查

#### 交互式检查

```bash
# 使用菜单选择检查类型
python multi_lakehouse_kb_builder/check_connections.py
```

#### 直接运行检查

```bash
# 检查所有连接
python multi_lakehouse_kb_builder/run_direct.py check-conn

# 检查知识库健康状态
python multi_lakehouse_kb_builder/run_direct.py check-kb

# 完整诊断
python multi_lakehouse_kb_builder/run_direct.py check

# 管理知识库内容
python multi_lakehouse_kb_builder/run_direct.py manage-kb
```

### 知识库内容管理

可以对已部署的知识库进行内容管理，包括添加自定义知识、删除过时内容、搜索现有知识等。

#### 交互式管理

```bash
# 启动知识库管理工具
python multi_lakehouse_kb_builder/manage_knowledge_simple.py
```

支持的操作：
1. **添加单条知识**：交互式输入知识内容
2. **从文件批量添加**：支持JSON/CSV/TXT格式
3. **搜索知识**：按关键词或来源筛选
4. **删除知识**：按ID、来源或批量删除
5. **查看统计信息**：显示各来源的知识数量

#### 支持的文件格式

**JSON格式示例**：
```json
[
  {
    "text": "ClickZetta是云器、Singdata的技术品牌。云器是ClickZetta的中国商业公司，Singdata是ClickZetta的海外商业公司。",
    "source": "CompanyInfo",
    "languages": ["zh-cn"]
  },
  {
    "text": "Lakehouse支持SQL查询、实时分析和机器学习等多种数据处理场景。",
    "source": "ProductInfo",
    "languages": ["zh-cn"]
  }
]
```

**CSV格式示例**：
```csv
text,source,languages
"ClickZetta提供高性能的向量搜索功能，支持余弦相似度计算。","TechnicalDoc","['zh-cn']"
"云器科技总部位于中国，Singdata是其海外品牌。","CompanyInfo","['zh-cn']"
```

**TXT格式**：每行一条知识
```
ClickZetta Lakehouse是一个云原生的数据仓库平台。
支持结构化、半结构化和非结构化数据的统一管理。
提供高性能的向量搜索和相似度计算功能。
```

#### Python代码使用

```python
from multi_lakehouse_kb_builder.kb_knowledge_manager import (
    KnowledgeEntry, 
    BatchKnowledgeManager
)

# 创建知识条目
entry = KnowledgeEntry(
    text="ClickZetta是云器科技的技术品牌，提供云原生数据仓库解决方案。",
    source="CompanyInfo",
    languages=["zh-cn"]
)

# 创建批量管理器
manager = BatchKnowledgeManager()

# 添加到所有Lakehouse
results = manager.add_to_all_lakehouse([entry])

# 搜索知识
search_results = manager.search_across_lakehouse(query="ClickZetta")

# 删除知识
delete_results = manager.delete_from_all_lakehouse(["knowledge_id"])

# 获取统计信息
stats = manager.get_all_statistics()
```

#### 使用场景

1. **添加公司信息**：如ClickZetta/云器/Singdata的关系说明
2. **添加FAQ**：常见问题和解答
3. **添加最佳实践**：使用技巧和经验总结
4. **删除过时内容**：清理不再适用的知识
5. **统计分析**：了解知识库内容分布

##### 交互式添加

```bash
uv run python multi_lakehouse_kb_builder/add_knowledge_simple.py
```

选项说明：
1. **交互式输入**：直接输入知识内容
2. **从文件导入**：支持JSON、CSV、TXT格式
3. **创建示例文件**：生成示例模板
4. **预定义知识**：内置常用知识条目

##### 命令行方式

```bash
# 添加单条知识
uv run python multi_lakehouse_kb_builder/kb_knowledge_adder.py --text "ClickZetta支持实时数据同步"

# 从文件添加
uv run python multi_lakehouse_kb_builder/kb_knowledge_adder.py --file knowledge.json

# 创建示例文件
uv run python multi_lakehouse_kb_builder/kb_knowledge_adder.py --create-sample
```

##### 知识文件格式

**JSON格式**：
```json
[
  {
    "text": "知识内容",
    "source": "CompanyInfo",
    "languages": ["zh-cn"]
  }
]
```

**CSV格式**：
```csv
text,source
"TRUNCATE TABLE语句用于清空表中的所有数据","SQLReference"
```

**TXT格式**：每行一条知识

#### Python代码检查

```python
from multi_lakehouse_kb_builder.check_connections import ConnectionChecker, KnowledgeBaseHealthChecker

# 检查连接
checker = ConnectionChecker()
conn_results = checker.check_all_connections()
checker.print_summary(conn_results)

# 检查知识库健康
kb_checker = KnowledgeBaseHealthChecker()
kb_results = kb_checker.check_all_kb_health()
kb_checker.print_health_summary(kb_results)
```

## 表结构说明

### Raw表
- 表名：`dashscope_v4_1024_2048_20250611_yunqi_raw_elements`
- 用途：存储原始处理的文档数据和向量

### Silver表
- 表名：`dashscope_v4_1024_2048_20250611_yunqi_elements`
- 用途：存储清洗后的数据，包含索引
- 索引：
  - 倒排索引：用于文本搜索
  - 向量索引：用于相似度搜索

## 部署流程

1. **环境准备阶段**
   - 读取连接配置
   - 验证连接参数
   - 检查文档目录

2. **Schema管理阶段**
   - 检查`clickzetta_doc_kb` schema是否存在
   - 如不存在则创建

3. **表管理阶段**
   - 检查Raw表和Silver表是否存在
   - 如存在则执行`TRUNCATE TABLE`清空数据
   - 如不存在则创建新表

4. **知识库构建阶段**
   - 读取文档文件
   - 文档分片处理
   - 使用DashScope生成向量
   - 写入Raw表
   - 执行数据转换到Silver表

5. **数据验证阶段**（自动执行）
   - 验证Raw表和Silver表行数是否匹配
   - 检测问题向量（全零或大量零值）
   - 验证向量维度一致性

6. **结果汇总阶段**
   - 收集统计信息
   - 包含验证结果
   - 生成部署报告
   - 保存结果到JSON文件

## 注意事项

### 安全性
- ⚠️ **绝对不会删除表结构**，只会清空数据
- 建议在部署前备份重要数据
- 密码等敏感信息应妥善保管

### 性能优化
- 串行模式更稳定，适合首次部署
- 并行模式更快，但注意API限流
- 建议并行工作线程数不超过5个

### 错误处理
- 单个Lakehouse部署失败不影响其他
- 所有错误都会记录在日志文件中
- 部署结果会保存到JSON文件便于分析

## 注意事项

### VECTOR_DIM 函数支持
本系统不依赖 VECTOR_DIM 函数。在验证向量维度时，我们使用采样方法直接计算向量长度，避免了不同 ClickZetta 实例版本差异的问题。

### 文件组织
- **logs/** - 所有运行日志
- **reports/** - 所有生成的报告
- 这些目录已被添加到 .gitignore，不会被提交到版本控制

## 常见问题

### Q1: 如何只部署到特定的Lakehouse？
使用`--filter`参数指定连接名称模式：
```bash
uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --filter production
```

### Q2: 部署失败如何重试？
系统会保存部署结果，可以根据失败的连接名称单独重试：
```bash
uv run python multi_lakehouse_kb_builder/deploy_kb_simple.py
# 选择选项3，输入失败的连接名称
```

### Q3: 如何查看详细的部署日志？
日志文件会自动生成在当前目录：
```bash
kb_deployment_20250612_142530.log
```

### Q4: 如何验证部署结果？
部署完成后会自动进行数据验证，包括：
- Raw表记录数
- Silver表记录数
- 有嵌入向量的记录数
- 行数匹配验证
- 问题向量检测
- 向量维度验证

您也可以单独运行验证：
```bash
uv run python multi_lakehouse_kb_builder/validate_kb_simple.py
```

## 输出文件

所有日志和报告文件都会自动保存到对应的子目录中：

### logs/ 目录
- **部署日志**：`kb_deployment_YYYYMMDD_HHMMSS.log`
- 其他运行日志

### reports/ 目录
- **部署结果**：`kb_deployment_result_YYYYMMDD_HHMMSS.json`
- **验证报告**：`kb_validation_report_YYYYMMDD_HHMMSS.json`
- **连接检查**：`connection_check_YYYYMMDD_HHMMSS.json`
- **健康报告**：`kb_health_check_YYYYMMDD_HHMMSS.json`
- **完整诊断**：`full_diagnostic_YYYYMMDD_HHMMSS.json`

### 整理历史文件
如果你有旧的日志和报告文件在主目录中，可以运行：
```bash
python multi_lakehouse_kb_builder/organize_files.py
```

## 技术支持

如遇到问题，请检查：
1. 运行测试脚本确保环境正确
2. 查看日志文件了解详细错误
3. 确认连接配置和权限正确
4. 确认DashScope API可用
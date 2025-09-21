# ClickZetta 多Lakehouse知识库批量部署系统

## 概述

该系统用于将文档知识库批量部署到多个ClickZetta Lakehouse实例。支持自动创建schema、管理表结构、处理文档向量化，并提供串行和并行两种部署模式。

## 快速开始

### 📋 前置检查

在开始之前，请确保以下条件满足：

```bash
# 1. 检查项目根目录
pwd  # 应该在 unstructured-ingest-clickzetta 目录下

# 2. 检查配置文件
ls ~/.clickzetta/connections.json  # 配置文件必须存在

# 3. 检查文档目录（可选，如有自定义文档）
export LOCAL_FILE_INPUT_DIR="/path/to/your/documents"  # 设置文档路径
ls $LOCAL_FILE_INPUT_DIR  # 验证文档目录存在

# 4. 检查API密钥
echo $DASHSCOPE_API_KEY  # 验证API密钥已设置
```

### 🚀 快速启动

我们提供了简化的启动方式：

```bash
# 推荐：智能启动脚本（自动检测和适配环境）
./multi_lakehouse_kb_builder/run.sh

# 或者直接执行命令
./multi_lakehouse_kb_builder/run.sh deploy

# 命令行接口（适合脚本调用）
python multi_lakehouse_kb_builder/run_direct.py deploy

# 传统方式：Makefile
make -C multi_lakehouse_kb_builder deploy
```

### 🧪 一键验证脚本

在开始部署前，运行验证脚本确保环境正常：

```bash
# 创建并运行验证脚本
cat > validate_environment.sh << 'EOF'
#!/bin/bash
echo "🔍 开始环境验证..."

# 检查Python版本
echo "1. Python版本检查:"
python --version 2>/dev/null || echo "❌ Python未安装"

# 检查配置文件
echo "2. 配置文件检查:"
if [ -f ~/.clickzetta/connections.json ]; then
    echo "✅ connections.json存在"
    python -c "import json; json.load(open('$HOME/.clickzetta/connections.json'))" 2>/dev/null && echo "✅ JSON格式正确" || echo "❌ JSON格式错误"
else
    echo "❌ 配置文件不存在: ~/.clickzetta/connections.json"
fi

# 检查环境变量
echo "3. 环境变量检查:"
[ -n "$DASHSCOPE_API_KEY" ] && echo "✅ DASHSCOPE_API_KEY已设置" || echo "⚠️  DASHSCOPE_API_KEY未设置"
[ -n "$LOCAL_FILE_INPUT_DIR" ] && echo "✅ LOCAL_FILE_INPUT_DIR已设置: $LOCAL_FILE_INPUT_DIR" || echo "⚠️  LOCAL_FILE_INPUT_DIR未设置，将使用默认路径"

# 检查文档目录
echo "4. 文档目录检查:"
DOC_DIR=${LOCAL_FILE_INPUT_DIR:-"./documents"}
if [ -d "$DOC_DIR" ]; then
    FILE_COUNT=$(find "$DOC_DIR" -type f | wc -l)
    echo "✅ 文档目录存在: $DOC_DIR ($FILE_COUNT 个文件)"
else
    echo "❌ 文档目录不存在: $DOC_DIR"
fi

# 检查依赖
echo "5. 关键依赖检查:"
python -c "import clickzetta" 2>/dev/null && echo "✅ clickzetta" || echo "❌ clickzetta未安装"
python -c "import dashscope" 2>/dev/null && echo "✅ dashscope" || echo "❌ dashscope未安装"
python -c "import pandas" 2>/dev/null && echo "✅ pandas" || echo "❌ pandas未安装"

echo "🏁 环境验证完成!"
EOF

chmod +x validate_environment.sh
./validate_environment.sh
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

#### 自动环境检查

运行以下脚本来检查和验证环境：

```bash
# 检查Python版本（推荐3.11+）
python --version

# 检查是否有uv
which uv || echo "需要安装uv: curl -LsSf https://astral.sh/uv/install.sh | sh"

# 检查虚拟环境
echo "当前环境: $VIRTUAL_ENV"
```

#### 依赖包

环境依赖会通过 `uv sync` 自动安装，主要包括：
- **Python 3.11+**（推荐版本）
- **clickzetta** - ClickZetta连接器
- **dashscope** - 阿里云通义千问API
- **pandas** - 数据处理
- **unstructured-ingest** - 文档处理（从本地安装）

#### 自动安装依赖

```bash
# 使用uv自动同步依赖
uv sync

# 或手动安装核心依赖
pip install clickzetta dashscope pandas python-dotenv
```

### 3. 配置文件

#### 创建配置文件

确保 `~/.clickzetta/connections.json` 存在且格式正确：

```bash
# 创建配置目录
mkdir -p ~/.clickzetta

# 检查配置文件是否存在
if [ ! -f ~/.clickzetta/connections.json ]; then
    echo "⚠️  配置文件不存在，请创建 ~/.clickzetta/connections.json"
    echo "📖 参考下面的配置模板"
fi
```

#### 配置文件模板

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
      "service": "your-service-url.com",
      "username": "your_username",
      "password": "your_password",
      "instance": "your_instance_id",
      "workspace": "your_workspace",
      "schema": "your_schema",
      "vcluster": "your_vcluster"
    },
    {
      "connection_name": "development",
      "service": "dev-service-url.com",
      "username": "dev_username",
      "password": "dev_password",
      "instance": "dev_instance_id",
      "workspace": "dev_workspace",
      "schema": "dev_schema",
      "vcluster": "dev_vcluster"
    }
  ]
}
```

#### 配置验证

```bash
# 验证配置文件格式
python -c "
import json
try:
    with open('~/.clickzetta/connections.json'.replace('~', '$HOME'), 'r') as f:
        config = json.load(f)
    print('✅ 配置文件格式正确')
    print(f'📊 发现 {len(config.get(\"connections\", []))} 个连接配置')
except Exception as e:
    print(f'❌ 配置文件错误: {e}')
"
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

#### 配置文档路径

支持多种方式配置文档目录：

```bash
# 方法1：设置环境变量（推荐）
export LOCAL_FILE_INPUT_DIR="/path/to/your/documents"

# 方法2：在.env文件中配置
echo "LOCAL_FILE_INPUT_DIR=/path/to/your/documents" >> .env

# 方法3：使用默认路径（如果存在）
# 系统会依次检查以下路径：
#   1. $LOCAL_FILE_INPUT_DIR
#   2. ~/Documents
#   3. ./documents
#   4. 当前目录
```

#### 验证文档目录

```bash
# 检查文档目录是否存在且包含文件
if [ -d "$LOCAL_FILE_INPUT_DIR" ]; then
    echo "✅ 文档目录存在: $LOCAL_FILE_INPUT_DIR"
    echo "📁 文件数量: $(find "$LOCAL_FILE_INPUT_DIR" -type f | wc -l)"
else
    echo "❌ 文档目录不存在，请设置 LOCAL_FILE_INPUT_DIR 环境变量"
fi
```

#### 支持的文档格式

- **Markdown文件** (*.md, *.markdown)
- **PDF文档** (*.pdf)
- **Word文档** (*.docx, *.doc)
- **文本文件** (*.txt)
- **HTML文件** (*.html, *.htm)
- **其他格式** - 参考unstructured支持的格式

## 使用方法

### 方法一：智能启动脚本（推荐）

新版本提供了智能启动脚本，自动检测和适配不同的Python环境：

```bash
# 1. 进入项目目录
cd unstructured-ingest-clickzetta  # 替换为你的实际项目路径

# 2. 运行智能启动脚本
./multi_lakehouse_kb_builder/run.sh

# 3. 或直接执行命令
./multi_lakehouse_kb_builder/run.sh deploy    # 交互式部署
./multi_lakehouse_kb_builder/run.sh test      # 环境测试
./multi_lakehouse_kb_builder/run.sh validate  # 数据验证
./multi_lakehouse_kb_builder/run.sh check     # 健康检查
```

**智能环境检测** - 脚本会按优先级自动选择：
1. 🎯 当前激活的虚拟环境 (`$VIRTUAL_ENV`)
2. 📁 项目本地 `.venv` 环境
3. 🛠️ uv 管理的环境（自动同步依赖）
4. 🐍 系统 Python3（不推荐生产使用）

### 方法二：命令行接口

适合脚本调用和自动化场景：

```bash
# 直接命令行调用（无交互界面）
python multi_lakehouse_kb_builder/run_direct.py deploy
python multi_lakehouse_kb_builder/run_direct.py check-conn
python multi_lakehouse_kb_builder/run_direct.py manage-kb

# 查看可用命令
python multi_lakehouse_kb_builder/run_direct.py
```


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
export LOCAL_FILE_INPUT_DIR="/path/to/your/docs"
uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py
```

### 方法四：Python代码使用

```python
from multi_lakehouse_kb_builder import BatchKnowledgeBaseDeployer

# 创建部署器
deployer = BatchKnowledgeBaseDeployer(
    config_path="~/.clickzetta/connections.json",
    doc_path=os.getenv("LOCAL_FILE_INPUT_DIR", "./documents"),  # 使用环境变量或默认路径
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

## 故障排除和常见问题

### 🔧 环境问题

#### Q1: Python版本不兼容
```bash
# 检查Python版本
python --version
# 如果版本低于3.11，请升级Python或使用pyenv管理版本
pyenv install 3.11.9
pyenv local 3.11.9
```

#### Q2: 依赖安装失败
```bash
# 清理并重新安装依赖
rm -rf .venv
uv sync --reinstall

# 或手动安装关键依赖
pip install --upgrade pip
pip install clickzetta dashscope pandas
```

### 🔗 连接问题

#### Q3: ClickZetta连接失败
```bash
# 1. 检查网络连通性
ping your-service-url.com

# 2. 验证连接参数
python -c "
from clickzetta.zettapark.session import Session
# 使用你的实际连接参数测试
config = {
    'username': 'your_username',
    'password': 'your_password',
    'service': 'your_service',
    'instance': 'your_instance',
    'workspace': 'your_workspace',
    'schema': 'your_schema',
    'vcluster': 'your_vcluster'
}
try:
    session = Session.builder.configs(config).create()
    print('✅ 连接成功')
except Exception as e:
    print(f'❌ 连接失败: {e}')
"
```

#### Q4: DashScope API错误
```bash
# 检查API密钥
echo $DASHSCOPE_API_KEY
# 验证API密钥有效性
python -c "
import dashscope
import os
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
from dashscope import TextEmbedding
try:
    response = TextEmbedding.call(model='text-embedding-v4', input='测试')
    print('✅ DashScope API正常')
except Exception as e:
    print(f'❌ API错误: {e}')
"
```

### 📁 文件和配置问题

#### Q5: 配置文件格式错误
```bash
# 验证JSON格式
python -m json.tool ~/.clickzetta/connections.json
# 如果报错，检查JSON语法
```

#### Q6: 文档目录为空或不存在
```bash
# 检查并创建文档目录
mkdir -p "$LOCAL_FILE_INPUT_DIR"
# 复制示例文档
cp -r examples/sample_docs/* "$LOCAL_FILE_INPUT_DIR/"
```

### 🚀 部署问题

#### Q7: 如何只部署到特定的Lakehouse？
```bash
# 使用filter参数
uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --filter production

# 或使用interactive模式
python multi_lakehouse_kb_builder/deploy_kb_simple.py
# 选择选项3，输入连接名称
```

#### Q8: 部署失败如何重试？
```bash
# 1. 查看失败日志
tail -f logs/kb_deployment_*.log

# 2. 单独重试失败的连接
python multi_lakehouse_kb_builder/deploy_kb_simple.py
# 选择选项3，输入失败的连接名称

# 3. 清理并重新部署
python -c "
from multi_lakehouse_kb_builder.multi_lakehouse_kb_builder import BatchKnowledgeBaseDeployer
deployer = BatchKnowledgeBaseDeployer()
deployer.cleanup_failed_deployments()  # 清理失败的部署
"
```

#### Q9: 向量维度不匹配
```bash
# 检查向量维度配置
grep -r "embeddings_dimensions" multi_lakehouse_kb_builder/
# 确保所有地方都使用1024维度（DashScope v4）
```

#### Q10: 内存不足
```bash
# 减少并行度
export MAX_WORKERS=2  # 默认是5
# 或使用串行模式
python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --mode serial
```

### 📊 验证和监控

#### Q11: 如何验证部署结果？
```bash
# 自动验证
python multi_lakehouse_kb_builder/validate_kb_simple.py

# 手动检查表数据
python -c "
# 连接到数据库检查记录数
# ... 验证代码
"
```

#### Q12: 如何查看详细日志？
```bash
# 查看最新部署日志
ls -lt logs/kb_deployment_*.log | head -1
tail -f logs/kb_deployment_$(date +%Y%m%d)_*.log

# 查看错误日志
grep -i error logs/kb_deployment_*.log
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